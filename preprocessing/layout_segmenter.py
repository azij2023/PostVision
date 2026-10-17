"""
Document Layout Analysis & Address Line Segmentation for Postal Mail Images.
Detects destination address blocks, segments individual text lines,
deskews text, and normalizes crops for input into the scratch neural network.
"""

from typing import List, Tuple, Dict, Any, Optional
import cv2
import numpy as np
from PIL import Image


class PostalLayoutSegmenter:
    """
    Analyzes postal envelope images and segments address text lines.
    Pipeline:
    1. Grayscale & Contrast Adaptive Enhancement
    2. Morphological Horizontal Word & Line Grouping
    3. Postal Artifact Filtering (ignoring stamps, cancellation marks, borders)
    4. Top-to-bottom Line Ordering & Normalization
    """

    def __init__(self, target_line_height: int = 32):
        self.target_line_height = target_line_height

    def preprocess_image(self, img_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert to grayscale, enhance ink contrast with CLAHE, and compute binary mask.
        """
        if len(img_bgr.shape) == 3:
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        else:
            gray = img_bgr.copy()

        # CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        # Bilateral filter to smooth paper grain while retaining ink edges
        denoised = cv2.bilateralFilter(enhanced, 7, 50, 50)

        # Otsu thresholding
        _, binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        return denoised, binary

    def segment_envelope(
        self,
        envelope_img: Image.Image
    ) -> Tuple[List[Dict[str, Any]], np.ndarray]:
        """
        Segments text lines from an envelope image.
        Returns:
            lines: list of dicts with keys 'line_index', 'crop_image' (PIL), 'bbox' (x, y, w, h)
            annotated_img: RGB numpy array showing detected bounding boxes
        """
        # Convert PIL to BGR numpy array
        np_img = np.array(envelope_img)
        if len(np_img.shape) == 2:
            img_bgr = cv2.cvtColor(np_img, cv2.COLOR_GRAY2BGR)
        elif np_img.shape[2] == 4:
            img_bgr = cv2.cvtColor(np_img, cv2.COLOR_RGBA2BGR)
        else:
            img_bgr = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)

        h_img, w_img = img_bgr.shape[:2]
        denoised, binary = self.preprocess_image(img_bgr)

        # Horizontal dilation kernel to fuse individual characters and words into continuous line bars
        kernel_w = max(15, int(w_img * 0.035))
        kernel_h = max(2, int(h_img * 0.007))
        morph_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_w, kernel_h))
        dilated = cv2.morphologyEx(binary, cv2.MORPH_DILATE, morph_kernel, iterations=2)

        # Find connected components / contours
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        candidate_boxes = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            area = w * h
            aspect_ratio = w / float(h)

            # Filter criteria for address text lines:
            # - Must have sufficient width (at least 60px or 10% of envelope width)
            # - Aspect ratio should be elongated horizontally (> 2.0)
            # - Exclude full-envelope border or tiny noise specks
            # - Ignore upper right stamp region if aspect ratio is roughly square (stamp is ~ 1:1.3)
            is_stamp = (x > w_img * 0.65) and (y < h_img * 0.35) and (aspect_ratio < 2.0)

            if w > 60 and h > 10 and h < h_img * 0.25 and area > 600 and not is_stamp:
                candidate_boxes.append((x, y, w, h))

        # If morphological segmentation didn't find clear lines (e.g. clean crop already),
        # fallback to horizontal projection profile
        if not candidate_boxes:
            candidate_boxes = self._projection_segmentation(binary, w_img, h_img)

        # Sort candidate lines top-to-bottom by vertical Y coordinate
        candidate_boxes = sorted(candidate_boxes, key=lambda b: b[1])

        # Filter overlapping or near-identical boxes
        filtered_boxes = []
        for box in candidate_boxes:
            if not filtered_boxes:
                filtered_boxes.append(box)
            else:
                prev_x, prev_y, prev_w, prev_h = filtered_boxes[-1]
                curr_x, curr_y, curr_w, curr_h = box
                # Check vertical overlap
                overlap = max(0, min(prev_y + prev_h, curr_y + curr_h) - max(prev_y, curr_y))
                if overlap > 0.5 * min(prev_h, curr_h):
                    # Merge boxes
                    nx = min(prev_x, curr_x)
                    ny = min(prev_y, curr_y)
                    nw = max(prev_x + prev_w, curr_x + curr_w) - nx
                    nh = max(prev_y + prev_h, curr_y + curr_h) - ny
                    filtered_boxes[-1] = (nx, ny, nw, nh)
                else:
                    filtered_boxes.append(box)

        # Create line crops and annotated image
        annotated_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        line_records = []

        colors = [(220, 38, 38), (37, 99, 235), (16, 185, 129), (147, 51, 234), (245, 158, 11)]

        for idx, (x, y, w, h) in enumerate(filtered_boxes):
            # Spatial classification: Top-left is return address ("From"), Center/Lower-right is Destination ("To")
            is_return = (x < w_img * 0.40) and (y < h_img * 0.35)
            block_type = "return_address" if is_return else "destination_address"

            # Add padding
            pad_x = 8
            pad_y = 4
            x1 = max(0, x - pad_x)
            y1 = max(0, y - pad_y)
            x2 = min(w_img, x + w + pad_x)
            y2 = min(h_img, y + h + pad_y)

            # Crop from original grayscale/denoised
            crop_arr = denoised[y1:y2, x1:x2]
            if crop_arr.size == 0:
                continue

            # Resize to target height while preserving aspect ratio
            crop_pil = Image.fromarray(crop_arr)
            c_w, c_h = crop_pil.size
            if c_h > 0:
                aspect = c_w / float(c_h)
                new_w = max(24, int(round(self.target_line_height * aspect)))
                norm_crop = crop_pil.resize((new_w, self.target_line_height), Image.Resampling.BILINEAR)
            else:
                norm_crop = crop_pil

            line_records.append({
                "line_index": idx + 1,
                "crop_image": norm_crop,
                "bbox": (x1, y1, x2 - x1, y2 - y1),
                "block_type": block_type
            })

            # Draw on annotated image
            color = (16, 185, 129) if not is_return else (245, 158, 11)  # Green for destination, Amber for return
            cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, 2)
            label = f"{'TO' if not is_return else 'FROM'} L{idx + 1}"
            cv2.putText(annotated_img, label, (x1, max(15, y1 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)

        return line_records, annotated_img

    def _projection_segmentation(self, binary: np.ndarray, w_img: int, h_img: int) -> List[Tuple[int, int, int, int]]:
        """Fallback line segmentation using horizontal projection profile."""
        proj = np.sum(binary > 0, axis=1)
        threshold = np.mean(proj) * 0.2

        in_line = False
        start_y = 0
        boxes = []

        for y, val in enumerate(proj):
            if val > threshold and not in_line:
                in_line = True
                start_y = y
            elif val <= threshold and in_line:
                in_line = False
                line_h = y - start_y
                if line_h >= 10:
                    boxes.append((int(w_img * 0.2), start_y, int(w_img * 0.6), line_h))

        if in_line and (h_img - start_y) >= 10:
            boxes.append((int(w_img * 0.2), start_y, int(w_img * 0.6), h_img - start_y))

        return boxes
