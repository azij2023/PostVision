"""
Advanced Postal Envelope Vision & Preprocessing Pipeline.
Provides:
- Envelope boundary detection & 4-point quadrilateral perspective rectification
- Text orientation deskewing (Hough & contour minimum area bounding rect)
- Morphological background estimation & shadow removal
- Ink edge crisping & bilateral paper grain filtering
- Multi-stage diagnostic telemetry generation
"""

import cv2
import numpy as np
from PIL import Image
from typing import Dict, Any, Tuple, Optional, List


def order_quad_points(pts: np.ndarray) -> np.ndarray:
    """Order 4 quadrilateral coordinates: [top-left, top-right, bottom-right, bottom-left]."""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]      # Top-left has smallest sum
    rect[2] = pts[np.argmax(s)]      # Bottom-right has largest sum

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]   # Top-right has smallest difference
    rect[3] = pts[np.argmax(diff)]   # Bottom-left has largest difference
    return rect


class EnvelopeVisionPreprocessor:
    """
    Modular vision engine for document envelope photography rectification,
    shadow neutralization, ink stroke enhancement, and multi-stage diagnostics.
    """

    def __init__(self, target_line_height: int = 32):
        self.target_line_height = target_line_height

    @classmethod
    def detect_boundary_and_rectify(cls, img_bgr: np.ndarray) -> Tuple[np.ndarray, bool, float, Optional[np.ndarray]]:
        """
        Detects outer envelope boundary against background table/desk.
        If a convex 4-point polygon is found, warps perspective to a clean rectangular envelope crop.
        Otherwise, estimates text orientation angle and deskews rotation.
        Returns:
            (rectified_bgr, was_perspective_warped, deskew_angle_deg, quad_points)
        """
        h, w = img_bgr.shape[:2]
        total_area = h * w

        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 35, 120)

        # Morphological bridge to connect faint envelope border lines
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        envelope_quad = None
        for cnt in contours[:6]:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.025 * peri, True)
            area = cv2.contourArea(approx)

            # Must occupy between 25% and 96% of total camera frame
            if len(approx) == 4 and (0.25 * total_area < area < 0.96 * total_area):
                if cv2.isContourConvex(approx):
                    envelope_quad = approx.reshape(4, 2)
                    break

        if envelope_quad is not None:
            pts = order_quad_points(envelope_quad)
            (tl, tr, br, bl) = pts

            widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
            widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
            maxWidth = max(int(widthA), int(widthB))

            heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
            heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
            maxHeight = max(int(heightA), int(heightB))

            # Maintain sensible envelope aspect ratio
            if maxWidth > 100 and maxHeight > 60 and (maxWidth / maxHeight) >= 1.1:
                dst = np.array([
                    [0, 0],
                    [maxWidth - 1, 0],
                    [maxWidth - 1, maxHeight - 1],
                    [0, maxHeight - 1]
                ], dtype="float32")

                M = cv2.getPerspectiveTransform(pts, dst)
                warped = cv2.warpPerspective(img_bgr, M, (maxWidth, maxHeight), flags=cv2.INTER_CUBIC)
                return warped, True, 0.0, pts

        # Fallback: Document fills the frame -> perform text orientation deskewing
        deskew_angle, deskewed_img = cls.deskew_image(img_bgr)
        return deskewed_img, False, deskew_angle, None

    @classmethod
    def deskew_image(cls, img_bgr: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        Estimates skew angle of text lines using minimum area bounding box on edge pixels
        and counter-rotates image to level text horizontally.
        """
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        coords = np.column_stack(np.where(edges > 0))

        if len(coords) < 80:
            return 0.0, img_bgr

        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = -(90 + angle)
        elif angle > 45:
            angle = 90 - angle
        else:
            angle = -angle

        # Cap correction to realistic envelope skew (+/- 20 degrees)
        if abs(angle) > 20:
            angle = 0.0

        if abs(angle) > 0.6:
            h, w = img_bgr.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(
                img_bgr, M, (w, h),
                flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_REPLICATE
            )
            return round(float(angle), 2), rotated

        return 0.0, img_bgr

    @classmethod
    def remove_shadows_and_flatten(cls, img_bgr: np.ndarray) -> np.ndarray:
        """
        Removes ambient illumination gradients and mobile phone cast shadows
        using morphological background surface approximation and division normalization.
        """
        planes = cv2.split(img_bgr)
        result_planes = []

        h, w = img_bgr.shape[:2]
        # Dynamic kernel size proportional to image resolution
        k_size = max(15, (min(h, w) // 35) | 1)
        k_blur = max(21, (min(h, w) // 25) | 1)

        struct_elem = cv2.getStructuringElement(cv2.MORPH_RECT, (k_size, k_size))

        for plane in planes:
            # Dilate to eliminate dark ink lines while retaining paper illumination
            dilated = cv2.dilate(plane, struct_elem)
            # Median blur to smooth paper lighting background
            bg_illumination = cv2.medianBlur(dilated, k_blur)
            # Difference and normalization
            diff = 255 - cv2.absdiff(plane, bg_illumination)
            norm = cv2.normalize(diff, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            result_planes.append(norm)

        shadow_free_bgr = cv2.merge(result_planes)
        return shadow_free_bgr

    @classmethod
    def enhance_ink_contrast(cls, img_bgr: np.ndarray) -> np.ndarray:
        """
        Applies Contrast-Limited Adaptive Histogram Equalization (CLAHE),
        bilateral edge-preserving smoothing, and unsharp masking to pop ink strokes.
        """
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        # CLAHE for localized contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.4, tileGridSize=(8, 8))
        clahe_enhanced = clahe.apply(gray)

        # Bilateral filter to smooth fibrous paper grain while preserving sharp ink stroke boundaries
        denoised = cv2.bilateralFilter(clahe_enhanced, 7, 45, 45)

        # Unsharp masking for crisp text edges
        gaussian = cv2.GaussianBlur(denoised, (0, 0), 2.0)
        unsharp = cv2.addWeighted(denoised, 1.3, gaussian, -0.3, 0)

        # Re-convert to 3-channel RGB for neural model consistency
        return cv2.cvtColor(unsharp, cv2.COLOR_GRAY2BGR)

    @classmethod
    def compute_diagnostics(
        cls,
        raw_bgr: np.ndarray,
        final_bgr: np.ndarray,
        deskew_angle: float,
        was_perspective_warped: bool
    ) -> Dict[str, Any]:
        """Calculates quantitative telemetry metrics describing the vision transformations."""
        raw_gray = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2GRAY)
        final_gray = cv2.cvtColor(final_bgr, cv2.COLOR_BGR2GRAY)

        raw_std = float(np.std(raw_gray))
        final_std = float(np.std(final_gray))

        # Illumination uniformity improvement (paper background variance reduction)
        raw_bg_mask = raw_gray > np.median(raw_gray)
        final_bg_mask = final_gray > np.median(final_gray)

        raw_bg_var = float(np.var(raw_gray[raw_bg_mask])) if np.any(raw_bg_mask) else 1.0
        final_bg_var = float(np.var(final_gray[final_bg_mask])) if np.any(final_bg_mask) else 1.0

        shadow_reduction_pct = max(0.0, min(99.0, (1.0 - (final_bg_var / (raw_bg_var + 1e-5))) * 100.0))
        contrast_boost = round(float(final_std / (raw_std + 1e-5)), 2)

        h, w = final_bgr.shape[:2]
        return {
            "resolution": f"{w} × {h} px",
            "aspect_ratio": f"{w / max(1, h):.2f}:1",
            "deskew_angle": round(deskew_angle, 2),
            "perspective_rectified": was_perspective_warped,
            "shadow_attenuation_pct": round(shadow_reduction_pct, 1),
            "contrast_boost_factor": f"{contrast_boost}×",
            "clahe_clip_limit": 2.4,
            "filter_kernel": "Bilateral (7px, σ=45)",
            "pipeline_status": "OPTIMIZED"
        }

    @classmethod
    def process(cls, input_image: Any) -> Dict[str, Any]:
        """
        Executes complete multi-stage vision preprocessing pipeline.
        Returns processed image ready for OCR, multi-stage images for Inspector, and diagnostic metrics.
        """
        # Ingest PIL Image or numpy array
        if isinstance(input_image, Image.Image):
            np_img = np.array(input_image)
            if len(np_img.shape) == 2:
                raw_bgr = cv2.cvtColor(np_img, cv2.COLOR_GRAY2BGR)
            elif np_img.shape[2] == 4:
                raw_bgr = cv2.cvtColor(np_img, cv2.COLOR_RGBA2BGR)
            else:
                raw_bgr = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)
        else:
            raw_bgr = input_image.copy()

        # Stage 1: Raw Image (convert to RGB for UI display)
        stage_1_raw = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB)

        # Stage 2: Boundary Rectification & Deskewing
        rectified_bgr, was_warped, deskew_angle, _ = cls.detect_boundary_and_rectify(raw_bgr)
        stage_2_rectified = cv2.cvtColor(rectified_bgr, cv2.COLOR_BGR2RGB)

        # Stage 3: Illumination Flattening & Shadow Removal
        shadow_free_bgr = cls.remove_shadows_and_flatten(rectified_bgr)
        stage_3_shadow_free = cv2.cvtColor(shadow_free_bgr, cv2.COLOR_BGR2RGB)

        # Stage 4: Contrast & Ink Edge Enhancement
        enhanced_bgr = cls.enhance_ink_contrast(shadow_free_bgr)
        stage_4_enhanced = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2RGB)

        # Quantitative Diagnostics
        diagnostics = cls.compute_diagnostics(raw_bgr, enhanced_bgr, deskew_angle, was_warped)

        # Postage Stamp & Postmark Cancellation Detection
        stamp_info = cls.detect_postage_stamp_and_cancellation(rectified_bgr)

        return {
            "ocr_input_image": stage_4_enhanced,
            "rectified_bgr": rectified_bgr,
            "stages": {
                "raw": stage_1_raw,
                "rectified": stage_2_rectified,
                "shadow_free": stage_3_shadow_free,
                "enhanced": stage_4_enhanced
            },
            "diagnostics": diagnostics,
            "stamp_info": stamp_info
        }

    @classmethod
    def detect_postage_stamp_and_cancellation(cls, img_bgr: np.ndarray) -> Dict[str, Any]:
        """
        Detects postage stamps in the upper-right postal region (standard postal zone)
        and verifies whether an ink cancellation seal or postal postmark is present.
        """
        h, w = img_bgr.shape[:2]
        x_min_zone = int(w * 0.58)
        y_max_zone = int(h * 0.48)
        zone_bgr = img_bgr[0:y_max_zone, x_min_zone:w]
        zh, zw = zone_bgr.shape[:2]

        if zh < 20 or zw < 20:
            return {
                "stamp_detected": False,
                "confidence": 0.0,
                "cancellation_detected": False,
                "cancellation_type": "NONE",
                "status": "NO_STAMP_DETECTED",
                "crop_rgb": None
            }

        zone_gray = cv2.cvtColor(zone_bgr, cv2.COLOR_BGR2GRAY)
        zone_hsv = cv2.cvtColor(zone_bgr, cv2.COLOR_BGR2HSV)

        # Edge & contour detection for stamp rectangular geometry
        blurred = cv2.GaussianBlur(zone_gray, (3, 3), 0)
        edges = cv2.Canny(blurred, 35, 130)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        envelope_area = h * w
        zone_area = zh * zw

        candidate_bbox = None
        best_score = 0.0

        for cnt in contours:
            bx, by, bw, bh = cv2.boundingRect(cnt)
            c_area = bw * bh
            aspect = float(bw) / max(1, bh)

            if 0.004 * envelope_area < c_area < 0.16 * envelope_area:
                if 0.52 < aspect < 1.85 and bw >= 20 and bh >= 20:
                    stamp_mask = np.zeros_like(zone_gray)
                    cv2.drawContours(stamp_mask, [cnt], -1, 255, -1)
                    mean_val = cv2.mean(zone_gray, mask=stamp_mask)[0]
                    paper_median = float(np.median(zone_gray))
                    contrast_diff = abs(mean_val - paper_median)
                    score = float(c_area) * (1.0 + contrast_diff / 40.0)

                    if score > best_score:
                        best_score = score
                        candidate_bbox = (bx, by, bw, bh)

        # Color saturation heuristic
        sat = zone_hsv[:, :, 1]
        high_sat_mask = sat > 35
        if candidate_bbox is None and np.sum(high_sat_mask) > (0.015 * zone_area):
            coords = np.column_stack(np.where(high_sat_mask))
            if len(coords) > 40:
                y0, x0 = coords.min(axis=0)
                y1, x1 = coords.max(axis=0)
                bw, bh = x1 - x0, y1 - y0
                if bw >= 20 and bh >= 20 and (bw * bh) < (0.65 * zone_area):
                    candidate_bbox = (x0, y0, bw, bh)

        if candidate_bbox is not None:
            bx, by, bw, bh = candidate_bbox
            pad = 6
            sx1 = max(0, bx - pad)
            sy1 = max(0, by - pad)
            sx2 = min(zw, bx + bw + pad)
            sy2 = min(zh, by + bh + pad)
            stamp_bgr = zone_bgr[sy1:sy2, sx1:sx2]
            stamp_rgb = cv2.cvtColor(stamp_bgr, cv2.COLOR_BGR2RGB)

            # Check for cancellation postmark ink lines / circular seals
            s_gray = cv2.cvtColor(stamp_bgr, cv2.COLOR_BGR2GRAY)
            paper_level = float(np.percentile(s_gray, 70))
            ink_mask = (s_gray < (paper_level - 30)).astype(np.uint8)

            circles = cv2.HoughCircles(
                s_gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=15,
                param1=50, param2=25, minRadius=8, maxRadius=max(12, min(bw, bh) // 2)
            )

            horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 2))
            horiz_lines = cv2.morphologyEx(ink_mask, cv2.MORPH_OPEN, horiz_kernel)
            has_wavy_lines = np.sum(horiz_lines) > 15

            has_circles = circles is not None and len(circles[0]) > 0
            dark_ink_ratio = np.sum(ink_mask) / float(max(1, stamp_bgr.shape[0] * stamp_bgr.shape[1]))

            cancellation_detected = has_circles or has_wavy_lines or (dark_ink_ratio > 0.07)
            cancel_type = "CIRCULAR_POSTMARK" if has_circles else ("WAVY_LINE_SEAL" if has_wavy_lines else ("INK_POSTMARK" if cancellation_detected else "NONE"))

            status = "VALID_CANCELLED" if cancellation_detected else "UNCANCELLED_WARNING"
            confidence = round(min(0.98, max(0.72, 0.70 + (0.24 if cancellation_detected else 0.12))), 2)

            return {
                "stamp_detected": True,
                "confidence": confidence,
                "bbox_in_zone": [int(bx), int(by), int(bw), int(bh)],
                "global_bbox": [int(x_min_zone + bx), int(by), int(bw), int(bh)],
                "cancellation_detected": cancellation_detected,
                "cancellation_type": cancel_type,
                "status": status,
                "crop_rgb": stamp_rgb
            }
        else:
            return {
                "stamp_detected": False,
                "confidence": 0.0,
                "cancellation_detected": False,
                "cancellation_type": "NONE",
                "status": "NO_STAMP_DETECTED",
                "crop_rgb": None
            }
