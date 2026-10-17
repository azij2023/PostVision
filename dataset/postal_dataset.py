"""
PyTorch Dataset and DataLoader collation for Postal Address OCR.
Supports dynamic generation, aspect-ratio preserving padding, and CTC label alignment.
"""

from typing import List, Tuple, Dict, Any, Optional
import random
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image

from dataset.vocab import text_to_indices, NUM_CLASSES
from dataset.address_generator import (
    render_text_line_image,
    generate_random_address_record,
    get_available_fonts
)


class PostalLineDataset(Dataset):
    """
    Dataset of postal address text lines.
    Supports three operating modes:
    1. 'emnist': Real human handwritten letters assembled from the EMNIST dataset.
    2. 'synthetic': Clean & noisy computer font rendered lines.
    3. 'hybrid': 50/50 mixture of real EMNIST handwriting and synthetic fonts for maximum generalization.
    """

    def __init__(
        self,
        num_samples: int = 1000,
        target_height: int = 32,
        max_width: int = 384,
        is_training: bool = True,
        samples: Optional[List[Tuple[Image.Image, str]]] = None,
        dataset_mode: str = "emnist"
    ):
        super().__init__()
        self.num_samples = num_samples
        self.target_height = target_height
        self.max_width = max_width
        self.is_training = is_training
        self.dataset_mode = dataset_mode.lower()
        self.fonts = get_available_fonts()
        self.cached_samples = samples
        self.emnist_manager = None

        if self.dataset_mode in ("emnist", "hybrid"):
            try:
                from dataset.emnist_loader import EMNISTGlyphManager
                self.emnist_manager = EMNISTGlyphManager.get_instance()
            except Exception as e:
                print(f"[PostalLineDataset] Warning: Could not initialize EMNIST manager ({e}). Falling back to synthetic.")
                self.dataset_mode = "synthetic"

    def __len__(self) -> int:
        if self.cached_samples is not None:
            return len(self.cached_samples)
        return self.num_samples

    def _generate_sample(self) -> Tuple[Image.Image, str]:
        # Determine whether to use real EMNIST handwriting for this sample
        use_emnist = False
        if self.dataset_mode == "emnist" and self.emnist_manager is not None:
            use_emnist = True
        elif self.dataset_mode == "hybrid" and self.emnist_manager is not None:
            use_emnist = random.random() < 0.50

        rec = generate_random_address_record()
        line_type = random.choice(["recipient", "line2", "line3", "city_state_pin", "pin_only"])

        if line_type == "recipient":
            text = rec["recipient"]
        elif line_type == "line2":
            text = rec["line2"]
        elif line_type == "line3":
            text = rec["line3"]
        elif line_type == "city_state_pin":
            text = rec["city_state_pin"]
        else:
            text = f"PIN: {rec['pin']}" if random.random() < 0.5 else f"PIN CODE - {rec['pin']}"

        if use_emnist:
            img, text = self.emnist_manager.render_handwritten_line(
                text=text,
                target_height=self.target_height,
                is_training=self.is_training
            )
            return img, text

        # Synthetic font fallback/mode
        font_path = random.choice(self.fonts) if self.fonts else None
        font_size = random.randint(18, 24)
        noise = random.uniform(0.01, 0.05) if self.is_training else 0.0
        slant = self.is_training

        img, text = render_text_line_image(
            text=text,
            font_path=font_path,
            font_size=font_size,
            target_height=self.target_height,
            noise_level=noise,
            add_slant=slant
        )
        return img, text

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if self.cached_samples is not None:
            img, text = self.cached_samples[idx]
        else:
            img, text = self._generate_sample()

        # Convert to numpy and normalize
        img_np = np.array(img, dtype=np.float32) / 255.0  # Range [0, 1]
        # Invert so text is foreground (1.0 = ink, 0.0 = paper) or keep standard normalized
        # Standard: (pixel - 0.5) / 0.5 -> [-1.0, 1.0]
        img_norm = (img_np - 0.5) / 0.5

        # Shape: (1, H, W)
        img_tensor = torch.from_numpy(img_norm).unsqueeze(0)

        # Label indices
        target_indices = torch.tensor(text_to_indices(text), dtype=torch.long)

        return {
            "image": img_tensor,
            "text": text,
            "target": target_indices,
            "target_length": len(target_indices),
            "width": img_tensor.shape[2]
        }


def postal_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Custom collate function for variable-width text lines:
    Pads all images in the batch to the maximum width in the batch with white padding (-1.0).
    Concatenates targets and records input_lengths and target_lengths for CTC Loss.
    """
    batch_size = len(batch)
    max_w = max(b["width"] for b in batch)
    # Ensure max_w is a multiple of 4 (due to model's 2x2 maxpools)
    max_w = int(np.ceil(max_w / 4.0) * 4)
    target_h = batch[0]["image"].shape[1]

    # Pre-allocate padded image tensor with 1.0 (paper background)
    padded_images = torch.full((batch_size, 1, target_h, max_w), 1.0, dtype=torch.float32)

    texts = []
    target_list = []
    target_lengths = []
    input_lengths = []

    for i, item in enumerate(batch):
        w = item["width"]
        padded_images[i, :, :, :w] = item["image"]
        texts.append(item["text"])
        target_list.append(item["target"])
        target_lengths.append(item["target_length"])
        # In our model, horizontal width is downsampled by factor of 4: T = max_w // 4
        # Calculate exact valid time dimension for this sample (ignoring white padding)
        t_sample = max(item["target_length"], w // 4)
        t_sample = min(max_w // 4, t_sample)
        input_lengths.append(t_sample)

    # Flattened targets for PyTorch CTCLoss
    concatenated_targets = torch.cat(target_list, dim=0)

    return {
        "images": padded_images,
        "texts": texts,
        "targets": concatenated_targets,
        "target_lengths": torch.tensor(target_lengths, dtype=torch.long),
        "input_lengths": torch.tensor(input_lengths, dtype=torch.long)
    }


def create_postal_dataloader(
    num_samples: int = 1000,
    batch_size: int = 32,
    shuffle: bool = True,
    is_training: bool = True,
    dataset_mode: str = "emnist"
) -> DataLoader:
    """Create a high-throughput DataLoader supporting EMNIST, synthetic, or hybrid data."""
    dataset = PostalLineDataset(
        num_samples=num_samples,
        is_training=is_training,
        dataset_mode=dataset_mode
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=postal_collate_fn,
        num_workers=0
    )
