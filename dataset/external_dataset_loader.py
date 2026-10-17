"""
External Dataset Loader for Postal Address OCR.
Enables training ScratchPostalCRNN on any downloaded public or private dataset
(e.g., IAM Lines, RIMES, Kaggle OCR, or custom scanned envelopes).

Expected directory formats supported:
1. CSV / TSV format:
   A directory containing images and a labels.csv file:
   | image_name | text |
   | line_001.png | 124 Baker Street |

2. IAM format:
   A folder of line images with a lines.txt ground truth file.

3. Image folder with filename as label:
   e.g. data/560001_01.png, data/MG_Road_02.png
"""

import os
import csv
from typing import List, Tuple, Dict, Any, Optional
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image

from dataset.vocab import text_to_indices
from dataset.postal_dataset import postal_collate_fn


class ExternalPostalDataset(Dataset):
    """
    Loads text line images from a local directory using a CSV or TSV ground truth file.
    """

    def __init__(
        self,
        data_dir: str,
        csv_file: Optional[str] = None,
        target_height: int = 32,
        delimiter: str = ","
    ):
        super().__init__()
        self.data_dir = data_dir
        self.target_height = target_height
        self.samples: List[Tuple[str, str]] = []

        # 1. Check if CSV file is provided
        csv_path = csv_file if csv_file else os.path.join(data_dir, "labels.csv")
        tsv_path = os.path.join(data_dir, "labels.txt")

        if os.path.exists(csv_path):
            with open(csv_path, "r", encoding="utf-8", errors="ignore") as f:
                reader = csv.reader(f, delimiter=delimiter)
                header = next(reader, None)
                for row in reader:
                    if len(row) >= 2:
                        img_path = os.path.join(data_dir, row[0].strip())
                        label = row[1].strip()
                        if os.path.exists(img_path):
                            self.samples.append((img_path, label))

        elif os.path.exists(tsv_path):
            with open(tsv_path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    parts = line.strip().split("\t")
                    if len(parts) >= 2:
                        img_path = os.path.join(data_dir, parts[0].strip())
                        label = parts[1].strip()
                        if os.path.exists(img_path):
                            self.samples.append((img_path, label))

        # 2. Fallback: load directly from images in folder
        if not self.samples:
            valid_exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
            for fname in os.listdir(data_dir):
                ext = os.path.splitext(fname)[1].lower()
                if ext in valid_exts:
                    img_path = os.path.join(data_dir, fname)
                    # Use filename stem as label (substituting underscores with spaces)
                    label = os.path.splitext(fname)[0].replace("_", " ")
                    self.samples.append((img_path, label))

        print(f"[External Dataset] Loaded {len(self.samples)} samples from '{data_dir}'")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        img_path, text = self.samples[idx]

        # Load image in grayscale
        img = Image.open(img_path).convert("L")
        w, h = img.size

        # Resize preserving aspect ratio so height = target_height
        if h != self.target_height:
            aspect = w / float(max(1, h))
            new_w = max(16, int(round(self.target_height * aspect)))
            img = img.resize((new_w, self.target_height), Image.Resampling.BILINEAR)

        # Normalize pixel values to [-1.0, 1.0]
        img_np = np.array(img, dtype=np.float32) / 255.0
        img_norm = (img_np - 0.5) / 0.5
        img_tensor = torch.from_numpy(img_norm).unsqueeze(0)  # (1, 32, W)

        # Label indices
        target_indices = torch.tensor(text_to_indices(text), dtype=torch.long)

        return {
            "image": img_tensor,
            "text": text,
            "target": target_indices,
            "target_length": len(target_indices),
            "width": img_tensor.shape[2]
        }


def create_external_dataloader(
    data_dir: str,
    csv_file: Optional[str] = None,
    batch_size: int = 32,
    shuffle: bool = True
) -> DataLoader:
    """Create a PyTorch DataLoader for an external dataset folder."""
    dataset = ExternalPostalDataset(data_dir=data_dir, csv_file=csv_file)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=postal_collate_fn,
        num_workers=0
    )


def load_emnist_dataset(
    num_samples: int = 1500,
    batch_size: int = 32,
    shuffle: bool = True,
    is_training: bool = True,
    use_all_splits: bool = True
) -> DataLoader:
    """Convenience helper to load EMNIST real handwriting (all splits by default: letters + digits + byclass + balanced)."""
    from dataset.emnist_loader import create_emnist_dataloader
    return create_emnist_dataloader(
        num_samples=num_samples,
        batch_size=batch_size,
        shuffle=shuffle,
        is_training=is_training,
        use_all_splits=use_all_splits,
    )
