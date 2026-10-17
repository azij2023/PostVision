"""
EMNIST Full Dataset Loader and Real Handwritten Address Line Assembler.
Loads ALL available EMNIST splits from archive.zip for maximum training coverage:
  - letters (A-Z): 88,800 train / 14,800 test
  - digits (0-9): 240,000 train / 40,000 test
  - balanced (47 classes): 112,800 train / 18,800 test
  - byclass (62 classes): 697,932 train / 116,323 test
  - bymerge (47 classes): 697,932 train / 116,323 test
  - mnist (digits-only): 60,000 train / 10,000 test

Key Capabilities:
1. Auto-extraction from archive.zip when CSV files are missing.
2. Per-split binary caching (.npz) for instant millisecond dataset loading.
3. Unified character-level glyph index for all handwritten letters (A-Z) and digits (0-9).
4. Realistic Handwritten Postal Line Synthesizer that concatenates real human-written
   EMNIST glyphs with natural baseline jitter, variable kerning, stroke variations,
   and paper background blending.
5. Seamless compatibility with ScratchPostalCRNN, CTC loss, and postal collate functions.
"""

import os
import random
import time
import zipfile
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
from PIL import Image, ImageFilter, ImageDraw, ImageFont
import torch
from torch.utils.data import Dataset, DataLoader

from dataset.vocab import text_to_indices, NUM_CLASSES
from dataset.postal_dataset import postal_collate_fn
from dataset.address_generator import (
    TITLES, FIRST_NAMES, LAST_NAMES, PREMISES, BUILDINGS,
    STREETS, LOCALITIES, CITIES_STATES_PINS
)

# ─────────────────────────────────────────────────────────────
# EMNIST label mapping tables (label_index → ASCII code)
# Parsed from the official mapping .txt files in archive.zip
# ─────────────────────────────────────────────────────────────
EMNIST_MAPPINGS = {
    "letters": {i: chr(ord('A') + i - 1) for i in range(1, 27)},
    "digits":  {i: chr(ord('0') + i) for i in range(10)},
    "balanced": {
        0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9',
        10: 'A', 11: 'B', 12: 'C', 13: 'D', 14: 'E', 15: 'F', 16: 'G', 17: 'H', 18: 'I',
        19: 'J', 20: 'K', 21: 'L', 22: 'M', 23: 'N', 24: 'O', 25: 'P', 26: 'Q', 27: 'R',
        28: 'S', 29: 'T', 30: 'U', 31: 'V', 32: 'W', 33: 'X', 34: 'Y', 35: 'Z',
        36: 'a', 37: 'b', 38: 'd', 39: 'e', 40: 'f', 41: 'g', 42: 'h', 43: 'n',
        44: 'q', 45: 'r', 46: 't',
    },
    "byclass": {
        **{i: chr(48 + i) for i in range(10)},          # 0-9 → '0'-'9'
        **{10 + i: chr(65 + i) for i in range(26)},     # 10-35 → 'A'-'Z'
        **{36 + i: chr(97 + i) for i in range(26)},     # 36-61 → 'a'-'z'
    },
    "bymerge": {
        0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9',
        10: 'A', 11: 'B', 12: 'C', 13: 'D', 14: 'E', 15: 'F', 16: 'G', 17: 'H', 18: 'I',
        19: 'J', 20: 'K', 21: 'L', 22: 'M', 23: 'N', 24: 'O', 25: 'P', 26: 'Q', 27: 'R',
        28: 'S', 29: 'T', 30: 'U', 31: 'V', 32: 'W', 33: 'X', 34: 'Y', 35: 'Z',
        36: 'a', 37: 'b', 38: 'd', 39: 'e', 40: 'f', 41: 'g', 42: 'h', 43: 'n',
        44: 'q', 45: 'r', 46: 't',
    },
    "mnist": {i: chr(ord('0') + i) for i in range(10)},
}

# All available EMNIST splits
ALL_EMNIST_SPLITS = ["letters", "digits", "balanced", "byclass", "bymerge", "mnist"]

# Dataset directory (relative to project root)
DATASET_DIR = "dataset"


def _ensure_csv_extracted(split_name: str, dataset_dir: str = DATASET_DIR) -> bool:
    """
    Checks if the CSV files for a given EMNIST split exist.
    If they don't, attempts to auto-extract them from archive.zip.
    Returns True if files are available after the check.
    """
    train_csv = os.path.join(dataset_dir, f"emnist-{split_name}-train.csv")
    test_csv = os.path.join(dataset_dir, f"emnist-{split_name}-test.csv")

    if os.path.exists(train_csv) and os.path.exists(test_csv):
        return True

    # Try to extract from archive.zip
    archive_path = os.path.join(dataset_dir, "archive.zip")
    if not os.path.exists(archive_path):
        print(f"[EMNIST] Warning: {split_name} CSVs not found and archive.zip is missing.")
        return False

    print(f"[EMNIST] Extracting {split_name} CSVs from archive.zip...")
    try:
        with zipfile.ZipFile(archive_path, 'r') as zf:
            target_files = [
                f"emnist-{split_name}-train.csv",
                f"emnist-{split_name}-test.csv",
                f"emnist-{split_name}-mapping.txt",
            ]
            for target in target_files:
                if target in zf.namelist():
                    zf.extract(target, dataset_dir)
                    print(f"  Extracted: {target}")
        return os.path.exists(train_csv) and os.path.exists(test_csv)
    except Exception as e:
        print(f"[EMNIST] Error extracting {split_name} from archive.zip: {e}")
        return False


class EMNISTGlyphManager:
    """
    Manages loading, caching, and sampling of real human handwritten glyphs
    from ALL available EMNIST dataset splits (letters, digits, balanced, byclass, bymerge, mnist).

    Builds a unified character→glyph index so that any letter A-Z or digit 0-9
    can be sampled from real handwriting with maximum diversity.
    """
    _instance = None

    def __init__(
        self,
        train_csv_path: str = "dataset/emnist-letters-train.csv",
        test_csv_path: str = "dataset/emnist-letters-test.csv",
        cache_path: str = "dataset/emnist_cache.npz",
        use_all_splits: bool = True,
        dataset_dir: str = DATASET_DIR,
    ):
        self.train_csv_path = train_csv_path
        self.test_csv_path = test_csv_path
        self.cache_path = cache_path
        self.use_all_splits = use_all_splits
        self.dataset_dir = dataset_dir

        # Primary letters arrays (backward compatible)
        self.train_images: Optional[np.ndarray] = None
        self.train_labels: Optional[np.ndarray] = None
        self.test_images: Optional[np.ndarray] = None
        self.test_labels: Optional[np.ndarray] = None

        # Per-split storage: {split_name: {"train_images": ..., "train_labels": ..., ...}}
        self.splits: Dict[str, Dict[str, np.ndarray]] = {}

        # Unified character glyph pools: {'A': [glyph_array, ...], '0': [...], ...}
        self.train_glyph_pool: Dict[str, List[np.ndarray]] = {}
        self.test_glyph_pool: Dict[str, List[np.ndarray]] = {}

        # Legacy indices (backward compatible)
        self.train_letter_indices: Dict[str, np.ndarray] = {}
        self.test_letter_indices: Dict[str, np.ndarray] = {}

        # Statistics
        self.total_train_samples = 0
        self.total_test_samples = 0
        self.loaded_splits: List[str] = []

        self._load_all_splits()
        self._build_unified_glyph_pool()
        self._build_letter_indices()

    @classmethod
    def get_instance(cls, **kwargs):
        if cls._instance is None:
            cls._instance = cls(**kwargs)
        return cls._instance

    @classmethod
    def reset_instance(cls):
        """Resets the singleton (useful for testing or reloading with different config)."""
        cls._instance = None

    def _load_all_splits(self):
        """Loads all available EMNIST splits from CSV or cache."""
        # 1. Always load the primary letters split (backward compatible)
        self._load_or_build_cache()

        if not self.use_all_splits:
            return

        # 2. Load additional splits
        additional_splits = ["digits", "balanced", "byclass", "bymerge", "mnist"]
        for split_name in additional_splits:
            self._load_split(split_name)

        print(f"[EMNIST] === Full Dataset Summary ===")
        print(f"  Loaded Splits   : {', '.join(self.loaded_splits)}")
        print(f"  Total Train     : {self.total_train_samples:,} real handwritten samples")
        print(f"  Total Test      : {self.total_test_samples:,} real handwritten samples")
        print(f"  Unique Glyphs   : {len(self.train_glyph_pool)} characters (train)")
        print(f"  ===================================")

    def _load_split(self, split_name: str):
        """Loads a single EMNIST split from cache or CSV, with auto-extraction from zip."""
        cache_path = os.path.join(self.dataset_dir, f"emnist_{split_name}_cache.npz")
        train_csv = os.path.join(self.dataset_dir, f"emnist-{split_name}-train.csv")
        test_csv = os.path.join(self.dataset_dir, f"emnist-{split_name}-test.csv")

        # Try loading from cache
        if os.path.exists(cache_path):
            try:
                t0 = time.time()
                data = np.load(cache_path)
                split_data = {
                    "train_images": data["train_images"],
                    "train_labels": data["train_labels"],
                    "test_images": data["test_images"],
                    "test_labels": data["test_labels"],
                }
                self.splits[split_name] = split_data
                self.loaded_splits.append(split_name)
                n_train = len(split_data["train_labels"])
                n_test = len(split_data["test_labels"])
                self.total_train_samples += n_train
                self.total_test_samples += n_test
                print(f"[EMNIST:{split_name}] Loaded from cache in {time.time()-t0:.2f}s "
                      f"(Train: {n_train:,}, Test: {n_test:,})")
                return
            except Exception as e:
                print(f"[EMNIST:{split_name}] Cache load failed ({e}), falling back to CSV.")

        # Auto-extract from zip if needed
        if not os.path.exists(train_csv) or not os.path.exists(test_csv):
            if not _ensure_csv_extracted(split_name, self.dataset_dir):
                print(f"[EMNIST:{split_name}] Skipping — CSVs unavailable.")
                return

        # Parse CSVs
        train_imgs, train_lbls = self._parse_csv(train_csv, max_rows=None)
        test_imgs, test_lbls = self._parse_csv(test_csv, max_rows=None)

        split_data = {
            "train_images": train_imgs,
            "train_labels": train_lbls,
            "test_images": test_imgs,
            "test_labels": test_lbls,
        }
        self.splits[split_name] = split_data
        self.loaded_splits.append(split_name)
        self.total_train_samples += len(train_lbls)
        self.total_test_samples += len(test_lbls)

        # Save per-split cache
        try:
            np.savez_compressed(
                cache_path,
                train_images=train_imgs,
                train_labels=train_lbls,
                test_images=test_imgs,
                test_labels=test_lbls,
            )
            print(f"[EMNIST:{split_name}] Saved cache to: {cache_path}")
        except Exception as e:
            print(f"[EMNIST:{split_name}] Warning: Could not save cache ({e})")

    def _build_unified_glyph_pool(self):
        """
        Builds a unified character→glyph lookup by merging all loaded EMNIST splits.
        Each character ('A'-'Z', '0'-'9', 'a'-'z') gets a pool of real handwritten images
        drawn from every split that contains it.
        """
        # Initialize pools
        for ch in "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789":
            self.train_glyph_pool[ch] = []
            self.test_glyph_pool[ch] = []

        # 1. Add glyphs from the primary letters split
        if self.train_images is not None:
            mapping = EMNIST_MAPPINGS["letters"]
            for label_idx, char in mapping.items():
                char_upper = char.upper()
                train_mask = np.where(self.train_labels == label_idx)[0]
                for idx in train_mask:
                    self.train_glyph_pool.setdefault(char_upper, []).append(self.train_images[idx])
                if self.test_labels is not None:
                    test_mask = np.where(self.test_labels == label_idx)[0]
                    for idx in test_mask:
                        self.test_glyph_pool.setdefault(char_upper, []).append(self.test_images[idx])

        # 2. Add glyphs from all additional splits
        for split_name, split_data in self.splits.items():
            mapping = EMNIST_MAPPINGS.get(split_name, {})
            train_imgs = split_data["train_images"]
            train_lbls = split_data["train_labels"]
            test_imgs = split_data["test_images"]
            test_lbls = split_data["test_labels"]

            for label_idx, char in mapping.items():
                char_upper = char.upper()
                if char_upper not in "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789":
                    continue

                train_mask = np.where(train_lbls == label_idx)[0]
                for idx in train_mask:
                    self.train_glyph_pool.setdefault(char_upper, []).append(train_imgs[idx])

                test_mask = np.where(test_lbls == label_idx)[0]
                for idx in test_mask:
                    self.test_glyph_pool.setdefault(char_upper, []).append(test_imgs[idx])

        # Convert lists to numpy arrays for efficient random sampling
        for ch in list(self.train_glyph_pool.keys()):
            if self.train_glyph_pool[ch]:
                self.train_glyph_pool[ch] = np.array(self.train_glyph_pool[ch], dtype=np.uint8)
            else:
                self.train_glyph_pool[ch] = np.zeros((1, 28, 28), dtype=np.uint8)

        for ch in list(self.test_glyph_pool.keys()):
            if self.test_glyph_pool[ch]:
                self.test_glyph_pool[ch] = np.array(self.test_glyph_pool[ch], dtype=np.uint8)
            else:
                self.test_glyph_pool[ch] = np.zeros((1, 28, 28), dtype=np.uint8)

        # Report glyph counts per character
        total_glyphs = sum(len(v) for v in self.train_glyph_pool.values())
        digit_glyphs = sum(len(self.train_glyph_pool.get(str(d), [])) for d in range(10))
        letter_glyphs = total_glyphs - digit_glyphs
        print(f"[EMNIST] Unified Glyph Pool: {total_glyphs:,} total "
              f"({letter_glyphs:,} letters + {digit_glyphs:,} digits)")

    def _load_or_build_cache(self):
        """Loads the primary letters split from .npz cache or parses CSVs."""
        if os.path.exists(self.cache_path):
            t0 = time.time()
            data = np.load(self.cache_path)
            self.train_images = data["train_images"]
            self.train_labels = data["train_labels"]
            self.test_images = data["test_images"]
            self.test_labels = data["test_labels"]
            n_train = len(self.train_labels)
            n_test = len(self.test_labels)
            self.total_train_samples += n_train
            self.total_test_samples += n_test
            self.loaded_splits.append("letters")
            print(f"[EMNIST:letters] Loaded from cache in {time.time()-t0:.2f}s "
                  f"(Train: {n_train:,}, Test: {n_test:,})")
            return

        # Auto-extract from zip if needed
        _ensure_csv_extracted("letters", self.dataset_dir)

        print(f"[EMNIST:letters] Initializing from CSV files...")
        train_imgs, train_lbls = self._parse_csv(self.train_csv_path, max_rows=None)
        test_imgs, test_lbls = self._parse_csv(self.test_csv_path, max_rows=None)

        self.train_images = train_imgs
        self.train_labels = train_lbls
        self.test_images = test_imgs
        self.test_labels = test_lbls
        self.total_train_samples += len(train_lbls)
        self.total_test_samples += len(test_lbls)
        self.loaded_splits.append("letters")

        # Save binary cache for fast subsequent reloads
        try:
            np.savez_compressed(
                self.cache_path,
                train_images=self.train_images,
                train_labels=self.train_labels,
                test_images=self.test_images,
                test_labels=self.test_labels
            )
            print(f"[EMNIST:letters] Saved compressed binary cache to: {self.cache_path}")
        except Exception as e:
            print(f"[EMNIST:letters] Warning: Could not save cache ({e})")

    def _parse_csv(self, csv_path: str, max_rows: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Parses EMNIST CSV and transposes 28x28 images so they are correctly oriented."""
        if not os.path.exists(csv_path):
            print(f"[EMNIST] Warning: {csv_path} not found. Creating synthetic fallback.")
            return np.zeros((100, 28, 28), dtype=np.uint8), np.ones((100,), dtype=np.int64)

        t0 = time.time()
        import pandas as pd
        df = pd.read_csv(csv_path, header=None, nrows=max_rows)
        labels = df.iloc[:, 0].values.astype(np.int64)
        raw_pixels = df.iloc[:, 1:].values.astype(np.uint8)

        # Transpose each 28x28 image to fix EMNIST NIST coordinate orientation
        num_samples = len(labels)
        images = raw_pixels.reshape(num_samples, 28, 28).transpose(0, 2, 1)

        print(f"[EMNIST] Parsed {num_samples:,} samples from {os.path.basename(csv_path)} in {time.time()-t0:.2f}s")
        return images, labels

    def _build_letter_indices(self):
        """Builds legacy lookup indices for letters A-Z (labels 1-26) from the primary letters split."""
        if self.train_labels is None:
            return
        for char_code in range(1, 27):
            char = chr(ord('A') + char_code - 1)
            train_mask = np.where(self.train_labels == char_code)[0]
            self.train_letter_indices[char] = train_mask

            if self.test_labels is not None and len(self.test_labels) > 0:
                test_mask = np.where(self.test_labels == char_code)[0]
                self.test_letter_indices[char] = test_mask

    def get_random_glyph(self, char: str, is_training: bool = True) -> np.ndarray:
        """
        Retrieves a randomly sampled 28x28 real handwritten glyph array for the given character.
        Uses the unified glyph pool which merges glyphs from ALL loaded EMNIST splits.
        Supports both letters (A-Z) and digits (0-9).
        """
        char_upper = char.upper()
        pool = self.train_glyph_pool if is_training else self.test_glyph_pool

        if char_upper in pool and len(pool[char_upper]) > 0:
            idx = random.randint(0, len(pool[char_upper]) - 1)
            glyph = pool[char_upper][idx]
        else:
            # Fallback: try the legacy letters index
            indices_dict = self.train_letter_indices if is_training else self.test_letter_indices
            img_source = self.train_images if is_training else self.test_images

            if char_upper in indices_dict and len(indices_dict[char_upper]) > 0:
                idx = random.choice(indices_dict[char_upper])
                glyph = img_source[idx]
            else:
                glyph = np.zeros((28, 28), dtype=np.uint8)

        # Invert from white-on-black to black-on-white ink
        inverted = 255 - glyph
        return inverted

    def get_glyph_pool_size(self, char: str, is_training: bool = True) -> int:
        """Returns the number of unique handwritten glyphs available for a character."""
        char_upper = char.upper()
        pool = self.train_glyph_pool if is_training else self.test_glyph_pool
        if char_upper in pool:
            return len(pool[char_upper])
        return 0

    def get_dataset_stats(self) -> Dict[str, Any]:
        """Returns comprehensive statistics about all loaded EMNIST data."""
        letter_counts = {}
        digit_counts = {}
        for ch in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
            letter_counts[ch] = self.get_glyph_pool_size(ch, is_training=True)
        for d in "0123456789":
            digit_counts[d] = self.get_glyph_pool_size(d, is_training=True)

        return {
            "loaded_splits": self.loaded_splits,
            "total_train_samples": self.total_train_samples,
            "total_test_samples": self.total_test_samples,
            "total_unique_characters": len(self.train_glyph_pool),
            "letter_glyph_counts": letter_counts,
            "digit_glyph_counts": digit_counts,
            "total_letter_glyphs": sum(letter_counts.values()),
            "total_digit_glyphs": sum(digit_counts.values()),
        }

    def render_handwritten_word(
        self,
        word: str,
        target_height: int = 32,
        is_training: bool = True
    ) -> Image.Image:
        """
        Renders a single word by stitching real EMNIST handwritten characters.
        Now supports BOTH letters AND digits from the unified glyph pool.
        Applies natural handwriting effects: baseline jitter, kerning variation,
        and subtle scaling.
        """
        rendered_glyphs = []
        base_h = 28

        for ch in word:
            if ch.isalpha() or ch.isdigit():
                glyph_arr = self.get_random_glyph(ch, is_training=is_training)
                g_img = Image.fromarray(glyph_arr.astype(np.uint8), mode="L")

                # Natural height scaling
                if is_training:
                    scale = random.uniform(0.92, 1.08)
                    gh = int(round(base_h * scale))
                    gw = int(round(26 * scale))
                else:
                    gh = base_h
                    gw = 26

                g_img = g_img.resize((max(12, gw), max(16, gh)), Image.Resampling.BILINEAR)
                rendered_glyphs.append((g_img, ch))
            elif ch in ".-/#,&:":
                # Render punctuation with ink-like font
                ch_img = self._render_digit_or_symbol(ch, height=base_h)
                rendered_glyphs.append((ch_img, ch))
            elif ch == " ":
                # Space token
                space_w = random.randint(12, 18)
                rendered_glyphs.append((Image.new("L", (space_w, base_h), color=255), " "))

        if not rendered_glyphs:
            return Image.new("L", (32, target_height), color=255)

        # Calculate word canvas width
        spacing = random.randint(1, 4) if is_training else 2
        total_w = sum(g[0].width for g in rendered_glyphs) + max(0, len(rendered_glyphs) - 1) * spacing + 12

        word_canvas = Image.new("L", (total_w, target_height), color=255)
        cur_x = 6

        for g_img, ch in rendered_glyphs:
            # Baseline jitter (-2 to +2 pixels)
            jitter_y = random.randint(-2, 2) if is_training and (ch.isalpha() or ch.isdigit()) else 0
            pos_y = max(0, min(target_height - g_img.height, (target_height - g_img.height) // 2 + jitter_y))

            word_canvas.paste(g_img, (cur_x, pos_y))
            cur_x += g_img.width + spacing

        return word_canvas

    def _render_digit_or_symbol(self, ch: str, height: int = 28) -> Image.Image:
        """Render digits or punctuation symbols using default PIL font with ink styling."""
        dummy = Image.new("L", (24, height), color=255)
        draw = ImageDraw.Draw(dummy)
        draw.text((4, max(0, (height - 18) // 2)), ch, fill=random.randint(15, 50))
        return dummy

    def render_handwritten_line(
        self,
        text: str,
        target_height: int = 32,
        is_training: bool = True
    ) -> Tuple[Image.Image, str]:
        """
        Renders a full address text line using real EMNIST handwritten glyphs.
        """
        words = text.split(" ")
        word_images = []

        for w in words:
            if not w:
                continue
            w_img = self.render_handwritten_word(w, target_height=target_height, is_training=is_training)
            word_images.append(w_img)

        if not word_images:
            empty_img = Image.new("L", (64, target_height), color=255)
            return empty_img, text

        # Inter-word space width (14-22 pixels)
        word_gap = random.randint(14, 22) if is_training else 18
        line_w = sum(w.width for w in word_images) + (len(word_images) - 1) * word_gap + 16

        line_canvas = Image.new("L", (line_w, target_height), color=255)
        cur_x = 8

        for i, w_img in enumerate(word_images):
            line_canvas.paste(w_img, (cur_x, 0))
            cur_x += w_img.width + word_gap

        # Add subtle paper variation and ink blur
        if is_training:
            if random.random() < 0.25:
                line_canvas = line_canvas.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.2, 0.5)))

            # Sensor / paper noise
            arr = np.array(line_canvas, dtype=np.float32)
            noise = np.random.normal(0, random.uniform(2.0, 7.0), arr.shape)
            arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
            line_canvas = Image.fromarray(arr)

        return line_canvas, text.upper()


class EMNISTPostalDataset(Dataset):
    """
    PyTorch Dataset providing address lines composed of real handwritten EMNIST characters.
    Now uses the full unified glyph pool including real handwritten digits for PIN codes.
    Provides identical batch items to PostalLineDataset for seamless drop-in training.
    """

    def __init__(
        self,
        num_samples: int = 1500,
        target_height: int = 32,
        is_training: bool = True,
        csv_train: str = "dataset/emnist-letters-train.csv",
        csv_test: str = "dataset/emnist-letters-test.csv",
        use_all_splits: bool = True,
    ):
        super().__init__()
        self.num_samples = num_samples
        self.target_height = target_height
        self.is_training = is_training

        self.glyph_manager = EMNISTGlyphManager.get_instance(
            train_csv_path=csv_train,
            test_csv_path=csv_test,
            use_all_splits=use_all_splits,
        )

        # Common postal vocabulary templates (expanded with digit-heavy content)
        self.postal_phrases = [
            # Recipient names
            "DR SUNITA RAO", "MR ARJUN SHARMA", "MS PRIYA PATEL", "PROF RAJESH VERMA",
            "ANANYA IYER", "VIKRAM NAIR", "SNEHA DESHMUKH", "KIRAN KUMAR", "AMIT CHATTERJEE",
            # Street & locality lines
            "MG ROAD NEAR METRO STATION", "BRIGADE ROAD", "CONNAUGHT PLACE",
            "PARK STREET", "BANDRA WEST", "KORAMANGALA", "INDIRANAGAR", "SALT LAKE SECTOR",
            "ANNA NAGAR", "JUBILEE HILLS", "CIVIL LINES", "RAILWAY COLONY",
            # Building & Premise lines
            "FLAT SUNSHINE HEIGHTS", "GREEN VALLEY APARTMENTS", "SHANTI SADAN",
            "LOTUS TOWERS", "KRISHNA RESIDENCY", "SAI ENCLAVE", "POORNIMA NIWAS",
            # City, State & Postal PIN lines
            "BENGALURU KARNATAKA", "MUMBAI MAHARASHTRA", "NEW DELHI DELHI",
            "CHENNAI TAMIL NADU", "KOLKATA WEST BENGAL", "HYDERABAD TELANGANA",
            "PUNE MAHARASHTRA", "JAIPUR RAJASTHAN", "AHMEDABAD GUJARAT",
            # Standard postal tokens
            "POST BOX NUMBER", "SPEED POST", "DELIVERY POST OFFICE", "AIR MAIL",
            "GENERAL POST OFFICE", "HEAD POST OFFICE", "REGISTERED POST",
            # Premises with numbers (exercises real digit glyphs!)
            "FLAT 402 SUNSHINE HEIGHTS", "ROOM 301 KRISHNA TOWER",
            "BLOCK 12 GREEN PARK", "HOUSE 55 SECTOR 21",
            "PLOT 78 INDUSTRIAL AREA", "DOOR 16 ROSE GARDEN",
        ]

        # PIN code templates for training digit recognition
        self.pin_templates = [
            "{pin}", "PIN {pin}", "PIN CODE {pin}",
            "POSTAL CODE {pin}", "ZIP {pin}",
            "{city} {pin}", "{city} {state} {pin}",
        ]

    def __len__(self) -> int:
        return self.num_samples

    def _generate_text(self) -> str:
        """Selects or generates a diverse postal text line including digit content."""
        choice_type = random.random()

        if choice_type < 0.40:
            # Pick from authentic postal phrases (many now include numbers)
            text = random.choice(self.postal_phrases)
        elif choice_type < 0.60:
            # Generate PIN-focused lines for digit training
            city, state, pin = random.choice(CITIES_STATES_PINS)
            template = random.choice(self.pin_templates)
            text = template.format(pin=pin, city=city.upper(), state=state.upper())
        elif choice_type < 0.80:
            # Combine random name or city with postal terms
            title = random.choice(TITLES).upper()
            first = random.choice(FIRST_NAMES).upper()
            last = random.choice(LAST_NAMES).upper()
            text = f"{title} {first} {last}"
        elif choice_type < 0.92:
            # Address line with premise numbers
            premise = random.choice(PREMISES) if PREMISES else "FLAT"
            building = random.choice(BUILDINGS) if BUILDINGS else "TOWER"
            number = random.randint(1, 999)
            text = f"{premise.upper()} {number} {building.upper()}"
        else:
            # Full city + state + PIN line
            city, state, pin = random.choice(CITIES_STATES_PINS)
            text = f"{city.upper()} {state.upper()} {pin}"

        return text.strip()

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        text = self._generate_text()

        # Render line image using real EMNIST handwritten characters
        line_img, clean_text = self.glyph_manager.render_handwritten_line(
            text,
            target_height=self.target_height,
            is_training=self.is_training
        )

        # Normalize pixels to [-1.0, 1.0]
        img_np = np.array(line_img, dtype=np.float32) / 255.0
        img_norm = (img_np - 0.5) / 0.5
        img_tensor = torch.from_numpy(img_norm).unsqueeze(0)  # Shape: (1, 32, W)

        # Encode labels
        target_indices = torch.tensor(text_to_indices(clean_text), dtype=torch.long)

        return {
            "image": img_tensor,
            "text": clean_text,
            "target": target_indices,
            "target_length": len(target_indices),
            "width": img_tensor.shape[2]
        }


def create_emnist_dataloader(
    num_samples: int = 1500,
    batch_size: int = 32,
    shuffle: bool = True,
    is_training: bool = True,
    csv_train: str = "dataset/emnist-letters-train.csv",
    csv_test: str = "dataset/emnist-letters-test.csv",
    use_all_splits: bool = True,
) -> DataLoader:
    """Create a high-speed DataLoader backed by real EMNIST handwriting from ALL splits."""
    dataset = EMNISTPostalDataset(
        num_samples=num_samples,
        is_training=is_training,
        csv_train=csv_train,
        csv_test=csv_test,
        use_all_splits=use_all_splits,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=postal_collate_fn,
        num_workers=0
    )
