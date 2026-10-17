"""
Vocabulary and character mapping for Postal Address OCR.
Contains alphanumeric characters, postal punctuation, space, and CTC blank token.
"""

from typing import List, Tuple, Optional
import torch

# Define postal character set:
# 0 is reserved for CTC blank token
BLANK_TOKEN = "<BLANK>"

DIGITS = "0123456789"
UPPERCASE = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
LOWERCASE = "abcdefghijklmnopqrstuvwxyz"
PUNCTUATION = " ,.-/#:&()'\";"

VOCAB = [BLANK_TOKEN] + list(DIGITS) + list(UPPERCASE) + list(LOWERCASE) + list(PUNCTUATION)

CHAR2IDX = {ch: idx for idx, ch in enumerate(VOCAB)}
IDX2CHAR = {idx: ch for idx, ch in enumerate(VOCAB)}
NUM_CLASSES = len(VOCAB)
BLANK_IDX = 0


def text_to_indices(text: str) -> List[int]:
    """Convert text string into integer indices using VOCAB."""
    indices = []
    for ch in text:
        if ch in CHAR2IDX:
            indices.append(CHAR2IDX[ch])
        else:
            # Fallback to uppercase or space if unknown
            if ch.upper() in CHAR2IDX:
                indices.append(CHAR2IDX[ch.upper()])
            else:
                indices.append(CHAR2IDX[" "])
    return indices


def indices_to_text(indices: List[int]) -> str:
    """Convert integer indices back to text string, ignoring blank."""
    chars = []
    for idx in indices:
        if idx != BLANK_IDX and idx in IDX2CHAR:
            chars.append(IDX2CHAR[idx])
    return "".join(chars)


def ctc_greedy_decode(
    logits_or_probs: torch.Tensor,
    lengths: Optional[List[int]] = None
) -> Tuple[List[str], List[float]]:
    """
    Greedy CTC decoding for batched outputs.
    Input shape: (Batch, Time, NumClasses) or (Time, Batch, NumClasses)
    Optional lengths: list of valid sequence timesteps per sample (ignoring right-padding).
    Returns:
        decoded_texts: list of decoded strings
        confidences: list of average sequence confidence scores
    """
    if logits_or_probs.dim() == 3 and logits_or_probs.shape[1] != NUM_CLASSES and logits_or_probs.shape[2] == NUM_CLASSES:
        # Shape is (Batch, Time, NumClasses)
        batch_probs = torch.softmax(logits_or_probs, dim=-1)
    else:
        # Transpose if needed
        batch_probs = torch.softmax(logits_or_probs.transpose(0, 1), dim=-1)

    batch_size, time_steps, _ = batch_probs.shape
    decoded_texts = []
    confidences = []

    for b in range(batch_size):
        probs = batch_probs[b]  # (Time, NumClasses)
        valid_t = time_steps if lengths is None or b >= len(lengths) else min(time_steps, lengths[b])

        max_probs, max_indices = torch.max(probs[:valid_t], dim=-1)

        indices = max_indices.tolist()
        step_confidences = max_probs.tolist()

        # CTC collapse: remove consecutive duplicate characters and remove blanks
        collapsed_chars = []
        char_confs = []
        prev_idx = -1

        for t, idx in enumerate(indices):
            if idx != prev_idx:
                if idx != BLANK_IDX:
                    ch = IDX2CHAR.get(idx, "")
                    # Prevent redundant consecutive whitespace
                    if not (ch == " " and collapsed_chars and collapsed_chars[-1] == " "):
                        collapsed_chars.append(ch)
                        char_confs.append(step_confidences[t])
                prev_idx = idx

        decoded_text = "".join(collapsed_chars).strip()
        decoded_texts.append(decoded_text)
        avg_conf = float(sum(char_confs) / len(char_confs)) if char_confs else 1.0
        confidences.append(avg_conf)

    return decoded_texts, confidences
