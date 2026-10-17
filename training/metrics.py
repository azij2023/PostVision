"""
Evaluation Metrics for Postal OCR:
- Levenshtein Edit Distance
- Character Error Rate (CER)
- Word Error Rate (WER)
- Exact Match Sequence Accuracy
"""

from typing import List, Tuple, Dict


def levenshtein_distance(s1: str, s2: str) -> int:
    """Computes Levenshtein edit distance between two strings using dynamic programming."""
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(
                    dp[i - 1][j],      # deletion
                    dp[i][j - 1],      # insertion
                    dp[i - 1][j - 1]   # substitution
                )
    return dp[m][n]


def compute_cer(predictions: List[str], ground_truths: List[str]) -> float:
    """Computes Character Error Rate (CER) across all sequences."""
    total_dist = 0
    total_len = 0
    for pred, gt in zip(predictions, ground_truths):
        dist = levenshtein_distance(pred, gt)
        total_dist += dist
        total_len += max(1, len(gt))
    return float(total_dist / total_len) if total_len > 0 else 0.0


def compute_wer(predictions: List[str], ground_truths: List[str]) -> float:
    """Computes Word Error Rate (WER) across all sequences."""
    total_dist = 0
    total_words = 0
    for pred, gt in zip(predictions, ground_truths):
        p_words = pred.strip().split()
        g_words = gt.strip().split()

        # Word-level edit distance
        m, n = len(p_words), len(g_words)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if p_words[i - 1] == g_words[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])

        total_dist += dp[m][n]
        total_words += max(1, len(g_words))
    return float(total_dist / total_words) if total_words > 0 else 0.0


def compute_sequence_accuracy(predictions: List[str], ground_truths: List[str]) -> float:
    """Computes exact string match ratio."""
    correct = sum(1 for p, g in zip(predictions, ground_truths) if p.strip() == g.strip())
    return float(correct / len(ground_truths)) if ground_truths else 0.0


def evaluate_predictions(predictions: List[str], ground_truths: List[str]) -> Dict[str, float]:
    """Computes full OCR evaluation metrics dictionary."""
    cer = compute_cer(predictions, ground_truths)
    wer = compute_wer(predictions, ground_truths)
    seq_acc = compute_sequence_accuracy(predictions, ground_truths)
    return {
        "character_error_rate_cer": cer,
        "word_error_rate_wer": wer,
        "exact_sequence_accuracy": seq_acc,
        "character_recognition_accuracy": max(0.0, 1.0 - cer)
    }
