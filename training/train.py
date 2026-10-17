"""
Training Pipeline for ScratchPostalCRNN.
Trains the custom neural network from scratch using CTC Loss, AdamW, and Cosine Annealing.
"""

import os
import json
import time
from typing import Dict, Any, Optional
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from models.custom_postal_crnn import ScratchPostalCRNN
from dataset.vocab import NUM_CLASSES, BLANK_IDX, ctc_greedy_decode
from dataset.postal_dataset import create_postal_dataloader, PostalLineDataset
from training.metrics import evaluate_predictions


def train_scratch_crnn(
    epochs: int = 15,
    train_samples: int = 1500,
    val_samples: int = 300,
    batch_size: int = 32,
    lr: float = 8e-4,
    checkpoint_dir: str = "checkpoints",
    device: Optional[str] = None,
    dataset_mode: str = "emnist"
) -> Dict[str, Any]:
    """
    Trains the ScratchPostalCRNN model from scratch using the specified dataset mode:
    'emnist' (real handwritten letters), 'synthetic', or 'hybrid'.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Postal OCR] Training Scratch Model on device: {device} | Dataset: {dataset_mode.upper()}")

    os.makedirs(checkpoint_dir, exist_ok=True)
    best_checkpoint_path = os.path.join(checkpoint_dir, "best_postal_crnn.pt")
    history_path = os.path.join(checkpoint_dir, "training_history.json")

    # 1. Initialize custom scratch model
    model = ScratchPostalCRNN(num_classes=NUM_CLASSES)
    model.to(device)

    # 2. Loss and Optimizer
    # PyTorch CTC Loss handles alignments without requiring character-level bounding boxes
    criterion = nn.CTCLoss(blank=BLANK_IDX, zero_infinity=True)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=5e-5)

    # 3. Create DataLoaders
    train_loader = create_postal_dataloader(
        num_samples=train_samples,
        batch_size=batch_size,
        shuffle=True,
        is_training=True,
        dataset_mode=dataset_mode
    )
    val_loader = create_postal_dataloader(
        num_samples=val_samples,
        batch_size=batch_size,
        shuffle=False,
        is_training=False,
        dataset_mode=dataset_mode
    )

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_cer": [],
        "val_wer": [],
        "val_seq_acc": []
    }

    best_cer = float("inf")
    start_time = time.time()

    print(f"\n{'='*70}")
    print(f"  POSTAL ADDRESS OCR - SCRATCH DEEP NEURAL NETWORK TRAINING")
    print(f"  Dataset: {dataset_mode.upper()} (Real Handwritten EMNIST Letters)")
    print(f"  Total Scratch Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Epochs: {epochs} | Batch Size: {batch_size} | Train: {train_samples} | Val: {val_samples}")
    print(f"{'='*70}\n")

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        num_batches = 0

        for batch in train_loader:
            images = batch["images"].to(device)
            targets = batch["targets"].to(device)
            target_lengths = batch["target_lengths"].to(device)
            input_lengths = batch["input_lengths"].to(device)

            optimizer.zero_grad()

            # Forward pass: shape (Batch, Time, Classes)
            logits = model(images)

            # CTCLoss expects log_probs of shape (Time, Batch, Classes)
            log_probs = logits.log_softmax(dim=-1).permute(1, 0, 2)

            loss = criterion(log_probs, targets, input_lengths, target_lengths)

            if torch.isnan(loss) or torch.isinf(loss):
                continue

            loss.backward()
            # Gradient clipping to stabilize recurrent layers
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            running_loss += loss.item()
            num_batches += 1

        avg_train_loss = running_loss / max(1, num_batches)
        scheduler.step()

        # Validation phase
        model.eval()
        val_running_loss = 0.0
        val_batches = 0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch in val_loader:
                images = batch["images"].to(device)
                targets = batch["targets"].to(device)
                target_lengths = batch["target_lengths"].to(device)
                input_lengths = batch["input_lengths"].to(device)
                texts = batch["texts"]

                logits = model(images)
                log_probs = logits.log_softmax(dim=-1).permute(1, 0, 2)
                v_loss = criterion(log_probs, targets, input_lengths, target_lengths)

                if not (torch.isnan(v_loss) or torch.isinf(v_loss)):
                    val_running_loss += v_loss.item()
                    val_batches += 1

                # Decode predictions
                decoded_batch, _ = ctc_greedy_decode(logits.cpu(), lengths=input_lengths.tolist())
                all_preds.extend(decoded_batch)
                all_targets.extend(texts)

        avg_val_loss = val_running_loss / max(1, val_batches)
        metrics = evaluate_predictions(all_preds, all_targets)

        history["train_loss"].append(round(avg_train_loss, 4))
        history["val_loss"].append(round(avg_val_loss, 4))
        history["val_cer"].append(round(metrics["character_error_rate_cer"], 4))
        history["val_wer"].append(round(metrics["word_error_rate_wer"], 4))
        history["val_seq_acc"].append(round(metrics["exact_sequence_accuracy"], 4))

        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | "
            f"CER: {metrics['character_error_rate_cer']*100:.2f}% | "
            f"WER: {metrics['word_error_rate_wer']*100:.2f}% | "
            f"Seq Acc: {metrics['exact_sequence_accuracy']*100:.2f}%"
        )

        # Print a sample comparison every 3 epochs
        if epoch % 3 == 0 or epoch == epochs:
            print(f"   [Sample GT]   : {all_targets[0]}")
            print(f"   [Sample Pred] : {all_preds[0]}")

        # Save best model based on CER
        if metrics["character_error_rate_cer"] < best_cer:
            best_cer = metrics["character_error_rate_cer"]
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_cer": best_cer,
                "model_summary": model.get_architecture_summary()
            }, best_checkpoint_path)

    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time/60:.2f} minutes.")
    print(f"Best CER: {best_cer*100:.2f}% saved to {best_checkpoint_path}")

    # Save history
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    return history


if __name__ == "__main__":
    train_scratch_crnn(epochs=10, train_samples=1000, val_samples=200, batch_size=32)
