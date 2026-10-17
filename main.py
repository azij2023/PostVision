"""
Main CLI Interface for Postal Address OCR System.
Usage:
  python main.py generate [--num 10]
  python main.py train [--epochs 10] [--train_samples 1000] [--val_samples 200]
  python main.py infer --image <path_to_image>
  python main.py evaluate [--num 300]
  python main.py ui
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import argparse
import json
from PIL import Image
import torch

from dataset.address_generator import render_postal_envelope
from training.train import train_scratch_crnn
from inference import PostalAddressOCRSystem
from training.metrics import evaluate_predictions
from dataset.postal_dataset import create_postal_dataloader
from dataset.vocab import ctc_greedy_decode


def cmd_generate(args):
    """Generate realistic synthetic postal envelopes."""
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    print(f"\n[Generate] Generating {args.num} synthetic postal envelopes in '{out_dir}'...")

    for i in range(1, args.num + 1):
        env_img, meta = render_postal_envelope()
        out_path = os.path.join(out_dir, f"sample_envelope_{i}.png")
        env_img.save(out_path)
        print(f"  [{i}/{args.num}] Saved: {out_path} (PIN: {meta['record']['pin']})")

    print(f"[Generate] Completed! {args.num} envelopes generated.\n")


def cmd_train(args):
    """Train the Scratch Deep Neural Network on EMNIST, synthetic, or hybrid dataset."""
    train_scratch_crnn(
        epochs=args.epochs,
        train_samples=args.train_samples,
        val_samples=args.val_samples,
        batch_size=args.batch_size,
        lr=args.lr,
        checkpoint_dir=args.checkpoint_dir,
        dataset_mode=args.dataset
    )


def cmd_infer(args):
    """Run end-to-end OCR and address extraction on an image."""
    if not os.path.exists(args.image):
        print(f"Error: Image file not found: {args.image}")
        sys.exit(1)

    print(f"\n[Inference] Processing image: {args.image}")
    ocr = PostalAddressOCRSystem(checkpoint_path=args.checkpoint)

    img = Image.open(args.image)
    result = ocr.process_envelope(img)

    print("\n" + "=" * 60)
    print("  POSTAL OCR TRANSCRIPTION & ADDRESS EXTRACTION RESULTS")
    print("=" * 60)
    print(f"Lines Detected: {result['num_lines_detected']}")

    for line in result["line_details"]:
        print(f"  Line {line['line_number']} [{line['confidence']*100:.1f}%]: {line['transcribed_text']}")

    print("\n--- Structured Address Fields ---")
    struct = result["structured_address"]
    print(f"  Recipient Name     : {struct.get('recipient_name', 'N/A')}")
    print(f"  Premise / Building : {struct.get('premise_building', 'N/A')}")
    print(f"  Street / Locality  : {struct.get('street_locality', 'N/A')}")
    print(f"  City               : {struct.get('city', 'N/A')}")
    print(f"  State              : {struct.get('state', 'N/A')}")
    print(f"  Postal / PIN Code  : {struct.get('postal_code', 'N/A')}")
    print(f"  Overall Confidence : {struct.get('overall_confidence', 0.0)*100:.1f}%")
    print("=" * 60 + "\n")

    if args.save_json:
        with open(args.save_json, "w") as f:
            json.dump(struct, f, indent=2)
        print(f"[Inference] Saved structured JSON to: {args.save_json}\n")


def cmd_evaluate(args):
    """Evaluate ScratchPostalCRNN on a test dataset."""
    print(f"\n[Evaluate] Evaluating model on {args.num} test postal lines (Dataset: {args.dataset.upper()})...")
    ocr = PostalAddressOCRSystem(checkpoint_path=args.checkpoint)

    test_loader = create_postal_dataloader(
        num_samples=args.num,
        batch_size=32,
        shuffle=False,
        is_training=False,
        dataset_mode=args.dataset
    )

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in test_loader:
            images = batch["images"].to(ocr.device)
            texts = batch["texts"]
            lengths = batch["input_lengths"].tolist()
            logits = ocr.model(images)
            decoded, _ = ctc_greedy_decode(logits.cpu(), lengths=lengths)
            all_preds.extend(decoded)
            all_targets.extend(texts)

    metrics = evaluate_predictions(all_preds, all_targets)

    print("\n" + "=" * 60)
    print(f"  BENCHMARK EVALUATION RESULTS ({args.dataset.upper()} DATASET)")
    print("=" * 60)
    print(f"  Character Error Rate (CER) : {metrics['character_error_rate_cer']*100:.2f}%")
    print(f"  Word Error Rate (WER)      : {metrics['word_error_rate_wer']*100:.2f}%")
    print(f"  Exact Sequence Accuracy    : {metrics['exact_sequence_accuracy']*100:.2f}%")
    print(f"  Char Recognition Accuracy  : {metrics['character_recognition_accuracy']*100:.2f}%")
    print("=" * 60 + "\n")


def cmd_emnist_demo(args):
    """Demonstrate real EMNIST handwritten letter assembly and OCR transcription."""
    print(f"\n[EMNIST Demo] Assembling handwritten postal line from real EMNIST glyphs...")
    from dataset.emnist_loader import EMNISTGlyphManager
    manager = EMNISTGlyphManager.get_instance()

    sample_text = args.text
    img, text = manager.render_handwritten_line(sample_text, is_training=False)
    print(f"  Target Text : '{text}'")
    print(f"  Line Image  : {img.size[0]}x{img.size[1]} px")

    ocr = PostalAddressOCRSystem(checkpoint_path=args.checkpoint)
    pred_text, conf = ocr.transcribe_line_image(img)

    print("\n" + "=" * 50)
    print(f"  Ground Truth   : {text}")
    print(f"  Transcribed    : {pred_text}")
    print(f"  Confidence     : {conf * 100:.1f}%")
    print("=" * 50 + "\n")


def cmd_dataset_stats(args):
    """Display comprehensive statistics about all loaded EMNIST dataset splits."""
    from dataset.emnist_loader import EMNISTGlyphManager
    print("\n[Dataset Stats] Loading ALL EMNIST splits from archive.zip...\n")
    EMNISTGlyphManager.reset_instance()
    manager = EMNISTGlyphManager.get_instance(use_all_splits=True)
    stats = manager.get_dataset_stats()

    print("\n" + "=" * 70)
    print("  EMNIST FULL DATASET STATISTICS")
    print("=" * 70)
    print(f"  Loaded Splits          : {', '.join(stats['loaded_splits'])}")
    print(f"  Total Train Samples    : {stats['total_train_samples']:,}")
    print(f"  Total Test Samples     : {stats['total_test_samples']:,}")
    print(f"  Unique Characters      : {stats['total_unique_characters']}")
    print(f"  Total Letter Glyphs    : {stats['total_letter_glyphs']:,}")
    print(f"  Total Digit Glyphs     : {stats['total_digit_glyphs']:,}")
    print("\n  Letter Glyph Counts (Train):")
    for ch, count in sorted(stats['letter_glyph_counts'].items()):
        bar = '#' * min(50, count // 500)
        print(f"    {ch}: {count:>8,}  {bar}")
    print("\n  Digit Glyph Counts (Train):")
    for d, count in sorted(stats['digit_glyph_counts'].items()):
        bar = '#' * min(50, count // 500)
        print(f"    {d}: {count:>8,}  {bar}")
    print("=" * 70 + "\n")


def cmd_ui(args):
    """Launch PostalVision interactive web dashboard."""
    import subprocess
    app_path = os.path.join(os.path.dirname(__file__), "webapp", "app.py")
    subprocess.run([sys.executable, app_path])


def main():
    parser = argparse.ArgumentParser(description="Postal Address Image/OCR to Text Data (Scratch Deep Neural Network)")
    subparsers = parser.add_subparsers(dest="command", help="Sub-commands")

    # Generate
    p_gen = subparsers.add_parser("generate", help="Generate synthetic postal envelopes")
    p_gen.add_argument("--num", type=int, default=5, help="Number of samples")
    p_gen.add_argument("--out_dir", type=str, default="sample_data", help="Output directory")

    # Train
    p_train = subparsers.add_parser("train", help="Train the scratch deep neural network")
    p_train.add_argument("--dataset", type=str, choices=["emnist", "synthetic", "hybrid"], default="emnist", help="Dataset source")
    p_train.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    p_train.add_argument("--train_samples", type=int, default=1200, help="Number of training samples")
    p_train.add_argument("--val_samples", type=int, default=200, help="Number of validation samples")
    p_train.add_argument("--batch_size", type=int, default=32, help="Batch size")
    p_train.add_argument("--lr", type=float, default=8e-4, help="Learning rate")
    p_train.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Checkpoint directory")

    # Infer
    p_infer = subparsers.add_parser("infer", help="Run OCR and address parsing on an envelope image")
    p_infer.add_argument("--image", type=str, required=True, help="Path to envelope image")
    p_infer.add_argument("--checkpoint", type=str, default="checkpoints/best_postal_crnn.pt", help="Path to checkpoint")
    p_infer.add_argument("--save_json", type=str, default=None, help="Save structured JSON to file")

    # Evaluate
    p_eval = subparsers.add_parser("evaluate", help="Evaluate model accuracy on test set")
    p_eval.add_argument("--dataset", type=str, choices=["emnist", "synthetic", "hybrid"], default="emnist", help="Dataset source")
    p_eval.add_argument("--num", type=int, default=200, help="Number of test samples")
    p_eval.add_argument("--checkpoint", type=str, default="checkpoints/best_postal_crnn.pt", help="Path to checkpoint")

    # EMNIST Demo
    p_edemo = subparsers.add_parser("emnist-demo", help="Render and test handwritten EMNIST line")
    p_edemo.add_argument("--text", type=str, default="BENGALURU KARNATAKA", help="Text to render with real handwriting")
    p_edemo.add_argument("--checkpoint", type=str, default="checkpoints/best_postal_crnn.pt", help="Path to checkpoint")

    # Dataset Stats
    subparsers.add_parser("dataset-stats", help="Show EMNIST full dataset statistics")

    # UI
    subparsers.add_parser("ui", help="Launch interactive PostalVision Web UI")

    args = parser.parse_args()

    if args.command == "generate":
        cmd_generate(args)
    elif args.command == "train":
        cmd_train(args)
    elif args.command == "infer":
        cmd_infer(args)
    elif args.command == "evaluate":
        cmd_evaluate(args)
    elif args.command == "emnist-demo":
        cmd_emnist_demo(args)
    elif args.command == "dataset-stats":
        cmd_dataset_stats(args)
    elif args.command == "ui":
        cmd_ui(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
