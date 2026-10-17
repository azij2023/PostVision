# ✉️ Postal Address Image / OCR to Structured Text Data

### Built with Deep Neural Networks Designed Strictly from Scratch
> **Course / Project**: MSDSM Sem 3 - AI & Machine Learning  
> **Key Constraint Satisfied**: **100% Custom Deep Neural Networks** designed from first principles with zero standard model dependencies (**No ResNet, No DenseNet, No MobileNet, No VGG, No EfficientNet**).

---

## 📌 Executive Summary

Manual sorting of postal envelopes and logistics packages is slow, error-prone, and expensive. This project implements an **end-to-end Deep Learning OCR and Address Extraction System** capable of taking raw postal mail envelope scans or photographs, performing automated document layout analysis and text line segmentation, transcribing address lines using a **custom Convolutional Recurrent Neural Network (CRNN) designed from scratch**, and parsing the unstructured text into clean digital fields:

- **Recipient Name**
- **Premise / Flat / Building**
- **Street / Locality**
- **City / District**
- **State / Union Territory**
- **Postal PIN / ZIP Code**

---

## 🚫 Why Standard Models Were Strictly Excluded

Standard computer vision backbones (ResNet-50, DenseNet-121, MobileNetV3) were developed for **symmetrical 2D object classification** (e.g., ImageNet $224 \times 224$). They enforce isotropic spatial downsampling of $32\times$ in **both** height and width ($\frac{H}{32}, \frac{W}{32}$).

For text transcription, isotropic downsampling has catastrophic consequences:
1. **Severe Sequence Compression**: In a horizontal text strip of dimensions $32 \times 320$, downsampling width by $32\times$ reduces temporal sequence length to only 10 timesteps, making it mathematically impossible to represent an address line containing 35–45 characters.
2. **Receptive Field Mismatch**: Text strokes have high vertical gradients but narrow horizontal inter-character spacing.
3. **Black-box Inductive Bias**: Pretrained weights conceal the underlying inductive biases.

### Our Solution: `ScratchPostalCRNN`
Our architecture employs **asymmetric pooling** ($2\times 2$ early, $2\times 1$ late) to compress height from $32 \to 1$ while preserving horizontal sequence length ($\frac{W}{4}$), followed by a **Spatial-to-Sequence bridge**, a **2-layer Bidirectional GRU (BiGRU)**, and a **Connectionist Temporal Classification (CTC)** transcription head.

---

## 🧠 System Architecture

```
[Raw Envelope / Post Image]
           │
           ▼
[Preprocessing & Layout Analysis]
   ├─ CLAHE Contrast Enhancement
   ├─ Bilateral Filtering
   ├─ Morphological Word/Line Dilation
   └─ Connected Component Line Bounding
           │
           ▼
[Segmented Text Line Crops (32 x W)]
           │
           ▼
[ScratchPostalCRNN (Designed from Scratch)]
   ├─ Stage 1: Stem Conv (5x5, 32 channels) + BN + LeakyReLU + MaxPool(2x2)
   ├─ Stage 2: Detail Block (3x3, 64 channels) + BN + LeakyReLU + MaxPool(2x2)
   ├─ Stage 3: Glyph Block (2x 3x3, 128 channels) + BN + Asymmetric MaxPool(2x1)
   ├─ Stage 4: Morphology Block (2x 3x3, 256 channels) + BN + Asymmetric MaxPool(2x1)
   ├─ Stage 5: Height Collapse (2x1 Conv, 256 channels) -> Height = 1
   ├─ Stage 6: Map-to-Sequence Bridge (Linear 256->192 + LayerNorm)
   ├─ Stage 7: Contextual Sequence Modeling (2-Layer BiGRU, hidden=128)
   └─ Stage 8: Transcription Head (Linear -> 76 Classes) + CTC Loss
           │
           ▼
[CTC Decoding (Greedy / Beam Search)]
   └─ Collapses repetitive character activations and blanks
           │
           ▼
[Postal Address Entity Parser]
   ├─ Recipient Name Extraction
   ├─ Premise & Street Disambiguation
   ├─ City & State Lexical Matching
   └─ 6-Digit / 5-Digit Postal PIN Extraction
           │
           ▼
[Structured JSON / CSV / Digital Record]
```

---

## 📁 Repository Structure

```
AIML/
├── dataset/
│   ├── archive.zip            # Full EMNIST dataset (auto-extracted on first run)
│   ├── emnist_loader.py       # Full EMNIST loader: letters + digits + byclass + balanced + bymerge + mnist
│   ├── vocab.py               # Vocabulary mapping, character indices & CTC decoding
│   ├── address_generator.py   # High-fidelity synthetic postal envelope & line renderer
│   ├── postal_dataset.py      # PyTorch Dataset & dynamic aspect-ratio batch collator
│   └── external_dataset_loader.py  # Loader for external/custom datasets
├── models/
│   ├── custom_postal_crnn.py  # 100% scratch CRNN + BiGRU + CTC architecture
│   └── postal_pin_net.py      # Dedicated 6-digit scratch multi-head PIN classifier
├── preprocessing/
│   ├── envelope_preprocessor.py # Perspective deskew, shadow neutralizer & CLAHE
│   └── layout_segmenter.py    # Envelope document layout analysis & text line segmenter
├── postprocessing/
│   ├── address_parser.py      # Regex & heuristic structured postal address extractor
│   └── postal_geocoder.py     # Offline 19,000+ PIN India Post geocoder & RMS hubs
├── training/
│   ├── train.py               # Training loop with CTC loss, AdamW, and Cosine Annealing
│   └── metrics.py             # Levenshtein distance, CER, WER, and Sequence Accuracy
├── webapp/
│   ├── app.py                 # Production Flask REST API (upload, webcam, geocoding)
│   └── static/                # Interactive UI (split slider, radial dial, speech TTS)
├── checkpoints/               # Trained model weights & training history logs
├── sample_data/               # Pre-generated test envelopes
├── inference.py               # End-to-end inference pipeline class
├── main.py                    # Unified Command-Line Interface (CLI)
└── requirements.txt           # Python dependency requirements
```

---

## 🚀 Quickstart Guide

### 1. Requirements
Ensure the following packages are installed:
```bash
pip install torch torchvision opencv-python pillow numpy pandas matplotlib flask
```

### 2. Generate Realistic Envelope Samples
Generate 5 photorealistic synthetic postal envelopes complete with stamps and cancellation postmarks:
```bash
python main.py generate --num 5
```

### 3. Run Inference on an Image
Process any envelope image into structured text:
```bash
python main.py infer --image sample_data/sample_envelope_1.png --save_json output.json
```

### 4. Train the Scratch Neural Network on Real Handwriting (Full EMNIST)
Train the custom neural network using the **Full EMNIST Dataset** (all 6 splits auto-extracted from `archive.zip`: ~1.9M real handwritten characters including letters A-Z AND digits 0-9):
```bash
# Train on Real EMNIST Handwriting with full dataset (letters + digits) — Default:
python main.py train --dataset emnist --epochs 10 --train_samples 1200 --val_samples 200 --batch_size 32

# Or train on hybrid mixture (50% real handwriting + 50% computer fonts):
python main.py train --dataset hybrid --epochs 10 --train_samples 1200 --val_samples 200
```

### 5. View Full Dataset Statistics
Inspect how many real handwritten glyphs are available per character across all loaded EMNIST splits:
```bash
python main.py dataset-stats
```

### 6. Test Real Handwriting OCR (EMNIST Live Demo)
Assemble any destination address using real human-drawn characters (now including digits!) and transcribe live:
```bash
python main.py emnist-demo --text "FLAT 402 MUMBAI 400001"
```

### 7. Benchmark Evaluation on Real Handwriting
Compute Character Error Rate (CER), Word Error Rate (WER), and Exact Match accuracy:
```bash
python main.py evaluate --dataset emnist --num 200
```

### 8. Launch the Interactive Web Dashboard
```bash
python main.py ui
# Or start directly:
python webapp/app.py
```
Open [http://127.0.0.1:5000](http://127.0.0.1:5000) in any browser.

---

## 📊 Evaluation Metrics

| Metric | Definition | Significance in Postal Automation |
| :--- | :--- | :--- |
| **CER (Character Error Rate)** | $\frac{S + D + I}{N}$ via Levenshtein Distance | Evaluates raw character-level transcription fidelity |
| **WER (Word Error Rate)** | Word-level normalized edit distance | Evaluates word tokenization and spelling accuracy |
| **Sequence Accuracy** | Exact string match percentage | Critical for automated high-speed mail sorting |
| **PIN Extraction Rate** | Accuracy of 6-digit PIN code extraction | Directly determines mail destination routing bin |

---

## 🎓 Academic Submission Compliance
- **No Pre-trained Weights**: The network weights are initialized from scratch using Kaiming Normal and Xavier Uniform distributions.
- **No Third-Party OCR Engines**: Tesseract, EasyOCR, PaddleOCR, and TrOCR are completely absent.
- **Architectural Traceability**: Every forward pass layer can be audited and inspected via `model.get_architecture_summary()` and `model.extract_features()`.
