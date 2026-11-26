# DermAssist — Federated-Learning-Based Skin Disease Detection

Final Year Project, Department of Electrical & Computer Engineering, COMSATS University Islamabad (Lahore Campus).

DermAssist is a mobile app that classifies skin images as **eczema** or **healthy**. The model is an EfficientNet-B3 that runs on the phone (TFLite), so a prediction works offline. Users can choose to contribute images, and a Flask backend then fine-tunes a global model with Federated Averaging (FedAvg).

> Screening aid, not a medical diagnosis.

## Architecture

```
Flutter app (TFLite, on-device inference)
   │  optional image contribution
   ▼
Flask backend ── Google Drive (images) + Firestore (metadata, quality metrics)
   │
   ├─ /download-images  → sorts contributed images into per-client folders
   └─ /run-fedavg       → local fine-tuning per client → FedAvg → new global model
                           → PyTorch → ONNX → TensorFlow → TFLite → shipped to app
```

| Part | Stack |
|---|---|
| Model | EfficientNet-B3 (`timm`), 300×300 input, 2 classes |
| Baselines compared | EfficientNet-B0, ViT-B/16 |
| Preprocessing | OpenCV lesion pipeline (hair removal, contrast enhancement, segmentation) |
| Backend | Flask, Google Drive API, Firestore, gunicorn |
| Mobile | Flutter + TFLite (in a separate repo) |

## Results

**Phase 2 (final model), balanced held-out test set: 154 images (77 eczema / 77 healthy)**

| Class | Precision | Recall | F1 |
|---|---|---|---|
| Eczema | 0.97 | 0.90 | 0.93 |
| Healthy | 0.90 | 0.97 | 0.94 |
| **Overall** | Accuracy **93.5%** (144/154) | AUC **0.989** | Macro F1 **0.94** |

Phase 1 baselines reached about 99% accuracy. That test set was heavily imbalanced (1212 eczema vs 98 healthy), so the Phase 1 number overstates performance and is not the headline result. Plots are in `results/`.

## Repository layout

```
backend/            Flask API: upload + quality check, Drive/Firestore sync, FedAvg
  config.py         all paths/IDs come from environment variables
training/
  phase1_baselines/ first experiments: EfficientNet-B0, B3, ViT
  efficientnet_b3/  final model: split, train (with/without DIP), evaluate
  vit_b16/          ViT comparison
  export/           PyTorch → ONNX → TF → TFLite
preprocessing/      lesion image-processing pipeline
results/            confusion matrices, ROC/AUC plots
docs/screenshots/   app screenshots
models/labels.txt   class labels (weights are in Releases)
```

## Setup

```bash
# backend
cd backend
pip install -r requirements.txt
cp ../.env.example .env        # fill in your own values
export $(cat .env | xargs)
python app.py                  # or: gunicorn app:app
```

You need your own Google Cloud service account with Drive and Firestore access. Point `GOOGLE_APPLICATION_CREDENTIALS` at its key file. **Never commit the key.** `.gitignore` blocks `*.json`.

Model weights (`eczema_classifier_latest.pth`, `.tflite`) are attached to the **GitHub Release**. Put them in `models/`.

Training scripts expect data under `data/` (for example `data/dermo`, `data/dermo_dip`). The dataset is not redistributed.

```bash
cd training && pip install -r requirements.txt
python efficientnet_b3/train.py
```

### Endpoints

| Method | Path | Purpose |
|---|---|---|
| POST | `/upload` (blueprint in `uploader.py`) | receive image, run skin/quality checks, classify, store to Drive + Firestore |
| GET/POST | `/download-images` | pull contributed images into per-client folders |
| GET/POST | `/run-fedavg` | run a federated round and save the new global model |
| GET | `/client-stats`, `/dataset-info`, `/health` | status and dataset info |

## Known limitations

These are stated plainly on purpose:

- **Simulated federated learning.** Contributed images are uploaded to central storage, and "clients" are folders on the server. Real FL keeps raw data on the device and shares only weight updates. The FedAvg logic is real; the privacy guarantee is not there yet.
- **Small FL run.** The recorded federated round used 1 client and 4 samples. It shows the pipeline works, not that FL improves accuracy.
- **Preprocessing mismatch.** `global_model.py` fine-tunes with ImageNet normalization, but `single_test.py` inference does not normalize. Fix this before you trust backend predictions.
- Binary task only: eczema vs healthy.
- No clinical validation.

## Roadmap

- On-device training (true FL) with secure aggregation
- Fix the normalization mismatch, then add tests
- More classes (psoriasis, acne, etc.)

## Team

- Muhammad Ali Imran Khan
- Fatima Saqib
- Neha Hamid

Supervisor: Dr. Jehangir Arshad · Co-supervisor: Dr. Usman Iqbal

<!-- TODO: Contributions — who built what -->
<!-- TODO: Demo video link -->
