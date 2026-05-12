# Sign Language Translator

Real-time sign language word recognition using MediaPipe hand landmarks and a Bidirectional LSTM model, trained on the MS-ASL dataset.

## Recognised words (21)

hello,      please,     sorry,      yes,        no,
help,       water,      more,       go,         good,
bad,        want,       like,       love,       know,
understand, name,       where,      what,       how,
again

---

## Project structure

```
sign_language_translator/
├── main.py                  ← entry point (run this)
├── config.py                ← all constants and paths
├── utils.py                 ← shared helpers
├── data_loader.py           ← Phase 2: download MS-ASL clips
├── feature_extraction.py    ← Phase 3: extract MediaPipe landmarks
├── preprocessing.py         ← Phase 3: split + augment dataset
├── model_training.py        ← Phase 4: build and train LSTM model
├── predictor.py             ← Phase 5: real-time inference engine
├── ui.py                    ← Phase 5: OpenCV overlay and display
├── requirements.txt
├── dataset/
│   └── ms_asl/              ← downloaded clips + features.npz go here
├── model/                   ← trained model saved here
└── logs/                    ← training curves and TensorBoard logs
```

---

## Setup

### 1. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 2. Install ffmpeg

ffmpeg is required to trim video clips. Install it for your OS:

- **macOS**:  `brew install ffmpeg`
- **Ubuntu**: `sudo apt install ffmpeg`
- **Windows**: download from https://ffmpeg.org/download.html and add to PATH

### 3. Verify your environment

```bash
python main.py --check
```

---

## Step-by-step workflow

### Phase 2 — Download the dataset

1. Go to https://www.microsoft.com/en-us/research/project/ms-asl/
2. Download the three JSON split files:
   - `MSASL_train.json`
   - `MSASL_val.json`
   - `MSASL_test.json`
3. Place them in `dataset/ms_asl/`
4. Run the downloader:

```bash
python main.py --download
```

This downloads up to 150 video clips per word (~3,750 clips total).
Expect this to take 1–3 hours depending on your internet speed.
Failed downloads (deleted YouTube videos) are automatically skipped.

### Phase 3 — Extract landmarks

```bash
python main.py --extract
```

Runs MediaPipe Hands on every clip, samples 30 frames per clip,
and saves the normalised landmark sequences to `dataset/ms_asl/features.npz`.

### Phase 4 — Train the model

```bash
python main.py --train
```

Trains a Bidirectional LSTM for up to 60 epochs with early stopping.
The best model is saved to `model/sign_model.h5`.
Training curves are saved to `logs/training_history.png`.

Typical training time: 10–30 minutes on CPU, 3–8 minutes on GPU.

### Phase 5 — Run the real-time translator

```bash
python main.py
```

Controls:
| Key     | Action                                      |
|---------|---------------------------------------------|
| `SPACE` | Confirm the current predicted word          |
| `R`     | Reset the sentence buffer                   |
| `Q`     | Quit                                        |

---

## Tips for best accuracy

- **Lighting**: make sure your hand is well-lit and the background is plain.
- **Distance**: keep your hand 40–70 cm from the camera.
- **Speed**: sign at a natural pace — the model uses 30-frame (~1 second) windows.
- **Confidence threshold**: if predictions are too jittery, raise `CONFIDENCE_THRESHOLD` in `config.py` (e.g. to 0.82).
- **Smoothing**: increase `SMOOTHING_BUFFER` in `config.py` for more stable but slightly slower predictions.

---

## Retraining on more words

1. Add new words to the `LABELS` list in `config.py`.
2. Re-run `--download`, `--extract`, and `--train`.
3. Delete the old `features.npz` before extracting so it is rebuilt cleanly.

---

## Requirements

- Python 3.9+
- Webcam
- ffmpeg (for video trimming)
- ~4 GB disk space for clips + features
