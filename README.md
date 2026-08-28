# sensor-to-llm

A LLaVA-inspired architecture for **Human Activity Recognition (HAR)** using wearable sensor data — where raw inertial signals are encoded into a single "soft token" and injected directly into a frozen Large Language Model, bypassing text tokenization of the sensor data entirely.

> Inspired by [LLaVA: Large Language and Vision Assistant](https://llava-vl.github.io/). Instead of patching vision tokens into an LLM, we patch a sensor embedding token.

---

## Architecture

```
raw sensor window (B, 128, 9)
        |
  SensorEncoder          1D-CNN, encoder_dim=128
  (trainable)            3x [Conv1d -> BatchNorm -> GELU -> MaxPool] + GlobalAvgPool
        |
  SensorProjector        2-layer MLP  128 -> 256 -> 960
  (trainable)            with GELU + LayerNorm
        |  (one soft token spliced into prompt)
  Frozen SmolLM2-360M-Instruct
        |
  Linear(960 -> 6)       Classification head (trainable)
        |
  Activity label (0-5)
```

**Key idea:** Instead of converting sensor readings to text and then tokenizing, the CNN encoder compresses the 128-timestep window into a 128-dim embedding. The projector then translates that into a single 960-dim "soft token" — the same dimensionality as the LLM's word embeddings — and splices it into the prompt at a `<SENSOR>` placeholder position. The LLM's hidden representation at the final `Activity:` token is then classified.

**Dataset:** [UCI HAR Dataset](https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones) — 6 activities (WALKING, WALKING_UPSTAIRS, WALKING_DOWNSTAIRS, SITTING, STANDING, LAYING), 9 sensor channels (3-axis accelerometer + gyroscope + body acceleration), 128-sample windows at 50 Hz.

---

## Setup

> Trained on **Google Colab** (T4 GPU). Local inference and evaluation tested on **Windows with an NVIDIA MX550 GPU**.

### 1. Clone and create environment

```bash
git clone https://github.com/mohana-pamidi/sensor-to-llm.git
cd sensor-to-llm
python -m venv venv
```

### 2. Activate virtual environment

```powershell
# Windows PowerShell
.\venv\Scripts\Activate.ps1
```

```bash
# macOS / Linux / Colab
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

**requirements.txt:**
```
torch>=2.5.0
transformers>=4.46.0
accelerate>=0.33.0
numpy>=2.1.0
scikit-learn>=1.5.0
pandas>=2.2.0
tqdm>=4.66.0
matplotlib>=3.9.0
ipykernel
```

---

## Running the Experiments

### Condition 1 — Direct Sensor Classifier (Baseline)

Encoder + head only, no LLM.

```bash
python src/train_direct.py --seed 42 --epochs 30 --batch_size 32 --lr 1e-3
```

### Condition 2 — Context-Embedding Model (LLaVA-style)

Encoder -> Projector -> Frozen LLM -> head. Trained on Google Colab with seed 90.

```bash
python src/train_context.py --seed 90 --epochs 10 --batch_size 4 --accum_steps 8 --lr 1e-4
```

### Condition 3 — Shuffled-Embedding Sanity Check

Evaluates the Condition 2 checkpoint with globally shuffled sensor embeddings to verify the model is not exploiting a static prompt shortcut.

```bash
.\venv\Scripts\python.exe src/sensor_dependence_check.py --seed 42 --batch_size 16 --ckpt src/checkpoints/context_model.pt
```

---

## Results

| Condition | Description | Train Seed | Eval Seed | Val Macro-F1 |
|-----------|-------------|:----------:|:---------:|:------------:|
| 1 | Direct classifier (no LLM) | 42 | — | — |
| 2 | Context model (encoder -> frozen LLM) | **90** | — | **0.9360** |
| 3 | Condition 2 checkpoint, globally shuffled embeddings | 90 | **42** | **0.1533** |

**Interpretation:**
The Condition 3 score (0.1533) is essentially random chance for 6 balanced classes (theoretical: 1/6 ≈ 0.1667). The massive drop from 0.9360 -> 0.1533 confirms the model is genuinely using the specific identity of each sensor window — not any static prompt bias or shortcut — to make predictions.

---

## Project Structure

```
sensor-to-llm/
├── data/
│   └── processed/          # X_train.npy, y_train.npy, X_val.npy, y_val.npy, ...
├── src/
│   ├── common.py                       # HARDataset, set_seed, macro_f1
│   ├── sensor_encoder.py               # 1D-CNN SensorEncoder
│   ├── projector.py                    # MLP SensorProjector
│   ├── testllm.py                      # LLM loading + build_inputs_embeds pipeline
│   ├── train_direct.py                 # Condition 1 training script
│   ├── train_context.py                # Condition 2 training script
│   ├── sensor_dependence_check.py      # Condition 3 shuffled eval
│   └── checkpoints/
│       ├── context_model.pt            # Condition 2 best checkpoint (seed=90, F1=0.9360)
│       └── direct_classifier.pt        # Condition 1 best checkpoint
├── requirements.txt
└── README.md
```
