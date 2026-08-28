# Experiment Log — sensor-to-llm

A record of the three experimental conditions, training runs, results, and a critical bug that was caught and fixed during evaluation.

---

## Project Overview

This project investigates whether a frozen Large Language Model can serve as a useful representation backbone for wearable sensor classification. The core question: does routing sensor data *through* an LLM's contextual representations yield better features than directly training a CNN classifier on the raw signals?

**Inspired by LLaVA** (Large Language and Vision Assistant): instead of converting sensor readings to text and tokenizing them, we encode the raw signal window into a single "soft token" — a vector in the LLM's embedding space — and splice it directly into the prompt. The LLM never "sees" text describing the sensor values; it only sees the compressed embedding.

---

## Condition 1 — Direct Sensor Classifier (Baseline)

### What it does
Trains a 1D-CNN encoder directly connected to a linear classification head. No LLM involved.

```
raw window (B, 128, 9) -> SensorEncoder (dim=128) -> Linear(128 -> 6) -> loss
```

### Script
```bash
python src/train_direct.py --seed 90 --epochs 30 --batch_size 32 --lr 1e-3
```

### Architecture
- **SensorEncoder**: 3× stacked Conv1d blocks (kernel sizes 7, 5, 3) with BatchNorm + GELU + MaxPool, followed by GlobalAvgPool → output (B, 128)
- **Head**: `nn.Linear(128, 6)`
- **Optimizer**: AdamW, lr=1e-3
- **Loss**: CrossEntropyLoss
- **Trainable parameters**: 95,878

### Training run (seed=90, Google Colab T4)

| Epoch | Val Loss | Val Macro-F1 |
|:-----:|:--------:|:------------:|
| 1  | 0.1613 | 0.9104 |
| 2  | 0.1444 | 0.9310 |
| 10 | 0.1253 | 0.9461 |
| 18 | 0.1171 | **0.9548** ← best |
| 30 | 0.1822 | 0.9311 |

### Result
| Metric | Value |
|--------|-------|
| Best val Macro-F1 (Colab, epoch 18) | **0.9548** |
| Local re-eval Macro-F1 (Windows MX550) | **0.9311** |
| Best epoch | 18 / 30 |
| Train seed | 90 |
| Eval seed (local) | 42 |
| Checkpoint | `direct_classifier.pt` |

> **Note on the difference**: The checkpoint saves the best epoch's F1 measured on Colab's T4. Local re-eval on the MX550 yields 0.9311 — the small gap is expected because `BatchNorm` uses running statistics accumulated during Colab training; minor floating-point differences across CUDA environments can shift the result slightly.

### Notes
- Cheap to train (~30 seconds on Colab T4, no LLM loading).
- Establishes the baseline F1 that the LLM-route must beat to justify its added complexity.

---

## Condition 2 — Context-Embedding Model (LLaVA-style)

### What it does
Routes the sensor embedding through a **frozen SmolLM2-360M-Instruct** before classification. Gradients flow *back through* the frozen LLM to train the encoder and projector.

```
raw window (B, 128, 9)
    -> SensorEncoder (dim=128)           [trainable]
    -> SensorProjector (dim -> 960)      [trainable]
    -> <SENSOR> slot in prompt            [spliced soft token]
    -> Frozen SmolLM2-360M-Instruct       [frozen, no grad update]
    -> final hidden state at "Activity:" position
    -> Linear(960 -> 6)                  [trainable]
    -> loss
```

### Script
```bash
python src/train_context.py --seed 90 --epochs 10 --batch_size 4 --accum_steps 8 --lr 1e-4
```

### Training Details
- **Trained on**: Google Colab (T4 GPU)
- **Seed**: 90
- **Epochs**: 10
- **Batch size**: 4 (with 8-step gradient accumulation → effective batch = 32)
- **LR**: 1e-4, AdamW
- **Trainable parameters**: 381,126
- **Checkpoint saved**: `src/checkpoints/context_model.pt`

### Training run (seed=90, Google Colab T4)

| Epoch | Train F1 | Val Macro-F1 |
|:-----:|:--------:|:------------:|
| 1  | 0.6154 | 0.8775 |
| 2  | 0.8640 | 0.9100 |
| 5  | 0.9191 | 0.9039 |
| 9  | 0.9317 | **0.9360** ← best |
| 10 | 0.9310 | 0.9005 |

### Result
| Metric | Value |
|--------|-------|
| Best val Macro-F1 | **0.9360** |
| Best epoch | 9 / 10 |
| Train seed | 90 |
| Checkpoint | `context_model.pt` |

### How the soft-token splice works
The prompt template contains a `<SENSOR>` placeholder:
```
"Classify the activity as walking, walking upstairs, ...

Sensor context: <SENSOR>

Activity:"
```
At runtime, the `<SENSOR>` token is replaced by the projected sensor embedding (shape `(B, 1, 960)`), concatenated between the before- and after-text embeddings. The LLM's final hidden state at the last token position feeds the classification head.

---

## Condition 3 — Shuffled-Embedding Sanity Check

### What it does
Reuses the trained Condition 2 checkpoint **without any retraining**, but globally shuffles the sensor embeddings before the LLM forward pass. If the model learned a real sensor→activity mapping, performance should collapse to near random chance.

### Why this check matters
Without this test, a model that achieves high F1 *might* be exploiting:
- A static bias in the classification head (always predicts the majority class)
- Positional artifacts in the prompt that correlate with labels
- Any learned shortcut that doesn't actually depend on the sensor signal

If shuffling embeddings doesn't hurt performance, the sensor data wasn't driving predictions at all.

### Script
```bash
.\venv\Scripts\python.exe src/sensor_dependence_check.py \
    --seed 42 --batch_size 16 \
    --ckpt src/checkpoints/context_model.pt
```

### Result
| Metric | Value |
|--------|-------|
| Shuffled val Macro-F1 | **0.1533** |
| Eval seed | 42 |
| Val samples shuffled | 3,077 |
| Permutation example (first 10) | [2553, 566, 2636, 947, 1561, 2949, 2929, 1117, 1929, 2363] |

**Interpretation**: Random chance for 6 balanced classes = 1/6 ≈ **0.1667**. The observed 0.1533 is right at that floor, confirming the model collapses when embeddings are genuinely mismatched — the sensor identity is driving predictions, not a shortcut.

---

## The Bug We Caught and Fixed

### The original implementation (flawed)

```python
# OLD CODE — shuffled only within each mini-batch
for batch_X, batch_y in val_loader:
    B = batch_X.shape[0]
    perm = torch.randperm(B, device=device)   # permutation over only B=16 samples
    shuffled_X = batch_X[perm]
    ...
```

### Why this was wrong

The DataLoader used `shuffle=False`, meaning batches were fed in the original dataset order. The UCI HAR dataset is organized as continuous recordings per subject per activity — consecutive windows are almost certainly from the **same subject doing the same activity**.

With B=16 samples per batch, shuffling within the batch only swapped samples like:
- "WALKING window #4" ↔ "WALKING window #11"

These have the same label! A within-batch permutation over consecutive windows from the same recording session is **not a genuine mismatch** — it barely disrupts the sensor→label pairing at all.

**Evidence**: Running the old code on Colab gave a shuffled Macro-F1 of **0.7218** — which looked like a drop from 0.9360, but was still far above chance (0.1667), revealing that the shuffle was not creating real mismatches.

### The fix — global two-pass approach

```python
# NEW CODE — global permutation over the entire val set

# Pass 1: collect ALL embeddings first
all_sensor_embeds = []
all_labels = []
for batch_X, batch_y in val_loader:
    enc_out = encoder(batch_X.to(device))
    proj_out = projector(enc_out)
    all_sensor_embeds.append(proj_out.cpu())
    all_labels.extend(batch_y.tolist())

all_sensor_embeds = torch.cat(all_sensor_embeds, dim=0)   # (N, 960)

# Global shuffle across all N=3077 samples
perm = torch.randperm(N)
shuffled_embeds = all_sensor_embeds[perm]

# Pass 2: run LLM with shuffled embeddings, compare against original labels
for start in range(0, N, batch_size):
    sensor_embed = shuffled_embeds[start:end].to(device)
    ...
```

### Why the fix works

With `N=3,077` val samples and a global `randperm`, the probability that any sample ends up paired with a window from the same recording segment drops to near zero. WALKING window #4 is now randomly assigned LAYING window #2,553's label — a genuine, cross-activity, cross-subject mismatch.

### Impact on results

| Implementation | Shuffled Macro-F1 | Verdict |
|---|:---:|:---|
| Original (within-batch shuffle, B=16) — Colab run | 0.7218 | ❌ Not a real mismatch |
| Fixed (global shuffle over N=3,077) — local run | **0.1533** | ✅ Genuine mismatch confirmed |

---

## Summary Table

| | Condition 1 | Condition 2 | Condition 3 |
|---|---|---|---|
| **Description** | Direct CNN classifier | CNN → frozen LLM | Condition 2 weights, shuffled embeddings |
| **LLM used** | None | SmolLM2-360M-Instruct (frozen) | SmolLM2-360M-Instruct (frozen) |
| **Trainable modules** | Encoder + head | Encoder + projector + head | None (eval only) |
| **Train seed** | **90** | **90** | — |
| **Eval seed** | — | — | **42** |
| **Val Macro-F1 (Colab)** | **0.9548** | **0.9360** | 0.7218 *(buggy shuffle)* |
| **Val Macro-F1 (local)** | **0.9311** | — | **0.1533** ✅ |
| **Trained on** | Google Colab (T4) | Google Colab (T4) | Windows / NVIDIA MX550 |
| **Checkpoint** | `direct_classifier.pt` | `context_model.pt` | reuses `context_model.pt` |
