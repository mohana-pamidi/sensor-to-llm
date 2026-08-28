"""
eval_direct.py
--------------
Evaluates the saved Condition 1 (direct_classifier.pt) checkpoint on the val set.
No retraining — just loads weights and reports Macro-F1.
"""

import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sensor_encoder import SensorEncoder
from common import HARDataset, set_seed, macro_f1

DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "processed"
)
CKPT_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "checkpoints", "direct_classifier.pt"
)

NUM_CLASSES = 6
ENCODER_DIM = 128


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(42)
    print(f"Condition 1 eval | device={device}")

    if not os.path.exists(CKPT_PATH):
        raise FileNotFoundError(f"No checkpoint at {CKPT_PATH}. Run train_direct.py first.")

    ckpt = torch.load(CKPT_PATH, map_location=device)
    print(f"Loaded checkpoint (train seed={ckpt.get('seed')}, "
          f"saved val_macro_f1={ckpt.get('val_macro_f1', float('nan')):.4f})")

    encoder = SensorEncoder(in_channels=9, encoder_dim=ENCODER_DIM).to(device)
    head = nn.Linear(ENCODER_DIM, NUM_CLASSES).to(device)
    encoder.load_state_dict(ckpt["encoder"])
    head.load_state_dict(ckpt["head"])
    encoder.eval()
    head.eval()

    val_ds = HARDataset(DATA_DIR, split="val")
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)
    print(f"Val samples: {len(val_ds)}")

    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            batch_X = batch_X.to(device)
            logits = head(encoder(batch_X))
            all_preds.extend(logits.argmax(dim=1).cpu().tolist())
            all_labels.extend(batch_y.tolist())

    f1 = macro_f1(all_labels, all_preds)
    print(f"\nMACRO-F1 (Condition 1 - Direct classifier, val set): {f1:.4f}")


if __name__ == "__main__":
    main()
