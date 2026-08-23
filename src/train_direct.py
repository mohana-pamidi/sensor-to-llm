"""
1. Direct sensor classification wiht mlp head *baseline 
 
Pipeline:
  Dataset -> SensorEncoder -> SensorProjector -> Classification Head -> Loss

everything else is constant sensor and dataset 
"""


import argparse
import os
import sys
 
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
 
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
 
from sensor_encoder import SensorEncoder
from common import HARDataset, set_seed, macro_f1, count_trainable_params, ACTIVITY_NAMES
 
DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "processed"
)
CKPT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")
os.makedirs(CKPT_DIR, exist_ok=True)
 
NUM_CLASSES = 6
ENCODER_DIM = 128
 
 
def evaluate(encoder, head, loader, device, criterion):
    encoder.eval()
    head.eval()
    all_preds, all_labels = [], []
    total_loss, total_n = 0.0, 0
 
    with torch.no_grad():
        for batch_X, batch_y in loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            logits = head(encoder(batch_X))
            loss = criterion(logits, batch_y)
            total_loss += loss.item() * batch_X.size(0)
            total_n += batch_X.size(0)
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(batch_y.cpu().tolist())
 
    f1 = macro_f1(all_labels, all_preds)
    return total_loss / total_n, f1, all_labels, all_preds
 
 
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()
 
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"1. (Baseline) - Direct sensor classifier | seed={args.seed} | device={device}")
 
    train_ds = HARDataset(DATA_DIR, split="train")
    val_ds = HARDataset(DATA_DIR, split="val")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    print(f"Train samples: {len(train_ds)} | Val samples: {len(val_ds)}")
 
    encoder = SensorEncoder(in_channels=9, encoder_dim=ENCODER_DIM).to(device)
    head = nn.Linear(ENCODER_DIM, NUM_CLASSES).to(device)
 
    n_params = count_trainable_params(encoder, head)
    print(f"Trainable parameters: {n_params:,}")
 
    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(head.parameters()), lr=args.lr
    )
    criterion = nn.CrossEntropyLoss()
 
    best_f1 = -1.0
    best_state = None
 
    for epoch in range(args.epochs):
        encoder.train()
        head.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]")
        for batch_X, batch_y in pbar:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            logits = head(encoder(batch_X))
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
 
        val_loss, val_f1, _, _ = evaluate(encoder, head, val_loader, device, criterion)
        print(f"Epoch {epoch+1}: val_loss={val_loss:.4f} val_macro_f1={val_f1:.4f}")
 
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = {
                "encoder": encoder.state_dict(),
                "head": head.state_dict(),
                "seed": args.seed,
                "val_macro_f1": best_f1,
            }
 
    ckpt_path = os.path.join(CKPT_DIR, "direct_classifier.pt")
    torch.save(best_state, ckpt_path)
    print(f"\nSaved best checkpoint to {ckpt_path}")
    print(f"BEST VAL MACRO-F1 (Condition 1 - Direct classifier): {best_f1:.4f}")
 
 
if __name__ == "__main__":
    main()