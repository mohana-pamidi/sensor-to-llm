"""
2. Context-embedding model
 
  
Pipeline:
  Dataset -> SensorEncoder -> SensorProjector -> Frozen LLM -> Classification Head -> Loss
 
Only SensorEncoder + SensorProjector + classifier head are trainable.
The LLM backbone is frozen 
"""
 
import argparse
import os
import sys
 
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
 
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
 
from common import HARDataset, set_seed, macro_f1, count_trainable_params
# NOTE: importing testllm triggers loading of the frozen LLM onto `device`.
from testllm import build_inputs_embeds, model, device, llm_hidden_size, encoder, projector
 
DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "processed"
)
CKPT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")
os.makedirs(CKPT_DIR, exist_ok=True)
 
NUM_CLASSES = 6
 
 
def run_epoch(loader, classifier_head, optimizer, criterion, accum_steps, train: bool):
    if train:
        encoder.train()
        projector.train()
        classifier_head.train()
    else:
        encoder.eval()
        projector.eval()
        classifier_head.eval()
    # model (frozen LLM) always stays in eval() mode regardless of split.
 
    total_loss, total_n = 0.0, 0
    all_preds, all_labels = [], []
 
    if train:
        optimizer.zero_grad()
 
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        pbar = tqdm(loader, desc="Train" if train else "Val  ")
        for step, (batch_X, batch_y) in enumerate(pbar):
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
 
            inputs_embeds, attention_mask, _ = build_inputs_embeds(batch_X)
            outputs = model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            final_hidden = outputs.hidden_states[-1][:, -1, :]
            logits = classifier_head(final_hidden.to(torch.float32))
            loss = criterion(logits, batch_y)
 
            if train:
                (loss / accum_steps).backward()
                if (step + 1) % accum_steps == 0 or (step + 1) == len(loader):
                    optimizer.step()
                    optimizer.zero_grad()
 
            total_loss += loss.item() * batch_X.size(0)
            total_n += batch_X.size(0)
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.detach().cpu().tolist())
            all_labels.extend(batch_y.cpu().tolist())
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
 
    f1 = macro_f1(all_labels, all_preds)
    return total_loss / total_n, f1
 
 
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--accum_steps", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()
 
    set_seed(args.seed)
    print(f"Condition 2: Context-embedding model | seed={args.seed} | device={device}")
 
    train_ds = HARDataset(DATA_DIR, split="train")
    val_ds = HARDataset(DATA_DIR, split="val")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    print(f"Train samples: {len(train_ds)} | Val samples: {len(val_ds)}")
 
    classifier_head = nn.Linear(llm_hidden_size, NUM_CLASSES).to(device)
 
    n_params = count_trainable_params(encoder, projector, classifier_head)
    print(f"Trainable parameters (encoder + projector + head): {n_params:,}")
 
    trainable_params = (
        list(encoder.parameters()) + list(projector.parameters()) + list(classifier_head.parameters())
    )
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr)
    criterion = nn.CrossEntropyLoss()
 
    best_f1 = -1.0
    best_state = None
 
    for epoch in range(args.epochs):
        train_loss, train_f1 = run_epoch(
            train_loader, classifier_head, optimizer, criterion, args.accum_steps, train=True
        )
        val_loss, val_f1 = run_epoch(
            val_loader, classifier_head, optimizer, criterion, args.accum_steps, train=False
        )
        print(f"Epoch {epoch+1}/{args.epochs}: "
              f"train_loss={train_loss:.4f} train_f1={train_f1:.4f} | "
              f"val_loss={val_loss:.4f} val_macro_f1={val_f1:.4f}")
 
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = {
                "encoder": encoder.state_dict(),
                "projector": projector.state_dict(),
                "classifier_head": classifier_head.state_dict(),
                "seed": args.seed,
                "val_macro_f1": best_f1,
            }
 
    ckpt_path = os.path.join(CKPT_DIR, "context_model.pt")
    torch.save(best_state, ckpt_path)
    print(f"\nSaved best checkpoint to {ckpt_path}")
    print(f"BEST VAL MACRO-F1 (Condition 2 - Context-embedding model): {best_f1:.4f}")
 
 
if __name__ == "__main__":
    main()