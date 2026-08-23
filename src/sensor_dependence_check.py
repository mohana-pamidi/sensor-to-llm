"""
3. Eval the trained context model from #2 after shuffling proejcted embeddings (without retraining)

The performance should drop signifcantly (otherwise model might have learnt a shortcut that doesn't even depedn on sensor readings - which is bad)
If the shortcut was present, then the F1 score of the #2 woul look artifically high

Basically a check to see whether the identity of the sensor window is driving the prediction or not 
if it is, then the performace scores shoudl be significnatly differnt. 


Just evals, so only uses the trianed weights from the previosu 
"""



import argparse
import os
import sys
 
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
 
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
 
from common import HARDataset, set_seed, macro_f1
from testllm import build_inputs_embeds, model, device, llm_hidden_size, encoder, projector
 
DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "processed"
)
CKPT_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "checkpoints", "context_model.pt"
)
 
NUM_CLASSES = 6
 
 
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--ckpt", type=str, default=CKPT_PATH)
    args = parser.parse_args()
 
    set_seed(args.seed)
    print(f"Condition 3: Shuffled-embedding check | seed={args.seed} | device={device}")
 
    if not os.path.exists(args.ckpt):
        raise FileNotFoundError(
            f"No checkpoint at {args.ckpt}. Run train_context.py first "
            "(Condition 3 reuses Condition 2's trained weights)."
        )
 
    ckpt = torch.load(args.ckpt, map_location=device)
    encoder.load_state_dict(ckpt["encoder"])
    projector.load_state_dict(ckpt["projector"])
    classifier_head = nn.Linear(llm_hidden_size, NUM_CLASSES).to(device)
    classifier_head.load_state_dict(ckpt["classifier_head"])
    print(f"Loaded Condition 2 checkpoint (its val_macro_f1 was "
          f"{ckpt.get('val_macro_f1', float('nan')):.4f}, seed={ckpt.get('seed')})")
 
    encoder.eval()
    projector.eval()
    classifier_head.eval()
 
    val_ds = HARDataset(DATA_DIR, split="val")
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    print(f"Val samples: {len(val_ds)}")
 
    all_preds, all_labels = [], []
 
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
 
            # Shuffle which sensor window is paired with which prompt/label.
            # Guard against the (unlikely) identity permutation on small batches.
            B = batch_X.shape[0]
            perm = torch.randperm(B, device=device)
            if B > 1:
                while torch.equal(perm, torch.arange(B, device=device)):
                    perm = torch.randperm(B, device=device)
            shuffled_X = batch_X[perm]
 
            inputs_embeds, attention_mask, _ = build_inputs_embeds(shuffled_X)
            outputs = model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            final_hidden = outputs.hidden_states[-1][:, -1, :]
            logits = classifier_head(final_hidden.to(torch.float32))
 
            preds = logits.argmax(dim=1)
            # Compare against the ORIGINAL (unshuffled) labels.
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(batch_y.cpu().tolist())
 
    f1 = macro_f1(all_labels, all_preds)
    print(f"\nMACRO-F1 (Condition 3 - Context model, shuffled embeddings): {f1:.4f}")
    print(f"For reference, Condition 2's (unshuffled) val macro-F1 was: "
          f"{ckpt.get('val_macro_f1', float('nan')):.4f}")
 
 
if __name__ == "__main__":
    main()
