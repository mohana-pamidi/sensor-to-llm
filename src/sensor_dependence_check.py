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
from testllm import (build_inputs_embeds, model, device, llm_hidden_size,
                      encoder, projector, tokenizer, PROMPT_TEXT, SENSOR_PLACEHOLDER)
 
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
    N = len(val_ds)
    print(f"Val samples: {N}")
 
    # ------------------------------------------------------------------
    # Pass 1: Collect ALL projected sensor embeddings (no shuffling yet).
    #
    # The original code only shuffled within each 16-sample mini-batch,
    # but consecutive rows come from the same continuous recording and
    # almost certainly share the same activity label — so a within-batch
    # permutation rarely creates a genuine sensor ↔ label mismatch.
    #
    # Instead we collect embeddings for the full val set, then apply a
    # single GLOBAL permutation so that window A's embedding is paired
    # with a truly unrelated window's label.
    # ------------------------------------------------------------------
    all_sensor_embeds = []          # will hold (B_i, llm_hidden_size) chunks
    all_labels = []
 
    print("Pass 1/2: collecting projected sensor embeddings …")
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            batch_X = batch_X.to(device)
 
            # Encode + project (same path as build_inputs_embeds, but we
            # only need the sensor soft-token, not the full LLM forward).
            enc_out = encoder(batch_X)                    # (B, 128)
            proj_out = projector(enc_out)                 # (B, llm_hidden_size)
            all_sensor_embeds.append(proj_out.cpu())
            all_labels.extend(batch_y.tolist())
 
    all_sensor_embeds = torch.cat(all_sensor_embeds, dim=0)   # (N, llm_hidden_size)
    assert all_sensor_embeds.shape[0] == N
 
    # ------------------------------------------------------------------
    # Global shuffle of embeddings (keep labels in original order).
    # Guard against the (astronomically unlikely) identity permutation.
    # ------------------------------------------------------------------
    perm = torch.randperm(N)
    while torch.equal(perm, torch.arange(N)):
        perm = torch.randperm(N)
    shuffled_embeds = all_sensor_embeds[perm]               # (N, llm_hidden_size)
    print(f"Global permutation applied over all {N} samples "
          f"(first 10 mapping: {perm[:10].tolist()})")
 
    # ------------------------------------------------------------------
    # Pass 2: Run the frozen LLM + classifier with shuffled embeddings.
    # ------------------------------------------------------------------
    all_preds = []
 
    print("Pass 2/2: running LLM forward with shuffled embeddings …")
    with torch.no_grad():
        model_dtype = next(model.parameters()).dtype
 
        # Pre-compute the fixed text embeddings once (they're identical
        # for every sample — only the sensor soft-token changes).
        messages = [{"role": "user", "content": PROMPT_TEXT}]
        input_text = tokenizer.apply_chat_template(messages, tokenize=False)
        before_text, after_text = input_text.split(SENSOR_PLACEHOLDER)
 
        before_ids = tokenizer(before_text, return_tensors="pt",
                               add_special_tokens=False).input_ids.to(device)
        after_ids  = tokenizer(after_text,  return_tensors="pt",
                               add_special_tokens=False).input_ids.to(device)
        embed_layer = model.get_input_embeddings()
        before_embeds_1 = embed_layer(before_ids)            # (1, n_before, H)
        after_embeds_1  = embed_layer(after_ids)             # (1, n_after,  H)
 
        for start in range(0, N, args.batch_size):
            end = min(start + args.batch_size, N)
            B = end - start
 
            sensor_embed = shuffled_embeds[start:end].to(device)
            sensor_embed = sensor_embed.to(dtype=model_dtype).unsqueeze(1)  # (B,1,H)
 
            before_exp = before_embeds_1.expand(B, -1, -1)
            after_exp  = after_embeds_1.expand(B, -1, -1)
            inputs_embeds = torch.cat([before_exp, sensor_embed, after_exp], dim=1)
 
            attention_mask = torch.ones(
                (B, inputs_embeds.shape[1]), dtype=torch.long, device=device
            )
            outputs = model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            final_hidden = outputs.hidden_states[-1][:, -1, :]
            logits = classifier_head(final_hidden.to(torch.float32))
            all_preds.extend(logits.argmax(dim=1).cpu().tolist())
 
    f1 = macro_f1(all_labels, all_preds)
    print(f"\nMACRO-F1 (Condition 3 - Context model, shuffled embeddings): {f1:.4f}")
    print(f"For reference, Condition 2's (unshuffled) val macro-F1 was: "
          f"{ckpt.get('val_macro_f1', float('nan')):.4f}")
 
 
if __name__ == "__main__":
    main()
