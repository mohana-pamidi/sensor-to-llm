"""
train.py
--------
Training loop for the Sensor-to-LLM architecture.

Pipeline:
  Dataset -> SensorEncoder -> SensorProjector -> Frozen LLM -> Classification Head -> Loss
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Ensure local imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sensor_encoder import SensorEncoder
from projector import SensorProjector
from testllm import build_inputs_embeds, model, device, llm_hidden_size, encoder, projector

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BATCH_SIZE = 4           # Lowered to fit in 2GB VRAM (MX550)
ACCUM_STEPS = 8          # Effective batch size = 32 (4 * 8)
LEARNING_RATE = 1e-4
EPOCHS = 3
NUM_CLASSES = 6
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "processed")


# ---------------------------------------------------------------------------
# Dataset Definition
# ---------------------------------------------------------------------------
class HARDataset(Dataset):
    def __init__(self, data_dir, split="train"):
        """
        Loads the preprocessed .npy files.
        Labels in HAR are 1-indexed (1 to 6). We convert them to 0-indexed (0 to 5).
        """
        self.X = np.load(os.path.join(data_dir, f"X_{split}.npy"))
        # Convert labels from [1, 6] to [0, 5]
        self.y = np.load(os.path.join(data_dir, f"y_{split}.npy")) - 1

        assert self.X.shape[0] == self.y.shape[0], "Mismatch between features and labels"
        
        # Convert to tensors
        self.X = torch.tensor(self.X, dtype=torch.float32)
        self.y = torch.tensor(self.y, dtype=torch.long)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ---------------------------------------------------------------------------
# Training Loop
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("Training Sensor-to-LLM Architecture")
    print("=" * 60)

    # 1. Load Data
    print("Loading data...")
    train_dataset = HARDataset(DATA_DIR, split="train")
    val_dataset = HARDataset(DATA_DIR, split="val")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples  : {len(val_dataset)}")

    # 2. Setup Classification Head
    # The LLM is frozen, so we need a linear layer to map the 960-dim hidden state to 6 classes
    classifier_head = nn.Linear(llm_hidden_size, NUM_CLASSES).to(device)

    # 3. Setup Optimizer and Loss
    # We only train: Encoder, Projector, and Classifier Head. LLM is already frozen.
    trainable_params = (
        list(encoder.parameters()) + 
        list(projector.parameters()) + 
        list(classifier_head.parameters())
    )
    optimizer = torch.optim.AdamW(trainable_params, lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    print(f"\nModel configured on {device}")
    print(f"Total trainable parameters: {sum(p.numel() for p in trainable_params):,}")

    # 4. Training Loop
    for epoch in range(EPOCHS):
        # -- Train --
        encoder.train()
        projector.train()
        classifier_head.train()
        # model (LLM) remains in eval() mode
        
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")
        optimizer.zero_grad()
        
        for step, (batch_X, batch_y) in enumerate(pbar):
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            # Forward pass: encoder -> projector -> splice into LLM prompt
            inputs_embeds, attention_mask, sensor_pos = build_inputs_embeds(batch_X)

            # Pass through frozen LLM
            outputs = model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            
            # Get the final hidden state of the LAST token
            last_hidden_state = outputs.hidden_states[-1]      # (B, seq_len, 960)
            final_token_hidden = last_hidden_state[:, -1, :]   # (B, 960)

            # Classify (cast back to float32 for stability with CrossEntropyLoss)
            logits = classifier_head(final_token_hidden.to(torch.float32))       # (B, 6)

            # Loss & Backprop (scale loss by accumulation steps)
            loss = criterion(logits, batch_y)
            (loss / ACCUM_STEPS).backward()
            
            # Step optimizer every ACCUM_STEPS
            if (step + 1) % ACCUM_STEPS == 0 or (step + 1) == len(train_loader):
                optimizer.step()
                optimizer.zero_grad()

            # Track metrics
            train_loss += loss.item() * batch_X.size(0)
            _, predicted = torch.max(logits.data, 1)
            train_total += batch_y.size(0)
            train_correct += (predicted == batch_y).sum().item()

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        train_loss /= train_total
        train_acc = 100.0 * train_correct / train_total

        # -- Eval --
        encoder.eval()
        projector.eval()
        classifier_head.eval()

        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch_X, batch_y in tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]  "):
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)

                inputs_embeds, attention_mask, _ = build_inputs_embeds(batch_X)

                outputs = model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                )
                
                last_hidden_state = outputs.hidden_states[-1]
                final_token_hidden = last_hidden_state[:, -1, :]
                
                logits = classifier_head(final_token_hidden.to(torch.float32))
                loss = criterion(logits, batch_y)

                val_loss += loss.item() * batch_X.size(0)
                _, predicted = torch.max(logits.data, 1)
                val_total += batch_y.size(0)
                val_correct += (predicted == batch_y).sum().item()

        val_loss /= val_total
        val_acc = 100.0 * val_correct / val_total

        print(f"Epoch {epoch+1} Summary:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss  : {val_loss:.4f} | Val Acc  : {val_acc:.2f}%")
        print("-" * 60)

    print("Training complete.")

if __name__ == "__main__":
    main()
