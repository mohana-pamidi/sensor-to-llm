"""
common.py
---------
Shared utilities used by train_direct.py, train_context.py, and eval_shuffled.py:
  - set_seed()      : full reproducibility (python/numpy/torch/cuda)
  - HARDataset      : loads the preprocessed .npy windows produced by
                       parse_har_data.py
  - macro_f1()      : thin wrapper around sklearn's macro-averaged F1
  - count_trainable_params() : for the "trainable parameter count" table
    in the technical note
"""
 
import os
import random
 
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.metrics import f1_score
 
ACTIVITY_NAMES = [
    "WALKING",
    "WALKING_UPSTAIRS",
    "WALKING_DOWNSTAIRS",
    "SITTING",
    "STANDING",
    "LAYING",
]
 
 
def set_seed(seed: int) -> None:
    """Seed python, numpy, and torch (CPU + all CUDA devices)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Make cuDNN deterministic where possible (small speed cost, worth it
    # for a "one reproducible run per condition" deliverable).
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
 
 
class HARDataset(Dataset):
    """Loads X_{split}.npy / y_{split}.npy produced by parse_har_data.py.
 
    Labels are stored 1-indexed (1-6) on disk; this class converts them to
    0-indexed (0-5) for CrossEntropyLoss / sklearn.
    """
 
    def __init__(self, data_dir: str, split: str = "train"):
        self.X = np.load(os.path.join(data_dir, f"X_{split}.npy"))
        self.y = np.load(os.path.join(data_dir, f"y_{split}.npy")) - 1
 
        assert self.X.shape[0] == self.y.shape[0], (
            "Mismatch between features and labels"
        )
 
        self.X = torch.tensor(self.X, dtype=torch.float32)
        self.y = torch.tensor(self.y, dtype=torch.long)
 
    def __len__(self):
        return self.X.shape[0]
 
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
 
 
def macro_f1(y_true, y_pred) -> float:
    """Macro-averaged F1 over all 6 activity classes.
 
    labels=range(6) is passed explicitly so that a class missing from a
    small eval batch/run doesn't silently change the averaging denominator.
    """
    return f1_score(
        y_true, y_pred, average="macro", labels=list(range(6)), zero_division=0
    )
 
 
def count_trainable_params(*modules) -> int:
    total = 0
    for m in modules:
        total += sum(p.numel() for p in m.parameters() if p.requires_grad)
    return total
 