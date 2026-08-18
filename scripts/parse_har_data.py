"""
parse_har_data.py
-----------------
Loads the UCI HAR (Human Activity Recognition) dataset from raw inertial signal
files and produces NumPy arrays ready for model training.

Pipeline
--------
1. Read the 9 inertial signal files for both the original train and test splits.
2. Concatenate them so we have the full dataset of 30 subjects.
3. Perform a *subject-disjoint* 70 / 30 split:
   - 70 % of unique subject IDs -> training set
   - 30 % of unique subject IDs -> validation set
   Subjects are never shared between the two sets.
4. Save the resulting arrays to data/processed/.

Output arrays (saved as .npy files)
-------------------------------------
  X_train.npy  : shape (N_train, 128, 9)  -- raw inertial windows
  y_train.npy  : shape (N_train,)          -- activity labels (1-indexed, 1-6)
  X_val.npy    : shape (N_val,   128, 9)
  y_val.npy    : shape (N_val,)
  subject_train.npy : shape (N_train,)    -- subject ID per window
  subject_val.npy   : shape (N_val,)

Signal channel order (last dimension of X)
-------------------------------------------
  0  body_acc_x
  1  body_acc_y
  2  body_acc_z
  3  body_gyro_x
  4  body_gyro_y
  5  body_gyro_z
  6  total_acc_x
  7  total_acc_y
  8  total_acc_z
"""

import os
import random
import numpy as np

# -- Paths ---------------------------------------------------------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "UCI HAR Dataset")
OUT_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
os.makedirs(OUT_DIR, exist_ok=True)

# -- Signal names (channel order) ---------------------------------------------

SIGNAL_NAMES = [
    "body_acc_x",
    "body_acc_y",
    "body_acc_z",
    "body_gyro_x",
    "body_gyro_y",
    "body_gyro_z",
    "total_acc_x",
    "total_acc_y",
    "total_acc_z",
]

ACTIVITY_LABELS = {
    1: "WALKING",
    2: "WALKING_UPSTAIRS",
    3: "WALKING_DOWNSTAIRS",
    4: "SITTING",
    5: "STANDING",
    6: "LAYING",
}

# -- Helpers ------------------------------------------------------------------


def load_signal_file(path: str) -> np.ndarray:
    """Load a single inertial signal file -> (N, 128) float32 array."""
    data = np.loadtxt(path, dtype=np.float32)
    assert data.ndim == 2 and data.shape[1] == 128, (
        f"Unexpected shape {data.shape} in {path}"
    )
    return data  # (N, 128)


def load_split(split: str) -> tuple:
    """
    Load one original split ('train' or 'test').

    Returns
    -------
    X        : (N, 128, 9)
    y        : (N,)   activity labels
    subjects : (N,)   subject IDs
    """
    split_dir = os.path.join(RAW_DIR, split)
    signals_dir = os.path.join(split_dir, "Inertial Signals")

    # Stack 9 signal channels -> (N, 128, 9)
    channels = []
    for name in SIGNAL_NAMES:
        filename = f"{name}_{split}.txt"
        filepath = os.path.join(signals_dir, filename)
        print(f"    Loading {filename} ...", end=" ", flush=True)
        channel = load_signal_file(filepath)  # (N, 128)
        print(f"{channel.shape}")
        channels.append(channel)

    X = np.stack(channels, axis=-1)  # (N, 128, 9)

    # Labels (1-indexed, keep as-is: 1=WALKING ... 6=LAYING)
    y = np.loadtxt(
        os.path.join(split_dir, f"y_{split}.txt"), dtype=np.int32
    ).ravel()

    # Subject IDs
    subjects = np.loadtxt(
        os.path.join(split_dir, f"subject_{split}.txt"), dtype=np.int32
    ).ravel()

    assert X.shape[0] == y.shape[0] == subjects.shape[0], (
        "Length mismatch between signals, labels, and subject files."
    )

    print(f"  [{split}] X={X.shape}, y={y.shape}, "
          f"unique subjects={np.unique(subjects).tolist()}")
    return X, y, subjects


# -- Main ---------------------------------------------------------------------


def main(val_fraction: float = 0.30, seed: int = 42) -> None:
    print("=" * 60)
    print("UCI HAR Dataset -- inertial signal parser")
    print("=" * 60)

    # 1. Load & concatenate both original splits
    print("\nLoading original 'train' split ...")
    X_tr, y_tr, subj_tr = load_split("train")

    print("\nLoading original 'test' split ...")
    X_te, y_te, subj_te = load_split("test")

    X_all    = np.concatenate([X_tr,    X_te],    axis=0)   # (N_total, 128, 9)
    y_all    = np.concatenate([y_tr,    y_te],    axis=0)   # (N_total,)
    subj_all = np.concatenate([subj_tr, subj_te], axis=0)   # (N_total,)

    all_subjects = sorted(np.unique(subj_all).tolist())
    n_subjects = len(all_subjects)
    print(f"\nFull dataset: {X_all.shape[0]:,} windows, "
          f"{n_subjects} unique subjects: {all_subjects}")

    # 2. Subject-disjoint 70/30 split
    rng = random.Random(seed)
    shuffled = all_subjects.copy()
    rng.shuffle(shuffled)

    n_val_subjects   = max(1, round(n_subjects * val_fraction))
    n_train_subjects = n_subjects - n_val_subjects

    train_subjects = set(shuffled[:n_train_subjects])
    val_subjects   = set(shuffled[n_train_subjects:])

    # Sanity: no subject overlap
    assert train_subjects.isdisjoint(val_subjects), "Subject overlap detected!"

    print(f"\nSubject-disjoint split  (seed={seed}, val_fraction={val_fraction})")
    print(f"  Train subjects ({len(train_subjects)}): {sorted(train_subjects)}")
    print(f"  Val   subjects ({len(val_subjects)}):   {sorted(val_subjects)}")

    # Boolean masks
    train_mask = np.isin(subj_all, list(train_subjects))
    val_mask   = np.isin(subj_all, list(val_subjects))

    X_train = X_all[train_mask];    y_train = y_all[train_mask];    subj_train = subj_all[train_mask]
    X_val   = X_all[val_mask];      y_val   = y_all[val_mask];      subj_val   = subj_all[val_mask]

    print(f"\nResulting shapes")
    print(f"  X_train : {X_train.shape}   y_train : {y_train.shape}")
    print(f"  X_val   : {X_val.shape}   y_val   : {y_val.shape}")
    frac_actual = X_val.shape[0] / X_all.shape[0]
    print(f"  Actual val fraction (by windows): {frac_actual:.3f}")

    # 3. Save
    pairs = [
        ("X_train.npy",       X_train),
        ("y_train.npy",       y_train),
        ("subject_train.npy", subj_train),
        ("X_val.npy",         X_val),
        ("y_val.npy",         y_val),
        ("subject_val.npy",   subj_val),
    ]
    print(f"\nSaving to {OUT_DIR} ...")
    for fname, arr in pairs:
        path = os.path.join(OUT_DIR, fname)
        np.save(path, arr)
        print(f"  Saved {fname}  {arr.shape}  dtype={arr.dtype}")

    # 4. Label distribution summary
    print("\nLabel distribution")
    print(f"  {'Activity':<25} {'Train':>8} {'Val':>8}")
    print(f"  {'-'*25} {'-'*8} {'-'*8}")
    for label_id, name in ACTIVITY_LABELS.items():
        tr_count = int((y_train == label_id).sum())
        va_count = int((y_val   == label_id).sum())
        print(f"  {name:<25} {tr_count:>8} {va_count:>8}")

    print("\nDone.")


if __name__ == "__main__":
    main()
