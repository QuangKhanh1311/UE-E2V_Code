# =============================================================================
# 03_rgb_dataloader.py
# =============================================================================
#
# PURPOSE:
#   Generic data loader for RGB video clips stored as .npy files.
#   Adapt the CONFIG section and plug the DataLoaders into any model.
#
# INPUT .NPY FORMAT:
#   Each video is one .npy file with shape (T, H, W, C):
#     T  — number of frames  (variable per video)
#     H  — frame height (px)
#     W  — frame width  (px)
#     C  — channels (3, RGB, uint8 values 0–255)
#
# DIRECTORY STRUCTURE:
#   FEATURE_ROOT/
#     <Class>/
#       ...nested sub-folders...
#           <video_name>.npy
#
# CSV FORMAT:
#   Required columns:
#     video  — relative path, e.g. "Violence/scene01/Top/v001"
#     class  — label name matching a key in LABEL_MAP
#
#   Example:
#     video,class
#     Violence/Corridor/Top/Hitting/v001,Hitting
#     NonViolence/Corridor/Top/v012,NonViolence
#
# OUTPUT TENSOR (per sample):
#   shape  : (C, T, H, W)  float32 in [0.0, 1.0]
#   where T = TARGET_FRAMES after uniform sampling or padding.
#
# USAGE:
#   # Option A — import ready-made loaders
#   from 03_rgb_dataloader import train_loader, val_loader, test_loader
#
#   # Option B — build loaders yourself
#   from 03_rgb_dataloader import VideoDataset, get_dataloaders
#   train_loader, val_loader, test_loader = get_dataloaders()
#
#   # Inside your training loop:
#   for video, label in train_loader:
#       video = video.to(device)   # (B, C, T, H, W)
#       label = label.to(device)
#       ...
#
# =============================================================================

import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


# =============================================================================
# CONFIGURATION  — edit this section to match your setup
# =============================================================================

# Root directory that contains all per-video .npy files
FEATURE_ROOT = "/content/drive/MyDrive/EarlyViolenceDetection2025/Features/Video_NPY"

# Number of frames per clip after normalization
TARGET_FRAMES = 16

# Label map: class name → integer index
# ----- 3-Class -----
LABEL_MAP = {
    "NonViolence": 0,
    "PreViolence": 1,
    "Violence":    2,
}

# ----- 7-Class (uncomment to use) -----
# LABEL_MAP = {
#     "Pointing":       0,
#     "CollarGrabbing": 1,
#     "Pushing":        2,
#     "Strangling":     3,
#     "Hitting":        4,
#     "Headlock":       5,
#     "KickingAttack":  6,
# }

BATCH_SIZE_TRAIN = 16
BATCH_SIZE_EVAL  = 32
NUM_WORKERS      = 2


# =============================================================================
# DATASET
# =============================================================================

class VideoDataset(Dataset):
    """
    Loads RGB video clips stored as .npy files.

    Each clip is normalized to TARGET_FRAMES and returned as a float32
    tensor of shape (C, T, H, W) with pixel values in [0.0, 1.0].

    Frame normalization:
      - T >= TARGET_FRAMES : uniform sampling via np.linspace.
      - T <  TARGET_FRAMES : zero-pad at the end.
    """

    def __init__(self, csv_path, root):
        self.df   = pd.read_csv(csv_path)
        self.root = root

    # ------------------------------------------------------------------

    def find_feature(self, video_rel):
        """
        Locate the .npy file for a given relative video path.

        Searches recursively under FEATURE_ROOT/<Class>/ so the exact
        sub-folder structure does not need to be known in advance.

        Args:
            video_rel (str): value from the "video" CSV column.

        Returns:
            str | None: absolute path to the .npy file, or None if missing.
        """
        video_name  = os.path.basename(video_rel)
        npy_name    = os.path.splitext(video_name)[0] + ".npy"
        cls         = video_rel.split("/")[0]
        search_root = os.path.join(self.root, cls)

        for root, _, files in os.walk(search_root):
            if npy_name in files:
                return os.path.join(root, npy_name)
        return None

    # ------------------------------------------------------------------

    def sample_frames(self, video):
        """
        Normalize frame count to TARGET_FRAMES.

        Args:
            video (np.ndarray): shape (T, H, W, C)

        Returns:
            np.ndarray: shape (TARGET_FRAMES, H, W, C)
        """
        T = video.shape[0]

        if T >= TARGET_FRAMES:
            idx   = np.linspace(0, T - 1, TARGET_FRAMES).astype(int)
            video = video[idx]
        else:
            pad   = np.zeros(
                (TARGET_FRAMES - T, video.shape[1], video.shape[2], video.shape[3]),
                dtype=video.dtype,
            )
            video = np.concatenate([video, pad], axis=0)

        return video

    # ------------------------------------------------------------------

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row       = self.df.iloc[idx]
        video_rel = row["video"]
        label     = LABEL_MAP[row["class"]]

        npy_path = self.find_feature(video_rel)
        video    = np.load(npy_path)              # (T, H, W, C)
        video    = self.sample_frames(video)      # (TARGET_FRAMES, H, W, C)
        video    = video.astype(np.float32) / 255.0
        video    = torch.tensor(video).permute(3, 0, 1, 2)  # (C, T, H, W)

        return video, label


# =============================================================================
# DATALOADERS
# =============================================================================

def get_dataloaders():
    """
    Build and return train / val / test DataLoaders.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    train_ds = VideoDataset("train.csv", FEATURE_ROOT)
    val_ds   = VideoDataset("val.csv",   FEATURE_ROOT)
    test_ds  = VideoDataset("test.csv",  FEATURE_ROOT)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE_TRAIN, shuffle=True,  num_workers=NUM_WORKERS)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE_EVAL,  shuffle=False, num_workers=NUM_WORKERS)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE_EVAL,  shuffle=False, num_workers=NUM_WORKERS)

    print(f"Train : {len(train_ds)} samples")
    print(f"Val   : {len(val_ds)} samples")
    print(f"Test  : {len(test_ds)} samples")

    return train_loader, val_loader, test_loader


# =============================================================================
# READY-TO-USE LOADERS
# =============================================================================

train_loader, val_loader, test_loader = get_dataloaders()


# =============================================================================
# MAIN — quick shape check
# =============================================================================

if __name__ == "__main__":
    video_batch, label_batch = next(iter(train_loader))
    print(f"video : {tuple(video_batch.shape)}")   # (B, C, T, H, W)
    print(f"label : {tuple(label_batch.shape)}")   # (B,)
