# E2V — Early Violence Detection in School Environments

Source code for the paper:  
**"UE-E2V: A Multi-Level Dataset for Early-to-Violence Recognition in School Environments"**

---

## Table of Contents

1. [Overview](#overview)
2. [Dataset](#dataset)
3. [Project Structure](#project-structure)
4. [Pipeline](#pipeline)
5. [Models](#models)
6. [Quick Start](#quick-start)
7. [CSV Format](#csv-format)
8. [Label Maps](#label-maps)
9. [Requirements](#requirements)

---

## Overview

This project tackles **early violence detection** in school surveillance video.  
Rather than detecting violence only after it occurs, the system recognises three temporal stages:

| Stage | Description |
|---|---|
| **NonViolence** | Normal school activity — no threatening behaviour |
| **PreViolence** | Escalating behaviours that precede physical violence |
| **Violence** | Overt physical confrontation |

Two classification granularities are supported:

| Task | Classes | Use case |
|---|---|---|
| **3-Class** | NonViolence / PreViolence / Violence | Coarse, real-time alerting |
| **7-Class** | Pointing / CollarGrabbing / Pushing / Strangling / Hitting / Headlock / KickingAttack | Fine-grained behaviour recognition |

Two modalities are used independently:

- **RGB** — raw video frames extracted as `.npy` files
- **Skeleton** — body keypoints extracted via OpenPose, stored as `.npy` files

---

## Dataset

> **Download the dataset at:** [https://quangkhanh1311.github.io/E2VSchoolViolence/](https://quangkhanh1311.github.io/E2VSchoolViolence/)

### Directory Structure (after download)

```
Dataset/
  NonViolence/
    Classroom/
      Top/    Center/    Bottom/
    Corridor/
      Top/    Center/    Bottom/

  PreViolence/
    Classroom/
      Top/    Center/    Bottom/
        <Behavior>/
          *.mp4
    Corridor/
      ...

  Violence/
    Classroom/
      Top/    Center/    Bottom/
        <Behavior>/
          *.mp4
    Corridor/
      ...
```

### Pre-extracted Features (RGB .npy)

After feature extraction, RGB clips are stored as `.npy` files mirroring the same directory tree:

```
Features/Video_NPY/
  NonViolence/ ...
  PreViolence/ ...
  Violence/    ...
```

Each `.npy` file has shape **(T, 224, 224, 3)** — `uint8` pixel values.

---

## Project Structure

```
SourceCode/
│
├── Data/
│   ├── BehaviorTrainValTest/       # 7-class splits (behavior column)
│   │   ├── train.csv
│   │   ├── val.csv
│   │   └── test.csv
│   │
│   └── ClassTrainValTest/          # 3-class splits (class column)
│       ├── train.csv
│       ├── val.csv
│       └── test.csv
│
├── Tool/                           # Shared utilities
│   ├── 01_skeleton_extraction.py   # Step 1 — Extract skeleton keypoints from video
│   ├── 02_json_to_npy.py           # Step 2 — Convert OpenPose JSON → skeleton .npy
│   └── 03_rgb_dataloader.py        # Generic RGB data loader (plug into any model)
│
├── RGB/                            # RGB-based model notebooks
│   ├── hcmue-e2v-videomae-v1.ipynb # VideoMAE-v1 fine-tuning
│   ├── hcmue-e2v-vivit.ipynb       # ViViT fine-tuning
│   └── hcmue-e2v-slowfast.ipynb    # SlowFast R50 fine-tuning
│
└── Skeleton/                       # Skeleton-based model notebooks
    ├── hcmue-e2v-asgcn.ipynb       # AS-GCN
    ├── hcmue-e2v-degcn.ipynb       # DE-GCN
    ├── hcmue-e2v-graphlgsg-net.ipynb
    ├── hcmue-e2v-graphtsgcnext.ipynb
    └── hcmue-e2v-self-gcn.ipynb    # Self-GCN
```

---

## Pipeline

### RGB Modality

```
Raw video (.mp4 / .mov)
    │
    ▼
Feature extraction → RGB .npy  (T, 224, 224, 3)
    │
    ▼
03_rgb_dataloader.py  →  DataLoader  (B, C, T, H, W)
    │
    ▼
RGB model notebook (VideoMAE-v1 / ViViT / SlowFast)
    │
    ▼
Saved checkpoint (.pth) + Test metrics
```

### Skeleton Modality

```
Raw video (.mp4 / .mov)
    │
    ▼
01_skeleton_extraction.py  →  OpenPose JSON keypoints
    │
    ▼
02_json_to_npy.py  →  Skeleton .npy  (T, V, C)
    │
    ▼
Skeleton model notebook (AS-GCN / DE-GCN / Self-GCN / ...)
    │
    ▼
Saved checkpoint (.pth) + Test metrics
```

---

## Models

### RGB Models

| Notebook | Backbone | Frames | Library | Input shape |
|---|---|---|---|---|
| `hcmue-e2v-videomae-v1.ipynb` | `MCG-NJU/videomae-base` | 16 | HuggingFace | `(B, T, C, H, W)` |
| `hcmue-e2v-vivit.ipynb` | `google/vivit-b-16x2-kinetics400` | 32 | HuggingFace | `(B, T, C, H, W)` |
| `hcmue-e2v-slowfast.ipynb` | `slowfast_r50` (Kinetics-400) | Fast=32 / Slow=8 | pytorchvideo | `[slow, fast]` |

### Skeleton Models (GCN-based)

| Notebook | Model |
|---|---|
| `hcmue-e2v-self-gcn.ipynb` | Self-GCN |
| `hcmue-e2v-asgcn.ipynb` | AS-GCN |
| `hcmue-e2v-degcn.ipynb` | DE-GCN |
| `hcmue-e2v-graphlgsg-net.ipynb` | Graph-LGSG-Net |
| `hcmue-e2v-graphtsgcnext.ipynb` | Graph-TSGCNext |

---

## Quick Start

### 1. Download the dataset

Visit [https://quangkhanh1311.github.io/E2VSchoolViolence/](https://quangkhanh1311.github.io/E2VSchoolViolence/) and follow the instructions to download the raw video clips.

### 2. Extract RGB features

The RGB notebooks expect pre-extracted `.npy` files.  
Use your preferred frame extractor to save each video as a `.npy` of shape `(T, 224, 224, 3)`.

### 3. Extract skeleton features (optional)

```bash
# Step 1 — run OpenPose on all videos and save JSON keypoints
python Tool/01_skeleton_extraction.py

# Step 2 — convert JSON → .npy skeleton arrays
python Tool/02_json_to_npy.py
```

### 4. Choose CSV split

| Task | CSV folder |
|---|---|
| 3-Class (NonViolence / PreViolence / Violence) | `Data/ClassTrainValTest/` |
| 7-Class (Pointing / CollarGrabbing / ...) | `Data/BehaviorTrainValTest/` |

Copy or symlink the desired `train.csv`, `val.csv`, `test.csv` to your working directory.

### 5. Run a model notebook

Open any notebook in `RGB/` or `Skeleton/` in **Google Colab** (recommended) or JupyterLab.

Edit the **Configuration cell** (Cell 6 or 7 in RGB notebooks):

```python
FEATURE_ROOT  = "/content/drive/MyDrive/..."  # path to .npy files
TARGET_FRAMES = 16                             # 16 for VideoMAE / 32 for ViViT & SlowFast
NUM_CLASSES   = 3                              # 3 or 7
```

Then run all cells top to bottom.

---

## CSV Format

### 3-Class CSV (`Data/ClassTrainValTest/`)

```
video,class,scene,angle,behavior
Violence/Classroom/Bottom/Hitting/Bottom26T10_Hitting_001.mp4,Violence,Classroom,Bottom,Hitting
NonViolence/Corridor/Top/Top9T11_NonViolence_001.mp4,NonViolence,Corridor,Top,Top
PreViolence/Classroom/Top/Pointing/Top8T10_Pointing_001.mp4,PreViolence,Classroom,Top,Pointing
```

**Label column:** `class`

---

### 7-Class CSV (`Data/BehaviorTrainValTest/`)

```
video,class,scene,angle,behavior
Violence/Classroom/Bottom/Hitting/Bottom26T10_Hitting_001.mp4,Violence,Classroom,Bottom,Hitting
PreViolence/Classroom/Top/Pointing/Top8T10_Pointing_001.mp4,PreViolence,Classroom,Top,Pointing
```

**Label column:** `behavior`

---

## Label Maps

### 3-Class

| Index | Label | Description |
|---|---|---|
| 0 | NonViolence | Normal activity |
| 1 | PreViolence | Escalating / threatening behaviour |
| 2 | Violence | Active physical assault |

### 7-Class

| Index | Label | Stage |
|---|---|---|
| 0 | Pointing | PreViolence |
| 1 | CollarGrabbing | PreViolence |
| 2 | Pushing | PreViolence |
| 3 | Strangling | Violence |
| 4 | Hitting | Violence |
| 5 | Headlock | Violence |
| 6 | KickingAttack | Violence |

To switch between 3-class and 7-class in any notebook, comment / uncomment the corresponding `LABEL_MAP` block in the **Configuration cell**.

---

## Requirements

### RGB notebooks

```
transformers>=4.35
accelerate
pytorchvideo      # SlowFast only
fvcore            # SlowFast only
iopath            # SlowFast only
torch>=2.0
numpy
pandas
scikit-learn
tqdm
matplotlib
```

### Skeleton notebooks

```
torch>=2.0
numpy
pandas
scikit-learn
tqdm
matplotlib
```

### Skeleton extraction tools

```
# OpenPose must be installed separately — see:
# https://github.com/CMU-Perceptual-Computing-Lab/openpose

numpy
opencv-python
tqdm
```

---

## Citation

If you use this dataset or code in your research, please cite:

```bibtex
@misc{e2v2025,
  title   = {Early-to-Violence: Multi-Modal Recognition of Violence Escalation in School CCTV Footage},
  year    = {2025},
  url     = {https://quangkhanh1311.github.io/E2VSchoolViolence/}
}
```
