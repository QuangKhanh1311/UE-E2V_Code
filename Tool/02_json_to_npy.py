# =============================================================================
# 02_json_to_npy.py
# =============================================================================
#
# PURPOSE:
#   Read skeleton JSON files (output of 01_skeleton_extraction.py),
#   normalize each video to a fixed number of frames via uniform sampling
#   or padding, and build NumPy dataset arrays (.npy) ready for model
#   training.
#
# PIPELINE:
#   Skeleton JSON ──► sample / pad frames ──► tensor ──► stack ──► .npy
#
# INPUT JSON STRUCTURE (see 01_skeleton_extraction.py for details):
#   {
#     "frames": [
#       {
#         "frame_id": <int>,
#         "people": [
#           {
#             "id": <int>,
#             "keypoints": [[x, y, conf], ...]   # shape (25, 3)
#           }
#         ]
#       }
#     ]
#   }
#
# OUTPUT TENSOR SHAPE (per video):
#   (T, M, V, C)
#     T = TARGET_FRAMES   frames after normalization     (default: 60)
#     M = MAX_PERSON      maximum number of persons      (default:  9)
#     V = NUM_JOINT       skeleton joints (COCO 25-body) (default: 25)
#     C = CHANNEL         channels [x, y, confidence]   (default:  3)
#
# INPUT CSV FORMAT:
#   Three CSV files are required: train.csv, val.csv, test.csv
#   Each file must contain two columns:
#     video  : relative video path (e.g. "Violence/scene01/Top/v001")
#     class  : class label name (must match a key in LABEL_MAP)
#
# OUTPUT .NPY FILES:
#   X_train.npy  shape (N_train, T, M, V, C)   float32
#   y_train.npy  shape (N_train,)               int
#   X_val.npy    shape (N_val,   T, M, V, C)   float32
#   y_val.npy    shape (N_val,)                 int
#   X_test.npy   shape (N_test,  T, M, V, C)   float32
#   y_test.npy   shape (N_test,)                int
#
# USAGE:
#   1. Place train.csv / val.csv / test.csv in the working directory.
#   2. Adjust the CONFIG constants below for your dataset.
#   3. python 02_json_to_npy.py
#
# =============================================================================

import json
import os

import numpy as np
import pandas as pd


# =============================================================================
# CONFIGURATION  (adjust to your dataset)
# =============================================================================

# Number of frames per video after normalization
# (uniformly sampled if longer, padded with the last frame if shorter)
TARGET_FRAMES = 60

# Maximum number of persons kept per frame
MAX_PERSON = 9

# Number of skeleton joints (COCO 25-body layout)
NUM_JOINT = 25

# Number of channels per joint: [x, y, confidence]
CHANNEL = 3

# Root directory containing the skeleton JSON files
JSON_ROOT = "/content/drive/MyDrive/Scientific__Research/EarlyViolenceDetection2025/Features/OpenPose_JSON"

# Mapping from class label name to integer index.
# Adjust according to your dataset.
LABEL_MAP = {
    "Pointing":       0,
    "CollarGrabbing": 1,
    "Pushing":        2,
    "Strangling":     3,
    "Hitting":        4,
    "Headlock":       5,
    "KickingAttack":  6,
}

# Example for a 3-class problem (uncomment if needed):
# LABEL_MAP = {
#     "NonViolence": 0,
#     "PreViolence": 1,
#     "Violence":    2,
# }


# =============================================================================
# FRAME NORMALIZATION
# =============================================================================

def sample_frames(frames):
    """
    Normalize a video's frame list to exactly TARGET_FRAMES entries.

    - If the video has more frames than TARGET_FRAMES:
        uniformly sample TARGET_FRAMES indices via ``np.linspace``.
    - If the video has fewer frames than TARGET_FRAMES:
        repeat the last frame until the list reaches TARGET_FRAMES.

    Args:
        frames (list[dict]): list of frame dicts, each containing
                             "frame_id" and "people".

    Returns:
        list[dict]: normalized frame list of length TARGET_FRAMES.
    """
    T = len(frames)

    if T >= TARGET_FRAMES:
        idx    = np.linspace(0, T - 1, TARGET_FRAMES).astype(int)
        frames = [frames[i] for i in idx]
    else:
        pad    = [frames[-1]] * (TARGET_FRAMES - T)
        frames = frames + pad

    return frames


# =============================================================================
# JSON → NUMPY TENSOR
# =============================================================================

def json_to_tensor(json_path):
    """
    Load a skeleton JSON file for one video and convert it to a NumPy tensor.

    Steps:
      1. Read the frame list from the JSON file.
      2. Normalize to TARGET_FRAMES via sampling or padding.
      3. Fill keypoints into a zero-initialized tensor of shape (T, M, V, C).
         Slots with no person / joint data remain zero.

    Args:
        json_path (str): path to the skeleton JSON file for one video.

    Returns:
        np.ndarray: float32 tensor, shape
                    (TARGET_FRAMES, MAX_PERSON, NUM_JOINT, CHANNEL).
    """
    with open(json_path) as f:
        data = json.load(f)

    frames = sample_frames(data["frames"])
    tensor = np.zeros((TARGET_FRAMES, MAX_PERSON, NUM_JOINT, CHANNEL), dtype=np.float32)

    for t, frame in enumerate(frames):
        # Keep at most MAX_PERSON persons per frame
        for pid, person in enumerate(frame["people"][:MAX_PERSON]):
            tensor[t, pid] = np.array(person["keypoints"])  # shape (25, 3)

    return tensor


# =============================================================================
# FIND JSON FILE FOR A GIVEN VIDEO
# =============================================================================

def find_json(video_rel):
    """
    Locate the skeleton JSON file that corresponds to a video entry in the CSV.

    Search strategy:
      - Derive the JSON filename from the video basename (replace extension
        with ".json").
      - Recursively search under JSON_ROOT/<Class>/ to narrow the scope.

    Args:
        video_rel (str): relative video path from the CSV "video" column,
                         e.g. "Violence/scene01/Top/v001".

    Returns:
        str | None: absolute path to the JSON file if found, else None.
    """
    video_name = os.path.basename(video_rel)
    json_name  = os.path.splitext(video_name)[0] + ".json"

    # Use the top-level class directory to narrow the search
    cls         = video_rel.split("/")[0]
    search_root = os.path.join(JSON_ROOT, cls)

    for root, _, files in os.walk(search_root):
        if json_name in files:
            return os.path.join(root, json_name)

    return None


# =============================================================================
# DATASET STATISTICS — MAX PEOPLE
# =============================================================================

def find_max_people(json_root):
    """
    Scan all JSON files in ``json_root`` and report the frame with the most
    detected persons.

    Useful for choosing an appropriate value for MAX_PERSON before building
    the dataset.

    Args:
        json_root (str): root directory containing skeleton JSON files.

    Returns:
        tuple[int, str, int]: (max_people, video_path, frame_id)
    """
    max_people     = 0
    video_with_max = ""
    frame_with_max = -1

    for root, _, files in os.walk(json_root):
        for file in files:
            if not file.endswith(".json"):
                continue

            path = os.path.join(root, file)
            with open(path) as f:
                data = json.load(f)

            for frame in data["frames"]:
                n = len(frame["people"])
                if n > max_people:
                    max_people     = n
                    video_with_max = path
                    frame_with_max = frame["frame_id"]

    print("MAX PEOPLE IN DATASET:", max_people)
    print("VIDEO:", video_with_max)
    print("FRAME:", frame_with_max)

    return max_people, video_with_max, frame_with_max


# =============================================================================
# BUILD DATASET FROM CSV
# =============================================================================

def build_dataset(csv_path):
    """
    Read a split CSV file and build the (X, y) arrays for that split.

    For each row in the CSV:
      1. Locate the corresponding JSON file via ``find_json``.
      2. Convert the JSON to a tensor via ``json_to_tensor``.
      3. Append the tensor to X and the integer label to y.

    Videos whose JSON file cannot be found are skipped and counted as
    ``missing`` (reported at the end).

    Args:
        csv_path (str): path to the CSV file with columns "video" and "class".

    Returns:
        tuple[np.ndarray, np.ndarray]:
          X — shape (N, TARGET_FRAMES, MAX_PERSON, NUM_JOINT, CHANNEL), float32
          y — shape (N,), int
    """
    df      = pd.read_csv(csv_path)
    X       = []
    y       = []
    missing = 0

    for i, row in df.iterrows():
        video_rel = row["video"]
        label     = LABEL_MAP[row["class"]]
        json_path = find_json(video_rel)

        if json_path is None:
            print(f"[MISSING] {video_rel}")
            missing += 1
            continue

        skeleton = json_to_tensor(json_path)
        X.append(skeleton)
        y.append(label)

        if (i + 1) % 100 == 0:
            print(f"  Processed: {i + 1} / {len(df)}")

    X = np.stack(X)   # (N, T, M, V, C)
    y = np.array(y)   # (N,)

    print(f"  Missing files: {missing}")
    return X, y


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 60)
    print("JSON → NPY  DATASET BUILDER")
    print("=" * 60)
    print(f"TARGET_FRAMES : {TARGET_FRAMES}")
    print(f"MAX_PERSON    : {MAX_PERSON}")
    print(f"NUM_JOINT     : {NUM_JOINT}")
    print(f"CHANNEL       : {CHANNEL}")
    print(f"LABEL_MAP     : {LABEL_MAP}")
    print()

    # Optional: scan the JSON root to verify MAX_PERSON is large enough
    # find_max_people(JSON_ROOT)

    # ------------------------------------------------------------------
    # Build each dataset split
    # ------------------------------------------------------------------
    print("Building TRAIN...")
    X_train, y_train = build_dataset("train.csv")

    print("Building VAL...")
    X_val, y_val = build_dataset("val.csv")

    print("Building TEST...")
    X_test, y_test = build_dataset("test.csv")

    # ------------------------------------------------------------------
    # Save .npy files
    # ------------------------------------------------------------------
    np.save("X_train.npy", X_train)
    np.save("y_train.npy", y_train)
    np.save("X_val.npy",   X_val)
    np.save("y_val.npy",   y_val)
    np.save("X_test.npy",  X_test)
    np.save("y_test.npy",  y_test)

    # ------------------------------------------------------------------
    # Shape summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("DATASET SHAPE SUMMARY")
    print("=" * 60)
    print(f"X_train : {X_train.shape}  |  y_train : {y_train.shape}")
    print(f"X_val   : {X_val.shape}    |  y_val   : {y_val.shape}")
    print(f"X_test  : {X_test.shape}   |  y_test  : {y_test.shape}")
    print()
    print("Saved: X_train.npy, y_train.npy, X_val.npy, y_val.npy, X_test.npy, y_test.npy")


if __name__ == "__main__":
    main()
