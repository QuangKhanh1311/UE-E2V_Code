# =============================================================================
# 01_skeleton_extraction.py
# =============================================================================
#
# PURPOSE:
#   Extract skeleton keypoints (25 joints) from video frames using OpenPose,
#   then consistently track each person across frames using the Hungarian
#   Assignment algorithm. Results are saved as structured JSON files.
#
# PIPELINE:
#   Frames (images) ──► OpenPose ──► Raw JSON ──► Tracking ──► Clean JSON
#
# OUTPUT JSON STRUCTURE (one file per video):
#   {
#     "frames": [
#       {
#         "frame_id": <int>,          # 0-based frame index
#         "people": [
#           {
#             "id": <int>,            # consistent person ID across frames
#             "keypoints": [          # list of shape (25, 3)
#               [x, y, confidence],   # per joint: pixel coords + confidence
#               ...
#             ]
#           },
#           ...
#         ]
#       },
#       ...
#     ]
#   }
#
# INPUT DIRECTORY STRUCTURE:
#   INPUT_ROOT/
#     <Class>/                        # NonViolence | Violence | PreViolence
#       <Scene>/                      # Corridor | ...
#         <Angle>/                    # Top | ...
#           <VideoFolder>/            # folder containing frame images
#             frame_0001.jpg
#             frame_0002.jpg
#             ...
#
#   For Violence / PreViolence, an extra <Behavior>/ level is added:
#   INPUT_ROOT/<Class>/<Scene>/<Angle>/<Behavior>/<VideoFolder>/
#
# OUTPUT DIRECTORY STRUCTURE:
#   OUTPUT_ROOT/                      # mirrors INPUT_ROOT layout
#     <Class>/
#       <Scene>/
#         <Angle>/
#           <VideoFolder>.json        # tracking result for each video
#
# REQUIREMENTS:
#   - OpenPose built at the path specified in OPENPOSE_BIN
#   - Python packages: numpy, scipy
#   - Linux environment with GPU (e.g., Google Colab)
#
# USAGE:
#   python 01_skeleton_extraction.py
#
# =============================================================================

import json
import os
import subprocess

import numpy as np
from scipy.optimize import linear_sum_assignment


# =============================================================================
# CONFIGURATION  (adjust to your environment)
# =============================================================================

# Root directory containing extracted video frames
INPUT_ROOT = "/content/drive/MyDrive/Scientific__Research/EarlyViolenceDetection2025/AllVideo/Frames_Extracted"

# Root directory where output JSON files will be saved
OUTPUT_ROOT = "/content/drive/MyDrive/Scientific__Research/EarlyViolenceDetection2025/Features/OpenPose_JSON"

# Path to the compiled OpenPose binary
OPENPOSE_BIN = "/content/openpose/build/examples/openpose/openpose.bin"

# Temporary directory for raw OpenPose JSON output (cleared after each video)
TEMP_JSON_DIR = "/content/temp_openpose_json"

# Dataset filters — set to "ALL" to process everything
SELECT_CLASS = "ALL"   # e.g. "NonViolence" | "Violence" | "ALL"
SELECT_SCENE = "ALL"   # e.g. "Corridor" | "ALL"
SELECT_ANGLE = "ALL"   # e.g. "Top" | "ALL"

# Maximum pixel distance allowed when matching skeletons between frames.
# Pairs with distance > MAX_TRACK_DIST are treated as unmatched (new person
# or lost track).
MAX_TRACK_DIST = 150

os.makedirs(OUTPUT_ROOT,   exist_ok=True)
os.makedirs(TEMP_JSON_DIR, exist_ok=True)


# =============================================================================
# SKELETON DISTANCE
# =============================================================================

def skeleton_distance(kp1, kp2):
    """
    Compute the mean Euclidean distance between two skeletons.

    Only joints where both skeletons have confidence > 0.1 are used,
    avoiding distortion from occluded or undetected joints.

    Args:
        kp1 (np.ndarray): keypoints of skeleton 1, shape (25, 3) — [x, y, conf].
        kp2 (np.ndarray): keypoints of skeleton 2, shape (25, 3) — [x, y, conf].

    Returns:
        float: mean distance in pixels. Returns 1e6 if no valid joint pair
               exists (prevents division by zero).
    """
    valid = (kp1[:, 2] > 0.1) & (kp2[:, 2] > 0.1)

    if np.sum(valid) == 0:
        return 1e6

    diff = kp1[valid, :2] - kp2[valid, :2]
    return np.mean(np.linalg.norm(diff, axis=1))


# =============================================================================
# RUN OPENPOSE
# =============================================================================

def run_openpose(frame_dir):
    """
    Invoke the OpenPose binary to process all frames in ``frame_dir``.

    OpenPose writes one raw JSON file per frame into TEMP_JSON_DIR, each
    containing 2-D keypoints for every detected person.

    Args:
        frame_dir (str): absolute path to the folder of frame images for
                         one video clip.
    """
    cmd = [
        OPENPOSE_BIN,
        "--image_dir",   frame_dir,
        "--write_json",  TEMP_JSON_DIR,
        "--display",     "0",
        "--render_pose", "0",
    ]
    subprocess.run(cmd, cwd="/content/openpose")


# =============================================================================
# LOAD RAW OPENPOSE JSON
# =============================================================================

def load_openpose_json(json_path):
    """
    Parse one raw OpenPose JSON file and return the detected skeletons.

    Args:
        json_path (str): path to the OpenPose JSON file for a single frame.

    Returns:
        list[np.ndarray]: list of skeleton keypoints; each element has
                          shape (25, 3) where each row is [x, y, confidence].
    """
    with open(json_path) as f:
        data = json.load(f)

    kps = []
    for person in data.get("people", []):
        kp = np.array(person["pose_keypoints_2d"]).reshape(25, 3)
        kps.append(kp)

    return kps


# =============================================================================
# SKELETON TRACKING  (Hungarian Assignment)
# =============================================================================

def track_skeleton():
    """
    Track each person consistently across frames using the Hungarian algorithm.

    Algorithm:
      - First frame  : assign IDs 0, 1, 2, ... to detected persons in order.
      - Later frames : build a cost matrix of pairwise skeleton distances
                       between the previous frame's persons and the current
                       detections, then call ``linear_sum_assignment`` to find
                       the minimum-cost matching.
                       Pairs whose cost exceeds MAX_TRACK_DIST are discarded
                       (lost track or new person).

    Reads raw JSON files from TEMP_JSON_DIR.

    Returns:
        dict: video-level skeleton data in the format:
              {
                "frames": [
                  {
                    "frame_id": int,
                    "people":   [{"id": int, "keypoints": list}, ...]
                  },
                  ...
                ]
              }
    """
    frame_list = sorted(os.listdir(TEMP_JSON_DIR))
    prev_kps   = None
    video_data = {"frames": []}

    for frame_idx, jf in enumerate(frame_list):
        if not jf.endswith(".json"):
            continue

        json_path = os.path.join(TEMP_JSON_DIR, jf)
        kps       = load_openpose_json(json_path)

        # --- No people detected in this frame ---
        if len(kps) == 0:
            video_data["frames"].append({"frame_id": frame_idx, "people": []})
            continue

        # --- First frame: assign sequential IDs ---
        if prev_kps is None:
            prev_kps     = np.zeros((len(kps), 25, 3))
            frame_people = []

            for i, kp in enumerate(kps):
                prev_kps[i] = kp
                frame_people.append({"id": int(i), "keypoints": kp.tolist()})

            video_data["frames"].append({"frame_id": frame_idx, "people": frame_people})
            continue

        # --- Build cost matrix and solve assignment problem ---
        cost = np.full((len(prev_kps), len(kps)), 1e6)

        for i in range(len(prev_kps)):
            for j in range(len(kps)):
                cost[i, j] = skeleton_distance(prev_kps[i], kps[j])

        row, col     = linear_sum_assignment(cost)
        max_people   = max(len(prev_kps), len(kps))
        frame_skel   = np.zeros((max_people, 25, 3))
        frame_people = []

        for r, c in zip(row, col):
            if cost[r, c] > MAX_TRACK_DIST:
                continue  # distance too large — skip this pair

            frame_skel[r] = kps[c]
            frame_people.append({"id": int(r), "keypoints": kps[c].tolist()})

        prev_kps = frame_skel.copy()
        video_data["frames"].append({"frame_id": frame_idx, "people": frame_people})

    return video_data


# =============================================================================
# SAVE JSON
# =============================================================================

def save_json(video_data, save_path):
    """
    Write the tracked skeleton data for one video to a JSON file.

    Args:
        video_data (dict): tracking result in the standard format.
        save_path  (str):  destination JSON file path.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(video_data, f)
    print("Saved:", save_path)


# =============================================================================
# CLEAN TEMP DIRECTORY
# =============================================================================

def clean_temp():
    """Remove all files from TEMP_JSON_DIR after processing a video."""
    for f in os.listdir(TEMP_JSON_DIR):
        os.remove(os.path.join(TEMP_JSON_DIR, f))


# =============================================================================
# PROCESS ONE VIDEO
# =============================================================================

def process_video(video_folder):
    """
    Run the full extraction pipeline for a single video clip:
      1. Run OpenPose  → raw JSON files in TEMP_JSON_DIR.
      2. Track persons → clean video_data dict.
      3. Save output   → JSON file in OUTPUT_ROOT.
      4. Clean temp    → remove raw OpenPose files.

    Args:
        video_folder (str): absolute path to the folder of frame images.
    """
    print("\nProcessing:", video_folder)

    run_openpose(video_folder)
    video_data = track_skeleton()

    relative  = os.path.relpath(video_folder, INPUT_ROOT)
    save_path = os.path.join(OUTPUT_ROOT, relative + ".json")

    save_json(video_data, save_path)
    clean_temp()


# =============================================================================
# PROCESS ENTIRE DATASET
# =============================================================================

def process_dataset():
    """
    Walk the dataset directory tree and process each video clip.

    Supported directory layouts:
      NonViolence : INPUT_ROOT/Class/Scene/Angle/VideoFolder/
      Violence    : INPUT_ROOT/Class/Scene/Angle/Behavior/VideoFolder/

    The SELECT_CLASS / SELECT_SCENE / SELECT_ANGLE filters allow partial
    processing, which is useful when resuming after an interruption.
    """
    for cls in sorted(os.listdir(INPUT_ROOT)):

        if SELECT_CLASS != "ALL" and cls != SELECT_CLASS:
            continue

        class_path = os.path.join(INPUT_ROOT, cls)
        if not os.path.isdir(class_path):
            continue

        print("\nCLASS:", cls)

        for scene in sorted(os.listdir(class_path)):

            if SELECT_SCENE != "ALL" and scene != SELECT_SCENE:
                continue

            scene_path = os.path.join(class_path, scene)
            if not os.path.isdir(scene_path):
                continue

            print("  SCENE:", scene)

            for angle in sorted(os.listdir(scene_path)):

                if SELECT_ANGLE != "ALL" and angle != SELECT_ANGLE:
                    continue

                angle_path = os.path.join(scene_path, angle)
                if not os.path.isdir(angle_path):
                    continue

                print("    ANGLE:", angle)

                if cls == "NonViolence":
                    # No Behavior sub-level for NonViolence
                    for video_folder in sorted(os.listdir(angle_path)):
                        video_path = os.path.join(angle_path, video_folder)
                        if os.path.isdir(video_path):
                            process_video(video_path)

                else:
                    # Violence / PreViolence have an extra Behavior sub-level
                    for behavior in sorted(os.listdir(angle_path)):
                        behavior_path = os.path.join(angle_path, behavior)
                        if not os.path.isdir(behavior_path):
                            continue

                        print("      BEHAVIOR:", behavior)

                        for video_folder in sorted(os.listdir(behavior_path)):
                            video_path = os.path.join(behavior_path, video_folder)
                            if os.path.isdir(video_path):
                                process_video(video_path)


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 60)
    print("SKELETON EXTRACTION — OpenPose + Hungarian Tracking")
    print("=" * 60)
    print("INPUT :", INPUT_ROOT)
    print("OUTPUT:", OUTPUT_ROOT)
    print()

    process_dataset()

    print("\nDONE — all videos have been processed.")


if __name__ == "__main__":
    main()
