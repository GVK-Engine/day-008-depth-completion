# evaluate.py
# Honest evaluation of depth completion accuracy
# Uses 80/20 holdout method:
#   80% of LiDAR points used as completion input
#   20% held out as test ground truth
#
# Why holdout matters:
#   Naive evaluation tests where LiDAR already exists.
#   Error is near zero because we copied those values.
#   Holdout tests pixels the algorithm had to INTERPOLATE.
#   These are the pixels that actually matter.
#   This is how KITTI depth completion benchmark works.
#
# Results prove:
#   IP-Basic is far better than camera alone at all ranges
#   Error grows with distance (expected physics behavior)
#   Numbers are honest and defensible in any interview
#
# Nani | Day 8 of 90 | MS Robotics ASU

import numpy as np
import cv2
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from completion import ip_basic_completion, colorize_depth
from projection import (
    load_calibration,
    project_lidar_to_image,
    CALIB_DIR, LIDAR_DIR, IMAGE_DIR
)

RESULTS_DIR  = "results"
MAX_DEPTH    = 80.0
HOLDOUT_FRAC = 0.20   # 20% of points held out for testing

# distance bands matching KITTI benchmark standard
BANDS = [
    ("0-10m",   0,  10),
    ("10-20m", 10,  20),
    ("20-30m", 20,  30),
    ("30-50m", 30,  50),
    ("50m+",   50,  80),
]

# Day 2 camera-only MAE for comparison
# measured in Day 2 project on same KITTI sequence
CAMERA_ONLY_MAE = {
    "0-10m":  1.039,
    "10-20m": 2.302,
    "20-30m": 3.819,
    "30-50m": 5.377,
    "50m+":   8.813,
}


def holdout_split(sparse_depth, holdout_frac=HOLDOUT_FRAC,
                  seed=42):
    """
    Split LiDAR points into input (80%) and test (20%).

    Input pixels:  used to run completion algorithm
    Test pixels:   held out, used to measure accuracy

    The algorithm never sees test pixels.
    Test pixels represent real interpolation quality.
    This is the honest evaluation method.
    """
    rng = np.random.RandomState(seed)

    # find all pixels with LiDAR measurements
    valid_y, valid_x = np.where(sparse_depth > 0)
    n_valid          = len(valid_y)

    if n_valid == 0:
        return sparse_depth.copy(), sparse_depth.copy()

    # randomly select 20% to hold out
    n_holdout  = int(n_valid * holdout_frac)
    holdout_idx = rng.choice(n_valid, n_holdout,
                              replace=False)

    # build input depth (80% of points)
    input_depth = sparse_depth.copy()
    hy = valid_y[holdout_idx]
    hx = valid_x[holdout_idx]
    input_depth[hy, hx] = 0   # remove held-out points

    # build test mask (20% of points)
    test_depth       = np.zeros_like(sparse_depth)
    test_depth[hy, hx] = sparse_depth[hy, hx]

    return input_depth, test_depth


def compute_mae_per_band(test_depth, dense_pred):
    """
    Compute MAE only at held-out test pixels.
    These are pixels the algorithm never saw.
    Error here = real interpolation quality.
    """
    results = {}

    for label, d_min, d_max in BANDS:
        # pixels in this distance band with test ground truth
        band_mask = (
            (test_depth > d_min) &
            (test_depth <= d_max)
        )

        n_pixels = band_mask.sum()

        if n_pixels == 0:
            results[label] = {
                'mae': None, 'n': 0
            }
            continue

        gt   = test_depth[band_mask]
        pred = dense_pred[band_mask]
        mae  = float(np.abs(gt - pred).mean())

        results[label] = {
            'mae': mae,
            'n':   int(n_pixels)
        }

    return results


def evaluate_all_frames(frame_paths):
    P2, T_lidar_to_cam = load_calibration(CALIB_DIR)
    all_maes = {label: [] for label, _, _ in BANDS}

    print(f"\n  Method: {int((1-HOLDOUT_FRAC)*100)}% input "
          f"/ {int(HOLDOUT_FRAC*100)}% holdout test")
    print(f"  Frames: {len(frame_paths)}")
    print(f"  {'─'*58}")

    for fi, fpath in enumerate(frame_paths):
        fname    = os.path.basename(fpath)
        img_path = os.path.join(
            IMAGE_DIR, fname.replace('.bin', '.png')
        )

        # get full sparse depth
        sparse, img, _, _, _ = project_lidar_to_image(
            fpath, img_path, P2, T_lidar_to_cam
        )

        # split into input and test
        input_depth, test_depth = holdout_split(sparse)

        # run completion using only 80% of points
        dense = ip_basic_completion(input_depth, img)

        # evaluate at held-out 20%
        band_maes = compute_mae_per_band(test_depth, dense)

        for label, _, _ in BANDS:
            if band_maes[label]['mae'] is not None:
                all_maes[label].append(
                    band_maes[label]['mae']
                )

        mae_str = "  ".join([
            f"{l}:{band_maes[l]['mae']:.3f}m"
            for l, _, _ in BANDS
            if band_maes[l]['mae'] is not None
        ])
        print(f"  Frame {fi+1:02d}: {mae_str}")

    mean_maes = {
        label: float(np.mean(vals))
        for label, _, _ in BANDS
        if (vals := all_maes[label])
    }

    return mean_maes


def plot_comparison_chart(mean_maes):
    """
    Side by side bar chart:
      Left bar:  camera-only MAE (Day 2 baseline)
      Right bar: IP-Basic fusion MAE (Day 8 result)

    Shows the improvement from sensor fusion clearly.
    This is the single most important chart in Day 8.
    """
    fig, ax = plt.subplots(figsize=(14, 7))
    fig.patch.set_facecolor('#1a1a1a')
    ax.set_facecolor('#1a1a1a')

    labels     = [l for l, _, _ in BANDS
                  if mean_maes.get(l) is not None]
    fusion_mae = [mean_maes[l] for l in labels]
    camera_mae = [CAMERA_ONLY_MAE[l] for l in labels]

    x     = np.arange(len(labels))
    width = 0.35

    bars1 = ax.bar(x - width/2, camera_mae,
                   width, label='Camera Only (Day 2)',
                   color='#FF4444', alpha=0.85,
                   edgecolor='#333')
    bars2 = ax.bar(x + width/2, fusion_mae,
                   width, label='LiDAR+Camera Fusion (Day 8)',
                   color='#00C8FF', alpha=0.85,
                   edgecolor='#333')

    # value labels on bars
    for bar in bars1:
        ax.text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.1,
            f"{bar.get_height():.2f}m",
            ha='center', color='#FF8888',
            fontsize=9, fontweight='bold'
        )
    for bar in bars2:
        ax.text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.1,
            f"{bar.get_height():.3f}m",
            ha='center', color='#00C8FF',
            fontsize=9, fontweight='bold'
        )

    # improvement labels between bars
    for i, (cam, fus) in enumerate(
            zip(camera_mae, fusion_mae)):
        improvement = cam / fus
        ax.text(
            x[i], max(cam, fus) + 0.4,
            f"{improvement:.0f}x better",
            ha='center', color='#00FF88',
            fontsize=10, fontweight='bold'
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, color='white', fontsize=11)
    ax.set_xlabel('Distance Band', color='white',
                  fontsize=12)
    ax.set_ylabel('Mean Absolute Error (meters)',
                  color='white', fontsize=12)
    ax.set_title(
        'Sensor Fusion vs Camera Only - MAE per Distance Band\n'
        'Vamshikrishna Gadde  |  MS Robotics ASU  '
        '|  Day 8 of 90\n'
        '(Holdout evaluation: 20% of LiDAR points '
        'withheld as ground truth)',
        color='white', fontsize=12
    )
    ax.tick_params(colors='white')
    ax.legend(facecolor='#1a1a1a', labelcolor='white',
              fontsize=10, edgecolor='#444')
    for spine in ax.spines.values():
        spine.set_edgecolor('#444')

    ax.text(
        0.99, 0.98,
        'Lower bar = More accurate',
        transform=ax.transAxes,
        ha='right', va='top',
        color='#888888', fontsize=9
    )

    plt.tight_layout()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR,
                        'evaluation_chart.png')
    plt.savefig(path, dpi=130, bbox_inches='tight',
                facecolor='#1a1a1a')
    plt.close()
    print(f"\n  Saved: {path}")
    return path


if __name__ == "__main__":

    print("\n" + "="*58)
    print("  Depth Completion Evaluation - Day 8")
    print("  Holdout method (honest evaluation)")
    print("="*58)

    all_files = sorted([
        os.path.join(LIDAR_DIR, f)
        for f in os.listdir(LIDAR_DIR)
        if f.endswith('.bin')
    ])[:10]

    mean_maes = evaluate_all_frames(all_files)

    # print results table
    print(f"\n  {'='*58}")
    print(f"  FINAL RESULTS")
    print(f"  {'─'*58}")
    print(f"  {'Band':<10} {'Camera Only':<16} "
          f"{'Fusion (Ours)':<16} {'Improvement'}")
    print(f"  {'─'*58}")

    for label, _, _ in BANDS:
        cam = CAMERA_ONLY_MAE.get(label)
        fus = mean_maes.get(label)
        if cam and fus:
            imp = cam / fus
            print(f"  {label:<10} {cam:<16.3f} "
                  f"{fus:<16.3f} {imp:.0f}x better")

    # key finding for resume
    fus_close = mean_maes.get("0-10m")
    fus_far   = mean_maes.get("50m+")
    cam_close = CAMERA_ONLY_MAE.get("0-10m")
    cam_far   = CAMERA_ONLY_MAE.get("50m+")

    print(f"\n  KEY FINDING (for resume and LinkedIn)")
    print(f"  {'─'*58}")
    if fus_close and fus_far and cam_close and cam_far:
        imp_close = cam_close / fus_close
        imp_far   = cam_far   / fus_far
        print(f"  At 0-10m:  {cam_close:.3f}m → {fus_close:.3f}m"
              f"  ({imp_close:.0f}x improvement)")
        print(f"  At 50m+:   {cam_far:.3f}m → {fus_far:.3f}m"
              f"  ({imp_far:.0f}x improvement)")
        print(f"\n  Resume bullet:")
        print(f"  'IP-Basic LiDAR-camera fusion on KITTI:")
        print(f"   MAE {fus_close:.2f}m at 0-10m, "
              f"{fus_far:.2f}m at 50m+")
        print(f"   vs camera-only {cam_close:.2f}m and "
              f"{cam_far:.2f}m'")

    plot_comparison_chart(mean_maes)
    print(f"\n  Open results/evaluation_chart.png")
    print("="*58)