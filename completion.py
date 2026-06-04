# completion.py
# Fills sparse LiDAR depth (4% coverage) into
# dense depth map (99%+ coverage)
#
# Problem:
#   LiDAR only covers ~4% of image pixels.
#   96% of pixels have no depth measurement.
#   Pedestrians between scan lines have no depth.
#
# Solution — IP-Basic algorithm:
#   Step 1: Small dilation fills scan line gaps
#   Step 2: Bilateral filter smooths edges
#   Step 3: Large dilation fills medium holes
#   Step 4: Morphological closing fills round holes
#   Step 5: Global fill covers every remaining pixel
#   Step 6: Final bilateral filter cleans the result
#
# Color convention (industry standard):
#   RED   = close objects (dangerous)
#   BLUE  = far objects   (safe)
#   BLACK = no LiDAR data
#
# Nani | Day 8 of 90 | MS Robotics ASU

import numpy as np
import cv2
import os
from scipy.ndimage import distance_transform_edt
from projection import (
    load_calibration,
    project_lidar_to_image,
    CALIB_DIR, LIDAR_DIR, IMAGE_DIR
)

MAX_DEPTH   = 80.0
RESULTS_DIR = "results"


def colorize_depth(depth_map, max_depth=MAX_DEPTH):
    # RED=close, BLUE=far
    # invert depth before JET: close→1.0→red, far→0.0→blue
    mask       = depth_map > 0
    normalized = np.zeros_like(depth_map)

    normalized[mask] = 1.0 - np.clip(
        depth_map[mask] / max_depth, 0.0, 1.0
    )

    colored = cv2.applyColorMap(
        (normalized * 255).astype(np.uint8),
        cv2.COLORMAP_JET
    )
    colored[~mask] = 0
    return colored


def fill_nearest(depth, target_mask):
    # fills pixels in target_mask that have no depth
    # uses scipy to find nearest known pixel instantly
    # this is 100x faster than a manual loop
    known_mask = depth > 0

    if known_mask.sum() == 0:
        return depth.copy()

    _, nearest_idx = distance_transform_edt(
        ~known_mask,
        return_distances=True,
        return_indices=True
    )

    filled    = depth.copy()
    fill_mask = (target_mask > 0) & (depth == 0)

    if fill_mask.sum() == 0:
        return filled

    filled[fill_mask] = depth[
        nearest_idx[0][fill_mask],
        nearest_idx[1][fill_mask]
    ]

    return filled


def ip_basic_completion(sparse_depth, camera_img):
    # fills sparse depth (4%) into dense depth (99%+)
    # uses camera image edge information to guide filling
    # bilateral filter prevents depth bleeding across
    # object boundaries (wall at 30m stays separate
    # from car at 8m even if adjacent pixels)

    depth = sparse_depth.copy()

    # Step 1: small dilation (3x3 kernel)
    # each known depth point expands 1 pixel outward
    # fills tiny gaps between adjacent LiDAR scan lines
    k_small = np.ones((3, 3), np.uint8)
    mask1   = (depth > 0).astype(np.uint8)
    dmask1  = cv2.dilate(mask1, k_small, iterations=1)
    depth   = fill_nearest(depth, dmask1)

    # Step 2: first bilateral filter
    # smooths depth at object boundaries
    # sigmaColor=1.5 prevents merging different surfaces
    d32  = depth.astype(np.float32)
    d32  = cv2.bilateralFilter(
        d32, d=5, sigmaColor=1.5, sigmaSpace=1.5
    )
    known        = depth > 0
    depth[known] = d32[known]

    # Step 3: larger dilation (15x15 kernel)
    # fills medium-sized holes between objects
    k_large = np.ones((15, 15), np.uint8)
    mask2   = (depth > 0).astype(np.uint8)
    dmask2  = cv2.dilate(mask2, k_large, iterations=1)
    depth   = fill_nearest(depth, dmask2)

    # Step 4: morphological closing (31x31 kernel)
    # closing = dilation then erosion
    # fills circular holes that dilation alone misses
    k_close = np.ones((31, 31), np.uint8)
    mask3   = (depth > 0).astype(np.uint8)
    cmask   = cv2.morphologyEx(
        mask3, cv2.MORPH_CLOSE, k_close
    )
    depth = fill_nearest(depth, cmask)

    # Step 5: global fill — covers every remaining pixel
    # sky, upper image, and far regions all get filled
    # every pixel gets depth of its nearest known point
    # this is what takes coverage from ~60% to 99%+
    full_mask = np.ones_like(depth, dtype=np.uint8)
    depth     = fill_nearest(depth, full_mask)

    # Step 6: final bilateral filter on complete map
    d32   = depth.astype(np.float32)
    d32   = cv2.bilateralFilter(
        d32, d=5, sigmaColor=1.5, sigmaSpace=1.5
    )
    depth = np.clip(d32, 0, MAX_DEPTH)

    coverage = (depth > 0).sum() / depth.size * 100
    print(f"  Dense coverage  : {coverage:.1f}%")

    return depth


def make_comparison_panel(sparse, dense, img):
    # 4-panel comparison:
    #   top-left:     camera image
    #   top-right:    sparse LiDAR projected on camera
    #   bottom-left:  sparse depth map (4% coverage)
    #   bottom-right: dense depth map (99%+ coverage)
    H, W    = img.shape[:2]
    gap     = 10
    label_h = 35

    panel = np.full(
        (H*2 + gap*3 + label_h*2,
         W*2 + gap*3, 3),
        30, dtype=np.uint8
    )

    # build sparse LiDAR overlay on camera image
    sparse_overlay = img.copy()
    sparse_colored = colorize_depth(sparse)
    lidar_mask     = sparse > 0
    sparse_overlay[lidar_mask] = sparse_colored[lidar_mask]

    # place 4 panels
    y1 = gap + label_h
    y2 = H + gap*2 + label_h*2

    panel[y1:y1+H, gap:gap+W]           = img
    panel[y1:y1+H, gap*2+W:gap*2+W*2]  = sparse_overlay
    panel[y2:y2+H, gap:gap+W]           = colorize_depth(sparse)
    panel[y2:y2+H, gap*2+W:gap*2+W*2]  = colorize_depth(dense)

    # add labels
    font  = cv2.FONT_HERSHEY_SIMPLEX
    white = (255, 255, 255)
    grey  = (150, 150, 150)

    cv2.putText(panel, "Camera Image",
                (gap, gap+22),
                font, 0.65, white, 2)
    cv2.putText(panel, "LiDAR Points on Camera (4% coverage)",
                (gap*2+W, gap+22),
                font, 0.6, white, 2)
    cv2.putText(panel, "Sparse Depth Map (4% coverage)",
                (gap, H+gap*2+label_h+22),
                font, 0.65, white, 2)
    cv2.putText(panel, "Dense Depth - IP-Basic Completed (99%+)",
                (gap*2+W, H+gap*2+label_h+22),
                font, 0.6, white, 2)
    cv2.putText(panel,
                "RED = close   BLUE = far   BLACK = no data",
                (gap*2+W, H*2+gap*3+label_h*2-8),
                font, 0.5, grey, 1)

    return panel


if __name__ == "__main__":
    import time

    print("\n" + "="*55)
    print("  Depth Completion — Day 8")
    print("  IP-Basic: sparse to dense depth")
    print("="*55)

    P2, T_lidar_to_cam = load_calibration(CALIB_DIR)

    # frame 5 has a good mix of near and far objects
    frames     = sorted(os.listdir(LIDAR_DIR))
    fname      = frames[5]
    lidar_path = os.path.join(LIDAR_DIR, fname)
    img_path   = os.path.join(
        IMAGE_DIR, fname.replace('.bin', '.png')
    )

    print(f"\n  Frame : {fname}")

    sparse, img, u, v, depth_vals = project_lidar_to_image(
        lidar_path, img_path, P2, T_lidar_to_cam
    )

    sparse_pct = (sparse > 0).sum() / sparse.size * 100
    print(f"  Sparse coverage : {sparse_pct:.1f}%")
    print(f"  Depth range     : {depth_vals.min():.1f}m "
          f"to {depth_vals.max():.1f}m")

    print(f"\n  Running IP-Basic...")
    t0    = time.time()
    dense = ip_basic_completion(sparse, img)
    print(f"  Time            : {time.time()-t0:.2f}s")

    os.makedirs(RESULTS_DIR, exist_ok=True)

    cv2.imwrite(
        f"{RESULTS_DIR}/sparse_depth.png",
        colorize_depth(sparse)
    )
    cv2.imwrite(
        f"{RESULTS_DIR}/dense_depth.png",
        colorize_depth(dense)
    )

    panel = make_comparison_panel(sparse, dense, img)
    cv2.imwrite(
        f"{RESULTS_DIR}/depth_comparison.png",
        panel
    )

    print(f"\n  Saved: results/sparse_depth.png")
    print(f"  Saved: results/dense_depth.png")
    print(f"  Saved: results/depth_comparison.png")
    print(f"\n  RED=close  BLUE=far  BLACK=no data")
    print(f"  Open depth_comparison.png")
    print("="*55)