# projection.py
# Projects LiDAR points onto the camera image
# Uses KITTI calibration files for accurate mapping
#
# Color convention (industry standard):
#   RED  = close objects (< 10m)  dangerous
#   BLUE = far objects   (> 40m)  safe
#
# Why we project LiDAR onto camera:
#   Camera gives dense color but NO depth
#   LiDAR gives accurate depth but only 26% pixels
#   Projecting LiDAR onto camera tells us:
#     WHAT each object is (from camera)
#     HOW FAR each object is (from LiDAR)
#   completion.py then fills the remaining 74%
#
# Nani | Day 8 of 90 | MS Robotics ASU

import numpy as np
import os
import cv2

# KITTI paths
KITTI_BASE = (
    r"C:\Users\vamsh\Downloads\kitti"
    r"\2011_09_26_drive_0001_sync"
    r"\2011_09_26"
)
LIDAR_DIR = os.path.join(
    KITTI_BASE,
    r"2011_09_26_drive_0001_sync\velodyne_points\data"
)
IMAGE_DIR = os.path.join(
    KITTI_BASE,
    r"2011_09_26_drive_0001_sync\image_02\data"
)
CALIB_DIR = r"C:\Users\vamsh\Downloads\kitti\2011_09_26"

MAX_DEPTH = 50.0  # max depth for visualization


def load_calibration(calib_dir):
    # load camera intrinsics and LiDAR-to-camera transform
    # these tell us exactly where each LiDAR point
    # lands in the camera image
    cam_file  = os.path.join(
        calib_dir, "calib_cam_to_cam.txt"
    )
    velo_file = os.path.join(
        calib_dir, "calib_velo_to_cam.txt"
    )

    # camera projection matrix P2 (left color camera)
    # contains focal length and image center
    P2 = None
    with open(cam_file, 'r') as f:
        for line in f:
            if line.startswith('P_rect_02:'):
                vals = line.strip().split()[1:]
                P2   = np.array(vals, dtype=np.float32)
                P2   = P2.reshape(3, 4)
                break

    # rotation and translation from LiDAR to camera frame
    R = None
    T = None
    with open(velo_file, 'r') as f:
        for line in f:
            if line.startswith('R:'):
                vals = line.strip().split()[1:]
                R    = np.array(vals, dtype=np.float32)
                R    = R.reshape(3, 3)
            if line.startswith('T:'):
                vals = line.strip().split()[1:]
                T    = np.array(vals, dtype=np.float32)

    # build 4x4 transform matrix
    T_lidar_to_cam         = np.eye(4, dtype=np.float32)
    T_lidar_to_cam[:3, :3] = R
    T_lidar_to_cam[:3,  3] = T

    return P2, T_lidar_to_cam


def project_lidar_to_image(lidar_path, image_path,
                            P2, T_lidar_to_cam):
    # load LiDAR points — (x, y, z, intensity) per point
    pts = np.fromfile(
        lidar_path, dtype=np.float32
    ).reshape(-1, 4)

    # only keep points in front of the car (x > 0)
    # behind-car points cannot be in the camera image
    pts = pts[pts[:, 0] > 0]

    # load camera image
    img  = cv2.imread(image_path)
    H, W = img.shape[:2]

    # Step 1: transform LiDAR points to camera frame
    # add homogeneous coordinate for matrix multiply
    ones    = np.ones((len(pts), 1), dtype=np.float32)
    pts_hom = np.hstack([pts[:, :3], ones])   # (N, 4)
    pts_cam = (T_lidar_to_cam @ pts_hom.T).T  # (N, 4)

    # keep only points in front of camera (z > 0.1)
    # points behind camera would project incorrectly
    valid   = pts_cam[:, 2] > 0.1
    pts_cam = pts_cam[valid]

    # Step 2: project 3D camera points to 2D pixels
    # u = fx * (X/Z) + cx
    # v = fy * (Y/Z) + cy
    pts_2d     = (P2 @ pts_cam[:, :4].T).T  # (N, 3)
    pts_2d[:, 0] /= pts_2d[:, 2]            # u = X/Z
    pts_2d[:, 1] /= pts_2d[:, 2]            # v = Y/Z
    depth_vals   = pts_2d[:, 2]             # actual depth

    # Step 3: keep only points inside image boundaries
    u    = pts_2d[:, 0].astype(int)
    v    = pts_2d[:, 1].astype(int)
    mask = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    u, v = u[mask], v[mask]
    depth_vals = depth_vals[mask]

    # Step 4: build sparse depth map
    # pixels without a LiDAR point stay at 0
    sparse_depth       = np.zeros((H, W), dtype=np.float32)
    sparse_depth[v, u] = depth_vals

    coverage = (sparse_depth > 0).sum() / (H * W) * 100
    print(f"  LiDAR coverage  : {coverage:.1f}% of pixels")
    print(f"  Points on image : {len(u):,}")

    return sparse_depth, img, u, v, depth_vals


def colorize_depth(depth_map, max_depth=MAX_DEPTH):
    # color convention: RED=close, BLUE=far
    # invert depth before JET colormap:
    #   close (small depth) → 1.0 → RED
    #   far   (large depth) → 0.0 → BLUE
    mask       = depth_map > 0
    normalized = np.zeros_like(depth_map)

    normalized[mask] = 1.0 - np.clip(
        depth_map[mask] / max_depth, 0.0, 1.0
    )

    colored = cv2.applyColorMap(
        (normalized * 255).astype(np.uint8),
        cv2.COLORMAP_JET
    )
    colored[~mask] = 0  # black = no LiDAR data
    return colored


def overlay_lidar_on_image(img, u, v, depth_vals,
                            max_depth=MAX_DEPTH):
    # draw each LiDAR point as a colored dot on the image
    # RED=close, BLUE=far (same convention as depth map)
    img_overlay = img.copy()

    # invert depth for color: close=1.0→red, far=0.0→blue
    depth_norm = 1.0 - np.clip(
        depth_vals / max_depth, 0.0, 1.0
    )
    colors = cv2.applyColorMap(
        (depth_norm * 255).astype(np.uint8).reshape(-1, 1),
        cv2.COLORMAP_JET
    ).reshape(-1, 3)

    for i in range(len(u)):
        color = tuple(int(c) for c in colors[i])
        cv2.circle(img_overlay, (u[i], v[i]),
                   2, color, -1)

    return img_overlay


if __name__ == "__main__":
    import time

    print("\n" + "="*55)
    print("  LiDAR-Camera Projection — Day 8")
    print("  Color: RED=close  BLUE=far")
    print("="*55)

    P2, T_lidar_to_cam = load_calibration(CALIB_DIR)

    print(f"\n  Calibration loaded")
    print(f"  Focal length fx : {P2[0,0]:.1f} px")
    print(f"  Image center cx : {P2[0,2]:.1f} px")
    print(f"  Image center cy : {P2[1,2]:.1f} px")

    # test on frame 0
    frames     = sorted(os.listdir(LIDAR_DIR))
    fname      = frames[0]
    lidar_path = os.path.join(LIDAR_DIR, fname)
    img_path   = os.path.join(
        IMAGE_DIR, fname.replace('.bin', '.png')
    )

    print(f"\n  Frame : {fname}")

    t0 = time.time()
    sparse, img, u, v, depth_vals = project_lidar_to_image(
        lidar_path, img_path, P2, T_lidar_to_cam
    )
    print(f"  Time  : {(time.time()-t0)*1000:.1f}ms")
    print(f"  Depth : {depth_vals.min():.1f}m "
          f"to {depth_vals.max():.1f}m")

    os.makedirs("results", exist_ok=True)

    # save sparse depth map (RED=close BLUE=far)
    depth_colored = colorize_depth(sparse)
    cv2.imwrite("results/sparse_depth.png", depth_colored)

    # save LiDAR dots on camera image (RED=close BLUE=far)
    overlay = overlay_lidar_on_image(img, u, v, depth_vals)
    cv2.imwrite("results/lidar_on_camera.png", overlay)

    print(f"\n  Saved: results/sparse_depth.png")
    print(f"  Saved: results/lidar_on_camera.png")
    print(f"\n  RED=close  BLUE=far  BLACK=no data")
    print("="*55)