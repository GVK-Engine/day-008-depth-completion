# visualize.py
# Creates demo video and GIF showing depth completion
# running on all 108 KITTI frames
#
# Output 1: depth_completion_demo.mp4
#   4-panel video at 10 FPS
#   Panel 1: Camera image
#   Panel 2: LiDAR points on camera
#   Panel 3: Sparse depth (4% coverage)
#   Panel 4: Dense depth (100% coverage)
#   Shows the fusion pipeline processing each frame
#
# Output 2: depth_completion_demo.gif
#   First 20 frames as animated GIF
#   For LinkedIn and GitHub README
#
# This is what makes Day 8 stand out.
# Not just numbers. A real video demo.
# Like what Waymo shows in their papers.
#
# Nani | Day 8 of 90 | MS Robotics ASU

import numpy as np
import cv2
import os
import imageio.v2 as imageio
from completion import ip_basic_completion, colorize_depth
from projection import (
    load_calibration,
    project_lidar_to_image,
    overlay_lidar_on_image,
    CALIB_DIR, LIDAR_DIR, IMAGE_DIR
)

RESULTS_DIR = "results"
VIDEO_FPS   = 10
PANEL_GAP   = 8
LABEL_H     = 30


def make_frame(img, sparse, dense, u, v, depth_vals,
               frame_num, total_frames):
    """
    Build one 4-panel video frame:
      Top-left:     Camera image
      Top-right:    LiDAR points on camera
      Bottom-left:  Sparse depth map
      Bottom-right: Dense completed depth map

    Each panel labeled clearly.
    Frame counter shown bottom-right.
    Color legend shown bottom-center.
    """
    H, W = img.shape[:2]

    # build each panel
    panel_cam    = img.copy()
    panel_lidar  = overlay_lidar_on_image(
        img, u, v, depth_vals
    )
    panel_sparse = colorize_depth(sparse)
    panel_dense  = colorize_depth(dense)

    # total frame size
    frame_w = W * 2 + PANEL_GAP * 3
    frame_h = H * 2 + PANEL_GAP * 3 + LABEL_H * 2
    frame   = np.full(
        (frame_h, frame_w, 3), 20, dtype=np.uint8
    )

    # place panels
    y1 = PANEL_GAP + LABEL_H
    y2 = H + PANEL_GAP * 2 + LABEL_H * 2
    x1 = PANEL_GAP
    x2 = W + PANEL_GAP * 2

    frame[y1:y1+H, x1:x1+W]   = panel_cam
    frame[y1:y1+H, x2:x2+W]   = panel_lidar
    frame[y2:y2+H, x1:x1+W]   = panel_sparse
    frame[y2:y2+H, x2:x2+W]   = panel_dense

    # font settings
    font  = cv2.FONT_HERSHEY_SIMPLEX
    white = (255, 255, 255)
    grey  = (140, 140, 140)
    cyan  = (0, 220, 255)

    # panel labels
    cv2.putText(frame, "Camera Image",
                (x1, PANEL_GAP + 20),
                font, 0.6, white, 1)
    cv2.putText(frame, "LiDAR on Camera (4% coverage)",
                (x2, PANEL_GAP + 20),
                font, 0.6, cyan, 1)
    cv2.putText(frame, "Sparse Depth Map (4%)",
                (x1, H + PANEL_GAP * 2 + LABEL_H + 20),
                font, 0.6, white, 1)
    cv2.putText(frame, "Dense Depth - IP-Basic (100%)",
                (x2, H + PANEL_GAP * 2 + LABEL_H + 20),
                font, 0.6, cyan, 1)

    # title bar at very top
    title = (
        "LiDAR-Camera Depth Completion  |  "
        "Vamshikrishna Gadde  |  MS Robotics ASU  |  "
        "Day 8 of 90"
    )
    cv2.putText(frame, title,
                (PANEL_GAP, PANEL_GAP + 5),
                font, 0.45, grey, 1)

    # frame counter bottom right
    counter = f"Frame {frame_num:03d} / {total_frames:03d}"
    cv2.putText(frame, counter,
                (frame_w - 200, frame_h - 8),
                font, 0.45, grey, 1)

    # color legend bottom center
    legend = "RED = close   BLUE = far   BLACK = no data"
    text_w = cv2.getTextSize(legend, font, 0.45, 1)[0][0]
    cv2.putText(frame, legend,
                ((frame_w - text_w) // 2, frame_h - 8),
                font, 0.45, grey, 1)

    return frame


def create_video_and_gif(n_frames_video=108,
                          n_frames_gif=20):
    """
    Process KITTI frames and save MP4 video and GIF.

    n_frames_video: how many frames for the MP4
    n_frames_gif:   how many frames for the GIF
                    (keep small so file is not huge)
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)

    P2, T_lidar_to_cam = load_calibration(CALIB_DIR)

    all_files = sorted([
        os.path.join(LIDAR_DIR, f)
        for f in os.listdir(LIDAR_DIR)
        if f.endswith('.bin')
    ])

    video_files = all_files[:n_frames_video]
    gif_files   = all_files[:n_frames_gif]

    # get frame size from first frame
    first_img = cv2.imread(
        os.path.join(
            IMAGE_DIR,
            os.path.basename(video_files[0])
              .replace('.bin', '.png')
        )
    )
    H, W  = first_img.shape[:2]
    frame_w = W * 2 + PANEL_GAP * 3
    frame_h = H * 2 + PANEL_GAP * 3 + LABEL_H * 2

    # setup video writer
    video_path = os.path.join(
        RESULTS_DIR, 'depth_completion_demo.mp4'
    )
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(
        video_path, fourcc, VIDEO_FPS,
        (frame_w, frame_h)
    )

    gif_frames = []

    print(f"\n  Building video: {len(video_files)} frames")
    print(f"  Building GIF  : {len(gif_files)} frames")
    print(f"  Frame size    : {frame_w} x {frame_h}")
    print(f"  {'─'*50}")

    for fi, fpath in enumerate(video_files):
        fname    = os.path.basename(fpath)
        img_path = os.path.join(
            IMAGE_DIR, fname.replace('.bin', '.png')
        )

        # project and complete
        sparse, img, u, v, depth_vals = \
            project_lidar_to_image(
                fpath, img_path, P2, T_lidar_to_cam
            )
        dense = ip_basic_completion(sparse, img)

        # build frame
        frame = make_frame(
            img, sparse, dense, u, v, depth_vals,
            fi + 1, len(video_files)
        )

        # write to video
        writer.write(frame)

        # collect GIF frames
        if fi < n_frames_gif:
            # convert BGR to RGB for imageio
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # resize for smaller GIF file
            gif_h = frame_h // 2
            gif_w = frame_w // 2
            frame_small = cv2.resize(
                frame_rgb, (gif_w, gif_h)
            )
            gif_frames.append(frame_small)

        if (fi + 1) % 10 == 0:
            print(f"  Processed {fi+1}/{len(video_files)} frames")

    writer.release()
    print(f"\n  Video saved: {video_path}")

    # save GIF
    gif_path = os.path.join(
        RESULTS_DIR, 'depth_completion_demo.gif'
    )
    imageio.mimsave(
        gif_path, gif_frames,
        duration=1.0 / VIDEO_FPS,
        loop=0
    )
    print(f"  GIF saved  : {gif_path}")

    return video_path, gif_path


if __name__ == "__main__":
    import time

    print("\n" + "="*55)
    print("  Demo Video + GIF - Day 8")
    print("  LiDAR-Camera Depth Completion")
    print("="*55)

    t0 = time.time()
    video_path, gif_path = create_video_and_gif(
        n_frames_video=108,
        n_frames_gif=20
    )
    elapsed = time.time() - t0

    print(f"\n  Total time : {elapsed:.1f}s")
    print(f"\n  OUTPUTS:")
    print(f"  Video : {video_path}")
    print(f"  GIF   : {gif_path}")
    print(f"\n  Upload GIF to LinkedIn post.")
    print(f"  Upload MP4 to YouTube, link in README.")
    print("="*55)