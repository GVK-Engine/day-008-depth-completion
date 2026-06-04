# Day 8 — LiDAR-Camera Depth Completion

> **Series 1: Perception | Project 8 of 12**
> MS Robotics & Autonomous Systems Engineering — Arizona State University — Dec 2026

---

## The Problem This Solves

Day 2 of this series proved that cameras fail beyond 35m.
Day 7 proved that LiDAR fails in dense fog.

Neither sensor is sufficient alone.
They fail in completely different conditions.

**Day 8 fuses both sensors to cover each other's failures.**

---

## Live Demo — 108 KITTI Frames

*4-panel pipeline: camera → LiDAR projection → sparse depth → dense completed depth*

![Depth Completion Demo](https://drive.google.com/uc?id=1F9BV6Hk5C9BT1JD8r5yv38egV6u4WLuz)

---

## Results

### Sparse vs Dense Depth

![Depth Comparison](https://drive.google.com/uc?id=10LB_X-bip_OFLJYxeIhN6PBBLWoz1GN_)

### LiDAR Points Projected on Camera

![LiDAR on Camera](https://drive.google.com/uc?id=1A_p_zWbYdJmC7A55ZUHmiUS1e01G6n0q)

### Sparse Depth Map — 4.1% Coverage

![Sparse Depth](https://drive.google.com/uc?id=184mB7nw8BqvmI8mwhE2F_PGuBgCSATVG)

### Dense Depth Map — 100% Coverage

![Dense Depth](https://drive.google.com/uc?id=1xEb9Ut1Agb8og_smxE9co5NnB-9jhYhw)

---

## Key Finding — Sensor Fusion Improvement

![Evaluation Chart](https://drive.google.com/uc?id=1ArAWNBpvsjSQRdJ1c1oTQw2iMH6KNhST)

| Distance Band | Camera Only (Day 2) | Fusion (Day 8) | Improvement |
|---|---|---|---|
| 0-10m | 1.039m MAE | 0.024m MAE | **44x better** |
| 10-20m | 2.302m MAE | 0.236m MAE | **10x better** |
| 20-30m | 3.819m MAE | 0.498m MAE | **8x better** |
| 30-50m | 5.377m MAE | 0.943m MAE | **6x better** |
| 50m+ | 8.813m MAE | 1.539m MAE | **6x better** |

> Evaluated using 80/20 holdout method — 20% of LiDAR points
> withheld as ground truth. The algorithm never saw test pixels.
> These are honest numbers, not biased evaluation.

---

## The Engineering Story

### Why Camera Alone Fails

A camera gives you a dense beautiful image.
But it has no depth information.
A car 5m away and a car 50m away
look different sizes but the camera
cannot tell you the exact distance.

Day 2 measured this failure directly:
camera depth error at 50m+ = 8.813m.
That is the length of a bus.
At highway speed this is fatal.

### Why LiDAR Alone Is Not Enough

LiDAR gives accurate depth but only covers
4.1% of image pixels on a forward camera.
The other 95.9% has no depth measurement.
A pedestrian standing between two laser scan lines
has no measured depth at all.

### Why Fusion Works

Project LiDAR onto the camera image.
Every LiDAR point now has a pixel location.
Use camera color to fill the gaps.
Pixels with similar color to a nearby LiDAR point
probably belong to the same surface.
Same surface = similar depth.
This is guided depth completion.

Result: 100% pixel coverage with LiDAR-accurate depths.

---

## Why the Second Car Has No LiDAR Points

Look at the sparse depth image carefully.
Some parked cars appear with no depth dots at all.

This is **LiDAR occlusion shadow** — a real physical phenomenon.

```
The LiDAR shoots laser pulses in straight lines.
Car A is in front of Car B.
The pulses hit Car A and return.
They never reach Car B.
Car B is in the laser shadow of Car A.

This is why Waymo uses 5 LiDAR units
at different positions on the vehicle.
Different positions = fewer occlusion shadows.
```

Depth completion assigns neighbor depth to occluded pixels.
This can be wrong. Neural-network based completion
learns to recognize car shapes and assign
plausible depths even without LiDAR hits.
That is why deep learning fusion outperforms
geometric methods like IP-Basic.

---

## Algorithm — IP-Basic Depth Completion

```
INPUT: Sparse depth map (4.1% coverage)
         |
Step 1: Small dilation (3x3 kernel)
         Expands each LiDAR point 1 pixel outward
         Fills gaps between adjacent scan lines
         |
Step 2: Bilateral filter
         Smooths depth at object boundaries
         Camera color prevents wrong blending
         Wall at 30m stays separate from car at 8m
         |
Step 3: Large dilation (15x15 kernel)
         Fills medium holes between objects
         |
Step 4: Morphological closing (31x31 kernel)
         Fills round holes that dilation misses
         |
Step 5: Global nearest-neighbor fill
         Every remaining pixel gets depth of
         its nearest known LiDAR point
         Coverage goes from ~60% to 100%
         |
Step 6: Final bilateral filter
         Clean smooth output

OUTPUT: Dense depth map (100% coverage)
```

---

## Performance

| Metric | Value |
|--------|-------|
| LiDAR input coverage | 4.1% of pixels |
| Dense output coverage | 100.0% |
| Completion time | 0.09s per frame |
| Frames processed | 108 KITTI frames |
| Evaluation method | 80/20 holdout |
| GPU | NVIDIA RTX 4050 |

---

## How This Connects to the Series

```
Day 2: Camera alone fails at 35m+
       MAE 8.813m — the problem identified

Day 7: LiDAR alone fails below 75m fog visibility
       The second failure mode identified

Day 8: Fuse both sensors
       Camera covers LiDAR in fog
       LiDAR covers camera at range
       44x improvement at close range
       6x improvement at far range

Day 9: Use dense depth to predict
       where pedestrians are going
       The next layer of the perception stack
```

---

## Run It Yourself

```bash
git clone https://github.com/GVK-Engine/day-008-depth-completion
cd day-008-depth-completion
pip install -r requirements.txt
```

Update `KITTI_BASE` and `CALIB_DIR` in each file to your KITTI path.

```bash
# Project LiDAR onto camera image
py -3.11 projection.py

# Run IP-Basic depth completion
py -3.11 completion.py

# Evaluate with holdout method
py -3.11 evaluate.py

# Generate 108-frame video and GIF
py -3.11 visualize.py
```

KITTI dataset: https://www.cvlibs.net/datasets/kitti/raw_data.php

---

## Project Structure

```
day-008-depth-completion/
├── projection.py       LiDAR-to-camera projection using calibration
├── completion.py       IP-Basic guided depth completion
├── evaluate.py         Holdout evaluation vs camera-only baseline
├── visualize.py        108-frame demo video and GIF generator
├── requirements.txt
└── results/
    ├── sparse_depth.png
    ├── dense_depth.png
    ├── depth_comparison.png
    ├── evaluation_chart.png
    ├── depth_completion_demo.gif
    └── depth_completion_demo.mp4
```

---

## Stack

`Python 3.11` `NumPy` `OpenCV` `SciPy` `Matplotlib` `imageio` `KITTI`

---

## Series 1 — Perception Progress

| # | Project | Key Finding | Status |
|---|---------|-------------|--------|
| P1.1 | LiDAR Obstacle Detection | 0.4m voxel creates ghost detections | ✅ |
| P1.2 | Stereo Camera Depth Safety | Camera unsafe beyond 10m — MAE 8.8m at 50m+ | ✅ |
| P1.3 | PointPillars 3D Detector | 98.9% loss reduction from scratch | ✅ |
| P1.4 | Multi-Camera BEV Perception | 178 objects from 6 cameras — IPM failure found | ✅ |
| P1.5 | Multi-Object Tracking SORT | Detector is bottleneck — tracker at 99.9% | ✅ |
| P1.6 | Semantic Segmentation ROS2 | 52.6 FPS — warmup cost measured | ✅ |
| P1.7 | Adverse Weather Analysis | Fog unsafe below 75m — rain safe at 100mm/hr | ✅ |
| P1.8 | LiDAR-Camera Depth Completion | 44x MAE improvement — 108-frame demo | ✅ |
