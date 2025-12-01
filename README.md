# 🤖 Robotics Algorithms Course

A comprehensive collection of robotics and computer vision projects demonstrating fundamental and advanced algorithms in perception, control, and decision-making systems.

---

## Assignment 1: Robot Kinematics & Reinforcement Learning

### Summary

Implementation of forward and inverse kinematics for planar robotic arms with varying degrees of freedom, combined with reinforcement learning for decision-making in Markov Decision Processes.

### Tech Stack

- **Language:** Python
- **Libraries:** NumPy, SciPy, Pygame, Gymnasium
- **Core Techniques:** Linear Algebra, SVD, Jacobian-based methods

### Tech Challenge

Solving the non-linear inverse kinematics problem while handling singularities and joint constraints; implementing optimal decision-making in stochastic environments.

### Solution

- **Damped Least Squares (DLS)** for robust IK near singularities
- **Null-space projection** for joint limit avoidance
- **Policy Iteration** algorithm for finding optimal policies in grid-world environments

### Impact

Successfully demonstrated precise robotic arm control with smooth motion near singularities, and developed optimal agents for FrozenLake and CliffWalking environments with convergence to optimal policies.

---

## Assignment 1.5: Autonomous Drone Control

### Summary

Development of three control strategies for a 2D drone simulation: PID, iLQR (Iterative Linear Quadratic Regulator), and MPC (Model Predictive Control) for balloon-popping task.

### Tech Stack

- **Language:** Python
- **Libraries:** NumPy, SciPy, Pygame
- **Control Methods:** PID, Optimal Control, Receding Horizon Planning

### Tech Challenge

Controlling an underactuated system with only 2 control inputs for 3 degrees of freedom; handling model mismatch and achieving real-time optimal control.

### Solution

- **Cascade PID architecture** for decoupled control loops
- **iLQR** with backward pass (Riccati equations) and forward pass (line search)
- **MPC wrapper** around iLQR for adaptive re-planning at each time step

### Impact

Demonstrated progression from classical to modern optimal control, with MPC achieving the highest robustness and performance by adapting to model uncertainties and disturbances.

---

## Assignment 2: Deep Learning & Panorama Stitching

### Summary

Image classification using ResNet-18 on CIFAR-10 dataset and panorama stitching using geometric computer vision techniques (SIFT, homography, RANSAC).

### Tech Stack

- **Language:** Python
- **Libraries:** PyTorch, TorchVision, OpenCV, NumPy, Matplotlib
- **Deep Learning:** CNN, ResNet-18, OneCycleLR scheduling

### Tech Challenge

Training deep networks efficiently with limited data; robust feature matching under noise and outliers; seamless image blending across overlapping regions.

### Solution

- **ResNet-18 with data augmentation** (random crops, flips) for CIFAR-10 classification
- **SIFT/ORB feature detection** and descriptor matching
- **RANSAC homography estimation** to filter outlier correspondences
- **Image warping and blending** for seamless panoramas

### Impact

Achieved competitive accuracy on CIFAR-10 with modern CNN architecture; successfully created wide-angle panoramic images from standard photos, demonstrating practical geometric computer vision.

---

## Assignment 2.5: Real-Time Rock-Paper-Scissors AI

### Summary

Real-time computer vision system for playing Rock-Paper-Scissors using YOLOv8 neural network and Finite State Machine for game logic on consumer GPU hardware.

### Tech Stack

- **Language:** Python 3.10
- **Frameworks:** PyTorch, Ultralytics (YOLOv8)
- **Libraries:** OpenCV, Roboflow
- **Hardware:** NVIDIA RTX 3050 Laptop (4GB VRAM)

### Tech Challenge

Achieving real-time performance (< 5ms latency) on limited VRAM; handling false positives from background clutter; synchronizing vision inference with game state.

### Solution

- **YOLOv8 Nano** trained with dataset containing ~39% negative examples for clutter rejection
- **Image resolution reduction** (416×416) for memory efficiency
- **State Machine** (Waiting → Countdown → Inference → Result) for game synchronization

### Impact

Achieved **95.3% mAP@50** with **~2.7ms inference time** (~370 FPS), demonstrating production-ready real-time AI system on consumer hardware with robust background handling.

---

## Assignment 3: Sensor Fusion & Object Tracking

### Summary

Implementation of Kalman Filter from first principles for multi-sensor fusion and development of robust multi-object tracking system with confidence gating mechanism.

### Tech Stack

- **Language:** Python
- **Libraries:** NumPy, SciPy, Matplotlib
- **Core Methods:** Probabilistic Filtering, State-Space Estimation, Gating

### Tech Challenge

Combining noisy high-frequency (15Hz) and clean low-frequency (2Hz) sensor streams; implementing robust tracking with clutter rejection; handling covariance propagation and uncertainty growth.

### Solution

- **Linear Kalman Filter** for sequential sensor fusion with adaptive trust weighting
- **4D state vector** [position_x, position_y, velocity_x, velocity_y] for motion prediction
- **Confidence gating** using 3-sigma rectangular gates that adapt to uncertainty growth over time
- **Clutter rejection** by filtering observations outside confidence window

### Impact

Successfully demonstrated superior position estimates by fusing heterogeneous sensor streams; robust tracking with false observation rejection; validated probabilistic estimation theory in practical scenario with smooth animation visualization.

---

## Repository Structure

```
Robotics/
├── Assignment_1/          # Kinematics & RL
├── Assignment_1.5/        # Drone Control (PID, iLQR, MPC)
├── Assignment_2/          # Deep Learning & Panorama Stitching
├── Assignment_2.5/        # Rock-Paper-Scissors AI
├── Assignment_3/          # Sensor Fusion & Object Tracking
└── README.md             # This file
```

---

## Key Technologies & Concepts

| Category | Technologies |
|----------|---------------|
| **Programming** | Python 3.8+ |
| **Deep Learning** | PyTorch, TorchVision, Ultralytics (YOLOv8) |
| **Computer Vision** | OpenCV, SIFT, RANSAC, Homography |
| **Control Systems** | PID, iLQR, MPC, Optimal Control |
| **Signal Processing** | Kalman Filtering, Sensor Fusion, State Estimation |
| **Reinforcement Learning** | Policy Iteration, MDP Solving |
| **Scientific Computing** | NumPy, SciPy, Matplotlib |
| **Simulation** | Pygame, Custom Physics Engines |

---

## Learning Outcomes

This course demonstrates mastery of:

✅ **Robotics Fundamentals:** Forward/Inverse kinematics, singularity handling, joint constraints  
✅ **Control Theory:** PID, optimal control (iLQR), model predictive control, robustness analysis  
✅ **Computer Vision:** Feature detection, matching, geometric transformations, object detection  
✅ **Deep Learning:** CNN architectures, training pipelines, optimization techniques  
✅ **Signal Processing:** Kalman filtering, sensor fusion, probabilistic estimation  
✅ **Software Engineering:** Real-time systems, simulation environments, reproducible pipelines  

---

## Quick Start

Each assignment is self-contained with its own README and requirements. To get started:

```bash
# Navigate to desired assignment
cd Assignment_X

# Install dependencies
pip install -r requirements.txt

# Run Jupyter notebooks or Python scripts
jupyter notebook
```

---

## Course Information

**Instructor:** Andrii Titarenko  
**Institution:** National Technical University of Ukraine "Igor Sikorsky Kyiv Polytechnic Institute" (NTUU "KPI"), Institute of Applied Systems Analysis, Department of Mathematical Methods of Systems Analysis

This repository contains solutions and implementations for assignments designed by **Andrii Titarenko**.

---

*Implementation & Documentation by Maksym Pastushenko | Course: Robotics Algorithms*
