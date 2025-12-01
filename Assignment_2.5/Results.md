# Assignment 2.5 Results

## Summary
This project involved developing a real-time computer vision system for playing "Rock-Paper-Scissors" against an AI. The system utilizes a **YOLOv8** neural network for gesture detection and a Finite State Machine (FSM) to manage the game logic. The goal was to create a responsive and accurate AI referee that runs efficiently on consumer hardware (NVIDIA RTX 3050 Laptop).

## Tech Stack
*   **Language:** Python 3.10
*   **Frameworks:** PyTorch, Ultralytics (YOLOv8)
*   **Libraries:**
    *   `opencv-python` (cv2): For video capture, image processing, and visualization.
    *   `roboflow`: For dataset management.
*   **Model:** YOLOv8 Nano (`yolov8n.pt`) - selected for its balance of speed and accuracy.
*   **Hardware:** NVIDIA GeForce RTX 3050 Laptop (4GB VRAM).

## Tech Challenge
The main technical challenges addressed in this assignment were:
1.  **Real-time Performance:** Ensuring the system runs with low latency (< 5ms) to provide a smooth gaming experience.
2.  **Resource Constraints:** Training and running the model on a GPU with limited VRAM (4GB), requiring optimization of batch sizes and image resolution.
3.  **False Positives:** Handling background noise and ensuring the model doesn't detect gestures when no hands are present.
4.  **Game Logic:** Implementing a robust state machine to handle game states (Waiting, Countdown, Result) and synchronize them with the computer vision inference.

## Solution
To address these challenges, the following solutions were implemented:
*   **Data Preparation:** Used the "Rock-Paper-Scissors SXSW" dataset from Roboflow, which includes ~39% "negative" examples (background images without hands) to train the model to ignore irrelevant objects.
*   **Model Training:**
    *   Trained **YOLOv8 Nano** for 30 epochs.
    *   Reduced input image size to **416x416** (from 640x640) to save memory and increase FPS.
    *   Used Automatic Mixed Precision (AMP) for faster training.
*   **Game Engine:** Developed a Python script (`main.py`) with a State Machine:
    *   **Waiting:** Waits for user input to start.
    *   **Countdown:** Displays a 3-2-1 timer.
    *   **Inference:** Performs instant detection at the end of the countdown.
    *   **Result:** Compares the player's move with the AI's random move and declares a winner.

## Impact
The resulting system demonstrated exceptional performance and reliability:
*   **Accuracy:** Achieved **mAP@50 of 95.3%** on the validation set.
    *   Precision: 0.949
    *   Recall: 0.924
*   **Speed:** Ultra-fast inference time of **~2.7ms** (theoretical ~370 FPS), significantly exceeding standard webcam frame rates.
*   **Robustness:** The model effectively ignores complex backgrounds and consistently detects gestures (Rock, Paper, Scissors) with high confidence.
