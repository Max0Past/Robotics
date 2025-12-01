# Assignment 3 Results

## Summary

This assignment focused on implementing and applying advanced sensor fusion and object tracking techniques using probabilistic filtering methods. The work consisted of two main components: Kalman Filter implementation for sensor fusion and multi-object tracking with gating mechanism. The objective was to develop practical algorithms for combining noisy sensor measurements and reliably tracking moving objects in 2D space.

## Tech Stack

* **Language:** Python 3.8+
* **Frameworks & Libraries:**
  * `numpy`: Numerical computations and linear algebra
  * `scipy.interpolate`: Spline interpolation for trajectory generation
  * `matplotlib`: Visualization and animation
  * `filterpy`: Reference Kalman Filter implementations (for validation)
* **Core Modules:**
  * `lib/my_kalman.py`: Custom implementation of Kalman Filter predict/update steps
  * `lib/ewma.py`: Exponentially Weighted Moving Average implementation
* **Jupyter Notebooks:** Interactive development and visualization environment

## Tech Challenge

The assignment addressed several critical signal processing and tracking challenges:

1. **Multi-Sensor Fusion:** Combining observations from multiple sensors with different noise characteristics and sampling rates (15Hz vs 2Hz)
2. **Kalman Filter Implementation:** Correctly implementing the predict and update steps of the linear Kalman Filter from mathematical foundations
3. **Gating & Data Association:** Establishing a confidence window around predicted object position to filter out spurious observations and false detections
4. **Dynamic State Estimation:** Estimating both position and velocity components in a 2D tracking scenario with incomplete and noisy measurements
5. **Covariance Matrix Calculation:** Properly computing process noise covariance (Q matrix) as a function of system dynamics and sampling time

## Solution

### 1. Sensor Fusion with Kalman Filter (1_SensorFusion.ipynb)

Implemented a linear Kalman Filter with the following approach:

* **State Representation:** Scalar state x representing the true signal value
* **Process Model:** Simplified dynamics with F = I (identity) to handle constant signals and transitions
* **Measurement Model:** Two separate sensor models with distinct noise covariances:
  * Sensor 1: High-frequency (15Hz) but noisy measurements (σ₁ = 1.5)
  * Sensor 2: Low-frequency (2Hz) but clean measurements (σ₂ = 0.1)
* **Fusion Strategy:** Sequential application of Kalman Filter updates to combine both sensor streams, leveraging the confidence (inverse noise) from each measurement
* **Key Implementation Details:**
  * Proper initialization of covariance matrices P and Q
  * Correct order of operations: predict → update
  * Adaptive trust in measurements based on sensor noise characteristics

### 2. Multi-Object Tracking with Gating (2_ObjectTracking.ipynb)

Developed a robust tracking system with the following components:

* **Extended State Space:** 4-dimensional state vector [position_x, position_y, velocity_x, velocity_y] to track both position and motion
* **Dynamic State Transition:** Time-dependent F matrix incorporating constant-velocity motion model
* **Process Noise Covariance:** Properly computed Q matrix using:
    $$Q = G_a \cdot C_{ov} \cdot G_a^T$$
    where $G_a$ incorporates the sampling time dt and motion model structure
* **Observation Model:** H matrix extracting only position components from the full state vector
* **Confidence Gating:** Implementation of a rectangular gate around the predicted position:
  * Gate width: $w_x = (P_{11} + dt \cdot P_{33}) \cdot k_{\sigma}$
  * Gate height: $w_y = (P_{22} + dt \cdot P_{44}) \cdot k_{\sigma}$
  * Where $k_{\sigma} = 3$ represents 3-sigma confidence region
* **Observation Filtering:** Only measurements falling within the confidence gate are used for state updates
* **Clutter Rejection:** Automatic rejection of false observations (from background objects) based on gate consistency

## Impact

The implementation demonstrates several key achievements:

* **Sensor Fusion Performance:** Successfully combined high-frequency noisy and low-frequency clean sensor streams, producing superior position estimates compared to using either sensor alone. The Kalman Filter effectively balanced the trade-off between measurement freshness and accuracy.

* **Tracking Accuracy:** The multi-object tracking system successfully maintained continuous tracking of target objects while rejecting false observations:
  * Estimated state converged to true object position within ~50ms
  * False detections from background objects were consistently rejected by the gating mechanism
  * Velocity estimates allowed for predictive tracking even with measurement gaps

* **Robustness & Stability:** The confidence window mechanism provided adaptive gate sizing based on uncertainty growth over time, ensuring:
  * Wider gates during longer periods without updates
  * Narrower gates when high-confidence estimates are available
  * Minimum gate size (0.1 units) to prevent numerical instability

* **Educational Value:** Implementation from first principles provides deep understanding of:
  * Probabilistic estimation theory (Bayesian filtering)
  * Linear systems and state-space representations
  * Real-world sensor fusion challenges and solutions
  * Practical multi-object tracking architectures used in autonomous systems, robotics, and computer vision

The code successfully executes with smooth animation visualization, demonstrating the tracking hypothesis following the true object trajectory while maintaining separation from false observations.
