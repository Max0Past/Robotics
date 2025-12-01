# Assignment 1.5 Results

## Summary
In this assignment, I implemented three distinct control strategies for a 2D drone simulation game where the objective is to pop balloons within a limited time. The controllers developed include a PID (Proportional-Integral-Derivative) controller, an iLQR (Iterative Linear Quadratic Regulator) controller, and an MPC (Model Predictive Control) controller based on iLQR. The project demonstrates the progression from classical model-free control to advanced model-based optimal control methods.

## Tech Stack
*   **Language:** Python
*   **Libraries:**
    *   `numpy`: For efficient numerical computations and matrix operations required by the control algorithms.
    *   `pygame`: For rendering the 2D simulation environment, drone sprites, and game UI.
    *   `scipy`: For additional scientific computing utilities.
*   **Environment:** Custom 2D drone simulation (`balloon.py`) with realistic physics dynamics (`dynamics.py`).

## Tech Challenge
The main technical challenges addressed in this assignment were:
1.  **Underactuated System Control:** The drone is an underactuated system (2 control inputs: left and right thrust, for 3 degrees of freedom: x, y, angle). Designing a controller to stabilize and navigate this system is non-trivial.
2.  **PID Tuning:** Implementing a cascade PID structure to decouple the control loops (e.g., controlling altitude via total thrust and horizontal position via pitch angle) and tuning the gains for stability and responsiveness.
3.  **Optimal Control Implementation:** Deriving and implementing the complex mathematics of iLQR, including the forward rollout, backward pass (computing gains using Riccati equations), and forward pass (line search for trajectory update).
4.  **Robustness to Model Mismatch:** Handling discrepancies between the internal model and the actual system dynamics (e.g., increased drone mass). Standard iLQR fails in such cases, requiring the implementation of MPC to constantly re-plan and adapt.

## Solution
To address these challenges, the following solutions were implemented:
*   **PID Controller:** A cascade PID architecture was developed.
    *   **Altitude Control:** A PID loop computes the desired vertical velocity based on altitude error, which drives the total thrust.
    *   **Horizontal Control:** A PID loop computes the desired pitch angle based on horizontal position error, which drives the thrust difference.
*   **iLQR Controller:** An Iterative Linear Quadratic Regulator was implemented from scratch.
    *   **Rollout:** Simulates the system forward using the current control sequence.
    *   **Backward Pass:** Computes optimal feedback gains by linearizing dynamics and quadratizing costs around the trajectory.
    *   **Forward Pass:** Updates the control sequence using the computed gains and a line search to minimize the cost function.
*   **MPC Controller:** Integrated Model Predictive Control with iLQR.
    *   Instead of executing the entire optimized trajectory, the MPC agent re-runs the iLQR optimization at every time step and executes only the first control action. This allows the controller to react to disturbances and model errors in real-time.

## Impact
The implemented solutions resulted in a capable autonomous drone agent:
*   **Performance:** The controllers successfully navigated the drone to pop balloons.
*   **Comparison:**
    *   **PID:** Provided a stable baseline but struggled with agility and precise tracking of moving targets.
    *   **iLQR:** Demonstrated superior performance with smooth, optimal trajectories when the model was accurate.
    *   **MPC:** Achieved the highest robustness and score, effectively handling cases where the drone's mass was altered (simulating model mismatch), where the standard iLQR failed.
*   **Quantitative Results:** The MPC approach yielded the highest game scores, significantly outperforming the PID baseline and showing greater reliability than the open-loop iLQR execution.
