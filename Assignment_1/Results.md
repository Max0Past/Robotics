# Assignment 1 Results

## Summary
This assignment focuses on fundamental robotics algorithms, specifically Kinematics and Reinforcement Learning. I implemented Forward Kinematics (FK) and Inverse Kinematics (IK) for planar robotic arms with varying degrees of freedom (DOF). Additionally, I developed an interactive IK solver allowing for real-time comparison of different numerical methods. Finally, I implemented the Policy Iteration algorithm to solve Markov Decision Processes (MDPs) in grid-world environments.

## Tech Stack
*   **Language:** Python
*   **Libraries:**
    *   `numpy`: For linear algebra operations (matrix multiplication, SVD, pseudo-inverse) essential for computing Jacobians and solving kinematic equations.
    *   `pygame`: For visualizing the robotic arms and creating an interactive simulation environment.
    *   `gymnasium`: For providing standard Reinforcement Learning environments (FrozenLake, CliffWalking).
    *   `scipy`: For auxiliary scientific computations.

## Tech Challenge
The key technical challenges addressed in this assignment included:
1.  **Inverse Kinematics (IK):** Solving the non-linear problem of finding joint angles for a desired end-effector position. This is ill-posed for redundant manipulators (more DOFs than task constraints).
2.  **Singularity Handling:** Standard inverse kinematic methods (like the Moore-Penrose pseudoinverse) become unstable near kinematic singularities where the Jacobian loses rank.
3.  **Joint Limits & Constraints:** Ensuring the robotic arm stays within physical joint limits while reaching the target.
4.  **MDP Solving:** Implementing Policy Iteration to find optimal policies in stochastic environments where actions have uncertain outcomes.

## Solution
To overcome these challenges, I implemented the following solutions:
*   **Forward Kinematics:** Implemented using transformation matrices/trigonometry to map joint angles to end-effector positions.
*   **Robust Inverse Kinematics:**
    *   **Damped Least Squares (DLS):** Implemented SVD-based damped pseudoinverse to robustly handle singularities by balancing error minimization and solution norm.
    *   **Null-Space Projection:** Utilized the redundancy of the arm to perform secondary tasks (like joint limit avoidance) without affecting the primary end-effector goal.
    *   **Newton-Raphson / Gauss-Newton:** Implemented iterative numerical solvers to minimize position error.
*   **Interactive Solver:** Created a real-time application (`task4.py`) that allows users to switch between IK methods (DLS, PINV, NR) and interactively control the arm target with the mouse.
*   **Policy Iteration:** Implemented the Policy Iteration algorithm (`task5.py`) consisting of Policy Evaluation (iteratively computing the value function) and Policy Improvement (greedily updating the policy) to solve the FrozenLake and CliffWalking environments.

## Impact
The completed assignment demonstrates:
*   **Precise Control:** The IK solvers enable the robotic arm to accurately track trajectories and draw complex shapes.
*   **Robustness:** The DLS and null-space projection methods ensure smooth motion even near singularities and joint limits.
*   **Optimal Decision Making:** The Policy Iteration agent successfully converges to the optimal path in both slippery (stochastic) and dangerous (CliffWalking) grid worlds, maximizing cumulative reward.
