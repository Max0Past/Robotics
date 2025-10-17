# Project Overview

This project is a 2D drone simulation game developed in Python using the Pygame library. The primary objective is to implement and compare different control strategies for a drone tasked with popping balloons. The controllers to be implemented and compared are PID (Proportional-Integral-Derivative), iLQR (iterative Linear Quadratic Regulator), and MPC (Model Predictive Control).

The project is structured as a Python package, with the core logic located in the `task` directory. The main entry point for the application is `task/balloon.py`, which initializes the game and allows the user to select which controller(s) to run.

## Key Files

*   `task/balloon.py`: The main script to run the game. It handles game logic, rendering, and player selection.
*   `task/pid_player.py`: Contains the implementation of the PID controller. The user is expected to complete the implementation of the `PID` and `PIDPlayer` classes.
*   `task/ilqr_player.py`: Contains the implementation of the iLQR and MPC controllers. The user is expected to complete the `rollout`, `backward_pass`, and `forward_pass` functions, as well as the MPC logic.
*   `task/dynamics.py`: Defines the physics model of the drone.
*   `task/cost.py`: Defines the cost function used by the iLQR controller.
*   `README_EN.md`: Provides a detailed explanation of the assignment, including the theoretical background of the controllers and implementation details.
*   `requirements.txt`: Lists the necessary Python packages for the project.
*   `setup.py`: The setup script for installing the project and its dependencies.

# Building and Running

1.  **Installation:**
    To install the project and its dependencies, run the following command in the root directory of the project:
    ```bash
    pip install -e .
    ```

2.  **Running the Game:**
    You can run the game with different controllers using the following commands:

    *   **Human Player:**
        ```bash
        python task/balloon.py --human
        ```
    *   **PID Player:**
        ```bash
        python task/balloon.py --pid
        ```
    *   **iLQR Player:**
        ```bash
        python task/balloon.py --ilqr
        ```
    *   **iLQR with MPC:**
        ```bash
        python task/balloon.py --mpc
        ```
    *   **Run multiple players to compare scores:**
        ```bash
        python task/balloon.py --pid --ilqr --mpc
        ```

    You can also use the `--fast` flag to run the simulation without rendering, which is useful for quickly evaluating the performance of the controllers.

# Development Conventions

The main development task in this project is to complete the implementation of the controllers in `task/pid_player.py` and `task/ilqr_player.py`. The `README_EN.md` file provides detailed instructions and theoretical background for these implementations.

The code is object-oriented, with each player type represented by a class that inherits from the base `Player` class. The controllers are expected to be implemented within these classes.

When implementing the controllers, you should refer to the instructions in `README_EN.md` and the existing code structure. The goal is to create functional controllers that can successfully navigate the drone to the balloons.
