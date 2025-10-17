## Project Overview

This is Assignment 1.5 from a Robotics course, focusing on implementing and comparing different control algorithms:
- PID (Proportional-Integral-Derivative) controller
- iLQR (iterative Linear Quadratic Regulator)
- MPC (Model Predictive Control) combined with iLQR

### Project Purpose
The main goal is to implement and compare different control strategies for a 2D drone that needs to pop balloons. The drone has two control inputs (left and right thrusts) and needs to navigate in a 2D space to reach balloons.

### Tech Stack
- Python 3.x
- Key dependencies:
  - numpy >= 1.26.0 (numerical computations)
  - pygame >= 2.5.1 (game visualization)
  - numba >= 0.60.0 (code optimization)
  - scipy (scientific computations)

### Core Components
1. `dynamics.py` - Contains the drone dynamics model
2. `pid_player.py` - PID controller implementation
3. `ilqr_player.py` - iLQR and MPC implementation
4. `balloon.py` - Main game implementation
5. Additional support files for visualization and game mechanics