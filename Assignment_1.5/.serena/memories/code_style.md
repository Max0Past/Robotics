## Code Style and Conventions

### Python Style Guidelines
1. **Code Formatting**
   - Uses standard Python style (PEP 8)
   - Classes use CamelCase: `PIDPlayer`, `iLQRPlayer`
   - Functions and variables use snake_case: `run_ilqr`, `goal_cost`

2. **Documentation**
   - Docstrings present in setup.py and main classes
   - Mathematical concepts explained in README_EN.md

3. **Project Structure**
   - Core implementation in `task/` directory
   - Separate modules for each controller type
   - Clear separation of concerns between dynamics, controllers, and game logic

4. **Type Hints**
   - NumPy arrays used extensively for state and control vectors
   - Mathematics-heavy code with clear variable naming reflecting mathematical concepts

5. **Design Patterns**
   - Strategy pattern for different controllers (PID, iLQR, MPC)
   - Inheritance used for player implementations
   - Separation of dynamics model from controllers