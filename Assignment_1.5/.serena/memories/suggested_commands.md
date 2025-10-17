## Essential Commands

### Installation
```bash
pip install -e .
```

### Running the Game
1. Human player mode:
```bash
python balloon.py --human
```

2. PID controller:
```bash
python balloon.py --pid
```

3. iLQR controller:
```bash
python balloon.py --ilqr
```

4. MPC (with iLQR):
```bash
python balloon.py --mpc
```

### Testing Different Configurations
1. Fast mode (for testing):
```bash
python balloon.py --fast
```

2. Multiple controllers comparison:
```bash
python balloon.py --mpc --ilqr --pid --fast
```

3. Testing with modified physics:
```bash
python balloon.py --mpc --ilqr --fast --mass 2.0
```

### Windows-specific Commands
- Directory listing: `dir`
- Change directory: `cd`
- Find in files: `findstr`
- Clear screen: `cls`
- Environment variables: `set`