# unitree_mujoco (telemetry fork)

Adds live telemetry visualization (GT body-frame velocity + motor thermal load)
and per-run CSV logging on top of the upstream simulator.

## Build

```bash
cd simulate
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc) unitree_mujoco
```

## Run

Two options:

```bash
# Plain run (no auto-plot on exit)
./simulate/build/unitree_mujoco

# Wrapper: runs the binary, then auto-plots the latest telemetry log on exit
./simulate/scripts/run.sh
```

### Key bindings

| Key | Action |
|-----|--------|
| `T` | Toggle live telemetry overlay (velocity + thermal load figures) |
| `C` | Toggle CMG ghost overlay — renders a transparent ghost of the CMG reference pose next to the robot, driven via `/cmg_viz_data` shared memory |
| `Backspace` | Reset simulation |
| `9` | Toggle elastic band (when enabled in config.yaml) |
| `7` / `↑` | Shorten elastic band |
| `8` / `↓` | Lengthen elastic band |

### Telemetry overlay (T key)

- **Top figure:** GT base linear velocity in body frame (fwd / lat / vert, m/s)
- **Bottom figure:** motor thermal load (EMA of tau²) for all 12 leg joints

Regardless of the `T` toggle, every run writes a CSV to `simulate/logs/` named
`telemetry_YYYYMMDD_HHMMSS.csv` — the toggle only controls whether figures are
drawn in the viewer.

### CMG ghost overlay (C key)

Requires a writer process publishing to the `/cmg_viz_data` shared memory
region (see `simulate/src/cmg_viz_shm.h`). When data is available and `C` is
pressed, a transparent blue ghost of the CMG reference pose is overlaid in the
viewer at a −0.5 m x-offset from the live robot.

## Plotting logs offline

```bash
# Plot the most recent log
./simulate/scripts/plot_telemetry.py

# Plot a specific log
./simulate/scripts/plot_telemetry.py simulate/logs/telemetry_20260415_013000.csv

# List all logs
./simulate/scripts/plot_telemetry.py --list
```

Requires `numpy` and `matplotlib`.

## Thermal Load Logs

Metric: EMA of tau² (Nm²).

| Policy | Cmd (m/s) | Peak Load (Nm²) | Hottest Joints |
|--------|-----------|-----------------|----------------|
| RuN | 1.0 | ~420 | Both knees + ankle pitch |
| unitree_rl_lab official | 1.0 | ~410 | Knees only |
| AMP policy1 | 1.0 | ~550 | Knees |
| AMP baseline | 1.0 | ~400 | Hip + knee |
| AMP baseline | 2.0 | ~880 | Knees |
| RuN | 2.0 | ~1000 | Knees |
