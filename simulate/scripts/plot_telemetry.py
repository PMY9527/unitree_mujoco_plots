#!/usr/bin/env python3
"""Plot GT forward velocity + motor thermal load from unitree_mujoco run logs.

Usage:
    ./plot_telemetry.py                   # plot most recent log
    ./plot_telemetry.py <path/to/csv>     # plot specific log
    ./plot_telemetry.py --list            # list available logs
"""
import argparse
import glob
import os

import numpy as np
import matplotlib.pyplot as plt

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs")

LEG_JOINT_NAMES = [
    "L hip_p", "L hip_r", "L hip_y", "L knee", "L ank_p", "L ank_r",
    "R hip_p", "R hip_r", "R hip_y", "R knee", "R ank_p", "R ank_r",
]


def latest_log():
    files = sorted(glob.glob(os.path.join(LOG_DIR, "telemetry_*.csv")))
    if not files:
        raise FileNotFoundError(f"No telemetry logs in {LOG_DIR}")
    return files[-1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", help="CSV file (default: latest)")
    parser.add_argument("--list", action="store_true", help="list available logs")
    args = parser.parse_args()

    if args.list:
        files = sorted(glob.glob(os.path.join(LOG_DIR, "telemetry_*.csv")))
        for f in files:
            size_kb = os.path.getsize(f) / 1024.0
            print(f"{os.path.basename(f)}  ({size_kb:.1f} KB)")
        return

    path = args.path or latest_log()
    print(f"Loading {path}")
    data = np.loadtxt(path, delimiter=",", skiprows=1)
    if data.ndim == 1:
        data = data[None, :]
    t = data[:, 0]
    vx = data[:, 1]
    loads = data[:, 2:]  # 29 joints

    _, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    axes[0].plot(t, vx, color="C0", linewidth=1.2)
    axes[0].set_ylabel("Forward velocity (m/s)")
    axes[0].set_title(f"GT body-frame fwd velocity  —  {os.path.basename(path)}")
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(0, color="k", linewidth=0.5)

    # Leg joints 0–11
    n_leg = min(12, loads.shape[1])
    cmap = plt.get_cmap("tab20")
    for i in range(n_leg):
        axes[1].plot(t, loads[:, i], label=LEG_JOINT_NAMES[i], color=cmap(i))
    axes[1].set_ylabel("Thermal load (Nm²)")
    axes[1].set_xlabel("Sim time (s)")
    axes[1].set_title("Motor thermal load — leg joints")
    axes[1].legend(loc="upper right", fontsize=7, ncol=2)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
