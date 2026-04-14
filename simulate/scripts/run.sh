#!/bin/bash
# Wrapper: run unitree_mujoco, then plot telemetry on exit (including Ctrl+C).

SCRIPT_DIR="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
BUILD_DIR="$SCRIPT_DIR/../build"
BINARY="$BUILD_DIR/unitree_mujoco"

if [ ! -x "$BINARY" ]; then
  echo "binary not found: $BINARY" >&2
  echo "build it first: cd $BUILD_DIR && make unitree_mujoco" >&2
  exit 1
fi

# Ignore SIGINT in this wrapper so Ctrl+C kills only the binary.
# The binary exits, then we fall through to the plot step.
trap '' INT
"$BINARY" "$@"
trap - INT

exec python3 "$SCRIPT_DIR/plot_telemetry.py"
