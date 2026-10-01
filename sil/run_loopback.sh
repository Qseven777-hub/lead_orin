#!/usr/bin/env bash
# End-to-end loopback of the SIL transport, on one machine and without CARLA.
#
#   roscore (ROS Noetic)  <-  bridge (ROS <-> ZeroMQ)  <-  host probe (py3.10)
#                                    ^
#                                    +-- stub Orin (echoes a control)
#
# Validates the wire contract, the bridge and the ZeroMQ transport. The real
# Orin replaces the stub later.
#
# Usage:
#   bash sil/run_loopback.sh
# NOTE: no `set -u`; the RoboStack activate hooks reference unset variables.
set -o pipefail

cd "$(dirname "$(realpath "${BASH_SOURCE:-$0}")")/.." || exit 1
CONDA_SH="${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
# shellcheck source=/dev/null
source "$CONDA_SH"

ROS_ENV="${ROS_ENV:-ros_noetic}"
COMPUTE_ENV="${COMPUTE_ENV:-cvci_project}"
LOG_DIR="${LOG_DIR:-/tmp/sil_loopback}"
mkdir -p "$LOG_DIR"

export ROS_MASTER_URI="${ROS_MASTER_URI:-http://127.0.0.1:11311}"
export ROS_IP="${ROS_IP:-127.0.0.1}"

ROSCORE_PID=""
BRIDGE_PID=""
STUB_PID=""

cleanup() {
  for pid in "$STUB_PID" "$BRIDGE_PID" "$ROSCORE_PID"; do
    [ -n "$pid" ] && kill -9 "$pid" 2>/dev/null
  done
  return 0
}
trap cleanup EXIT INT TERM

export PYTHONUNBUFFERED=1

echo "[loopback] starting roscore ($ROS_ENV)"
conda activate "$ROS_ENV" || exit 1
roscore -p 11311 >"$LOG_DIR/roscore.log" 2>&1 &
ROSCORE_PID=$!

for _ in $(seq 1 50); do
  if python -c "import rosgraph; rosgraph.Master('/probe').getPid()" >/dev/null 2>&1; then
    break
  fi
  sleep 0.2
done
if ! kill -0 "$ROSCORE_PID" 2>/dev/null; then
  echo "[loopback] roscore died; see $LOG_DIR/roscore.log"
  exit 1
fi
echo "[loopback] roscore up"

echo "[loopback] starting bridge"
python -u sil/ros_bridge/bridge_node.py >"$LOG_DIR/bridge.log" 2>&1 &
BRIDGE_PID=$!

echo "[loopback] starting stub Orin"
python -u sil/tools/stub_orin_node.py >"$LOG_DIR/stub.log" 2>&1 &
STUB_PID=$!

# Give the ROS graph a moment to connect before the probe sends.
for _ in $(seq 1 30); do
  if rostopic list 2>/dev/null | grep -q "^/lead/control$"; then
    break
  fi
  sleep 0.2
done
sleep 1

echo "[loopback] running host probe ($COMPUTE_ENV)"
conda activate "$COMPUTE_ENV" || exit 1
python sil/tools/local_probe.py
status=$?

if [ "$status" -ne 0 ]; then
  echo "[loopback] FAILED; logs in $LOG_DIR"
fi
exit "$status"
