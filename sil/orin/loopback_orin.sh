#!/usr/bin/env bash
# Orin-side SIL transport self-check: no CARLA, no engine.
#
#   roscore (native Noetic)  <-  bridge (py3.8)  <-  host probe (py3.10)
#                                     ^
#                                     +-- stub Orin (py3.8, echoes a control)
#
# Validates the wire contract, the bridge and the ZeroMQ transport on the Orin
# itself.  Uses the native ROS Noetic and the miniforge compute env.
set -o pipefail

ORIN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$ORIN_DIR/../.." && pwd)"

ROS_IP="${ROS_IP:-127.0.0.1}"
ROS_MASTER_URI="${ROS_MASTER_URI:-http://127.0.0.1:11311}"
CONDA_SH="${CONDA_SH:-/home/tjuae/miniforge3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-gqzl-py310}"
LOG_DIR="${LOG_DIR:-/tmp/sil_loopback}"
mkdir -p "$LOG_DIR"

export ROS_MASTER_URI ROS_IP PYTHONUNBUFFERED=1
source /opt/ros/noetic/setup.bash

ROSCORE_PID=""; BRIDGE_PID=""; STUB_PID=""
cleanup() {
  for pid in "$STUB_PID" "$BRIDGE_PID" "$ROSCORE_PID"; do
    if [ -n "$pid" ]; then
      pkill -9 -P "$pid" 2>/dev/null   # e.g. roscore's rosmaster/rosout children
      kill -9 "$pid" 2>/dev/null
    fi
  done
  return 0
}
trap cleanup EXIT INT TERM

echo "[loopback] roscore"
roscore -p 11311 >"$LOG_DIR/roscore.log" 2>&1 &
ROSCORE_PID=$!
for _ in $(seq 1 50); do
  /usr/bin/python3 -c "import rosgraph; rosgraph.Master('/probe').getPid()" >/dev/null 2>&1 && break
  sleep 0.2
done
kill -0 "$ROSCORE_PID" 2>/dev/null || { echo "[loopback] roscore died; $LOG_DIR/roscore.log"; exit 1; }

echo "[loopback] bridge (system py3.8)"
/usr/bin/python3 -u "$REPO/sil/ros_bridge/bridge_node.py" >"$LOG_DIR/bridge.log" 2>&1 &
BRIDGE_PID=$!

echo "[loopback] stub Orin (system py3.8)"
/usr/bin/python3 -u "$REPO/sil/tools/stub_orin_node.py" >"$LOG_DIR/stub.log" 2>&1 &
STUB_PID=$!

for _ in $(seq 1 40); do
  rostopic list 2>/dev/null | grep -q "^/lead/control$" && break
  sleep 0.25
done
sleep 1

echo "[loopback] host probe (conda $CONDA_ENV)"
# shellcheck source=/dev/null
source "$CONDA_SH"
conda activate "$CONDA_ENV" || exit 1
python "$REPO/sil/tools/local_probe.py"
status=$?

if [ "$status" -ne 0 ]; then
  echo "[loopback] FAILED; logs in $LOG_DIR"
  echo "--- bridge.log ---"; tail -15 "$LOG_DIR/bridge.log"
  echo "--- stub.log ---"; tail -15 "$LOG_DIR/stub.log"
fi
exit "$status"
