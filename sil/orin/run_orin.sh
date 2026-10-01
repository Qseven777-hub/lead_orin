#!/usr/bin/env bash
# Orin-side SIL launcher.
#
# Starts everything the Orin needs for a host<->Orin SIL run:
#   roscore (native Noetic, py3.8)          -- the ROS master
#   bridge  (native Noetic, py3.8)          -- ROS <-> ZeroMQ
#   engine  (system py3.8)                  -- TensorRT 8.4 engine service
#   agent   (miniforge gqzl-py310)          -- TransfuserCore + engine client
#
# All Orin-specific pieces live in sil/orin/**; the shared lead/bridge code is
# used as-is.  Override anything through the environment.
#
# Required:
#   CHECKPOINT                  checkpoint dir (config.yaml + one model*.pth)
#   LEAD_QUANTIZED_ENGINE       path to the FP16 .engine
# Common overrides:
#   ROS_IP                      Orin LAN IP for cross-machine runs (default 127.0.0.1)
#   ROS_MASTER_URI              default http://127.0.0.1:11311
#   SIL_ENGINE_ENDPOINT         default tcp://127.0.0.1:5562
#   CONDA_SH / CONDA_ENV        default miniforge3 / gqzl-py310
#   LEAD_DEVICE                 default cpu (this JetPack has no CUDA torch)
#   SKIP_ROS=1                  do not start roscore/bridge (managed elsewhere)
set -o pipefail

ORIN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$ORIN_DIR/../.." && pwd)"

ROS_IP="${ROS_IP:-127.0.0.1}"
ROS_MASTER_URI="${ROS_MASTER_URI:-http://127.0.0.1:11311}"
ENGINE_ENDPOINT="${SIL_ENGINE_ENDPOINT:-tcp://127.0.0.1:5562}"
CONDA_SH="${CONDA_SH:-/home/tjuae/miniforge3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-gqzl-py310}"
LEAD_DEVICE="${LEAD_DEVICE:-cpu}"
LEAD_CONFIG="${LEAD_CONFIG:-policy.target=orin_engine_policy:OrinEngineTransfuser}"
SKIP_ROS="${SKIP_ROS:-0}"
LOG_DIR="${LOG_DIR:-/tmp/sil_orin}"
mkdir -p "$LOG_DIR"

if [ -z "${CHECKPOINT:-}" ]; then echo "set CHECKPOINT" >&2; exit 2; fi
if [ -z "${LEAD_QUANTIZED_ENGINE:-}" ]; then echo "set LEAD_QUANTIZED_ENGINE" >&2; exit 2; fi
export CHECKPOINT LEAD_QUANTIZED_ENGINE LEAD_CONFIG LEAD_DEVICE SIL_ENGINE_ENDPOINT="$ENGINE_ENDPOINT"

ROSCORE_PID=""; BRIDGE_PID=""; ENGINE_PID=""; AGENT_PID=""
cleanup() {
  for pid in "$AGENT_PID" "$ENGINE_PID" "$BRIDGE_PID" "$ROSCORE_PID"; do
    [ -n "$pid" ] && kill -9 "$pid" 2>/dev/null
  done
  return 0
}
trap cleanup EXIT INT TERM

export ROS_MASTER_URI ROS_IP
source /opt/ros/noetic/setup.bash

if [ "$SKIP_ROS" != "1" ]; then
  echo "[orin] roscore (master) on $ROS_MASTER_URI"
  roscore -p 11311 >"$LOG_DIR/roscore.log" 2>&1 &
  ROSCORE_PID=$!
  for _ in $(seq 1 50); do
    /usr/bin/python3 -c "import rosgraph; rosgraph.Master('/probe').getPid()" >/dev/null 2>&1 && break
    sleep 0.2
  done
  kill -0 "$ROSCORE_PID" 2>/dev/null || { echo "[orin] roscore died; $LOG_DIR/roscore.log"; exit 1; }

  echo "[orin] bridge (system py3.8)"
  /usr/bin/python3 -u "$REPO/sil/ros_bridge/bridge_node.py" >"$LOG_DIR/bridge.log" 2>&1 &
  BRIDGE_PID=$!
fi

echo "[orin] engine service (system py3.8) on $ENGINE_ENDPOINT"
/usr/bin/python3 -u "$ORIN_DIR/engine_service.py" >"$LOG_DIR/engine.log" 2>&1 &
ENGINE_PID=$!
for _ in $(seq 1 120); do
  grep -q ENGINE_SERVICE_READY "$LOG_DIR/engine.log" 2>/dev/null && break
  kill -0 "$ENGINE_PID" 2>/dev/null || { echo "[orin] engine service died; $LOG_DIR/engine.log"; exit 1; }
  sleep 0.5
done
grep -q ENGINE_SERVICE_READY "$LOG_DIR/engine.log" 2>/dev/null || { echo "[orin] engine not ready; $LOG_DIR/engine.log"; exit 1; }

echo "[orin] agent node (conda $CONDA_ENV)"
# shellcheck source=/dev/null
source "$CONDA_SH"
conda activate "$CONDA_ENV" || exit 1
python -u "$ORIN_DIR/agent_node.py" \
  --checkpoint "$CHECKPOINT" \
  --device "$LEAD_DEVICE" \
  --in-endpoint "tcp://127.0.0.1:5561" \
  --out-endpoint "tcp://127.0.0.1:5560" \
  >"$LOG_DIR/agent.log" 2>&1 &
AGENT_PID=$!

echo "[orin] running. logs: $LOG_DIR (agent.log, engine.log, bridge.log, roscore.log)"
wait "$AGENT_PID"
