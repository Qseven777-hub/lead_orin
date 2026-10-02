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

ROSCORE_PID=""; BRIDGE_PID=""; ENGINE_PID=""; AGENT_PID=""; WATCH_PID=""
cleanup() {
  for pid in "$WATCH_PID" "$AGENT_PID" "$ENGINE_PID" "$BRIDGE_PID" "$ROSCORE_PID"; do
    if [ -n "$pid" ]; then
      pkill -9 -P "$pid" 2>/dev/null   # e.g. roscore's rosmaster/rosout children
      kill -9 "$pid" 2>/dev/null
    fi
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

# Wait until the policy is built and the node is listening, then print the
# one-glance self-check the host operator needs.
for _ in $(seq 1 300); do
  grep -q "Orin SIL agent node listening" "$LOG_DIR/agent.log" 2>/dev/null && break
  kill -0 "$AGENT_PID" 2>/dev/null || { echo "[orin] agent died; see $LOG_DIR/agent.log"; tail -20 "$LOG_DIR/agent.log"; exit 1; }
  sleep 0.5
done
if ! grep -q "Orin SIL agent node listening" "$LOG_DIR/agent.log" 2>/dev/null; then
  echo "[orin] agent not listening yet; see $LOG_DIR/agent.log"; exit 1
fi
POLICY_TARGET="$(grep -a "policy target:" "$LOG_DIR/agent.log" | tail -1 | sed 's/.*policy target: //')"

echo "[orin] ================== self-check =================="
echo "[orin] roscore:        up   ($ROS_MASTER_URI)"
if [ "$SKIP_ROS" != "1" ]; then echo "[orin] bridge:         up"; else echo "[orin] bridge:         skipped (SKIP_ROS=1)"; fi
echo "[orin] engine service: READY ($ENGINE_ENDPOINT)"
echo "[orin] agent:          listening (policy target: $POLICY_TARGET)"
echo "[orin] waiting for host: lead/session (ROS_IP=$ROS_IP)"
echo "[orin] ================================================"

# Report the two host-driven milestones as they happen.
(
  seen_core=0; seen_ctrl=0
  while kill -0 "$AGENT_PID" 2>/dev/null; do
    if [ "$seen_core" = 0 ] && grep -q "core ready" "$LOG_DIR/agent.log" 2>/dev/null; then
      echo "[orin] + session received: $(grep -a 'core ready' "$LOG_DIR/agent.log" | tail -1 | sed 's/.*\[INFO\] //')"
      seen_core=1
    fi
    if [ "$seen_ctrl" = 0 ] && grep -q "control seq=" "$LOG_DIR/agent.log" 2>/dev/null; then
      echo "[orin] + first control:   $(grep -a 'control seq=' "$LOG_DIR/agent.log" | tail -1 | sed 's/.*\[INFO\] //')"
      seen_ctrl=1
    fi
    sleep 1
  done
) &
WATCH_PID=$!

echo "[orin] running. logs: $LOG_DIR (agent.log, engine.log, bridge.log, roscore.log)"
wait "$AGENT_PID"
