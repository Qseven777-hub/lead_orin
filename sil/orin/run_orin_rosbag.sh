#!/usr/bin/env bash
# Orin-side SIL launcher *with rosbag recording*.
#
# Same as run_orin.sh (roscore + bridge + engine service + agent), plus a
# rosbag record of the SIL topics so a run can be replayed/visualized offline.
#
# The recording is a passive tap: the control path (lead/control -> host) is
# unchanged.  The model's plan is published on the extra ``lead/plan`` topic
# (visualization only) via SIL_SEND_PLAN=1.
#
# Required:
#   CHECKPOINT                  checkpoint dir (config.yaml + one model*.pth)
#   LEAD_QUANTIZED_ENGINE       path to the FP16 .engine
# Common overrides:
#   BAG_DIR                     where to write the bag (default ~/sil_bags).
#                               Point it at the USB drive to avoid using the
#                               Orin's internal storage, e.g.
#                               BAG_DIR="/media/tjuae/<usb>/sil_bags"
#   BAG_SCENARIO                subdirectory per scenario (default "unknown"),
#                               so bags land at BAG_DIR/<scenario>/*.bag
#   RECORD_TOPICS               space-separated topics to record
#   BAG_SPLIT_MB                split the bag every N MiB (0 = no split)
#   ROS_IP / ROS_MASTER_URI / SIL_ENGINE_ENDPOINT / CONDA_SH / CONDA_ENV /
#   LEAD_DEVICE                 same as run_orin.sh
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
SIL_SEND_PLAN="${SIL_SEND_PLAN:-1}"   # publish lead/plan for the recording
LOG_DIR="${LOG_DIR:-/tmp/sil_orin}"
BAG_DIR="${BAG_DIR:-$HOME/sil_bags}"
BAG_SCENARIO="${BAG_SCENARIO:-unknown}"   # subdir per scenario, e.g. 1773
RECORD_TOPICS="${RECORD_TOPICS:-/lead/session /lead/sensor_frame /lead/control /lead/plan /lead/heartbeat /lead/error}"
BAG_SPLIT_MB="${BAG_SPLIT_MB:-0}"
BAG_RUN_DIR="$BAG_DIR/$BAG_SCENARIO"
mkdir -p "$LOG_DIR" "$BAG_RUN_DIR"

if [ -z "${CHECKPOINT:-}" ]; then echo "set CHECKPOINT" >&2; exit 2; fi
if [ -z "${LEAD_QUANTIZED_ENGINE:-}" ]; then echo "set LEAD_QUANTIZED_ENGINE" >&2; exit 2; fi
export CHECKPOINT LEAD_QUANTIZED_ENGINE LEAD_CONFIG LEAD_DEVICE SIL_SEND_PLAN
export SIL_ENGINE_ENDPOINT="$ENGINE_ENDPOINT"

# Rebuilding the policy loads the pretrained timm backbone; keep HuggingFace
# offline so it uses the local cache instead of the (often blocked) network.
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

ROSCORE_PID=""; BRIDGE_PID=""; ENGINE_PID=""; AGENT_PID=""; BAG_PID=""
cleanup() {
  if [ -n "$BAG_PID" ]; then
    # rosbag needs SIGINT (not SIGKILL) to flush and write the bag index.
    kill -INT "$BAG_PID" 2>/dev/null
    wait "$BAG_PID" 2>/dev/null
  fi
  for pid in "$AGENT_PID" "$ENGINE_PID" "$BRIDGE_PID" "$ROSCORE_PID"; do
    if [ -n "$pid" ]; then
      pkill -9 -P "$pid" 2>/dev/null
      kill -9 "$pid" 2>/dev/null
    fi
  done
  return 0
}
trap cleanup EXIT INT TERM

export ROS_MASTER_URI ROS_IP
source /opt/ros/noetic/setup.bash

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
for _ in $(seq 1 300); do
  grep -q "Orin SIL agent node listening" "$LOG_DIR/agent.log" 2>/dev/null && break
  kill -0 "$AGENT_PID" 2>/dev/null || { echo "[orin] agent died; see $LOG_DIR/agent.log"; tail -20 "$LOG_DIR/agent.log"; exit 1; }
  sleep 0.5
done
grep -q "Orin SIL agent node listening" "$LOG_DIR/agent.log" 2>/dev/null || { echo "[orin] agent not listening; see $LOG_DIR/agent.log"; exit 1; }

# --- rosbag recording (passive tap; does not affect the control path) --------
BAG_BASE="$BAG_RUN_DIR/sil_$(date +%Y%m%d_%H%M%S)"
BAG_ARGS=(record -O "$BAG_BASE.bag")
if [ "$BAG_SPLIT_MB" -gt 0 ] 2>/dev/null; then
  BAG_ARGS+=(--split "--size=$BAG_SPLIT_MB")
fi
FREE_MB="$(df -Pm "$BAG_RUN_DIR" 2>/dev/null | awk 'NR==2{print $4}')"
echo "[orin] recording ${RECORD_TOPICS}"
echo "[orin] bag dir: $BAG_RUN_DIR (free ${FREE_MB:-?} MiB)"
rosbag "${BAG_ARGS[@]}" $RECORD_TOPICS >"$LOG_DIR/rosbag.log" 2>&1 &
BAG_PID=$!
sleep 1
kill -0 "$BAG_PID" 2>/dev/null || { echo "[orin] rosbag failed; see $LOG_DIR/rosbag.log"; tail -10 "$LOG_DIR/rosbag.log"; exit 1; }

POLICY_TARGET="$(grep -a "policy target:" "$LOG_DIR/agent.log" | tail -1 | sed 's/.*policy target: //')"
echo "[orin] ================== self-check =================="
echo "[orin] roscore:        up   ($ROS_MASTER_URI)"
echo "[orin] bridge:         up"
echo "[orin] engine service: READY ($ENGINE_ENDPOINT)"
echo "[orin] agent:          listening (policy target: $POLICY_TARGET)"
echo "[orin] plan topic:     $([ "$SIL_SEND_PLAN" = "1" ] && echo "on (lead/plan)" || echo "off")"
echo "[orin] recording ->    $BAG_BASE.bag"
echo "[orin] waiting for host: lead/session (ROS_IP=$ROS_IP)"
echo "[orin] ================================================"
echo "[orin] streaming controls below (one block per route; Ctrl+C stops and finalizes the bag)"

while kill -0 "$AGENT_PID" 2>/dev/null; do
  if ! IFS= read -r -t 1 line; then
    continue
  fi
  case "$line" in
    *"core ready"*)
      printf '\n──────── session %s ────────\n' "${line##*] }"
      ;;
    *"control seq="*)
      printf '  %s\n' "${line##*] }"
      ;;
  esac
done < <(tail -n 0 -F "$LOG_DIR/agent.log" 2>/dev/null)