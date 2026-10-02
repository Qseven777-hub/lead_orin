#!/usr/bin/env bash
# Host-side SIL launcher: the host ROS bridge + CARLA + leaderboard with the SIL
# remote agent. Symmetric to the Orin's sil/orin/run_orin.sh.
#
# The CARLA host runs the world and the leaderboard; the remote agent forwards
# each raw sensor frame to the Orin over ROS and applies the control it returns.
# The Orin must already be up (sil/orin/run_orin.sh on the Orin; see
# docs/orin_setup.md), and both sides must use the same `lead` commit and the
# same checkpoint/config.yaml.
#
# Required:
#   ROS_MASTER_URI   the Orin's master, e.g. http://192.168.110.50:11311
#   ROS_IP           this host's LAN IP (e.g. 192.168.110.51)
#   CHECKPOINT       checkpoint dir (config.yaml + exactly one model*.pth)
# Common overrides:
#   ROS_ENV          conda env that has rospy        (default ros_noetic)
#   CONDA_ENV        conda env that has lead + CARLA (default cvci_project)
#   CONDA_SH         conda activation script         (default ~/miniconda3/...)
#   GPU              CARLA graphics adapter          (default 0)
#   LOG_DIR          bridge log directory            (default /tmp/sil_host)
#   SKIP_DONE=0      re-run routes that already scored
#   LEAD_CONFIG      evaluator overrides; videos are off by default (SIL does
#                    not produce demo/debug videos yet)
#
# Usage (no args = all 220 Bench2Drive routes):
#   bash sil/run_host.sh 0          # route index 0
#   bash sil/run_host.sh 0-4        # a small batch
#   bash sil/run_host.sh            # full run
set -o pipefail

REPO="$(cd "$(dirname "$(realpath "${BASH_SOURCE:-$0}")")/.." && pwd)"
cd "$REPO" || exit 1

ROS_ENV="${ROS_ENV:-ros_noetic}"
CONDA_ENV="${CONDA_ENV:-cvci_project}"
CONDA_SH="${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
CHECKPOINT="${CHECKPOINT:-outputs/local_training_3cams_3000/posttrain}"
ROS_MASTER_URI="${ROS_MASTER_URI:-http://127.0.0.1:11311}"
ROS_IP="${ROS_IP:-127.0.0.1}"
SIL_BRIDGE_NODE_NAME="${SIL_BRIDGE_NODE_NAME:-lead_sil_bridge_host}"
LOG_DIR="${LOG_DIR:-/tmp/sil_host}"
LEAD_AGENT_MODULE="${LEAD_AGENT_MODULE:-src/lead/evaluation/agents/remote/remote_transfuser_agent.py}"
# SIL produces no demo/debug video yet, so keep the evaluator light unless the
# caller asks otherwise (a LEAD_CONFIG already in the environment wins).
export LEAD_CONFIG="${LEAD_CONFIG:-evaluation.produce_demo_image=false evaluation.produce_demo_video=false evaluation.produce_debug_image=false evaluation.produce_debug_video=false evaluation.produce_input_image=false evaluation.produce_input_video=false}"
export ROS_MASTER_URI ROS_IP CHECKPOINT LEAD_AGENT_MODULE
mkdir -p "$LOG_DIR"

if [ ! -d "$CHECKPOINT" ]; then
  echo "[host] no such checkpoint: $CHECKPOINT" >&2
  exit 2
fi
echo "[host] master=$ROS_MASTER_URI ip=$ROS_IP checkpoint=$CHECKPOINT"
if [ "$ROS_MASTER_URI" = "http://127.0.0.1:11311" ]; then
  echo "[host] note: ROS_MASTER_URI is localhost; set it to the Orin for a cross-machine run" >&2
fi

# The bridge is the only host process that imports rospy, so it runs in the ROS
# env while the evaluator runs in the lead env.
# shellcheck source=/dev/null
source "$CONDA_SH"
conda activate "$ROS_ENV" || { echo "[host] cannot activate ROS env '$ROS_ENV'" >&2; exit 1; }
echo "[host] starting bridge ($ROS_ENV) -> $LOG_DIR/bridge.log"
SIL_BRIDGE_NODE_NAME="$SIL_BRIDGE_NODE_NAME" \
  python -u sil/ros_bridge/bridge_node.py >"$LOG_DIR/bridge.log" 2>&1 &
BRIDGE_PID=$!

cleanup() {
  [ -n "$BRIDGE_PID" ] && kill -9 "$BRIDGE_PID" 2>/dev/null
  return 0
}
trap cleanup EXIT INT TERM

# Give the ROS graph a moment to connect the bridge before the evaluator starts.
for _ in $(seq 1 30); do
  kill -0 "$BRIDGE_PID" 2>/dev/null || break
  rostopic list >/dev/null 2>&1 && break
  sleep 0.3
done
if ! kill -0 "$BRIDGE_PID" 2>/dev/null; then
  echo "[host] bridge died; see $LOG_DIR/bridge.log" >&2
  tail -20 "$LOG_DIR/bridge.log"
  exit 1
fi

echo "[host] running evaluator with the SIL remote agent"
CONDA_ENV="$CONDA_ENV" bash scripts/common/run_bench2drive_per_route_v2.sh "$@"
status=$?
echo "[host] evaluator exited with status $status"
exit "$status"
