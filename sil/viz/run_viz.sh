#!/usr/bin/env bash
# Visualize a recorded SIL rosbag in RViz.
#
#   bash sil/viz/run_viz.sh <bag file or directory> [--loop] [--rate R]
#
# Starts roscore, the SIL visualization node (msgpack -> typed RViz messages)
# and RViz itself, then plays the bag back. Ctrl-C stops everything.
#
# The node republishes under /lead/viz/*: LiDAR and radar PointCloud2, the plan
# and route Paths, target/ego/control markers, the camera image, the control
# vector (rqt_plot) and a map->base_link Odometry. See sil/viz/viz_node.py.
#
# Env: ROS_SETUP (setup.bash to source), ROS_ENV (conda env, default
# ros_noetic), PYTHON (default python3 -- the Orin's `python` is Python 2),
# RVIS (1 = launch rviz), ROS_MASTER_URI, ROS_IP.
# NOTE: no `set -u`; the RoboStack activate hooks reference unset variables.
set -o pipefail

cd "$(dirname "$(realpath "${BASH_SOURCE:-$0}")")/../.." || exit 1

BAG_ARG=""
LOOP=""
RATE="1"
while [ "$#" -gt 0 ]; do
  case "$1" in
    --loop) LOOP="--loop" ;;
    --rate) RATE="$2"; shift ;;
    -h|--help) sed -n '2,12p' "$0"; exit 0 ;;
    *) BAG_ARG="$1" ;;
  esac
  shift
done

if [ -z "$BAG_ARG" ]; then
  BAG_ARG="$(ls -t outputs/local_evaluation/*/*/*.bag 2>/dev/null | head -1)"
fi
if [ -z "$BAG_ARG" ]; then
  echo "usage: bash sil/viz/run_viz.sh <bag file or directory> [--loop] [--rate R]" >&2
  exit 1
fi
if [ -d "$BAG_ARG" ]; then
  BAG="$(ls -t "$BAG_ARG"/*.bag 2>/dev/null | head -1)"
else
  BAG="$BAG_ARG"
fi
if [ ! -f "$BAG" ]; then
  echo "no .bag found at '$BAG_ARG'" >&2
  exit 1
fi

CONDA_SH="${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
# ROS environment: a RoboStack conda env on the host, or a system Noetic on the
# Orin (both are ROS1 Noetic). ROS_SETUP overrides either.
if [ -n "${ROS_SETUP:-}" ]; then
  # shellcheck source=/dev/null
  source "$ROS_SETUP" || exit 1
elif [ -f /opt/ros/noetic/setup.bash ]; then
  # shellcheck source=/dev/null
  source /opt/ros/noetic/setup.bash || exit 1
else
  # shellcheck source=/dev/null
  source "$CONDA_SH" || exit 1
  conda activate "${ROS_ENV:-ros_noetic}" || exit 1
fi
export ROS_MASTER_URI="${ROS_MASTER_URI:-http://127.0.0.1:11311}"
export ROS_IP="${ROS_IP:-127.0.0.1}"

LOG_DIR="${LOG_DIR:-/tmp/sil_viz}"
mkdir -p "$LOG_DIR"

ROSCORE_PID=""
NODE_PID=""
PLAY_PID=""
RVIZ_PID=""
cleanup() {
  for pid in "$RVIZ_PID" "$PLAY_PID" "$NODE_PID" "$ROSCORE_PID"; do
    [ -n "$pid" ] && kill -9 "$pid" 2>/dev/null
  done
  return 0
}
trap cleanup EXIT INT TERM

echo "[viz] bag: $BAG"
roscore -p 11311 >"$LOG_DIR/roscore.log" 2>&1 &
ROSCORE_PID=$!
for _ in $(seq 1 50); do
  rostopic list >/dev/null 2>&1 && break
  sleep 0.2
done

echo "[viz] starting visualization node"
"${PYTHON:-python3}" -u sil/viz/viz_node.py >"$LOG_DIR/viz.log" 2>&1 &
NODE_PID=$!

if [ "${RVIS:-1}" != "0" ] && command -v rviz >/dev/null 2>&1; then
  echo "[viz] starting rviz"
  rviz -d sil/viz/lead_sil.rviz >"$LOG_DIR/rviz.log" 2>&1 &
  RVIZ_PID=$!
else
  echo "[viz] rviz not launched (RViz=0 or 'rviz' not on PATH); logs in $LOG_DIR"
fi

echo "[viz] playing bag $LOOP at rate $RATE"
rosbag play $LOOP --rate "$RATE" "$BAG" >"$LOG_DIR/play.log" 2>&1 &
PLAY_PID=$!

wait "$PLAY_PID"
echo "[viz] bag finished; Open RViz displays under /lead/viz/*; logs in $LOG_DIR"
[ -n "$RVIZ_PID" ] && wait "$RVIZ_PID"
