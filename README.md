# lead_orin — Orin 侧（软件在环 SIL）

本仓库是主仓库 `lead_v1` 的 **Orin 侧子集**：只包含在本机 CARLA 软件在环（SIL）
中、**Orin 上运行驾驶 agent 所需**的内容。

## 用途

Orin 是"车上的计算单元"：接收本机转发的传感器帧，跑 `TransfuserCore` +
FP16 TensorRT 引擎，把 `steer/throttle/brake` 回给本机。

- 本机（CARLA 主机）跑 CARLA 世界、leaderboard、以及转发/记录逻辑。
- 两者通过 ROS1 通信。

## 为什么可以不要 CARLA

Orin 上的 agent 计算是 **CARLA-free** 的：已用 import 闭包验证，Orin 节点不
import `carla` / `srunner` / `leaderboard`，因此本仓库**不含** `3rd_party/`。

## 内容

| 路径 | 说明 |
|---|---|
| `src/lead/` | lead 包（Orin 节点及其依赖） |
| `sil/` | `ros_bridge/bridge_node.py`、`orin/agent_node.py`、`tools/stub_orin_node.py`、自检脚本 |
| `pyproject.toml` / `setup.py` | 安装依赖 |
| `docs/orin_setup.md` | **Orin 侧搭建步骤（先读这个）** |
| `docs/sil_ros.md` | 接口契约与设计 |
| `ROS软件在环.md` | 整体说明 |

## 快速开始

```bash
# 1. ROS1 Noetic（bridge 用）
sudo apt install -y ros-noetic-ros-base python3-zmq

# 2. Python 3.10 lead 环境
conda create -n lead_sil python=3.10 -y
conda activate lead_sil
pip install -e .          # torch 用 NVIDIA aarch64/JetPack 版本

# 3. bridge + agent 节点
source /opt/ros/noetic/setup.bash
export ROS_MASTER_URI=http://127.0.0.1:11311 ROS_IP=<Orin IP>
python3 sil/ros_bridge/bridge_node.py &

export LEAD_QUANTIZED_ENGINE=/path/to/model_fp16.engine
export LEAD_CONFIG="policy.target=lead.policy.transfuser.quantized_policy:QuantizedTransfuser"
python sil/orin/agent_node.py --checkpoint /path/to/checkpoint
```

详细步骤、自检清单与排错见 [`docs/orin_setup.md`](docs/orin_setup.md)。

## 与主仓库的关系

本仓库由主仓库 `lead_v1` 的对应提交导出。**两侧必须使用同一版本的 `lead`
与同一份 `config.yaml`**，否则 SIL 与本机评测的 parity 不成立。
