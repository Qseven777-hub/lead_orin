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
| `ROS软件在环.md` | 整体说明 |
| `ROS软件在环复现.md` | 0 基础复现手册（环境 / 网络 / 启动 / 排错） |
| `sil/orin/README.md` | Orin 侧细节（引擎服务、启动脚本） |

## 快速开始

一键启动 Orin 全套（`roscore + bridge + 引擎服务 + agent`）：

```bash
cd <lead_orin>
export ROS_IP=<Orin IP> ROS_MASTER_URI=http://<Orin IP>:11311
export CHECKPOINT=<checkpoint 目录>
export LEAD_QUANTIZED_ENGINE=<model_fp16.engine>
bash sil/orin/run_orin.sh
# 出现 "waiting for host: lead/session" 即就绪；终端会滚动显示每帧 control
```

主机侧、详细步骤与排错见 [`ROS软件在环复现.md`](ROS软件在环复现.md)。

## 与主仓库的关系

本仓库由主仓库 `lead_v1` 的对应提交导出。**两侧必须使用同一版本的 `lead`
与同一份 `config.yaml`**，否则 SIL 与本机评测的 parity 不成立。
