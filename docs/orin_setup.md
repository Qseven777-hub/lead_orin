# Orin 侧搭建说明（SIL）

Orin 是"车上的计算单元"：接收本机转发的传感器帧，跑驾驶 agent
（`TransfuserCore` + FP16 TensorRT 引擎），把控制回给本机。接口契约见
[`sil_ros.md`](sil_ros.md)，整体说明见根目录 [`ROS软件在环.md`](../ROS%E8%BD%AF%E4%BB%B6%E5%9C%A8%E7%8E%AF.md)。

> 关键点：Orin 上的 agent 计算是 **CARLA-free** 的——已用 import 闭包验证，
> 它 **不需要 CARLA / leaderboard / scenario_runner（即不需要 `3rd_party/`）**。

## 1. Orin 需要哪些源码

### 1.1 必须

| 路径                          | 说明                                                                         |
| ----------------------------- | ---------------------------------------------------------------------------- |
| `src/lead/`                   | 整个 lead 包（Orin 节点及其依赖都在里面）                                    |
| `sil/`                        | `ros_bridge/bridge_node.py`、`orin/agent_node.py`、`tools/stub_orin_node.py` |
| `pyproject.toml` + `setup.py` | 安装依赖用                                                                   |

### 1.2 不需要

`3rd_party/`（CARLA、leaderboard、scenario_runner）、`scripts/`、`tests/`、
`outputs/`、`data/`、`notebooks/`、`lead.egg-info/`。

### 1.3 还需要的数据（不是源码）

| 文件            | 说明                                                                                                             |
| --------------- | ---------------------------------------------------------------------------------------------------------------- |
| checkpoint 目录 | 含 `config.yaml` 和 **恰好一个** `model*.pth`（`PolicyRunner` 的硬要求；跑 engine 时该权重不加载，但文件必须在） |
| `.engine`       | 你在 Orin 上量化出的 FP16 TensorRT 引擎                                                                          |

### 1.4 交付方式（推荐 git clone，保证版本一致）

Orin 专属仓库 `lead_orin` 正好就是这个子集，且同步脚本已把它的
`pyproject.toml` 改成不含 `carla` / `open3d` / `pyqt5`（这三个在 aarch64 上没有
wheel，Orin 的 agent 节点运行时也用不到）：

```bash
# Orin 上
git clone <lead_orin 仓库> lead_orin
cd lead_orin
```

若不能 clone，最小打包（从 `lead_orin` 打包，避免带上 3rd_party 与 x86-only 依赖）：

```bash
# 本机
rsync -a --exclude 3rd_party --exclude outputs --exclude data \
      --exclude tests --exclude notebooks --exclude docs \
      ./ <orin>:/home/<user>/lead_orin/
```

## 2. 环境

- Orin：Ubuntu 20.04 + JetPack。
- **ROS1 Noetic（原生 apt）**：只给 bridge 用，Python 3.8/3.9 均可。
- **Python 3.10 的 lead 环境**：`lead` 要求 ≥3.10，与 ROS 分开（传输/计算分离）。

```bash
# 2.1 ROS1 Noetic（bridge 用）
sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu focal main" \
    > /etc/apt/sources.list.d/ros1-latest.list'
curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | sudo apt-key add -
sudo apt update
sudo apt install -y ros-noetic-ros-base python3-zmq

# 2.2 lead 计算环境
conda create -n lead_sil python=3.10 -y
conda activate lead_sil
# torch 用 NVIDIA aarch64 / JetPack 版本（不要用 pip 的 cu128 轮子）
# 注意：Orin 仓库的 pyproject.toml 已由 sync 脚本去掉 carla/open3d/pyqt5
# （它们没有 aarch64 wheel，且 Orin 的 agent 节点运行时用不到）。
pip install -e .
```

> aarch64 风险：`py123d`（git 依赖）的传递依赖里 `embreex / DracoPy / manifold3d / open3d / ray` 等可能没有现成 wheel，需要逐个确认/编译。这是
> 第二轮的主要风险点。

TensorRT：`LEAD_QUANTIZED_ENGINE` 指向 `.engine`；`LD_LIBRARY_PATH` 指向
TensorRT 库（`quantized_policy._import_tensorrt` 也会尝试从环境前缀预加载）。

## 3. 网络与 ROS master

两台机器只有一个 master，**推荐 Orin 当 master**：

```bash
# Orin
export ROS_MASTER_URI=http://127.0.0.1:11311
export ROS_IP=<Orin 的网卡 IP>        # 例如 192.168.110.50，不是 127.0.0.1
roscore -p 11311
```

本机侧设 `ROS_MASTER_URI=http://<Orin-IP>:11311 ROS_IP=<本机有线 IP>`（见
`scripts/common/run_bench2drive_remote_v2.sh`）。

> `ROS_IP` 必须是对端可访问的地址；填错会导致话题能列出但收不到消息。
> 有防火墙时放行同网段（`sudo ufw allow from 192.168.110.0/24`）。

## 4. 跑 bridge

```bash
source /opt/ros/noetic/setup.bash
export ROS_MASTER_URI=http://127.0.0.1:11311 ROS_IP=<Orin IP>
python3 sil/ros_bridge/bridge_node.py
```

| 变量                   | 默认                                                                    | 作用                          |
| ---------------------- | ----------------------------------------------------------------------- | ----------------------------- |
| `SIL_BRIDGE_IN`        | `5560`                                                                  | 计算进程 → ROS 的 ZeroMQ 端口 |
| `SIL_BRIDGE_OUT`       | `5561`                                                                  | ROS → 计算进程的 ZeroMQ 端口  |
| `SIL_BRIDGE_SUBSCRIBE` | `lead/control,lead/heartbeat,lead/error,lead/session,lead/sensor_frame` | 转发给计算进程的话题          |

## 5. 先验证链路（stub，不需要 engine）

```bash
source /opt/ros/noetic/setup.bash
export ROS_MASTER_URI=http://127.0.0.1:11311 ROS_IP=<Orin IP>
python3 -u sil/tools/stub_orin_node.py
```

保持 `roscore`、`bridge`、`stub` 三个进程在跑，本机执行
`scripts/common/run_bench2drive_remote_v2.sh`（或 `sil/run_loopback.sh`），
本机日志出现 `OK: control seq=...` 即链路打通。

## 6. 跑真正的 agent 节点

```bash
conda activate lead_sil
export ROS_MASTER_URI=http://127.0.0.1:11311 ROS_IP=<Orin IP>
export LEAD_QUANTIZED_ENGINE=/path/to/model_fp16.engine
export LEAD_CONFIG="policy.target=lead.policy.transfuser.quantized_policy:QuantizedTransfuser"
python sil/orin/agent_node.py --checkpoint /path/to/checkpoint
```

节点行为（与 `sil/tools/stub_orin_node.py` 相同的话题）：

- 收到 `lead/session` → 用 `map_name / gnss_uses_transverse_mercator / global_plan_gps / lat_ref / lon_ref / camera_indices` 初始化 `TransfuserCore`
  （这些正是本机替 Orin 补齐的"世界专有"信息）；
- 收到 `lead/sensor_frame` → `core.run_step(sensors)`，把 `steer/throttle/brake`
  经 `lead/control` 回给本机；
- 周期性发 `lead/heartbeat`。

## 7. 自检清单

- [ ] 与本机有线同网段、固定 IP、能互相 ping。
- [ ] `roscore` 已启动，双方 `ROS_MASTER_URI`/`ROS_IP` 正确；`rostopic list` 能看到 `/lead/*`。
- [ ] stub 联调返回 `OK: control seq=...`。
- [ ] `conda activate lead_sil` 后 `import lead` / `import py123d` / `import numba` / `import cv2` 成功。
- [ ] `.engine` 能在 Orin 本地加载（`LEAD_QUANTIZED_ENGINE` 指向它，`import tensorrt_bindings` 成功）。
- [ ] checkpoint 目录里有 `config.yaml` 和恰好一个 `model*.pth`。

## 8. 排错

| 现象                                  | 排查                                                                     |
| ------------------------------------- | ------------------------------------------------------------------------ |
| 话题能列出但收不到                    | `ROS_IP` 填错 / 防火墙                                                   |
| `sensor_frame arrived before session` | 本机 adapter 未发 session（检查本机日志）                                |
| 控制迟迟不回                          | engine 加载失败 / `LD_LIBRARY_PATH` 未设 / aarch64 依赖缺失              |
| `Expected exactly one 'model*.pth'`   | checkpoint 目录里权重文件不是恰好一个                                    |
| `set LEAD_QUANTIZED_ENGINE` 报错      | 未设 `LEAD_QUANTIZED_ENGINE` 或 `LEAD_CONFIG` 未切到 QuantizedTransfuser |
