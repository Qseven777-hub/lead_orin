# ROS 软件在环（SIL）复现手册（0 基础版）

这份文档假设你**完全没接触过 ROS、也没跑过这套代码**。跟着从头做一遍，就能在两台机器上
把「CARLA 仿真 + 模型在 Orin 上推理」的闭环跑通。

> 想先看设计原理，见 [`ROS软件在环.md`](ROS%E8%BD%AF%E4%BB%B6%E5%9C%A8%E7%8E%AF.md)。
> 本文只讲**怎么从 0 配到跑通**，并解释我们加的东西。

______________________________________________________________________

## 0. 三分钟速览

- 我们要做的是**软件在环（SIL, Software-In-the-Loop）**：让驾驶模型**不在仿真机本地跑**，
  而是跑在另一台算力机上（本项目是 NVIDIA **Orin**），两台机通过 **ROS** 交换数据，
  模拟「实车部署」。
- **两台机**：
  - **主机（跑来 CARLA，Ubuntu 22.04 笔记本/台式机）**：负责世界、评测、把传感器发出去、
    把收到的控制施加到车上。
  - **Orin（Ubuntu 20.04 / JetPack）**：负责驾驶模型的计算，把控制算好回传。
- **两条命令跑通**（配好之后）：
  - Orin：`bash sil/orin/run_orin.sh`
  - 主机：`bash scripts/common/run_bench2drive_orin_v2.sh 0-4`
- 本文后面每一节，都对应你实际要敲的命令。

______________________________________________________________________

## 1. 这套东西到底在干嘛（用大白话）

想象两台电脑：

- **主机**是「游戏机」：它运行 CARLA 这个驾驶仿真游戏，里面有一个虚拟城市、天气、车流，
  还能记录你有没有闯红灯、超速（这就是**评测**）。
- **Orin** 是「车上的大脑」：它拿不到游戏机里的世界，只能看到**一帧帧传感器画面**
  （摄像头、激光雷达、雷达、GPS、IMU、速度），据此算出「方向盘打多少、油门/刹车踩多少」。

两者怎么连？**主机把传感器帧发过去，Orin 把控制发回来**，不断循环，直到这条路线跑完。
用 ROS（机器人领域最常用的通信中间件）来传，是为了贴近真实车上的部署方式。

**为什么要费这个劲？** 因为最终模型要跑在车规算力（Orin）上，而仿真在开发机（主机）上。
SIL 就是「用仿真当世界、用真算力当大脑」的实车预演。

______________________________________________________________________

## 2. 名词小词典（看一眼就行）

| 名词                          | 一句话解释                                                                            |
| ----------------------------- | ------------------------------------------------------------------------------------- |
| **CARLA**                     | 开源的自动驾驶仿真器（那台「游戏机」）                                                |
| **leaderboard / Bench2Drive** | CARLA 官方的评测框架与数据集；Bench2Drive 有 220 条路线                               |
| **agent（智能体）**           | 驾驶模型 + 把模型输出变成控制的逻辑；本项目的 agent 是 TransFuser                     |
| **ROS**                       | 机器人通信中间件。`roscore` 是「电话总机」，各个程序通过\*\*话题（topic）\*\*收发消息 |
| **topic（话题）**             | 像微信群：谁发谁收按名字订阅。本项目话题都叫 `lead/xxx`                               |
| **ZMQ / ZeroMQ**              | 一个轻量通信库（本机进程间用的「管道」），不是 ROS                                    |
| **Orin**                      | NVIDIA Jetson AGX Orin，车规级算力板                                                  |
| **TensorRT**                  | NVIDIA 的推理加速引擎，把模型编译成 `.engine` 文件，跑得飞快                          |
| **checkpoint**                | 训练好的模型权重目录（含 `config.yaml` 和 `model*.pth`）                              |

______________________________________________________________________

## 3. 两台机器的分工与数据流

```
   ┌────────────── 主机 (CARLA) ──────────────┐         ┌──────────── Orin ────────────┐
   │ CARLA 仿真世界                            │         │ roscore（ROS 总机）           │
   │ leaderboard 评测                          │  ROS1   │ ros_bridge（ROS↔ZeroMQ）      │
   │ remote agent：发传感器 / 施加控制          │◄──────►│ agent_node（跑模型）           │
   │ ros_bridge（ROS↔ZeroMQ）                  │ 局域网  │ engine_service（TensorRT）     │
   └───────────────────────────────────────────┘         └───────────────────────────────┘
```

一次循环（一个 tick，20Hz）的流程：

1. 主机 CARLA 采到一帧传感器 → `remote agent` 打包 → 经 ROS 发到 Orin；
1. Orin 的 `agent_node` 收到，跑模型（TensorRT 引擎）得到**规划**，再由跟踪器转成**控制**；
1. Orin 把控制经 ROS 发回主机；
1. 主机把控制施加到车上，推进仿真，进入下一帧。

> 关键点：**过网的是控制**（方向盘/油门/刹车三个数），主机再包成 CARLA 的控制。
> 所以 Orin 上**不需要 CARLA**。

### 3.1 模型输出的是「规划」，为什么回传的是「控制」？

TransFuser 网络真正的输出是**规划**，不是控制：

| 网络输出                    | 含义                           |
| --------------------------- | ------------------------------ |
| `future_waypoints`          | 未来一段轨迹点（车打算怎么走） |
| `target_speed`（标量/分布） | 目标速度                       |
| `route`                     | 预测的路线 / 目标点            |

**规划 ≠ 控制**。把「车打算怎么走 + 想跑多快」变成「方向盘打多少、油门/刹车踩多少」，
是一层**控制跟踪器**（PID 类）做的事。这一层就在 Orin 上跑，而且**和本机跑同一份共享代码**：

```
sensor → tick → build_scene_data → 特征化 → 网络前向(TensorRT 引擎)
                                                   │
                     输出「规划」Prediction：       ▼
                     future_waypoints / target_speed / route
                                                   │
                     TransfuserControlMixin（共享的「控制跟踪」层）
                     ├─ WaypointTracker ：未来轨迹点 → steer（方向盘）
                     └─ PathSpeedTracker：目标速度   → throttle / brake
                                                   │
                              ControlCommand(steer, throttle, brake)
                                                   │
                        经 ROS 的 lead/control 传回主机
```

对应代码：

- `src/lead/evaluation/agents/transfuser/transfuser_control.py`：`compute_control` 把 `Prediction` 变成 `ControlCommand`；
- `sil/orin/agent_node.py` 调 `core.run_step(...)`，而 `TransfuserCore = TransfuserControlMixin + PolicyAgentCore`。

**为什么只回传控制、不回传规划？**

1. 主机开车只需要 `steer/throttle/brake`；规划是中间量，主机用不上。
1. 评测指标（压实线、超速等）看的是**车实际怎么开**，不需要规划。
1. 本机跑时用的也是这同一套跟踪器，所以 SIL 只是把「网络前向 + 跟踪」整段搬到 Orin，保证两端一致（parity）。
1. 若以后要在主机上看可视化，契约里预留了可选字段 `control.aux`，可把规划张量一起回传——目前未启用。

______________________________________________________________________

## 4. 一次性环境准备

> 只有第一次需要做；以后直接跳到「6. 启动」。

### 4.1 主机（Ubuntu 22.04）

1. 有 NVIDIA 显卡（跑 CARLA），装好驱动。
1. 准备 conda 环境：
   - **`cvci_project`**：Python 3.10，装本项目（含 CARLA 的 Python 客户端、模型依赖）。
   - **`ros_noetic`**：RoboStack 的 ROS1 Noetic（Python 3.9），只给 bridge 用。
1. 装项目与 ROS 环境：

```bash
# 项目环境（名字按你们的来；本项目用 cvci_project）
conda create -n cvci_project python=3.10 -y
conda activate cvci_project
cd <lead_v1 仓库目录>
pip install -e .

# ROS1 Noetic（RoboStack，无需 root）
conda create -n ros_noetic -c robostack-staging -c conda-forge \
    ros-noetic-ros-base python=3.9 msgpack-python pyzmq -y
```

4. CARLA 引擎放在 `3rd_party/CARLA/standard_0916/`（仓库里已有），`.env` 里 `CARLA_ROOT` 指向它。

### 4.2 Orin（Ubuntu 20.04 / JetPack）

> **从白板机完整配置 Orin 环境（含版本锁定清单）见 [`针对orin的环境配置.md`](%E9%92%88%E5%AF%B9orin%E7%9A%84%E7%8E%AF%E5%A2%83%E9%85%8D%E7%BD%AE.md)。** 下面是速览。

Orin 上要**两套 Python**，这是本项目一个关键设计：

| 用途                                 | Python                                    | 说明                                            |
| ------------------------------------ | ----------------------------------------- | ----------------------------------------------- |
| ROS + bridge + **TensorRT 引擎服务** | **系统 Python 3.8**                       | JetPack 的 TensorRT 8.4 只有 3.8 的 Python 绑定 |
| 驾驶模型的特征/控制逻辑（`lead`）    | **miniforge `gqzl-py310`（Python 3.10）** | `lead` 要求 ≥3.10                               |

准备（一次性）：

```bash
# 1) ROS1 Noetic（原生 apt）——给 bridge 用
sudo apt install -y ros-noetic-ros-base python3-zmq python3-msgpack

# 2) 系统 Python 3.8 的引擎服务依赖
/usr/bin/python3 -m pip install --user "pyzmq==26.4.0" "msgpack==1.1.1"

# 3) Python 3.10 计算环境（复用 miniforge 的 gqzl-py310）
source ~/miniforge3/etc/profile.d/conda.sh && conda activate gqzl-py310
python -m pip install filterpy==1.4.5 pyzmq==27.2
python -m pip install "lightning==2.6.1" --no-deps
cd <lead_orin 仓库目录> && python -m pip install -e . --no-deps
```

> 为什么 Orin 上的 `lead_orin` 的 `pyproject.toml` 看起来少了依赖？因为它被同步脚本改过：
> **去掉了 `carla/open3d/pyqt5`**（这三个没有 aarch64 轮子，Orin 也用不到）。

______________________________________________________________________

## 5. 网络配置（网线直连，静态 IP）

两台机用**一根网线直连**（或同一交换机），配成同一网段。示例：

| 机器 | 有线网口                            | IP                  |
| ---- | ----------------------------------- | ------------------- |
| 主机 | `enp129s0`（换成你的口）            | `192.168.110.51/24` |
| Orin | `eth0`/`enP8p1s0`（换成 Orin 的口） | `192.168.110.50/24` |

**主机**（用 NetworkManager）：

```bash
sudo nmcli con add type ethernet ifname enp129s0 con-name sil-wired \
     ipv4.method manual ipv4.addresses 192.168.110.51/24
sudo nmcli con up sil-wired
ip -4 addr show enp129s0        # 应看到 192.168.110.51/24
```

**Orin**：

```bash
sudo ip addr add 192.168.110.50/24 dev <orin网口>
sudo ip link set <orin网口> up
```

**互通检查**（两台都敲，能 ping 通即可）：

```bash
ping -c3 192.168.110.50     # 在主机敲
ping -c3 192.168.110.51     # 在 Orin 敲
```

> **`ROS_IP` 是什么？** ROS 用它告诉别人「我的地址」。跨机时必须填**本机的有线 IP**
> （主机 `192.168.110.51`，Orin `192.168.110.50`）。**不能填 `127.0.0.1`**，
> 否则对方收不到你的消息（能列出话题但收不到数据，是经典坑）。

______________________________________________________________________

## 6. 代码与数据准备

### 6.1 两个仓库

| 仓库                     | 内容                                                                      | 在哪改     |
| ------------------------ | ------------------------------------------------------------------------- | ---------- |
| `lead_v1`（主仓库）      | 共享代码：`src/lead/**`、`sil/ros_bridge`、`sil/tools`、`docs/`、构建文件 | 在主机改   |
| `lead_orin`（Orin 仓库） | Orin 专属：`sil/orin/**`（agent 节点、引擎服务、启动脚本）                | 在 Orin 改 |

**纪律**：Orin 上**只有 `sil/orin/**` 和它的 `README.md` 是你的**；其它（`src/lead`、bridge、契约、`pyproject.toml`）都由主仓库同步覆盖，别在 Orin 上改。

- **主机**：`cd <lead_v1> && git pull --ff-only`
- **Orin**：`cd <lead_orin> && git pull --ff-only`

> 两端必须用**同一个 commit**，否则消息/算法不一致。

### 6.2 数据（不是源码）

| 东西                | 要求                                                               |
| ------------------- | ------------------------------------------------------------------ |
| **checkpoint 目录** | 含 `config.yaml` 和**恰好一个** `model*.pth`                       |
| **`.engine`**       | 你在 Orin 上用 TensorRT 编译出的 FP16 引擎，与上面 checkpoint 配对 |

> **parity（一致性）**：两端用**同一份 `config.yaml`**（同一 checkpoint 目录）和**同一个 `.engine`**，
> 结果才可比。可用 `md5sum config.yaml` 在两端核对是否一致。

______________________________________________________________________

## 7. 启动与验证（照抄即可）

> **顺序很重要：先起 Orin，再起主机。** 否则主机先发数据没人接。

### 7.1 启动 Orin（一条命令起全套）

```bash
cd <lead_orin 仓库目录>
export ROS_IP=192.168.110.50
export ROS_MASTER_URI=http://192.168.110.50:11311
export CHECKPOINT=<checkpoint 目录>
export LEAD_QUANTIZED_ENGINE=<model_fp16.engine 路径>
bash sil/orin/run_orin.sh
```

它会依次启动 `roscore`（ROS 总机）→ `bridge`（ROS↔ZeroMQ）→ `engine_service`（TensorRT）→
`agent_node`（模型节点），并打印自检：

```
[orin] roscore:        up   (http://192.168.110.50:11311)
[orin] engine service: READY (tcp://127.0.0.1:5562)
[orin] agent:          listening (policy target: orin_engine_policy:OrinEngineTransfuser)
[orin] waiting for host: lead/session (ROS_IP=192.168.110.50)
```

看到 **`waiting for host: lead/session`** 才算就绪。日志在 `/tmp/sil_orin/`。

### 7.2 启动主机（一条命令起全套）

**另开一个终端**（或另一台机）：

```bash
cd <lead_v1 仓库目录>
export ROS_IP=192.168.110.51
export ROS_MASTER_URI=http://192.168.110.50:11311
export CHECKPOINT=<与 Orin 同一份 checkpoint 目录>

# 选一种：
bash scripts/common/run_bench2drive_orin_v2.sh 0        # 只跑 1 条路线（最快验证）
bash scripts/common/run_bench2drive_orin_v2.sh 0-4      # 跑 5 条
bash scripts/common/run_bench2drive_orin_v2.sh          # 全量 220 条
```

它会启动主机 `bridge`、CARLA、leaderboard 评测，并用 `remote agent` 转发数据。
默认已算过的路线会跳过；要强制重算加 `SKIP_DONE=0`。

### 7.3 验证跑通

- **Orin** 终端会打印（或看 `/tmp/sil_orin/agent.log`）：
  ```
  [orin] + session received: core ready: route=1833 map=Carla/Maps/Town12/Town12
  [orin] + first control:   control seq=2 step=2 steer=... throttle=... brake=... infer_ms=97.1
  ```
- **主机**日志里看到控制回流，最后打印每条路线的得分：
  ```
  [summary] index 0   id 1833   score 100.0   status: Completed
  [summary] route outcomes: 5 Completed, 0 not Completed
  ```

起步会有 **1–2 帧 `no control`**，那是 Orin 第一次建模型/引擎的预热，正常。

______________________________________________________________________

## 8. 我们加的这些功能是什么（逐项解释）

> 这一节回答「代码里多出来的这些东西各是干嘛的」。按数据从主机到 Orin 再回来的顺序讲。

### 8.1 消息契约：`src/lead/evaluation/sil/contract.py`

定义了两端交换的**消息格式**（就像约定信封里写什么）。核心是几个话题：

| 话题                | 方向      | 频率         | 内容                                                                     |
| ------------------- | --------- | ------------ | ------------------------------------------------------------------------ |
| `lead/session`      | 主机→Orin | 每条路线一次 | 路线元信息（地图名、GPS 计划、相机编号…）——Orin 没有世界，推不出来的东西 |
| `lead/sensor_frame` | 主机→Orin | 每帧         | 原始传感器（图像、激光、雷达、GPS、IMU、速度）                           |
| `lead/control`      | Orin→主机 | 每帧         | `steer/throttle/brake`（纯数值）                                         |
| `lead/heartbeat`    | Orin→主机 | 2Hz          | 心跳（证明 Orin 活着）                                                   |
| `lead/error`        | 双向      | 事件         | 出错信息                                                                 |

### 8.2 编解码：`src/lead/evaluation/sil/codec.py`

把上面的消息变成**字节**（用 msgpack，紧凑、能装下几 MB 的图像数组）。
我们做的关键修复：**让 tuple 在编解码后仍然是 tuple**。
（msgpack 没有 tuple 类型，会变成 list，而模型代码按 tuple 做类型检查，会直接报错——所以加了
`__tuple__` 标记。）

### 8.3 传输：`src/lead/evaluation/sil/transport.py`（ZeroMQ）

计算进程和本机 bridge 之间用一个\*\*轻量管道（ZeroMQ）\*\*通信。
我们做的关键修复：**给收发加锁**。因为 Orin 节点有两条线程（主循环发控制、心跳线程发心跳）
共用同一个 socket，而 ZeroMQ socket 不是线程安全的，并发发送会把数据帧交错，导致 bridge 崩塌、
整段超时。

### 8.4 ROS↔ZeroMQ 桥：`sil/ros_bridge/bridge_node.py`

唯一 `import rospy` 的代码。它一边说 ROS、一边说 ZeroMQ，做「翻译」。
**为什么需要它？** 因为 `lead` 要 Python 3.10，而 ROS Noetic 是 3.8/3.9，同一个进程装不下
两者；于是把 ROS 单独拆成一个进程，计算进程只跟它用本地 ZeroMQ 说话。

### 8.5 主机适配器：`src/lead/evaluation/agents/remote/remote_transfuser_agent.py`

替换掉「本地跑模型」的 agent：它保留 CARLA 侧的一切（传感器、评测、录像），
但**把每帧传感器发出去、把回来的控制施加到车上**。

- 若某帧超时没等到控制，先回退安全控制（`brake=1`）；
- **会重发 `session`**：ROS 话题不「latch」，第一次发的 session 若在链路接通前发出会丢，
  所以收到首个控制之前，超时就重发；
- 每次路线运行生成一个唯一的 **`session_id`**，让 Orin 能区分「同一次运行的重发」和「新的一次运行」。

### 8.6 Orin 节点：`sil/orin/agent_node.py`

接收 `session` 建模型、接收 `sensor_frame` 跑模型、回 `control`、发心跳。

- 只有 **`session_id` 变了才重建 core**（避免同一条路线复跑时用上次的旧状态）；
- **每帧打印** `control seq=... infer_ms=...`（可用 `SIL_LOG_EVERY_FRAME=0` 关掉只打首帧）。

### 8.7 Orin 引擎服务：`sil/orin/engine_service.py`

因为 Orin 上 TensorRT 8.4 的 Python 绑定只有系统 Python 3.8 有，而模型逻辑在 3.10，
所以把**网络前向**单独放进 3.8 的服务进程：3.10 每帧通过本地 ZeroMQ（5562 端口）把输入发给它，
它用 `.engine` 推理，把输出发回。

### 8.8 两个启动脚本

| 脚本                                        | 起什么                                                 |
| ------------------------------------------- | ------------------------------------------------------ |
| `sil/orin/run_orin.sh`                      | Orin：`roscore + bridge + engine_service + agent_node` |
| `scripts/common/run_bench2drive_orin_v2.sh` | 主机：`bridge + CARLA + leaderboard + remote agent`    |

端口约定（排错时有用）：

| 端口          | 用途                          |
| ------------- | ----------------------------- |
| `11311`       | ROS master（总机）            |
| `5560 / 5561` | 计算进程 ↔ bridge 的 ZeroMQ   |
| `5562`        | Orin：agent ↔ 引擎服务        |
| `2000`        | CARLA 服务                    |
| `8000`        | CARLA Traffic Manager（车流） |

______________________________________________________________________

## 9. 常见问题排查

| 现象                                                        | 原因 / 处理                                                                                      |
| ----------------------------------------------------------- | ------------------------------------------------------------------------------------------------ |
| 起步一直 `no control`、Orin 无 `core ready`                 | 主机先起了；或 `ROS_MASTER_URI`/`ROS_IP` 填错（必须是对端可达的**本机有线 IP**，不能 127.0.0.1） |
| 能列出话题但收不到数据                                      | 同上：`ROS_IP` 错 / 防火墙（`sudo ufw allow from <同网段>/24`）                                  |
| `Failed to connect to CARLA Traffic Manager`                | 上一轮遗留的评测/CARLA 进程占着 **8000** 端口，杀掉后重跑                                        |
| Orin 启动报 `Address already in use (tcp://127.0.0.1:5562)` | 上一轮的引擎服务没退干净；先清理（见下）再起                                                     |
| `sensor_frame arrived before session`                       | 主机没先发 session（remote agent 会重发，一般自愈；频繁出现检查版本是否一致）                    |
| 同一路线复跑结果异常                                        | 历史 bug（已修）：确保两端 `git pull` 到最新（有 `session_id` 修复）                             |
| 中途整段超时、bridge 日志有 `too many values`               | 历史 bug（已修）：确保 `transport.py` 是最新（线程安全）                                         |
| `Expected exactly one 'model*.pth'`                         | checkpoint 目录里权重文件不是恰好一个                                                            |

**启动前清理孤立进程**（防止端口占用）：

```bash
# Orin
pkill -9 -f "[e]ngine_service.py"; pkill -9 -f "[a]gent_node.py"; pkill -9 -f "[b]ridge_node.py"
pkill -9 -f "[r]osmaster"; pkill -9 -f "[r]oscore"; pkill -9 -f "[r]osout"

# 主机
pkill -9 -f "[C]arlaUE4"; pkill -9 -f "[l]eaderboard_evaluator"; pkill -9 -f "[b]ridge_node.py"
```

______________________________________________________________________

## 10. 收尾

- **主机**：在运行的终端按 `Ctrl+C`（脚本会杀主机 bridge，评测脚本会杀 CARLA）。
- **Orin**：按 `Ctrl+C`（`run_orin.sh` 会清掉它起的全部进程）。
- 确认端口都释放：`ss -ltn | grep -E '2000|8000|5560|5561|5562|11311'` 应无输出。

______________________________________________________________________

## 11. 一页速查（我们这次实验的具体值）

> 下面就是本次两台机的真实命令与路径，直接复制即可（换机器时替换成你的值）。

**本次实验的环境（记牢）：**

| 项          | 主机                                          | Orin                                                                                               |
| ----------- | --------------------------------------------- | -------------------------------------------------------------------------------------------------- |
| IP / 网口   | `192.168.110.51/24`（`enp129s0`）             | `192.168.110.50/24`                                                                                |
| 用户 / 仓库 | `q` @ `~/gqzl/cvci_project/lead_v1`           | `tjuae` @ `/home/tjuae/GQZL/lead_orin`                                                             |
| 计算环境    | conda `cvci_project`（py3.10）                | miniforge `gqzl-py310`（py3.10）                                                                   |
| ROS 环境    | conda `ros_noetic`（RoboStack）               | 系统 py3.8 原生 Noetic                                                                             |
| checkpoint  | `outputs/local_training_3cams_3000/posttrain` | `/home/tjuae/GQZL/model_quantization/model_pth/local_training_3cams_3000/posttrain`                |
| engine      | —（在 Orin 用）                               | `/home/tjuae/GQZL/model_quantization/outputs/orin_quantization/planning/engines/model_fp16.engine` |

```bash
# ===== 0) 网络（已配好可跳过）=====
# 主机
sudo nmcli con add type ethernet ifname enp129s0 con-name sil-wired \
     ipv4.method manual ipv4.addresses 192.168.110.51/24 && sudo nmcli con up sil-wired
# Orin
sudo ip addr add 192.168.110.50/24 dev <orin网口> && sudo ip link set <orin网口> up
# 互通
ping -c2 192.168.110.50     # 主机
ping -c2 192.168.110.51     # Orin

# ===== 1) 拉代码（两端同一 commit）=====
# 主机
cd ~/gqzl/cvci_project/lead_v1 && git pull --ff-only
# Orin
cd ~/GQZL/lead_orin && git pull --ff-only

# ===== 2) Orin（先起；一条命令起 roscore+bridge+引擎服务+agent）=====
cd ~/GQZL/lead_orin
export ROS_IP=192.168.110.50
export ROS_MASTER_URI=http://192.168.110.50:11311
export CHECKPOINT=/home/tjuae/GQZL/model_quantization/model_pth/local_training_3cams_3000/posttrain
export LEAD_QUANTIZED_ENGINE=/home/tjuae/GQZL/model_quantization/outputs/orin_quantization/planning/engines/model_fp16.engine
bash sil/orin/run_orin.sh
# 等 "waiting for host: lead/session"；之后终端会滚动显示每帧 control（按 route 分隔）

# ===== 3) 主机（后起；一条命令起 bridge+CARLA+leaderboard+remote agent）=====
cd ~/gqzl/cvci_project/lead_v1
export ROS_IP=192.168.110.51
export ROS_MASTER_URI=http://192.168.110.50:11311
export CHECKPOINT=outputs/local_training_3cams_3000/posttrain
bash scripts/common/run_bench2drive_orin_v2.sh 0     # 单条；0-4 小批；不带参数=全量 220
# 已算过的 route 默认跳过；要重算：SKIP_DONE=0 bash scripts/common/run_bench2drive_orin_v2.sh 0

# ===== 4) 看结果 =====
# Orin：终端滚动显示 control seq=... infer_ms=...；或
#       ssh tjuae@192.168.110.50 'tail -f /tmp/sil_orin/agent.log'
# 主机：终端里每条 [summary] index N id XXXX score .. status: Completed
#       详细日志：outputs/watchdog_logs/bench2drive_per_route_v2/route_<id>_attempt_1.log

# ===== 5) 收尾清理 =====
# 主机（Ctrl+C 后若仍有残留）
pkill -9 -f "[C]arlaUE4"; pkill -9 -f "[l]eaderboard_evaluator"; pkill -9 -f "[b]ridge_node.py"
ss -ltn | grep -E '2000|8000|5560|5561' || echo clean
# Orin（Ctrl+C 会触发 run_orin.sh 清理；必要时手动）
pkill -9 -f "[e]ngine_service.py"; pkill -9 -f "[a]gent_node.py"; pkill -9 -f "[b]ridge_node.py"
pkill -9 -f "[r]osmaster"; pkill -9 -f "[r]oscore"; pkill -9 -f "[r]osout"
```
