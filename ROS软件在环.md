# ROS 软件在环（SIL）

本机跑 CARLA，Orin 跑驾驶 agent 的计算，两者通过 ROS1 通信，用来模拟
"实车部署"：CARLA 是**世界**，Orin 是**车上的计算单元**。

> 相关文件：
>
> - 接口契约与设计：[`docs/sil_ros.md`](docs/sil_ros.md)
> - Orin 侧搭建步骤：[`docs/orin_setup.md`](docs/orin_setup.md)
> - 传输层代码：[`sil/`](sil/)、[`src/lead/evaluation/sil/`](src/lead/evaluation/sil/)

## 1. 形态

```
┌──────────────── 本机 Ubuntu 22.04 ────────────────┐        ┌──────── Orin Ubuntu 20.04 ────────┐
│ CARLA（世界 / 仿真器）                              │        │  roscore（原生 Noetic）            │
│ leaderboard evaluator                               │        │  ros_bridge（ROS <-> ZeroMQ）      │
│ 本机 adapter：收传感器 → 转发；收 control → 施加      │  ROS1  │  orin agent_node                   │
│ ros_bridge（ROS <-> ZeroMQ）                        │◄─────►│    TransfuserCore + TensorRT 引擎   │
│ infraction / video / metrics（保持在本机）           │        │                                    │
└────────────────────────────────────────────────────┘        └────────────────────────────────────┘
```

- **CARLA 始终在本机启动，是世界本身**，本次改动没有把 CARLA 从本机评测里去掉。
- **Orin 只拿到一帧帧传感器数据**，没有世界，所以它跑的那段计算不能访问
  `self._world` / `CarlaDataProvider` / `carla.VehicleControl` 这些"世界专有"的东西。
  这一步是 S0（Orin 跑整个 agent）的必然结果，不是额外目标。
- 控制以纯数值（`steer/throttle/brake`）回传，本机再包成 `carla.VehicleControl`
  施加——因此 **Orin 侧不需要 CARLA**。

## 2. 为什么是"传输 / 计算分离"（R2）

`lead` 需要 Python ≥ 3.10，而 ROS1 Noetic 是 Python 3.8/3.9，让同一进程同时
`import rospy` 和 `import lead` 两边都装不干净。所以：

- ROS1 由**独立 bridge 进程**承担（本机用 RoboStack conda 的 ROS Noetic，
  Orin 用原生 Noetic）；
- Python 3.10 的计算进程只通过**本地 ZeroMQ** 与 bridge 通信，不 import rospy。

## 3. 接口契约

所有 ROS 话题都是 `std_msgs/UInt8MultiArray`，`data` 是 msgpack 字节。

| 话题                | 方向        | 频率            | 内容                                                                                                                         |
| ------------------- | ----------- | --------------- | ---------------------------------------------------------------------------------------------------------------------------- |
| `lead/session`      | 本机 → Orin | 每 route 一次   | `route_id / scenario_type / map_name / gnss_uses_transverse_mercator / global_plan_gps / lat_ref / lon_ref / camera_indices` |
| `lead/sensor_frame` | 本机 → Orin | 每 tick（20Hz） | `seq / step / sim_time_us / sensors{rgb_i, lidar1,2, radar1..4, gps, imu, speed} / camera_indices`                           |
| `lead/control`      | Orin → 本机 | 每 tick         | `seq / step / steer / throttle / brake / infer_ms`                                                                           |
| `lead/heartbeat`    | Orin → 本机 | ~2Hz            | `status / engine_ready / last_infer_ms`                                                                                      |
| `lead/error`        | 双向        | 事件            | `code / message`                                                                                                             |

`map_name`、`gnss_uses_transverse_mercator`、`lat_ref/lon_ref`、`global_plan_gps`
就是"世界专有"的信息，由本机在 `session` 里一次性发给 Orin。

## 4. 代码结构

### 4.1 agent 计算的抽取（本机与 Orin 共用一份实现）

原来每个 tick 的计算散在 `BaseAgent` / `AbstractDrivingAgent` / `TransfuserAgent` 里，
其中大部分与 CARLA 无关。现在抽成：

| 模块                                                          | 角色                                                                                |
| ------------------------------------------------------------- | ----------------------------------------------------------------------------------- |
| `src/lead/common/driving_state.py`                            | `DrivingStateBase`：定位/Kalman/路线规划/`tick`/历史位姿；`ControlCommand`          |
| `src/lead/api/agent_scene.py`                                 | `ScenePipelineMixin`：`tick` 队列、`build_scene_data`、相机/雷达/激光处理           |
| `src/lead/evaluation/inference/agent_core.py`                 | `PolicyAgentCore`：把两步编排成 `run_step()`（tick→scene→features→forward→control） |
| `src/lead/evaluation/agents/transfuser/transfuser_control.py` | `TransfuserControlMixin`：prediction → steer/throttle/brake                         |
| `src/lead/evaluation/agents/transfuser/transfuser_core.py`    | `TransfuserCore`：Orin 上跑的完整 core                                              |

本机原有的
`BaseAgent` / `AbstractDrivingAgent` / `TransfuserAgent` **改为继承这些共享模块**，
方法体是**搬移**不是重写，因此本机评测行为不变。

### 4.2 传输层

| 路径                                                           | 运行环境                | 说明                                                                  |
| -------------------------------------------------------------- | ----------------------- | --------------------------------------------------------------------- |
| `src/lead/evaluation/sil/`                                     | py3.10                  | `contract` / `codec`(msgpack) / `transport`(ZeroMQ)                   |
| `sil/ros_bridge/bridge_node.py`                                | ROS1 Noetic (py3.8/3.9) | ROS `UInt8MultiArray` ↔ 本地 ZeroMQ                                   |
| `sil/orin/agent_node.py`                                       | py3.10                  | Orin 节点：`TransfuserCore` + `PolicyRunner` + `SilTransport`         |
| `src/lead/evaluation/agents/remote/remote_transfuser_agent.py` | py3.10                  | 本机 adapter：转发传感器、施加远端控制，保留 infraction/video/metrics |
| `scripts/common/run_bench2drive_remote_v2.sh`                  | —                       | 本机入口：起 bridge + 指定 remote agent + 复用 fast 脚本/watchdog     |
| `sil/tools/stub_orin_node.py`                                  | ROS1 Noetic             | 临时 Orin 替身（回固定 control）                                      |
| `sil/tools/local_probe.py`                                     | py3.10                  | 发一帧、等 control                                                    |
| `sil/run_loopback.sh`                                          | —                       | 本机端到端自检（无需 CARLA/Orin）                                     |
| `sil/docker/`                                                  | docker                  | 可选：把 bridge 容器化                                                |

## 5. 本机评测功能保持不变

- `tick` / `build_scene_data` / `build_features` / `compute_control` 是同一份代码，
  本机 `TransfuserAgent` 通过继承使用；
- `run_bench2drive_fast_v2.sh`、engine 评测、expert、SafeComfort 等脚本与流程不变；
- `infraction / video / metrics / SafeComfort` 仍留在本机（它们需要世界）；
- 校验：`lint-imports` 7 条契约全部 KEPT；单测全绿。

## 6. 怎么跑

### 6.1 本机端到端自检（不需要 CARLA / Orin）

```bash
# 一次性：本机无 root 也能装的 ROS1 Noetic
conda create -n ros_noetic -c robostack-staging -c conda-forge \
    ros-noetic-ros-base python=3.9 msgpack-python pyzmq

bash sil/run_loopback.sh     # 期望输出：OK: control seq=... steer=... throttle=... brake=...
```

### 6.2 跨机（Orin 当 ROS master）

```bash
# Orin
source /opt/ros/noetic/setup.bash
export ROS_MASTER_URI=http://127.0.0.1:11311 ROS_IP=<orin-ip>
roscore -p 11311 &
python3 sil/ros_bridge/bridge_node.py &

# 本机
export ROS_MASTER_URI=http://<orin-ip>:11311 ROS_IP=<本机有线IP>
python sil/ros_bridge/bridge_node.py &
```

Orin 上把 stub 换成真正的 agent 节点：

```bash
LEAD_QUANTIZED_ENGINE=/path/to/model_fp16.engine \
LEAD_CONFIG="policy.target=lead.policy.transfuser.quantized_policy:QuantizedTransfuser" \
python sil/orin/agent_node.py --checkpoint /path/to/checkpoint
```

（详细步骤见 [`docs/orin_setup.md`](docs/orin_setup.md)。）

## 7. 待办

- [x] **本机 adapter**：`remote_transfuser_agent.py`——收传感器 → 发 `sensor_frame`、
  收 `control` 施加到车，保留 infraction/video/metrics。
- [x] `run_bench2drive_remote_v2.sh`：起 bridge + 指定 remote agent，接入现有
  fast 脚本与 watchdog。
- [ ] **parity 验证**：同一 engine，本机本地跑 vs 走 Orin 跑，比对控制/得分。
- [ ] Orin 依赖落地（py123d/numba/cv2/torch aarch64）与 engine 加载。
- [ ] 本机 adapter 的 video/可视化（需要把 Orin 的 features/prediction 回传或
  在本机重算，当前 SIL 下暂不产出视频）。

## 8. 注意

- 请求/应答是同步单步；超时可回退安全控制（`brake=1`），心跳缺失判定掉线。
- CARLA leaderboard 用**仿真时间**判定超时，墙钟变慢不会污染评测结果，只影响吞吐。
- 两侧必须使用**同一套 `lead` 代码与同一份 `config.yaml`**，否则 parity 不成立。

## 9. 交付给 Orin 的东西

**源码**（推荐 `git clone` 同一 commit，保证版本一致）：

- 需要：`src/lead/`、`sil/`、`pyproject.toml`、`setup.py`
- 不需要：`3rd_party/`（CARLA/leaderboard/scenario_runner）、`scripts/`、`tests/`、
  `outputs/`、`data/`、`notebooks/`

> 依据：Orin 节点（`sil/orin/agent_node.py`）的 import 闭包**不含
> carla / srunner / leaderboard**，所以整套 agent 计算可以脱离 CARLA 运行。

**数据**：checkpoint 目录（`config.yaml` + 恰好一个 `model*.pth`）、FP16 `.engine`。

**Orin 要做的事**：装 ROS1 Noetic（跑 bridge）+ Python 3.10 lead 环境（跑
agent 节点）→ 起 `bridge_node.py` → 起 `agent_node.py`（`LEAD_QUANTIZED_ENGINE` +
`LEAD_CONFIG` 选 QuantizedTransfuser）。详见 [`docs/orin_setup.md`](docs/orin_setup.md)。

## 10. 仓库维护

代码分两个仓库：

| 仓库                     | 内容                                                                                                                     | 谁改       |
| ------------------------ | ------------------------------------------------------------------------------------------------------------------------ | ---------- |
| `lead_v1`（主仓库）      | 共用代码：`src/lead/**`、`sil/ros_bridge/**`、`sil/tools/**`、`sil/docker/**`、`sil/maintain_repo.sh`、`docs/`、构建文件 | 在本机改   |
| `lead_orin`（Orin 仓库） | Orin 专属：`sil/orin/**`（agent 节点）、它自己的 `README.md`                                                             | 在 Orin 改 |

### 10.1 Orin 侧改动边界（能改 / 不能改）

同步脚本（`sync_orin_repo.sh`）只同步共享路径，所以规则很硬：

**可以改（Orin 专属，改动能保留）：**

| 路径                    | 说明                                                               |
| ----------------------- | ------------------------------------------------------------------ |
| `sil/orin/**`           | agent 节点及其下新增文件（TensorRT 加载、Orin 接线、设备相关调整） |
| `README.md`             | `lead_orin` 自己的说明（同步脚本刻意不覆盖）                       |
| 根目录你新建的文件/目录 | 不在同步范围内的都不会被动（如本地脚本、`.env`、配置）             |

**不能改（共享代码，改了下次同步会被覆盖甚至删除）：**

| 路径                                                                                                                 | 同步行为                                      |
| -------------------------------------------------------------------------------------------------------------------- | --------------------------------------------- |
| `src/lead/**`                                                                                                        | 整棵覆盖（`src/lead/routes/` 除外，主机专属） |
| `sil/ros_bridge/**`、`sil/tools/**`、`sil/docker/**`、`sil/maintain_repo.sh`、`sil/README.md`、`sil/run_loopback.sh` | `sil/` 覆盖，仅 `sil/orin/` 被保留            |
| `docs/**`                                                                                                            | 整棵覆盖                                      |
| `pyproject.toml`、`setup.py`、`.env.example`、`.gitignore`、`LICENSE`、`ROS软件在环.md`                              | 用主仓库版本覆盖                              |

> 判断口诀：**除了 `sil/orin/` 和 `lead_orin/README.md`（以及你在根目录自建的东西），其余都不是 Orin 的。**
> 要改共享行为（agent core、契约、bridge），去主仓库 `lead_v1` 改，再 `bash sil/maintain_repo.sh "msg"` 同步过来。

- **部署**：Orin 上 `git clone lead_orin` 即可，含运行所需全部代码；`.env` 自己填（不提交）。
- **纪律**：不要直接在 Orin 上改共用代码（`src/lead`、bridge、契约）——下次同步会被主仓库版本覆盖；要改就改主仓库，再同步。
- 两侧必须用**同一版本的 `lead` 与同一份 `config.yaml`**，否则 parity 不成立。

**收工前一条命令**（两端通用，自动判断角色）：

```bash
# 本机：自检 + 提交推送 lead_v1 + 同步并推送 lead_orin
cd ~/gqzl/cvci_project/lead_v1
bash sil/maintain_repo.sh "你的提交信息"

# Orin：只提交推送 lead_orin
cd ~/lead_orin
bash sil/maintain_repo.sh "你的提交信息"
```

可选环境变量：`SKIP_CHECKS=1`（跳过 ruff）、`RUN_TESTS=1`（额外跑评测单测）、
`INCLUDE_3RDPARTY=1`（连 `3rd_party/` 一起提交，默认排除）、
`ORIN_REPO=/path/to/lead_orin`（指定 Orin 仓库路径，默认 `../lead_orin`）。
