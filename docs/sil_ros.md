# SIL：本机 CARLA + Orin 推理（ROS1）

软件在环（SIL）：本机跑 CARLA 与 leaderboard，Orin 跑驾驶 agent 的计算
（定位、建场景、特征化、网络前向、控制跟踪），两者通过 ROS1 交换数据。

本文件是接口契约与设计说明；Orin 侧的具体搭建步骤见
[`orin_setup.md`](orin_setup.md)。

## 1. 形态与边界（S0）

按约定，Orin 负责"整个 agent（除 CARLA 控制）"，本机负责 CARLA 相关部分：

| 侧                  | 负责                                                                                                                                                 |
| ------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| 本机 (Ubuntu 22.04) | CARLA 仿真、leaderboard evaluator、传感器产生与解码、把原始传感器帧发出去、应用返回的控制、infraction/video/metrics                                  |
| Orin (Ubuntu 20.04) | agent 计算：`tick`（定位/滤波）、`build_scene_data`、`build_features`、`features_to_batch`、网络前向、`compute_control`（返回 steer/throttle/brake） |

控制以纯数值（steer/throttle/brake）过网，本机再包成 `carla.VehicleControl`，
因此 **Orin 侧不需要 CARLA PythonAPI**。

## 2. 为什么是"传输/计算分离"（R2）

`lead` 需要 Python ≥ 3.10，而 ROS1 Noetic 是 Python 3.8/3.9。让同一个进程
既 `import rospy` 又 `import lead` 在 22.04 和 Orin 上都装不干净。所以：

- ROS1 由**独立的 bridge 进程**承担（本机用 RoboStack conda 的 ROS Noetic，
  或 Noetic 容器；Orin 用原生 Noetic）。
- Python 3.10 的计算进程只通过**本地 ZeroMQ** 与 bridge 通信，不 import rospy。

```
[本机 compute py3.10] --ZMQ tcp://127.0.0.1:5560/5561-- [bridge: ros_noetic] --ROS1/TCPROS over LAN-- [bridge: 原生 Noetic] --ZMQ-- [Orin compute py3.10]
```

## 3. 接口契约

所有 ROS 话题都是 `std_msgs/UInt8MultiArray`，`data` 是下列消息的 msgpack 字节。

| 话题                | 方向        | 频率            | 说明                                                     |
| ------------------- | ----------- | --------------- | -------------------------------------------------------- |
| `lead/session`      | 本机 → Orin | 每 route 一次   | 路线元信息与导航计划（Orin 无 CARLA 无法自行推导的部分） |
| `lead/sensor_frame` | 本机 → Orin | 每 tick（20Hz） | 原始传感器帧                                             |
| `lead/control`      | Orin → 本机 | 每 tick         | 车辆控制                                                 |
| `lead/heartbeat`    | Orin → 本机 | ~2Hz            | 存活与计时                                               |
| `lead/error`        | 双向        | 事件            | 错误                                                     |

### 3.1 `lead/session`

```
v, kind="session", route_id, scenario_type,
map_name,                            # 替代 self._world.get_map().name
gnss_uses_transverse_mercator,       # 替代 CarlaDataProvider.get_client()
global_plan_gps: [{lat, lon, z?, command}],
config_source                        # 仅供日志（checkpoint 路径/override）
```

### 3.2 `lead/sensor_frame`

```
v, kind="sensor_frame", seq, step, sim_time_us,
sensors: { "rgb_<i>": uint8 ndarray, "lidar1": float32 Nx4, "radar1": ..., "gps", "imu", "speed" },
camera_indices: [1,2,3]
```

只发 `policy.input_cameras` 需要的相机（本训练是 3 个前置相机），其余留给本机
的 video/metrics。数组以 msgpack 的 ndarray 扩展编码（dtype/shape/bytes）。

### 3.3 `lead/control`

```
v, kind="control", seq, step,
steer, throttle, brake,              # 纯数值
infer_ms,                            # Orin 侧推理耗时（可选）
aux                                  # 可选：prediction 张量，供本机可视化
```

### 3.4 `lead/heartbeat`

```
v, kind="heartbeat", seq, status, engine_ready, last_infer_ms, gpu_mem_mb
```

## 4. 同步、超时与安全回退

- leaderboard 是同步单步：本机发 `sensor_frame(seq)`，阻塞等 `control(seq)`。
- 超时（建议 200ms）→ 本机回退安全控制（`steer=0, throttle=0, brake=1`）并记录。
- `heartbeat` 缺失（如 >2s）→ 判定 Orin 掉线，评测脚本非零退出，交给 watchdog。
- CARLA leaderboard 用**仿真时间**判定超时，墙钟变慢不会污染评测结果，只影响吞吐。

## 5. 配置一致性与 parity

- 两侧必须使用**同一套 `lead` 代码与同一份 `config.yaml`**（同一 checkpoint 目录）。
- 同一 engine + 同一输入 → 同一输出；用本机 FP16 engine 跑与远端跑对比，
  验证 SIL 不改结果（允许 GPU 内核差异带来的极小数值差）。

## 6. 目录与运行

```
src/lead/evaluation/sil/          # py3.10：契约/编解码/传输
sil/ros_bridge/bridge_node.py     # ROS1 <-> ZeroMQ 桥（py3.8/3.9）
sil/tools/{local_probe,stub_orin_node}.py
sil/run_loopback.sh               # 本机端到端自检（无需 CARLA/Orin）
sil/docker/                       # 可选的容器化 bridge
```

本机（RoboStack，无需 root，已验证）：

```bash
conda create -n ros_noetic -c robostack-staging -c conda-forge \
    ros-noetic-ros-base python=3.9 msgpack-python pyzmq   # 一次性
bash sil/run_loopback.sh
```

## 7. 待办

- [ ] 抽 `AgentCore`（CARLA-free 的 agent 核心），本机本地跑通并与基线 parity。
- [ ] 本机的 leaderboard adapter：发 `sensor_frame`、应用远端 `control`，保留
  infraction/video/metrics。
- [ ] Orin 侧 agent 节点（复用同一 `AgentCore`，见 `orin_setup.md`）。
- [ ] `run_bench2drive_remote_v2.sh` 与 watchdog 接入。
