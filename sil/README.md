# sil/ — 软件在环（SIL）

把驾驶 agent 的计算从仿真机搬到另一台算力机（NVIDIA Orin），两者通过 ROS1 通信，
模拟「实车部署」：**主机 = 世界**，**Orin = 车上的计算单元**。

> - 总说明（形态 / 契约 / 维护）：[`../ROS软件在环.md`](../ROS%E8%BD%AF%E4%BB%B6%E5%9C%A8%E7%8E%AF.md)
> - 0 基础复现手册（环境 / 网络 / 启动 / 排错）：[`../ROS软件在环复现.md`](../ROS%E8%BD%AF%E4%BB%B6%E5%9C%A8%E7%8E%AF%E5%A4%8D%E7%8E%B0.md)

## 1. 仓库划分（完全分开）

| 仓库                         | 负责                  | 内容                                                                                           |
| ---------------------------- | --------------------- | ---------------------------------------------------------------------------------------------- |
| **`lead_v1`**（主仓库）      | **主机侧 + 共享代码** | `src/lead/**`、`sil/ros_bridge/**`、`sil/tools/**`、`sil/docker/**`、本文件、`docs/`、构建文件 |
| **`lead_orin`**（Orin 仓库） | **Orin 侧**           | `sil/orin/**`（agent 节点、引擎服务、启动脚本）、它自己的 `README.md`                          |

- 共享代码由 `lead_v1` 拥有，通过 `scripts/common/sync_orin_repo.sh` 同步进 `lead_orin`；
- **`sil/orin/` 只存在于 `lead_orin`**（本仓库不再包含它），Orin 侧改动不回灌。
- 纪律：改共享行为（agent core、契约、bridge）在 `lead_v1` 改；改 Orin 侧（引擎/接线）在 `lead_orin` 改。

## 2. 本仓库里的组件

| 路径                                                           | 运行环境                | 说明                                                             |
| -------------------------------------------------------------- | ----------------------- | ---------------------------------------------------------------- |
| `src/lead/evaluation/sil/`                                     | py3.10                  | 契约 `contract`、msgpack 编解码 `codec`、ZeroMQ 传输 `transport` |
| `src/lead/evaluation/agents/remote/remote_transfuser_agent.py` | py3.10                  | 主机 adapter：转发传感器、施加远端控制，保留 infraction/metrics  |
| `ros_bridge/bridge_node.py`                                    | ROS1 Noetic (py3.8/3.9) | ROS `UInt8MultiArray` ↔ 本地 ZeroMQ（唯一 import rospy 的代码）  |
| `tools/stub_orin_node.py`                                      | ROS1 Noetic             | 临时 Orin 替身（回固定 control），用于链路自检                   |
| `tools/local_probe.py`                                         | py3.10                  | 发一帧并等待 control                                             |
| `run_loopback.sh`                                              | —                       | 单机端到端自检（无需 CARLA/Orin）                                |
| `maintain_repo.sh`                                             | —                       | 收工：检查 → 提交推送 `lead_v1` → 同步并推送 `lead_orin`         |
| `docker/`                                                      | docker/podman           | 可选：把 bridge 容器化                                           |

Orin 侧的组件（**在 `lead_orin`**）：`sil/orin/agent_node.py`（节点）、`engine_service.py`
（系统 py3.8 的 TensorRT 引擎服务）、`engine_codec.py`、`orin_policy_runner.py`、
`orin_engine_policy.py`、`run_orin.sh`（一键启动）。

## 3. 一次循环的数据流

```
主机: CARLA 采传感器 → remote agent → [ZMQ 5560] → 主机 bridge → ROS(局域网)
        → Orin bridge → [ZMQ 5560] → Orin agent_node
        → 引擎服务(TensorRT) 出「规划」→ 控制跟踪器 出「控制」
        → [ZMQ 5560] → Orin bridge → ROS → 主机 bridge → [ZMQ 5561] → remote agent 施加
```

话题（都是 `std_msgs/UInt8MultiArray`，`data` 是 msgpack 字节）：

| 话题                | 方向      | 频率          | 内容                                                                                                                      |
| ------------------- | --------- | ------------- | ------------------------------------------------------------------------------------------------------------------------- |
| `lead/session`      | 主机→Orin | 每 route 一次 | `route_id / session_id / map_name / gnss_uses_transverse_mercator / global_plan_gps / lat_ref / lon_ref / camera_indices` |
| `lead/sensor_frame` | 主机→Orin | 每 tick       | `seq / step / sim_time_us / sensors{...} / camera_indices`                                                                |
| `lead/control`      | Orin→主机 | 每 tick       | `seq / step / steer / throttle / brake / infer_ms / aux`                                                                  |
| `lead/heartbeat`    | Orin→主机 | ~2Hz          | `status / engine_ready / last_infer_ms / gpu_mem_mb`                                                                      |
| `lead/error`        | 双向      | 事件          | `code / message / seq`                                                                                                    |

> 模型输出的是**规划**（未来轨迹 / 目标速度 / 路线），在 Orin 上由**控制跟踪器**变成
> `steer/throttle/brake` 再回传；主机只施加控制，不需要规划。详见复现手册 §3.1。

## 4. 源码剥离：agent 去 CARLA 化（SIL 的关键改动）

Orin 上没有 CARLA，所以原来「和 CARLA 绑死」的 agent 必须拆出一份**可在无 CARLA 环境运行**
的计算核心，且**主机与 Orin 共用同一份实现**（否则两端行为会漂移，parity 不成立）。

改动方式：把 `BaseAgent` / `AbstractDrivingAgent` / `TransfuserAgent` 里**与 CARLA 无关**的部分
抽成共享模块，原有的 CARLA 类改为**继承**它们（方法体是搬移，不是重写）：

| 抽出的共享模块                                                                           | 角色                                                                 | CARLA-free 的证据                              |
| ---------------------------------------------------------------------------------------- | -------------------------------------------------------------------- | ---------------------------------------------- |
| `src/lead/common/driving_state.py` → `DrivingStateBase`                                  | 定位/Kalman、路线规划、`tick`、历史位姿；`ControlCommand`            | 模块 docstring 明确「不 import carla/srunner」 |
| `src/lead/api/agent_scene.py` → `ScenePipelineMixin`                                     | `tick` 队列、`build_scene_data`、相机/雷达/激光处理                  | 同上；读 `map_name` 字符串而非 `self._world`   |
| `src/lead/evaluation/inference/agent_core.py` → `PolicyAgentCore`                        | 一步编排 `run_step`：tick→scene→features→forward→control             | 同上；无 CARLA                                 |
| `src/lead/evaluation/agents/transfuser/transfuser_control.py` → `TransfuserControlMixin` | 规划（prediction）→ `ControlCommand`                                 | CARLA-free                                     |
| `src/lead/evaluation/agents/transfuser/transfuser_core.py` → `TransfuserCore`            | Orin 上跑的完整 core（= `TransfuserControlMixin + PolicyAgentCore`） | CARLA-free                                     |

原有的 CARLA 类保持不变的行为，只继承这些共享模块：

- `src/lead/common/base_agent.py` → `BaseAgent(DrivingStateBase)`：只补 CARLA 相关的
  GNSS 投影标志与 `carla.VehicleControl` 占位；
- `src/lead/api/abstract_driving_agent.py` → `AbstractDrivingAgent(...)`：保留 leaderboard 协议、
  传感器装配、infraction/video/metrics；
- `src/lead/evaluation/agents/transfuser/transfuser_agent.py` → `TransfuserAgent(...)`：只留 CARLA 侧可视化。

配套的「去 CARLA 化」：

- `src/lead/common/localization/gps.py`、`kalman_filter.py`、`src/lead/common/carla_to_123d.py`：
  把 `import carla` 放进 `TYPE_CHECKING`，`carla.Rotation` 换成等价的 `EulerRotation` 值对象，
  用鸭子类型协议（`ControlLike`）替代 `carla.VehicleControl`。

判断口径：Orin 节点（`sil/orin/agent_node.py`）的 import 闭包**不含** `carla / srunner / leaderboard`，
所以整套 agent 计算可以脱离 CARLA 运行（这也是本仓库不含 `3rd_party/` 的原因）。

## 5. 运行入口（两条命令）

- **Orin**（在 `lead_orin`）：`bash sil/orin/run_orin.sh`
- **主机**（在本仓库）：`bash scripts/common/run_bench2drive_orin_v2.sh [routes]`

完整步骤见 [`../ROS软件在环复现.md`](../ROS%E8%BD%AF%E4%BB%B6%E5%9C%A8%E7%8E%AF%E5%A4%8D%E7%8E%B0.md)。

## 6. 自检与测试

```bash
# 单机端到端自检（无需 CARLA/Orin）：一次性建 ros_noetic 环境后
conda create -n ros_noetic -c robostack-staging -c conda-forge \
    ros-noetic-ros-base python=3.9 msgpack-python pyzmq
bash sil/run_loopback.sh

# 单元测试
python -m pytest tests/unittests/evaluation/sil -q
```
