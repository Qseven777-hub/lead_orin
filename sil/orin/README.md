# sil/orin — Orin 侧 SIL 部署

Orin 是"车上的计算单元"：接收主机（CARLA）转发的 session/sensor_frame，跑
`TransfuserCore` + FP16 TensorRT 引擎，把 `steer/throttle/brake` 回给主机。

本目录是 **Orin 专属**，同步脚本保留；共享代码（`src/lead/**`、`sil/ros_bridge/**`、
`sil/tools/**`、`pyproject.toml`…）一律不改，改了会被主仓库覆盖。

## 相关文档

- 部署与复现（从白板机）：[`../../README.md`](../../README.md)
- Orin 环境与版本锁定清单：[`../../针对orin的环境配置.md`](../../针对orin的环境配置.md)
- 主机 / 总说明：[`../../ROS软件在环.md`](../../ROS软件在环.md)、[`../../ROS软件在环复现.md`](../../ROS软件在环复现.md)

## 本机约束（为什么需要这一层）

本 Orin 是 JetPack 5.0.2 / L4T 35.1 / **TensorRT 8.4.1**，而 `lead` 需要 Python 3.10：

- TensorRT 8.4 的 Python 绑定**只有系统 Python 3.8 有**，PyPI 无 aarch64/CUDA11 轮子；
- `gqzl-py310` 的 torch 是 **CPU 版**（没有 CUDA）。

所以网络前向不在 py3.10 进程里跑，而是放到系统 py3.8 的**引擎服务**里，py3.10
通过本地 ZeroMQ 每帧调用。特征管线（tick/建场景/特征化/控制跟踪）仍是共享代码。

## 文件

| 文件 | 运行环境 | 作用 |
|---|---|---|
| `run_orin.sh` | — | 一键起 roscore + bridge + 引擎服务 + agent 节点 |
| `engine_service.py` | 系统 py3.8 | 加载 `.engine`，ZeroMQ REP 上跑推理（`execute_v2` + `ctypes` 调 `libcudart`） |
| `engine_codec.py` | py3.8 / py3.10 | 引擎链路的 msgpack+ndarray 编解码（与 SIL codec 同格式） |
| `orin_engine_policy.py` | py3.10 | `OrinEngineTransfuser`：只重写 `forward`，转发到引擎服务 |
| `orin_policy_runner.py` | py3.10 | 共享 `PolicyRunner` 的 Orin 版：只 `.to(device)`，不做 `.cuda()` |
| `agent_node.py` | py3.10 | 共享（本目录版本）的 SIL agent 节点，改用上面的 runner |

## 前置（一次性）

依赖清单在本仓库根目录（本机 aarch64 已验证）：

```bash
# 1. 系统 py3.8 的 bridge/引擎依赖（免 sudo，用户级安装）
/usr/bin/python3 -m pip install --user -r requirements-orin-system.txt   # pyzmq==26.4.0 msgpack==1.1.1

# 2. py3.10 计算环境（miniforge gqzl-py310）
cd <lead_orin>
source ~/miniforge3/etc/profile.d/conda.sh && conda activate gqzl-py310
pip install -r requirements-orin.txt     # torch(CPU)/onnx/py123d/...
pip install -e . --no-deps
```

完整从白板机配置见仓库根 [`README.md`](../../README.md)。

## 运行

```bash
export CHECKPOINT=<checkpoint 目录>           # config.yaml + 恰好一个 model*.pth
export LEAD_QUANTIZED_ENGINE=<model_fp16.engine>
export ROS_IP=<Orin 的有线网卡 IP>             # 跨机测试必填；单机可用 127.0.0.1
export ROS_MASTER_URI=http://<Orin IP>:11311   # 默认 127.0.0.1 即可（roscore 在本机起）
bash sil/orin/run_orin.sh
```

起好后这个终端会：

- 打印 self-check：`engine service: READY` / `agent: listening` / `waiting for host: lead/session`；
- **每帧**滚动显示 `control seq=… step=… steer=… throttle=… brake=… infer_ms=…`，
  并按 route（session）用分隔条隔开；设 `SIL_LOG_EVERY_FRAME=0` 则只打首帧。

`run_orin.sh` 默认 `LEAD_CONFIG=policy.target=orin_engine_policy:OrinEngineTransfuser`（即走引擎服务）。
每收到新的一次 `session_id` 才重建 core（同一条 route 的重发不会重置状态）。

日志在 `/tmp/sil_orin/`（`agent.log`、`engine.log`、`bridge.log`、`roscore.log`）。

可选环境变量：`SIL_ENGINE_ENDPOINT`（默认 `tcp://127.0.0.1:5562`）、
`CONDA_ENV`（默认 `gqzl-py310`）、`LEAD_DEVICE`（默认 `cpu`）、
`SKIP_ROS=1`（roscore/bridge 由别处管理）、`SIL_LOG_EVERY_FRAME=0`（只打首帧）。

## 单独调试

```bash
# 只起引擎服务
LEAD_QUANTIZED_ENGINE=<...>.engine /usr/bin/python3 sil/orin/engine_service.py
```

## 联调 checklist（主机 <-> Orin）

### 0. 前置一致性

- [ ] 两侧使用**同一版本 `lead`（同一 commit）与同一份 `config.yaml`**（同一个 checkpoint 目录）。
- [ ] Orin 的 `.engine` 与该 `config.yaml` 是同一对（由 `model_quantization` 导出/构建）。

### 1. 网络与 ROS

- [ ] 两台机有线同网段、固定 IP，能互相 `ping`。
- [ ] 防火墙放行（如 `sudo ufw allow from <同网段>/24`）。
- [ ] **Orin 当 master**：`ROS_MASTER_URI=http://<Orin IP>:11311`。
- [ ] `ROS_IP` 两侧都填**对端可达的本机 IP**（不是 `127.0.0.1`）。
- [ ] 起完后 `rostopic list` 在两侧都能看到 `/lead/*`。

### 2. Orin 侧

- [ ] `export CHECKPOINT=<ckpt 目录>`（含 `config.yaml` + 恰好一个 `model*.pth`）。
- [ ] `export LEAD_QUANTIZED_ENGINE=<model_fp16.engine>`。
- [ ] `export ROS_IP=<Orin IP>`。
- [ ] `bash sil/orin/run_orin.sh`。
- [ ] `/tmp/sil_orin/engine.log` 出现 `ENGINE_SERVICE_READY tcp://127.0.0.1:5562`。
- [ ] `/tmp/sil_orin/agent.log` 出现 `policy target: orin_engine_policy:OrinEngineTransfuser` 与 `Orin SIL agent node listening`。

### 3. 主机侧

- [ ] `source <ROS>/setup.bash`，`ROS_MASTER_URI=http://<Orin IP>:11311 ROS_IP=<主机 IP>`。
- [ ] 起主机 bridge（`sil/ros_bridge/bridge_node.py`）。
- [ ] 起 CARLA + leaderboard，用 remote agent 入口（`scripts/common/run_bench2drive_orin_v2.sh`）。

### 4. 联调验证点（按顺序看日志）

- [ ] Orin 收到 `lead/session` → `agent.log` 出现 `core ready: route=... map=...`。
- [ ] Orin 收到 `lead/sensor_frame` → 每帧回 `lead/control`，`agent.log`/主机能看到 `infer_ms`。
- [ ] 主机心跳 `lead/heartbeat` 正常（Orin 未掉线）。
- [ ] 控制不是安全回退（`brake=1`）——若一直回退，见排错。
- [ ] parity：同一 engine 本机本地跑 vs 走 Orin，比对控制/得分（允许极小数值差）。

### 5. 排错

| 现象 | 排查 |
|---|---|
| 话题能列出但收不到 | `ROS_IP` 填错 / 防火墙 |
| `sensor_frame arrived before session` | 主机 adapter 未先发 session |
| 控制迟迟不回 | 引擎服务未起 / `SIL_ENGINE_ENDPOINT` 不一致 / `engine.log` 有报错 |
| engine 一直不 READY | `.engine` 路径 / TensorRT 版本 / `LD_LIBRARY_PATH` |
| `Expected exactly one 'model*.pth'` | checkpoint 目录权重文件不是恰好一个 |
| `set LEAD_QUANTIZED_ENGINE` 报错 | 未设该环境变量 |

> 两侧必须使用**同一版本 `lead` 与同一份 `config.yaml`**，否则 parity 不成立。
> 引擎的输入/输出名与顺序见 `model_quantization/lead_model.py`。
