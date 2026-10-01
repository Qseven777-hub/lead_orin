# sil/orin — Orin 侧 SIL 部署

Orin 是"车上的计算单元"：接收主机（CARLA）转发的 session/sensor_frame，跑
`TransfuserCore` + FP16 TensorRT 引擎，把 `steer/throttle/brake` 回给主机。

本目录是 **Orin 专属**，同步脚本保留；共享代码（`src/lead/**`、`sil/ros_bridge/**`、
`sil/tools/**`、`docs/**`、`pyproject.toml`…）一律不改，改了会被主仓库覆盖。

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

```bash
# 1. 系统 py3.8 的 bridge/stub 依赖（免 sudo，用户级安装）
/usr/bin/python3 -m pip install --user "pyzmq==26.4.0" "msgpack==1.1.1"

# 2. py3.10 计算环境（复用 miniforge 的 gqzl-py310）
source /home/tjuae/miniforge3/etc/profile.d/conda.sh && conda activate gqzl-py310
python -m pip install filterpy==1.4.5 pyzmq==27.2
python -m pip install "lightning==2.6.1" --no-deps
cd <lead_orin> && python -m pip install -e . --no-deps
```

## 运行

```bash
export CHECKPOINT=<checkpoint 目录>          # config.yaml + 恰好一个 model*.pth
export LEAD_QUANTIZED_ENGINE=<model_fp16.engine>
export ROS_IP=<Orin 的有线网卡 IP>            # 跨机测试必填；单机可用 127.0.0.1
bash sil/orin/run_orin.sh
```

日志在 `/tmp/sil_orin/`（`agent.log`、`engine.log`、`bridge.log`、`roscore.log`）。

可选环境变量：`SIL_ENGINE_ENDPOINT`（默认 `tcp://127.0.0.1:5562`）、
`CONDA_ENV`（默认 `gqzl-py310`）、`LEAD_DEVICE`（默认 `cpu`）、
`SKIP_ROS=1`（roscore/bridge 由别处管理）。

## 单独调试

```bash
# 只起引擎服务
LEAD_QUANTIZED_ENGINE=<...>.engine /usr/bin/python3 sil/orin/engine_service.py

# 传输链路自检（stub，不需要 engine）
bash sil/orin/loopback_orin.sh
```

> 两侧必须使用**同一版本 `lead` 与同一份 `config.yaml`**，否则 parity 不成立。
> 引擎的输入/输出名与顺序见 `model_quantization/lead_model.py`。
