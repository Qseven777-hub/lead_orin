# lead_orin — Orin 侧（软件在环 SIL）部署与复现

本仓库是主仓库 `lead_v1` 的 **Orin 侧**：只包含在本机 CARLA 软件在环（SIL）中、
**Orin 上运行驾驶 agent 所需**的内容。照着本文从上到下做，就能在一台**全新的 Orin** 上复现。

- 主仓库 / 主机侧、整体设计与复现手册：[`ROS软件在环.md`](ROS软件在环.md)、[`ROS软件在环复现.md`](ROS软件在环复现.md)
- 环境变量清单：[`针对orin的环境配置.md`](针对orin的环境配置.md)
- Orin 侧细节（引擎服务、启动脚本）：[`sil/orin/README.md`](sil/orin/README.md)

---

## 0. 本机基线（迁移时请对齐这些版本）

| 项 | 值 |
| --- | --- |
| 硬件 | NVIDIA Jetson AGX Orin（**aarch64**） |
| 系统 | Ubuntu 20.04 + **JetPack 5.0.2 / L4T R35.1** |
| CUDA | **11.4**（`/usr/local/cuda-11.4`） |
| TensorRT | **8.4.1**（apt，`+cuda11.4`）；`trtexec` 在 `/usr/src/tensorrt/bin/trtexec` |
| 系统 Python | 3.8.10（跑 ROS / bridge / **引擎服务**） |
| 计算 Python | miniforge `gqzl-py310` = **Python 3.10**（跑 `lead` / **模型转换**） |

> 为什么要两个 Python：`lead` 要求 ≥3.10，而 **TensorRT 8.4 的 Python 绑定只有系统 py3.8 有**。
> 所以网络前向单独放在 py3.8 的**引擎服务**里，py3.10 通过本地 ZeroMQ 每帧调用。

---

## 1. 内容

| 路径 | 说明 |
| --- | --- |
| `src/lead/` | lead 包（Orin 节点及其依赖；由主仓库同步，勿改） |
| `sil/orin/**` | **Orin 专属**：`agent_node.py`、`engine_service.py`、`engine_codec.py`、`orin_policy_runner.py`、`orin_engine_policy.py`、`run_orin.sh`、`loopback_orin.sh` |
| `sil/ros_bridge/`、`sil/tools/` | 共享（由主仓库同步） |
| `requirements-orin.txt` | `gqzl-py310`（py3.10）的完整依赖清单（本机实测，见 §3.3） |
| `requirements-orin-system.txt` | 系统 py3.8 的依赖（bridge / 引擎服务） |
| `pyproject.toml` / `setup.py` | 安装用；已去掉无 aarch64 轮子的 `carla/open3d/pyqt5` |

---

## 2. 系统层（第一步，apt + 系统 pip）

```bash
# ROS1 Noetic（给 bridge 用）
sudo apt update
sudo apt install -y ros-noetic-ros-base python3-zmq python3-msgpack

# 常用工具
sudo apt install -y git build-essential cmake pkg-config

# 系统 py3.8 的 ZeroMQ 依赖（bridge 与引擎服务）
/usr/bin/python3 -m pip install --user -r requirements-orin-system.txt
```

校验：

```bash
/usr/bin/python3 -c "import zmq, msgpack; print(zmq.__version__, msgpack.__version__)"   # 26.4.0 1.1.1
/usr/src/tensorrt/bin/trtexec --version | head -1                                        # TensorRT v8401
```

---

## 3. 计算环境 `gqzl-py310`（py3.10：lead 推理 + 模型转换）

### 3.1 安装 miniforge（aarch64）并建环境

```bash
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-aarch64.sh
bash Miniforge3-Linux-aarch64.sh -b -p ~/miniforge3
source ~/miniforge3/etc/profile.d/conda.sh
conda create -n gqzl-py310 python=3.10 -y
conda activate gqzl-py310
```

### 3.2 安装依赖（直接用本仓库的清单）

```bash
pip install -r requirements-orin.txt     # 91 项，含 torch/onnx/py123d 等
pip install -e . --no-deps               # 可编辑安装 lead（跳过依赖解析）
```

> - torch 是 **CPU aarch64 版**（`torch==2.5.1`，`torch.version.cuda is None`）。CUDA 只在
>   TensorRT 引擎里用，推理不需要 torch CUDA。
> - `requirements-orin.txt` 里已包含**模型转换**的依赖：`onnx==1.16.2`、`onnxruntime==1.23.2`、
>   `onnxconverter-common==1.16.0`、`ml-dtypes==0.3.2`、`typeguard==4.6.0`。
> - **不要**直接 `pip install py123d`：它会拖 `ray/geopandas/pyogrio/mapbox-earcut/viser/...`，
>   部分无 aarch64 轮子；按清单装即可。

### 3.3 依赖清单怎么来的 / 怎么更新

- `requirements-orin.txt` = 本机 `conda activate gqzl-py310 && pip list --format=freeze`
  去掉 `lead/pip/setuptools/wheel` 后的结果（本机 91 项，已在 aarch64 验证可装）。
- 需要新增/升级依赖时，在 Orin 上装好后重新导出，或手工编辑该文件。
- 系统侧同理：`/usr/bin/python3 -m pip freeze` → `requirements-orin-system.txt`。

---

## 4. 数据（不是源码）

| 东西 | 要求 |
| --- | --- |
| checkpoint 目录 | 含 `config.yaml` 和**恰好一个** `model*.pth` |
| `.engine` | 与上面 checkpoint 配对的 FP16 引擎，**必须在目标机上构建**（TensorRT 与机器绑定） |

两侧（主机 / Orin）必须用**同一份 `config.yaml`**；用 `md5sum` 比对。

---

## 5. 启动与验证

```bash
cd ~/GQZL/lead_orin
export ROS_IP=<Orin 有线 IP>                       # 例如 192.168.110.50
export ROS_MASTER_URI=http://<Orin 有线 IP>:11311
export CHECKPOINT=<checkpoint 目录>
export LEAD_QUANTIZED_ENGINE=<model_fp16.engine>
bash sil/orin/run_orin.sh
```

自检通过会打印：

```
[orin] engine service: READY (tcp://127.0.0.1:5562)
[orin] agent:          listening (policy target: orin_engine_policy:OrinEngineTransfuser)
[orin] waiting for host: lead/session (ROS_IP=...)
```

之后**每帧**会在该终端滚动显示（`SIL_LOG_EVERY_FRAME=0` 可关）：

```
──────── session core ready: route=1833 map=Carla/Maps/Town12/Town12 ────────
  control seq=2 step=2 steer=0.000 throttle=1.000 brake=0.000 infer_ms=97.1
  ...
```

主机侧启动（**等 Orin 就绪后再起**）：在本机 `lead_v1` 仓库里
`bash scripts/common/run_bench2drive_orin_v2.sh 0`（详见 [`ROS软件在环复现.md`](ROS软件在环复现.md)）。

---

## 6. 与主仓库的关系

- `lead_v1` 拥有**主机侧 + 共享代码**；本仓库的 `src/lead`、`sil/ros_bridge`、`sil/tools` 等由
  `scripts/common/sync_orin_repo.sh` 同步过来，**不要在本仓库改这些**（改了会被覆盖）。
- `sil/orin/**` 和本 `README.md`、两个 `requirements-orin*.txt` 是 **Orin 侧**，在本仓库改。
- 两侧必须使用**同一版本的 `lead` 与同一份 `config.yaml`**，否则 SIL 与本机评测的 parity 不成立。

---

## 7. aarch64 常见坑

| 现象 | 处理 |
| --- | --- |
| `pip install -e .` 报 `carla/open3d/pyqt5` 无轮子 | 这三个没有 aarch64 轮子；本仓库 `pyproject.toml` 已去掉它们，且用 `--no-deps` 安装 |
| `pip install py123d` 拖一堆编译失败 | 不要整装；用 `requirements-orin.txt` |
| 引擎服务 `Address already in use :5562` | 上一轮残留；先 `pkill -9 -f "[e]ngine_service.py"` 再起 |
| `.engine` 换机器不能用 | engine 与 TensorRT/机器绑定，在目标机重建 |
| `import tensorrt` 在 py3.10 失败 | TensorRT 8.4 绑定只有系统 py3.8；引擎服务用 `/usr/bin/python3` |
| `trtexec` 找不到 | 完整路径 `/usr/src/tensorrt/bin/trtexec` |
