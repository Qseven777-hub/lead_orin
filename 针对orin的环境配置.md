# 针对 Orin 的环境配置（aarch64，从白板机）

本文档面向**一台全新的 NVIDIA Orin（aarch64）**，从零把软件在环（SIL）所需的**三套环境**
配好，并保证迁移到**另一台 Orin** 也能复现。

> 只想跑通流程看 [`ROS软件在环复现.md`](ROS%E8%BD%AF%E4%BB%B6%E5%9C%A8%E7%8E%AF%E5%A4%8D%E7%8E%B0.md)；本文只讲 Orin 这台机器的环境怎么搭。

______________________________________________________________________

## 0. 为什么 Orin 上要三套东西

Orin 上同时存在**两个 Python**，这是硬约束，不是选择：

| 环境             | Python             | 作用                                             | 为什么单独                                   |
| ---------------- | ------------------ | ------------------------------------------------ | -------------------------------------------- |
| **系统环境**     | 3.8.10（系统自带） | ROS Noetic、`ros_bridge`、**TensorRT 引擎服务**  | TensorRT 8.4 的 Python 绑定只有系统 py3.8 有 |
| **`gqzl-py310`** | 3.10               | `lead` 特征/控制逻辑、`agent_node`、**模型转换** | `lead` 要求 Python ≥ 3.10                    |

- **推理**：`agent_node`（py3.10）通过本地 ZeroMQ 调**引擎服务**（py3.8 + TensorRT）跑网络前向。
- **模型转换**：ONNX 导出 / TensorRT 图降级 / engine 构建，也在 `gqzl-py310` 里做（用 `trtexec`）。

______________________________________________________________________

## 1. 目标机器的基线（本机实测，作为参照）

| 项          | 值                                                                                                          |
| ----------- | ----------------------------------------------------------------------------------------------------------- |
| 硬件        | NVIDIA Jetson AGX Orin（aarch64）                                                                           |
| 系统        | Ubuntu 20.04 + **JetPack 5.0.2 / L4T R35.1**（`/etc/nv_tegra_release` 显示 `R35 (release), REVISION: 1.0`） |
| CUDA        | **11.4**（`/usr/local/cuda-11.4`）                                                                          |
| TensorRT    | **8.4.1**（apt，`+cuda11.4`），`trtexec` 在 `/usr/src/tensorrt/bin/trtexec`                                 |
| 系统 Python | 3.8.10                                                                                                      |
| ROS         | ROS1 Noetic（原生 apt）                                                                                     |
| miniforge   | `~/miniforge3`（aarch64）                                                                                   |

> **迁移到另一台 Orin 时，请刷**同一档 JetPack（5.x，L4T R35.1），保证 CUDA 11.4 / TensorRT 8.4.1
> 与本文一致；TensorRT 版本不同会导致 `.engine` 不兼容（engine 必须在目标机上重建）。

______________________________________________________________________

## 2. 系统层（第一步，用 apt / 系统 pip）

```bash
# 2.1 ROS1 Noetic + 系统 py3.8 的桥接/引擎依赖
sudo apt update
sudo apt install -y ros-noetic-ros-base python3-zmq python3-msgpack

# 2.2 常用工具（git / 编译 / 进程查看；ffmpeg 在 Orin 上非必需）
sudo apt install -y git build-essential cmake pkg-config

# 2.3 系统 py3.8 的 ZeroMQ 传输依赖（bridge 与引擎服务用）
/usr/bin/python3 -m pip install --user "pyzmq==26.4.0" "msgpack==1.1.1"
```

校验：

```bash
/usr/bin/python3 -c "import zmq, msgpack; print(zmq.__version__, msgpack.__version__)"   # 26.4.0 1.1.1
/usr/src/tensorrt/bin/trtexec --version | head -1                                       # TensorRT v8401
```

______________________________________________________________________

## 3. Python 3.10 计算环境（`gqzl-py310`：推理 + 模型转换）

### 3.1 安装 miniforge（aarch64）

```bash
# 下载 aarch64 版 Miniforge3 安装到 ~/miniforge3
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-aarch64.sh
bash Miniforge3-Linux-aarch64.sh -b -p ~/miniforge3
source ~/miniforge3/etc/profile.d/conda.sh
```

### 3.2 创建环境

```bash
conda create -n gqzl-py310 python=3.10 -y
conda activate gqzl-py310
python -V                      # 3.10.x
```

### 3.3 安装依赖（用 Orin 侧仓库的清单文件）

清单在 **`lead_orin` 仓库**里（单一来源，已在本机 aarch64 验证）：

- `requirements-orin.txt` —— `gqzl-py310`（py3.10）全部依赖（含 torch / onnx / py123d 等）；
- `requirements-orin-system.txt` —— 系统 py3.8 的依赖（bridge / 引擎服务）。

```bash
cd <lead_orin 仓库>
pip install -r requirements-orin.txt        # py3.10 环境
pip install -e . --no-deps                  # 可编辑安装 lead（跳过依赖解析）
```

关键说明：

- **torch 用 CPU 版**（本机 `torch==2.5.1`，`torch.version.cuda is None`）。
  JetPack 上的 PyTorch 是 aarch64 CPU 轮子；**推理的 CUDA 在 TensorRT 引擎里**，不需要 torch CUDA。
  ```bash
  pip install torch==2.5.1 torchvision==0.20.1      # PyPI 的 aarch64 轮子
  ```
- **不要直接 `pip install py123d`**：它声明的直接依赖很多（`ray/geopandas/pyogrio/mapbox-earcut/viser/notebook/...`），
  部分在 aarch64 上没有轮子，而且 SIL 推理并不需要它们。本机是按冻结清单装**子集**
  （`py123d==0.7.0` + `av/trimesh/laspy/DracoPy/pyarrow/...`）。
- **`lead` 用 editable 安装，跳过依赖解析**：
  ```bash
  pip install -e <lead_orin 仓库路径> --no-deps
  ```
- **模型转换依赖**已包含在冻结清单里：`onnx==1.16.2`、`onnxruntime==1.23.2`、
  `onnxconverter-common==1.16.0`、`ml-dtypes==0.3.2`、`typeguard==4.6.0`
  （来源见量化仓库的 `quantization-requirements.txt`）。

______________________________________________________________________

## 4. 校验（装完必跑）

```bash
source ~/miniforge3/etc/profile.d/conda.sh && conda activate gqzl-py310
python - <<'PY'
import importlib
for m in ["lead","py123d","numba","cv2","torch","torchvision","jaxtyping",
          "omegaconf","yaml","zmq","msgpack","onnx","onnxruntime"]:
    try:
        importlib.import_module(m); print("OK  ", m)
    except Exception as e:
        print("FAIL", m, type(e).__name__, e)
PY
```

引擎可用性：

```bash
export LEAD_QUANTIZED_ENGINE=<...>/model_fp16.engine
/usr/src/tensorrt/bin/trtexec --loadEngine="$LEAD_QUANTIZED_ENGINE" \
    --iterations=50 --warmUp=10 --duration=5 | tail -5
```

端到端（真正跑）：见 [`ROS软件在环复现.md`](ROS%E8%BD%AF%E4%BB%B6%E5%9C%A8%E7%8E%AF%E5%A4%8D%E7%8E%B0.md) 的 §7 —— Orin 起
`bash sil/orin/run_orin.sh`，主机起 `bash scripts/common/run_bench2drive_orin_v2.sh`。

______________________________________________________________________

## 5. 迁移到另一台 Orin（复现步骤）

1. 刷**同档 JetPack（5.x / L4T R35.1）**，确认 CUDA 11.4 / TensorRT 8.4.1。
1. **配网络**（与主机同网段；Orin 侧，持久与临时二选一）：
   - 持久（NetworkManager，重启后自动）：
     ```bash
     sudo nmcli con add type ethernet ifname eth0 con-name sil-wired \
          ipv4.method manual ipv4.addresses 192.168.110.50/24 ipv6.method ignore
     sudo nmcli con mod sil-wired connection.autoconnect yes && sudo nmcli con up sil-wired
     ```
   - 临时（重启即失效）：`sudo ip addr add 192.168.110.50/24 dev eth0 && sudo ip link set eth0 up`
   - 主机侧与互通检查见 [`ROS软件在环复现.md`](ROS%E8%BD%AF%E4%BB%B6%E5%9C%A8%E7%8E%AF%E5%A4%8D%E7%8E%B0.md) §5。
1. 执行 **§2**（ROS Noetic + 系统 py3.8 依赖）。
1. 执行 **§3**（miniforge + `gqzl-py310` + 冻结清单 + editable 装 `lead_orin`）。
1. 拉代码（与主机**同一 commit**）：`git clone <lead_orin> && cd lead_orin && git checkout <commit>`。
1. 准备数据：
   - checkpoint 目录（`config.yaml` + **恰好一个** `model*.pth`）；
   - `.engine`：**必须在目标机上重新构建**（TensorRT 与机器绑定）。用本机的量化仓库
     `trtexec` 构建，或直接拷贝同型号/同 TensorRT 版本的 `.engine` 试跑。
1. 起 `bash sil/orin/run_orin.sh`，确认打印 `waiting for host: lead/session`。
1. 两侧 `config.yaml` 必须一致（`md5sum` 比对），否则 parity 不成立。

______________________________________________________________________

## 6. 附录

### 附录 A：版本锁定（本机实测）

| 组件                                      | 版本                                   |
| ----------------------------------------- | -------------------------------------- |
| L4T / JetPack                             | R35.1 / 5.0.2                          |
| CUDA                                      | 11.4                                   |
| TensorRT                                  | 8.4.1（`trtexec` 报 `TensorRT v8401`） |
| 系统 Python                               | 3.8.10（pyzmq 26.4.0 / msgpack 1.1.1） |
| `gqzl-py310` Python                       | 3.10.21                                |
| torch / torchvision                       | 2.5.1 / 0.20.1（**CPU**，`cuda=None`） |
| numpy                                     | 1.26.4                                 |
| py123d                                    | 0.7.0                                  |
| onnx / onnxruntime / onnxconverter-common | 1.16.2 / 1.23.2 / 1.16.0               |
| pyzmq / msgpack（3.10）                   | 27.2.0 / 1.2.2                         |
| lightning                                 | 2.6.1                                  |
| jaxtyping / beartype / wadler_lindig      | 0.3.7 / 0.22.9 / 0.1.7                 |

### 附录 B：依赖清单（单一来源）

完整清单不在本文重复，以 **`lead_orin` 仓库**为准（已在本机 aarch64 验证）：

- `lead_orin/requirements-orin.txt` —— `gqzl-py310`（py3.10）全部依赖；
- `lead_orin/requirements-orin-system.txt` —— 系统 py3.8 依赖（bridge / 引擎服务）。

安装见 §3.3 与 §2。

## 7. aarch64 常见坑

| 现象                                              | 原因 / 处理                                                                                      |
| ------------------------------------------------- | ------------------------------------------------------------------------------------------------ |
| `pip install -e .` 报 `carla/open3d/pyqt5` 无轮子 | 这三个没有 aarch64 轮子；`lead_orin` 的 `pyproject.toml` 已被同步脚本去掉它们，用 `--no-deps` 装 |
| `pip install py123d` 拖一堆编译失败               | 不要整装 py123d 的依赖；按附录 B 的冻结子集装                                                    |
| torch 装了 CUDA 版                                | JetPack 上应装 aarch64 **CPU** 版；CUDA 只在 TensorRT 引擎里用                                   |
| `import tensorrt` 在 py3.10 失败                  | TensorRT 8.4 的 Python 绑定只有**系统 py3.8**；引擎服务必须用 `/usr/bin/python3`                 |
| `.engine` 换了台机不能用                          | engine 与 TensorRT/机器绑定，**在目标机上重建**                                                  |
| `trtexec` 找不到                                  | 完整路径 `/usr/src/tensorrt/bin/trtexec`（或加到 PATH）                                          |
| 引擎服务报 `Address already in use :5562`         | 上一轮残留；先 `pkill -9 -f "[e]ngine_service.py"` 再起                                          |
