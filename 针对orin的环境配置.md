# 针对 Orin 的环境配置（aarch64，从白板机）

本文档面向**一台全新的 NVIDIA Orin（aarch64）**，从零把软件在环（SIL）所需的**三套环境**
配好，并保证迁移到**另一台 Orin** 也能复现。

> 只想跑通流程看 [`ROS软件在环复现.md`](ROS软件在环复现.md)；本文只讲 Orin 这台机器的环境怎么搭。

---

## 0. 为什么 Orin 上要三套东西

Orin 上同时存在**两个 Python**，这是硬约束，不是选择：

| 环境 | Python | 作用 | 为什么单独 |
| --- | --- | --- | --- |
| **系统环境** | 3.8.10（系统自带） | ROS Noetic、`ros_bridge`、**TensorRT 引擎服务** | TensorRT 8.4 的 Python 绑定只有系统 py3.8 有 |
| **`gqzl-py310`** | 3.10 | `lead` 特征/控制逻辑、`agent_node`、**模型转换** | `lead` 要求 Python ≥ 3.10 |

- **推理**：`agent_node`（py3.10）通过本地 ZeroMQ 调**引擎服务**（py3.8 + TensorRT）跑网络前向。
- **模型转换**：ONNX 导出 / TensorRT 图降级 / engine 构建，也在 `gqzl-py310` 里做（用 `trtexec`）。

---

## 1. 目标机器的基线（本机实测，作为参照）

| 项 | 值 |
| --- | --- |
| 硬件 | NVIDIA Jetson AGX Orin（aarch64） |
| 系统 | Ubuntu 20.04 + **JetPack 5.0.2 / L4T R35.1**（`/etc/nv_tegra_release` 显示 `R35 (release), REVISION: 1.0`） |
| CUDA | **11.4**（`/usr/local/cuda-11.4`） |
| TensorRT | **8.4.1**（apt，`+cuda11.4`），`trtexec` 在 `/usr/src/tensorrt/bin/trtexec` |
| 系统 Python | 3.8.10 |
| ROS | ROS1 Noetic（原生 apt） |
| miniforge | `~/miniforge3`（aarch64） |

> **迁移到另一台 Orin 时，请刷**同一档 JetPack（5.x，L4T R35.1），保证 CUDA 11.4 / TensorRT 8.4.1
> 与本文一致；TensorRT 版本不同会导致 `.engine` 不兼容（engine 必须在目标机上重建）。

---

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

---

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

### 3.3 安装依赖（推荐：按本机冻结清单，aarch64 已验证）

把下面 **§6 附录 B** 的清单存成 `orin-requirements.txt`（**删掉 `lead==...` 那一行**），然后：

```bash
pip install -r orin-requirements.txt
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

---

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

端到端（真正跑）：见 [`ROS软件在环复现.md`](ROS软件在环复现.md) 的 §7 —— Orin 起
`bash sil/orin/run_orin.sh`，主机起 `bash scripts/common/run_bench2drive_orin_v2.sh`。

---

## 5. 迁移到另一台 Orin（复现步骤）

1. 刷**同档 JetPack（5.x / L4T R35.1）**，确认 CUDA 11.4 / TensorRT 8.4.1。
2. 执行 **§2**（ROS Noetic + 系统 py3.8 依赖）。
3. 执行 **§3**（miniforge + `gqzl-py310` + 冻结清单 + editable 装 `lead_orin`）。
4. 拉代码（与主机**同一 commit**）：`git clone <lead_orin> && cd lead_orin && git checkout <commit>`。
5. 准备数据：
   - checkpoint 目录（`config.yaml` + **恰好一个** `model*.pth`）；
   - `.engine`：**必须在目标机上重新构建**（TensorRT 与机器绑定）。用本机的量化仓库
     `trtexec` 构建，或直接拷贝同型号/同 TensorRT 版本的 `.engine` 试跑。
6. 起 `bash sil/orin/run_orin.sh`，确认打印 `waiting for host: lead/session`。
7. 两侧 `config.yaml` 必须一致（`md5sum` 比对），否则 parity 不成立。

---

## 6. 附录

### 附录 A：版本锁定（本机实测）

| 组件 | 版本 |
| --- | --- |
| L4T / JetPack | R35.1 / 5.0.2 |
| CUDA | 11.4 |
| TensorRT | 8.4.1（`trtexec` 报 `TensorRT v8401`） |
| 系统 Python | 3.8.10（pyzmq 26.4.0 / msgpack 1.1.1） |
| `gqzl-py310` Python | 3.10.21 |
| torch / torchvision | 2.5.1 / 0.20.1（**CPU**，`cuda=None`） |
| numpy | 1.26.4 |
| py123d | 0.7.0 |
| onnx / onnxruntime / onnxconverter-common | 1.16.2 / 1.23.2 / 1.16.0 |
| pyzmq / msgpack（3.10） | 27.2.0 / 1.2.2 |
| lightning | 2.6.1 |
| jaxtyping / beartype / wadler_lindig | 0.3.7 / 0.22.9 / 0.1.7 |

### 附录 B：`gqzl-py310` 完整 `pip freeze`（本机 96 项）

> 迁移时以此为准。删掉 `lead==1.5.0`（用 `pip install -e <lead_orin> --no-deps` 代替）。

```
antlr4-python3-runtime==4.9.3
attrs==26.1.0
av==15.1.0
beartype==0.22.9
cachetools==7.2.0
certifi==2026.7.22
charset-normalizer==3.5.1
click==8.5.0
coloredlogs==15.0.1
colorlog==6.12.0
contourpy==1.3.2
cycler==0.12.1
DracoPy==2.1.0
exceptiongroup==1.3.1
filelock==3.32.5
filterpy==1.4.5
flatbuffers==25.12.19
fonttools==4.65.0
fsspec==2026.7.0
gmpy2==2.3.1
hf-xet==1.6.0
huggingface_hub==0.36.2
humanfriendly==10.0
hydra-colorlog==1.2.0
hydra-core==1.3.2
idna==3.20
imagecodecs==2025.3.30
importlib_metadata==8.7.1
iniconfig==2.3.0
jaxtyping==0.3.7
Jinja2==3.1.6
jsonschema==4.26.0
jsonschema-specifications==2025.9.1
kiwisolver==1.5.1
laspy==2.7.0
lazrs==0.8.2
lead==1.5.0
lightning==2.6.1
lightning-utilities==0.15.3
llvmlite==0.49.0
MarkupSafe==3.0.3
matplotlib==3.10.9
ml-dtypes==0.3.2
mpmath==1.3.0
msgpack==1.2.2
networkx==3.4.2
numba==0.67.0
numpy==1.26.4
omegaconf==2.3.0
onnx==1.16.2
onnxconverter-common==1.16.0
onnxruntime==1.23.2
opencv-python==4.10.0.84
opentelemetry-api==1.44.0
packaging==26.3
pandas==2.3.3
pillow==12.0.0
pip==26.2.1
pluggy==1.6.0
protobuf==7.36.2
psutil==7.2.2
py123d==0.7.0
pyarrow==25.0.1
Pygments==2.21.0
pyparsing==3.3.3
pyquaternion==0.9.9
pytest==9.1.1
python-dateutil==2.9.0.post0
pytz==2026.4
PyYAML==6.0.3
pyzmq==27.2.0
ray==2.58.0
referencing==0.37.0
requests==2.34.2
rpds-py==0.30.0
safetensors==0.8.0
scipy==1.15.3
setuptools==83.0.0
shapely==2.1.2
six==1.17.0
sympy==1.14.0
timm==1.0.25
tomli==2.4.1
torch==2.5.1
torchmetrics==1.9.0
torchvision==0.20.1
tqdm==4.70.1
trimesh==4.11.5
typeguard==4.6.0
typing_extensions==4.16.0
tzdata==2026.4
urllib3==2.8.0
wadler_lindig==0.1.7
wheel==0.47.0
zipp==4.1.0
```

---

## 7. aarch64 常见坑

| 现象 | 原因 / 处理 |
| --- | --- |
| `pip install -e .` 报 `carla/open3d/pyqt5` 无轮子 | 这三个没有 aarch64 轮子；`lead_orin` 的 `pyproject.toml` 已被同步脚本去掉它们，用 `--no-deps` 装 |
| `pip install py123d` 拖一堆编译失败 | 不要整装 py123d 的依赖；按附录 B 的冻结子集装 |
| torch 装了 CUDA 版 | JetPack 上应装 aarch64 **CPU** 版；CUDA 只在 TensorRT 引擎里用 |
| `import tensorrt` 在 py3.10 失败 | TensorRT 8.4 的 Python 绑定只有**系统 py3.8**；引擎服务必须用 `/usr/bin/python3` |
| `.engine` 换了台机不能用 | engine 与 TensorRT/机器绑定，**在目标机上重建** |
| `trtexec` 找不到 | 完整路径 `/usr/src/tensorrt/bin/trtexec`（或加到 PATH） |
| 引擎服务报 `Address already in use :5562` | 上一轮残留；先 `pkill -9 -f "[e]ngine_service.py"` 再起 |
