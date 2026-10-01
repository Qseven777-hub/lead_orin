# sil/

本机 CARLA 与 Orin 推理之间的软件在环（SIL）通信层。

- 设计、接口契约：[`../docs/sil_ros.md`](../docs/sil_ros.md)
- Orin 侧搭建说明：[`../docs/orin_setup.md`](../docs/orin_setup.md)

## 组件

| 路径 | 运行环境 | 说明 |
|---|---|---|
| `src/lead/evaluation/sil/` | py3.10 | 契约、msgpack 编解码、ZeroMQ 传输 |
| `ros_bridge/bridge_node.py` | ROS1 Noetic (py3.8/3.9) | ROS `UInt8MultiArray` ↔ 本地 ZeroMQ |
| `tools/stub_orin_node.py` | ROS1 Noetic | 临时 Orin 替身（回固定 control） |
| `tools/local_probe.py` | py3.10 | 发一帧并等待 control |
| `run_loopback.sh` | - | 本机端到端自检（无需 CARLA/Orin） |
| `docker/` | docker/podman | 可选：把 bridge 容器化 |

## 本机自检

```bash
# 一次性：本机无 root 也能装的 ROS1 Noetic
conda create -n ros_noetic -c robostack-staging -c conda-forge \
    ros-noetic-ros-base python=3.9 msgpack-python pyzmq

bash sil/run_loopback.sh
```

## 单元测试

```bash
python -m pytest tests/unittests/evaluation/sil -q
```
