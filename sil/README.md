# sil/

本机 CARLA 与 Orin 推理之间的软件在环（SIL）通信层。

- 设计与总说明：[`../ROS软件在环.md`](../ROS%E8%BD%AF%E4%BB%B6%E5%9C%A8%E7%8E%AF.md)
- 0 基础复现手册：[`../ROS软件在环复现.md`](../ROS%E8%BD%AF%E4%BB%B6%E5%9C%A8%E7%8E%AF%E5%A4%8D%E7%8E%B0.md)

## 组件

| 路径                        | 运行环境                | 说明                                |
| --------------------------- | ----------------------- | ----------------------------------- |
| `src/lead/evaluation/sil/`  | py3.10                  | 契约、msgpack 编解码、ZeroMQ 传输   |
| `ros_bridge/bridge_node.py` | ROS1 Noetic (py3.8/3.9) | ROS `UInt8MultiArray` ↔ 本地 ZeroMQ |
| `tools/stub_orin_node.py`   | ROS1 Noetic             | 临时 Orin 替身（回固定 control）    |
| `tools/local_probe.py`      | py3.10                  | 发一帧并等待 control                |
| `run_loopback.sh`           | -                       | 本机端到端自检（无需 CARLA/Orin）   |
| `docker/`                   | docker/podman           | 可选：把 bridge 容器化              |

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
