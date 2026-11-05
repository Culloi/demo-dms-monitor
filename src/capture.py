"""摄像头打开与红外画面预处理。

独立成无 Qt 依赖的模块，主程序（camera.py）与离线工具（collect_eyes.py）共用，
保证训练采集与在线推理使用完全相同的预处理。
"""

import glob
import os

import cv2
import numpy as np

# 模型目录：默认取项目根的 models/（src 的上一级），可用环境变量 DMS_MODELS 覆盖
MODELS_DIR = os.environ.get(
    "DMS_MODELS",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models"),
)

_PREFERRED_KEYWORDS = ("usb", "camera", "webcam", "uvc")
_CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def looks_gray(frame) -> bool:
    """通道几乎完全一致时判定为灰度（红外）画面。抽样比较，开销可忽略。"""
    sample = frame[::8, ::8]
    b = sample[:, :, 0].astype(np.int16)
    r = sample[:, :, 2].astype(np.int16)
    return float(np.abs(r - b).mean()) < 1.0


def ir_prepare(frame, ir: bool):
    """红外/灰度适配：统一转为 3 通道；灰度画面再做 CLAHE 增强。"""
    if frame.ndim == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    if ir:
        gray = _CLAHE.apply(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    return frame


def try_open_camera(index):
    """打开指定 /dev/videoN 并按 DMS 需求配置 MJPG 1280x720。"""
    cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
    if not cap.isOpened():
        cap.release()
        return None
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    for _ in range(3):
        ok, frame = cap.read()
        if ok and frame is not None:
            return cap
    cap.release()
    return None


def all_video_nodes():
    """按名字里是否含 usb/camera 等关键字排序的 (index, name) 列表。"""
    nodes = []
    for p in glob.glob("/sys/class/video4linux/video*"):
        idx = int(p.rsplit("video", 1)[-1])
        try:
            with open(p + "/name") as f:
                name = f.read().strip()
        except OSError:
            continue
        nodes.append((idx, name))
    nodes.sort(key=lambda n: n[0])
    preferred = [n for n in nodes if any(k in n[1].lower() for k in _PREFERRED_KEYWORDS)]
    others = [n for n in nodes if n not in preferred]
    return preferred + others
