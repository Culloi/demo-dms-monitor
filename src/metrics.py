"""DMS 指标计算：EAR（眼睛纵横比）、MAR（嘴部纵横比）、头部姿态、PERCLOS。

关键点来自 FaceMesh 的 468 点，索引沿用 MediaPipe 约定。
"""

from collections import deque
import time

import cv2
import numpy as np

# MediaPipe FaceMesh 眼睛 6 点（顺序：外角, 上1, 上2, 内角, 下2, 下1）
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
LEFT_EYE = [362, 385, 387, 263, 373, 380]

# 嘴部：13 上内唇中点, 14 下内唇中点, 61 右嘴角, 291 左嘴角
MOUTH_UPPER, MOUTH_LOWER, MOUTH_RIGHT, MOUTH_LEFT = 13, 14, 61, 291

# 头部姿态用的 6 个关键点（nose, chin, 双眼外角, 双嘴角）
POSE_IDX = [1, 152, 33, 263, 61, 291]
# 对应的 3D 人脸模型点（mm），与上面的顺序一一对应。
# 约定：x=左右(右为正)、y=上下(下为正)、z=前后(后为正，即指向头内)，
# 与图像坐标 (x右, y下) 方向一致，避免 solvePnP 出现 180° 翻转。
POSE_3D = np.array([
    (0.0, 0.0, 0.0),          # 鼻尖
    (0.0, 330.0, 65.0),       # 下巴
    (-225.0, -170.0, 135.0),  # 眼（画左）
    (225.0, -170.0, 135.0),   # 眼（画右）
    (-150.0, 150.0, 125.0),   # 嘴角（画左）
    (150.0, 150.0, 125.0),    # 嘴角（画右）
], dtype=np.float64)


def _pts(landmarks: np.ndarray, idx) -> np.ndarray:
    return np.array([[landmarks[i][0], landmarks[i][1]] for i in idx], dtype=np.float64)


def _ear_one_eye(landmarks: np.ndarray, eye_idx) -> float:
    p = _pts(landmarks, eye_idx)
    v1 = np.linalg.norm(p[1] - p[5])
    v2 = np.linalg.norm(p[2] - p[4])
    h = np.linalg.norm(p[0] - p[3])
    if h < 1e-6:
        return 0.0
    return (v1 + v2) / (2.0 * h)


def eye_aspect_ratio(landmarks: np.ndarray) -> float:
    """双眼 EAR 平均值。"""
    return (_ear_one_eye(landmarks, RIGHT_EYE) + _ear_one_eye(landmarks, LEFT_EYE)) / 2.0


def mouth_aspect_ratio(landmarks: np.ndarray) -> float:
    """嘴部纵横比 = 内唇上下距离 / 嘴角水平距离。"""
    upper = np.array(landmarks[MOUTH_UPPER][:2], dtype=np.float64)
    lower = np.array(landmarks[MOUTH_LOWER][:2], dtype=np.float64)
    right = np.array(landmarks[MOUTH_RIGHT][:2], dtype=np.float64)
    left = np.array(landmarks[MOUTH_LEFT][:2], dtype=np.float64)
    width = np.linalg.norm(right - left)
    if width < 1e-6:
        return 0.0
    return np.linalg.norm(upper - lower) / width


def head_pose(landmarks: np.ndarray, frame_w: int, frame_h: int):
    """用 solvePnP 估计头部姿态，返回 (pitch, yaw, roll)（度）。

    pitch=点头(上下)，yaw=左右转头，roll=画面内倾斜。
    受通用 3D 模型与真实人脸比例差异影响，正视前方时各角度存在一个近似
    恒定的零点偏置，由上层的行为判定做零位校准后再使用。
    """
    image_pts = _pts(landmarks, POSE_IDX)
    focal = float(frame_w)
    center = (frame_w / 2.0, frame_h / 2.0)
    cam = np.array([[focal, 0, center[0]],
                    [0, focal, center[1]],
                    [0, 0, 1]], dtype=np.float64)
    dist = np.zeros((4, 1))
    ok, rvec, _ = cv2.solvePnP(POSE_3D, image_pts, cam, dist,
                               flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return 0.0, 0.0, 0.0
    rot, _ = cv2.Rodrigues(rvec)
    # 旋转矩阵分解为欧拉角（度）。
    # solvePnP 得到的是 模型->相机 的旋转；在模型轴向 x=左右/y=上下/z=前后 的
    # 约定下，绕 x/y/z 的旋转分别对应 点头(pitch) / 左右转头(yaw) / 画面内倾斜(roll)。
    sy = np.sqrt(rot[0, 0] ** 2 + rot[1, 0] ** 2)
    if sy > 1e-6:
        pitch = np.degrees(np.arctan2(rot[2, 1], rot[2, 2]))
        yaw = np.degrees(np.arctan2(-rot[2, 0], sy))
        roll = np.degrees(np.arctan2(rot[1, 0], rot[0, 0]))
    else:
        pitch = np.degrees(np.arctan2(rot[2, 1], rot[2, 2]))
        yaw = 0.0
        roll = np.degrees(np.arctan2(-rot[1, 2], rot[1, 1]))
    return float(pitch), float(yaw), float(roll)


class Perclos:
    """滑动时间窗内的闭眼时间占比（PERCLOS）。"""

    def __init__(self, window_sec: float = 30.0, min_samples: int = 1) -> None:
        self.window = window_sec
        self.min_samples = min_samples
        self._samples = deque()  # (timestamp, closed: bool)

    def update(self, closed: bool) -> float:
        now = time.time()
        self._samples.append((now, closed))
        while self._samples and now - self._samples[0][0] > self.window:
            self._samples.popleft()
        # 样本太少时占比不可靠（例如启动瞬间或眨眼），先返回 0
        if len(self._samples) < self.min_samples:
            return 0.0
        closed_n = sum(1 for _, c in self._samples if c)
        return closed_n / len(self._samples)

    def reset(self) -> None:
        self._samples.clear()