"""眼睛区域裁剪：数据采集、离线训练与在线推理共用同一套逻辑。

训练样本和推理输入必须走完全相同的裁剪（同一位置、同一尺度、同一预处理），
否则模型在板端的表现会和验证集对不上。

裁剪规则：以 6 个眼部关键点的中心为圆心，边长 = 2.2 × 眼宽（内外眼角距离），
正方形裁剪后缩放到 EYE_SIZE×EYE_SIZE 灰度图。2.2 倍能覆盖眼睑与眼角，
同时不会把眉毛、镜框大量带进来。
"""

import cv2
import numpy as np

from metrics import LEFT_EYE, RIGHT_EYE

EYE_SIZE = 96        # 网络输入边长
BOX_SCALE = 2.2      # 裁剪框边长 / 眼宽
MIN_SIDE = 20.0      # 眼宽过小（太远/侧脸）时放弃该帧，避免放大出糊图
# 归一化参数（ImageNet 均值方差，灰度图复制成 3 通道），训练脚本必须一致
NORM_MEAN = (0.485, 0.456, 0.406)
NORM_STD = (0.229, 0.224, 0.225)


def eye_boxes(landmarks, frame_w: int, frame_h: int):
    """由 FaceMesh 关键点算出左右眼方形裁剪框，返回 [(x1,y1,x2,y2), ...]。

    顺序固定为 [画面左眼(RIGHT_EYE), 画面右眼(LEFT_EYE)]，与 metrics 一致。
    """
    boxes = []
    for idx in (RIGHT_EYE, LEFT_EYE):
        pts = np.array([landmarks[i][:2] for i in idx], dtype=np.float32)
        cx, cy = pts.mean(axis=0)
        side = max(float(np.linalg.norm(pts[0] - pts[3])) * BOX_SCALE, MIN_SIDE)
        half = side / 2.0
        x1 = int(round(cx - half))
        y1 = int(round(cy - half))
        x2 = int(round(cx + half))
        y2 = int(round(cy + half))
        boxes.append((max(0, x1), max(0, y1),
                      min(frame_w - 1, x2), min(frame_h - 1, y2)))
    return boxes


def crop_eye(frame, box, size: int = EYE_SIZE):
    """按框裁剪并缩放为 size×size 灰度图，返回 (crop, ok)。"""
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    if w < 8 or h < 8:
        return None, False
    crop = frame[y1:y2, x1:x2]
    if crop.ndim == 3:
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    crop = cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)
    return crop, min(w, h) >= MIN_SIDE


def to_input(crop: np.ndarray) -> np.ndarray:
    """灰度图 -> 网络输入张量 (1,3,H,W)，与训练脚本的归一化保持一致。"""
    x = crop.astype(np.float32) / 255.0
    x = np.stack([x, x, x], axis=0)                       # 复制成 3 通道
    x = (x - np.array(NORM_MEAN, np.float32)[:, None, None]) / \
        np.array(NORM_STD, np.float32)[:, None, None]
    return x[None, :, :, :]
