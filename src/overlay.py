"""小窗预览叠加：人脸框、眼部/嘴部关键点、手部骨架（不含文字，指标由 Qt 绘制）。"""

import cv2
import numpy as np

from metrics import LEFT_EYE, MOUTH_LEFT, MOUTH_LOWER, MOUTH_RIGHT, MOUTH_UPPER, RIGHT_EYE
from models_mp import HAND_CONNECTIONS

LEVEL_COLOR = {0: (110, 220, 150), 1: (60, 190, 250), 2: (72, 72, 240)}


def draw_mini(frame: np.ndarray, result: dict, scale: float = 1.0) -> np.ndarray:
    """在已缩放的画面上绘制轻量标注，坐标按 scale 从原图映射。"""
    state = result.get("state", {})
    color = LEVEL_COLOR.get(state.get("level", 0), LEVEL_COLOR[0])

    bbox = result.get("bbox")
    if bbox is not None:
        x1, y1, x2, y2 = [int(round(v * scale)) for v in bbox]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)

    landmarks = result.get("landmarks")
    if landmarks is not None:
        # 四舍五入而不是截断：预览缩放比约 0.31，截断会让相邻两帧的点整体偏 1px，
        # 看起来像在抖。
        for idx in list(RIGHT_EYE) + list(LEFT_EYE) + [
                MOUTH_UPPER, MOUTH_LOWER, MOUTH_RIGHT, MOUTH_LEFT]:
            x = int(round(landmarks[idx][0] * scale))
            y = int(round(landmarks[idx][1] * scale))
            cv2.circle(frame, (x, y), 1, (255, 214, 120), -1, cv2.LINE_AA)

    for hand in result.get("hands") or []:
        pts = np.rint(np.asarray(hand)[:, :2] * scale).astype(int)
        for a, b in HAND_CONNECTIONS:
            cv2.line(frame, tuple(pts[a]), tuple(pts[b]), (150, 224, 170), 1, cv2.LINE_AA)
        for x, y in pts:
            cv2.circle(frame, (int(x), int(y)), 1, (0, 209, 255), -1, cv2.LINE_AA)
    return frame
