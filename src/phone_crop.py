"""手-耳区域裁剪：打电话识别的采集、训练与在线推理共用同一套逻辑。

打电话的关键证据是「手托着手机贴到耳侧」，所以 ROI 取「耳侧区域」与「手部
外接框」的并集（无手时退化为耳侧区域），再补成正方形并缩放到 PHONE_SIZE。

负样本（托腮、挠头、扶眼镜、撩头发、手放耳后）同样落在该区域内，模型只能靠
「手是否贴在耳朵上、有没有手机的轮廓」来区分。
"""

import cv2
import numpy as np

from hand_crop import crop_roi, face_width, to_input  # noqa: F401  推理侧共用

PHONE_SIZE = 128      # 网络输入边长
EAR_HALF = 0.55       # 耳侧区域半边长 / 脸宽
HAND_MARGIN = 0.18    # 手部外接框外扩比例
PAD = 0.10            # 并集后再外扩的比例
MIN_ROI = 40.0        # ROI 短边下限
NEAR_EAR = 0.80       # 手到耳侧锚点距离 < NEAR_EAR × 脸宽 视为「手在耳侧」

# FaceMesh 耳侧锚点（与 behaviors.hand_near_ear 保持一致）
EAR_RIGHT, EAR_LEFT = 234, 454


def ear_points(landmarks) -> np.ndarray:
    return np.array([landmarks[EAR_RIGHT][:2], landmarks[EAR_LEFT][:2]], dtype=np.float32)


def hand_near_ear(landmarks, hands) -> bool:
    """手（任一指/掌点）是否靠近任一侧耳根。"""
    if not hands or landmarks is None:
        return False
    ears = ear_points(landmarks)
    limit = NEAR_EAR * max(face_width(landmarks), 1.0)
    for hand in hands:
        pts = np.asarray(hand)[:, :2]
        d = np.linalg.norm(pts[:, None, :] - ears[None, :, :], axis=2)
        if float(d.min()) < limit:
            return True
    return False


def phone_roi(landmarks, hands, frame_w: int, frame_h: int):
    """返回手-耳联合 ROI (x1,y1,x2,y2)；关键点缺失时返回 None。"""
    if landmarks is None or len(landmarks) < 468:
        return None
    fw = max(face_width(landmarks), 1.0)
    ears = ear_points(landmarks)
    pts = None
    side = 0
    if hands:
        # 只取「离耳朵最近的那只手」：PalmDetection 在 0.3 阈值下偶尔会在袖子、
        # 桌面、键盘上误检出手，若把所有手的外接框并集进来，ROI 会被撑到
        # 800px 宽（正常约 500px），构图完全变样——板端实测这种畸形 ROI 会把
        # 空手贴耳打成 0.79（误报），也可能把真打电话打成 0.2（漏报）。
        cur = [np.asarray(h, dtype=np.float32)[:, :2] for h in hands]
        nearest = min(
            range(len(cur)),
            key=lambda i: float(np.linalg.norm(cur[i][:, None, :] - ears[None, :, :],
                                               axis=2).min()))
        pts = cur[nearest]
        d = np.linalg.norm(pts[:, None, :] - ears[None, :, :], axis=2)
        side = int(d.mean(axis=0).argmin())   # 取离手更近的那只耳朵

    half = max(EAR_HALF * fw, MIN_ROI / 2.0)
    ec = ears[side]
    x1, y1 = float(ec[0]) - half, float(ec[1]) - half
    x2, y2 = float(ec[0]) + half, float(ec[1]) + half

    if pts is not None:
        hx1, hy1 = pts.min(axis=0)
        hx2, hy2 = pts.max(axis=0)
        hw, hh = float(hx2 - hx1), float(hy2 - hy1)
        x1, y1 = min(x1, hx1 - hw * HAND_MARGIN), min(y1, hy1 - hh * HAND_MARGIN)
        x2, y2 = max(x2, hx2 + hw * HAND_MARGIN), max(y2, hy2 + hh * HAND_MARGIN)

    # 补成正方形，避免缩放拉伸
    side_len = max(x2 - x1, y2 - y1) * (1.0 + 2.0 * PAD)
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    return (int(max(0, round(cx - side_len / 2))), int(max(0, round(cy - side_len / 2))),
            int(min(frame_w - 1, round(cx + side_len / 2))),
            int(min(frame_h - 1, round(cy + side_len / 2))))


def crop_phone(frame, box, size: int = PHONE_SIZE):
    """按框裁剪并缩放为 size×size 灰度图，返回 (crop, ok)。"""
    return crop_roi(frame, box, size)
