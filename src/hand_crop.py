"""手-嘴区域裁剪：抽烟识别的采集、训练与在线推理共用同一套逻辑。

抽烟的关键证据是「手把细长物体送到嘴边」，所以 ROI 取「嘴部框」与「手部外接框」
的并集（无手时退化为嘴部框），再扩边、补成正方形，缩放到 SMOKE_SIZE。

这样无论烟在手里还是已经碰到嘴唇，都落在同一块区域内；负样本（托腮、擦嘴、
喝水）也落在同一区域，模型只能靠「嘴边有没有烟」这个细节来区分。
"""

import cv2
import numpy as np

from metrics import MOUTH_LEFT, MOUTH_RIGHT

SMOKE_SIZE = 128      # 网络输入边长
MOUTH_SCALE = 1.9     # 嘴部框边长 / 嘴宽
HAND_MARGIN = 0.18    # 手部外接框外扩比例
PAD = 0.10            # 并集后再外扩的比例
MIN_ROI = 40.0        # ROI 短边下限（太远/太小丢帧）
NEAR_MOUTH = 0.30     # 手到嘴中心距离 < NEAR_MOUTH × 脸宽 视为「手在嘴边」
EAR_MIN = 0.35        # 手到耳根距离 > EAR_MIN × 脸宽 才算「不在耳侧」（排除贴耳打电话）

NORM_MEAN = (0.485, 0.456, 0.406)
NORM_STD = (0.229, 0.224, 0.225)

# FaceMesh 脸部宽度锚点（左右耳根）
FACE_RIGHT, FACE_LEFT = 234, 454


def mouth_center(landmarks) -> np.ndarray:
    return (np.array(landmarks[MOUTH_RIGHT][:2], dtype=np.float32) +
            np.array(landmarks[MOUTH_LEFT][:2], dtype=np.float32)) / 2.0


def face_width(landmarks) -> float:
    return float(np.linalg.norm(
        np.array(landmarks[FACE_RIGHT][:2], dtype=np.float32) -
        np.array(landmarks[FACE_LEFT][:2], dtype=np.float32)))


def hand_near_mouth(landmarks, hands) -> bool:
    """手是否在嘴边，且没抬到眼睛以上、也不是把手机贴在耳侧。

    板端实测（量纲都是脸宽）：抽烟时手到嘴中心 0.04~0.09、到耳根 0.47~0.59；
    打电话时手到耳根 0.10~0.31（贴耳）或 0.33~0.64（指节掠过嘴），到嘴 0.07~0.64。
    只判「离嘴近」会把贴耳打电话时掠过嘴角的指节也算成抽烟（实测 smoke 0.85），
    因此要求离嘴 < 0.30 且离耳 > 0.35：手在嘴前但不在耳侧，才是抽烟的手。
    """
    if not hands or landmarks is None:
        return False
    fw = max(face_width(landmarks), 1.0)
    mc = mouth_center(landmarks)
    eye_y = (float(landmarks[33][1]) + float(landmarks[263][1])) / 2.0
    ear_r = np.array(landmarks[FACE_RIGHT][:2], dtype=np.float32)
    ear_l = np.array(landmarks[FACE_LEFT][:2], dtype=np.float32)
    limit = NEAR_MOUTH * fw
    for hand in hands:
        pts = np.asarray(hand)[:, :2]
        if float(np.linalg.norm(pts - mc, axis=1).min()) >= limit:
            continue
        d_ear = min(float(np.linalg.norm(pts - ear_r, axis=1).min()),
                    float(np.linalg.norm(pts - ear_l, axis=1).min()))
        if d_ear < EAR_MIN * fw:      # 手机贴在耳侧，交给打电话判定
            continue
        if float(pts[:, 1].mean()) < eye_y + 0.55 * fw:
            continue
        return True
    return False


def smoke_roi(landmarks, hands, frame_w: int, frame_h: int):
    """返回手-嘴联合 ROI (x1,y1,x2,y2)；人脸关键点缺失时返回 None。"""
    if landmarks is None or len(landmarks) < 468:
        return None
    mr = np.array(landmarks[MOUTH_RIGHT][:2], dtype=np.float32)
    ml = np.array(landmarks[MOUTH_LEFT][:2], dtype=np.float32)
    mc = (mr + ml) / 2.0
    mw = float(np.linalg.norm(mr - ml))
    mside = max(mw * MOUTH_SCALE, MIN_ROI)

    x1, y1 = mc[0] - mside / 2.0, mc[1] - mside / 2.0
    x2, y2 = mc[0] + mside / 2.0, mc[1] + mside / 2.0

    if hands:
        # 只取离嘴最近的那只手：多只手时如果把所有手的外接框并集进来，
        # 误检在桌面/键盘上的手会把 ROI 撑到整帧，构图和训练样本完全不符。
        cur = [np.asarray(h, dtype=np.float32)[:, :2] for h in hands]
        nearest = min(range(len(cur)),
                      key=lambda i: float(np.linalg.norm(cur[i] - mc, axis=1).min()))
        pts = cur[nearest]
        hx1, hy1 = pts.min(axis=0)
        hx2, hy2 = pts.max(axis=0)
        hw, hh = hx2 - hx1, hy2 - hy1
        hx1, hx2 = hx1 - hw * HAND_MARGIN, hx2 + hw * HAND_MARGIN
        hy1, hy2 = hy1 - hh * HAND_MARGIN, hy2 + hh * HAND_MARGIN
        x1, y1, x2, y2 = min(x1, hx1), min(y1, hy1), max(x2, hx2), max(y2, hy2)

    # 补成正方形，避免缩放拉伸
    side = max(x2 - x1, y2 - y1)
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    side *= 1.0 + 2.0 * PAD
    x1, y1 = cx - side / 2.0, cy - side / 2.0
    x2, y2 = cx + side / 2.0, cy + side / 2.0

    return (int(max(0, round(x1))), int(max(0, round(y1))),
            int(min(frame_w - 1, round(x2))), int(min(frame_h - 1, round(y2))))


def crop_roi(frame, box, size: int = SMOKE_SIZE):
    """按框裁剪并缩放为 size×size 灰度图，返回 (crop, ok)。"""
    x1, y1, x2, y2 = box
    if x2 - x1 < 8 or y2 - y1 < 8:
        return None, False
    crop = frame[y1:y2, x1:x2]
    if crop.ndim == 3:
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    crop = cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)
    return crop, min(x2 - x1, y2 - y1) >= MIN_ROI


def to_input(crop: np.ndarray) -> np.ndarray:
    """灰度图 -> 网络输入张量 (1,3,S,S)，与训练脚本归一化一致。"""
    x = crop.astype(np.float32) / 255.0
    x = np.stack([x, x, x], axis=0)
    x = (x - np.array(NORM_MEAN, np.float32)[:, None, None]) / \
        np.array(NORM_STD, np.float32)[:, None, None]
    return x[None, :, :, :]
