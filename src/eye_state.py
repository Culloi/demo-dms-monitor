"""眼睛睁/闭分类推理（ONNX）。

输入：左右眼两张 96×96 灰度裁剪图（与训练完全一致的裁剪与归一化）
输出：每只眼的闭眼概率；对外返回可信眼的融合值（取较小者，即两只眼都判闭才算闭）。

为什么不用 EAR：红外 940nm 下镜片强反光，实测戴眼镜时闭眼 EAR 反而高于睁眼
（0.226 vs 0.20），阈值法失效。分类模型直接学纹理，不受该问题影响。

为什么还要做「可信眼筛选」：取较小者意味着任意一只眼判闭就报闭眼。侧脸、转头
时远侧眼被透视压扁成细缝，裁剪图看起来就是一条线，模型会稳定地判成闭眼；
手举到耳侧打电话时，手部关键点还可能压进眼框把那只眼的纹理盖掉。这两种情况下
那只眼的概率不可信，必须排除，否则睁着眼也会持续报闭眼、PERCLOS 一路涨到告警。
"""

import os

import numpy as np
import onnxruntime as ort

from eye_crop import crop_eye, eye_boxes, to_input
from metrics import LEFT_EYE, RIGHT_EYE

# 双眼融合后的闭眼判定阈值（在采集数据上验证：0.5 时误报 0、闭眼漏检 1 帧）
CLOSED_PROB = 0.5

# 眼宽像素下限：低于此值说明人脸太远或完全侧过去，裁剪放大后全是糊图
MIN_EYE_PX = 14.0
# 两眼宽度比：低于此值说明画面里一只眼明显被透视压扁（侧脸），窄的那只不可信
WIDTH_RATIO = 0.65

DEBUG = os.environ.get("DMS_EYE_DEBUG") == "1"


def eye_widths(landmarks) -> list:
    """左右眼的内外眼角距离（像素），顺序与 eye_boxes 一致。"""
    out = []
    for idx in (RIGHT_EYE, LEFT_EYE):
        p = np.array([landmarks[i][:2] for i in idx], dtype=np.float32)
        out.append(float(np.linalg.norm(p[0] - p[3])))
    return out


def _occluded(box, hands) -> bool:
    """手部关键点是否盖住眼框（手挡脸时该眼纹理被覆盖，模型结果不可信）。

    要求至少 6 个点落进框内：真正的手挡眼会把十几个点压进眼框，而手掌检测
    在正脸/眼镜上偶发误报时只有零星两三个点飘进来。板端实测：正对镜头时
    两眼宽度比已经到 1.00，却因为 3 个误报点被判「手挡眼」，整段眼态全被跳过。
    """
    if not hands:
        return False
    x1, y1, x2, y2 = box
    for hand in hands:
        pts = np.asarray(hand)[:, :2]
        inside = ((pts[:, 0] >= x1) & (pts[:, 0] <= x2) &
                  (pts[:, 1] >= y1) & (pts[:, 1] <= y2))
        if int(inside.sum()) >= 6:
            return True
    return False


class EyeState:
    def __init__(self, model_path: str, threads: int = 2) -> None:
        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        opts.intra_op_num_threads = threads
        opts.inter_op_num_threads = 1
        opts.add_session_config_entry("session.intra_op.allow_spinning", "0")
        self.sess = ort.InferenceSession(model_path, sess_options=opts,
                                         providers=["CPUExecutionProvider"])
        self.input = self.sess.get_inputs()[0].name
        self.path = model_path
        self._dump = 0
        self._frames = 0

    def probs(self, frame, landmarks, hands=None, pose=None):
        """返回 (可信眼融合概率, 各眼概率列表)；没有可信眼时返回 (None, None)。

        hands 为空时不排除遮挡；有手部结果时把手压眼框的那只眼排除掉。
        pose 仅用于调试打印（pitch/yaw），不参与判定。
        """
        h, w = frame.shape[:2]
        widths = eye_widths(landmarks)
        widest = max(widths)
        if widest < MIN_EYE_PX:
            if DEBUG:
                print(f"[eye] skip: too small widths={widths}", flush=True)
            return None, None

        boxes = eye_boxes(landmarks, w, h)
        keep, reasons = [], []
        for i in range(2):
            why = ""
            if widths[i] < WIDTH_RATIO * widest:
                why += "narrow "
            if _occluded(boxes[i], hands):
                why += "hand "
            if not why:
                keep.append(i)
            reasons.append(why or "ok")
        crops = [crop_eye(frame, boxes[i])[0] for i in range(2)]
        if any(c is None for c in crops):
            return None, None

        # 调试模式下两只眼都过模型（即使已判为不可信），用来量化「单眼在侧脸
        # 角度下到底可不可信」。正式判定仍然只认 keep 里的眼。
        use = [0, 1] if DEBUG else keep
        if not use:
            return None, None
        x = np.concatenate([to_input(crops[i]) for i in use], axis=0)
        logits = self.sess.run(None, {self.input: x})[0]
        logits = logits - logits.max(axis=1, keepdims=True)
        e = np.exp(logits)
        p = e[:, 1] / e.sum(axis=1)
        p_by_eye = {i: float(p[k]) for k, i in enumerate(use)}

        # 必须两只眼都可信才判：转头/歪头时远侧眼被压成细缝（实测宽度 14~18px，
        # 另一只 46px），筛掉窄眼后剩下的那只也处在斜视角，模型会稳定输出
        # 0.94~0.98「闭眼」——板端实测 474 帧里 257 帧被判闭，全是 keep=[1]
        # 这一类。只用单眼判闭眼在侧脸完全不可靠，宁可放弃这几帧。
        if len(keep) < 2:
            if DEBUG:
                self._frames += 1
                print(f"[eye] skip: single-eye widths={widths[0]:.0f}/{widths[1]:.0f} "
                      f"reasons={reasons} hands={len(hands or [])} "
                      f"p={[round(p_by_eye[i], 3) for i in range(2)]} "
                      f"pose={'n/a' if pose is None else (round(pose[0]), round(pose[1]))}",
                      flush=True)
                if self._frames % 20 == 0:
                    self._dump_full(frame, boxes, landmarks, hands)
                self._dump_crops(crops)
            return None, None

        if DEBUG:
            tag = "" if pose is None else f" pitch={pose[0]:+.0f} yaw={pose[1]:+.0f}"
            print(f"[eye] keep widths={widths[0]:.0f}/{widths[1]:.0f} reasons={reasons} "
                  f"p={[round(p_by_eye[i], 3) for i in range(2)]} "
                  f"box={boxes[0].tolist()}/{boxes[1].tolist()}{tag}", flush=True)
            self._frames += 1
            if self._frames % 20 == 0:
                self._dump_full(frame, boxes, landmarks)
            if min(p_by_eye[i] for i in keep) > CLOSED_PROB:
                self._dump_crops(crops)
        return min(p_by_eye[i] for i in keep), [p_by_eye[i] for i in sorted(keep)]

    def _dump_crops(self, crops) -> None:
        """每 5 次落两张眼区裁剪图，用来肉眼核对裁的是不是眼睛。"""
        self._dump += 1
        if self._dump % 5:
            return
        import cv2
        import time as _t
        os.makedirs("/tmp/eye_dbg", exist_ok=True)
        ts = int(_t.time() * 1000)
        for i, c in enumerate(crops):
            cv2.imwrite(f"/tmp/eye_dbg/{ts}_{i}.png", c)

    @staticmethod
    def _dump_full(frame, boxes, landmarks, hands=None) -> None:
        """落一张整帧：画出眼框、468 点与手部点，用来判断是框的位置错了还是脸被判错。"""
        import cv2
        import time as _t
        os.makedirs("/tmp/eye_dbg", exist_ok=True)
        vis = frame.copy()
        for b in boxes:
            cv2.rectangle(vis, (b[0], b[1]), (b[2], b[3]), (0, 255, 0), 3)
        for i in range(len(landmarks)):
            cv2.circle(vis, (int(landmarks[i][0]), int(landmarks[i][1])), 1, (0, 0, 255), -1)
        for hand in (hands or []):
            for pt in np.asarray(hand)[:, :2]:
                cv2.circle(vis, (int(pt[0]), int(pt[1])), 4, (255, 0, 0), -1)
        cv2.imwrite(f"/tmp/eye_dbg/{int(_t.time() * 1000)}_full.png", vis)
