"""抽烟行为识别（ONNX 二分类）。

输入：手-嘴联合 ROI 的 128×128 灰度裁剪（裁剪规则见 hand_crop.py，与采集/训练一致）
输出：抽烟概率。

只在这几个条件都满足时才跑模型：检测到人脸关键点、手在嘴边、ROI 可用。
其余情况直接返回 0，避免每帧都付一次推理开销。

输出做了一阶平滑（EMA）：手部关键点本身有抖动，逐帧概率会在阈值附近跳，
平滑后再交给 behaviors 做持续时间判定，避免告警闪烁。
"""

import os
import time

import numpy as np
import onnxruntime as ort

from hand_crop import crop_roi, hand_near_mouth, smoke_roi, to_input

# 现场排查用：置 DMS_SMOKE_DEBUG=1 时逐帧打印手部/概率，便于判断是手部跟踪还是模型的问题
DEBUG = os.environ.get("DMS_SMOKE_DEBUG") == "1"

# 平滑系数：越大越灵敏。0.4 约等于 3~4 帧（0.4s）的等效窗口
EMA_ALPHA = 0.4
# 抽烟判定阈值（在采集数据上验证：验证集召回 0.94、负样本误报 0）
SMOKE_PROB = 0.6


class SmokeDetector:
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
        self._ema = 0.0
        self._dump = 0

    def prob(self, frame, landmarks, hands) -> float:
        """返回平滑后的抽烟概率；条件不满足时返回 0。"""
        if landmarks is None or not hand_near_mouth(landmarks, hands):
            self._ema *= (1.0 - EMA_ALPHA)
            if DEBUG:
                print(f"[smoke] hands={len(hands or [])} near=0 ema={self._ema:.3f}",
                      flush=True)
            return self._ema
        h, w = frame.shape[:2]
        box = smoke_roi(landmarks, hands, w, h)
        if box is None:
            self._ema *= (1.0 - EMA_ALPHA)
            return self._ema
        crop, _ok = crop_roi(frame, box)
        if crop is None:
            self._ema *= (1.0 - EMA_ALPHA)
            return self._ema
        logits = self.sess.run(None, {self.input: to_input(crop)})[0][0]
        logits = logits - logits.max()
        e = np.exp(logits)
        p = float(e[1] / e.sum())
        self._ema += EMA_ALPHA * (p - self._ema)
        if DEBUG:
            print(f"[smoke] hands={len(hands)} near=1 raw={p:.3f} ema={self._ema:.3f} "
                  f"roi={box[2] - box[0]}x{box[3] - box[1]}", flush=True)
            self._dump += 1
            if self._dump % 3 == 0:
                import cv2
                os.makedirs("/tmp/smoke_dbg", exist_ok=True)
                ts = int(time.time() * 1000)
                x1, y1 = box[0], box[1]
                big = frame[y1:box[3], x1:box[2]].copy()
                for h in hands or []:
                    for px, py in np.asarray(h)[:, :2]:
                        cv2.circle(big, (int(px) - x1, int(py) - y1), 2, (0, 255, 0), -1)
                cv2.imwrite(f"/tmp/smoke_dbg/{ts}_roi.png", big)
                vis = frame.copy()
                cv2.rectangle(vis, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 3)
                cv2.imwrite(f"/tmp/smoke_dbg/{ts}_full.jpg", vis)
        return self._ema
