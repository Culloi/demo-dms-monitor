"""打电话行为识别（ONNX 二分类）。

输入：手-耳联合 ROI 的 128×128 灰度裁剪（裁剪规则见 phone_crop.py，与采集/训练一致）
输出：打电话概率。

只在检测到人脸关键点、手在耳侧、ROI 可用时才跑模型，其余情况返回 0（EMA 衰减），
避免每帧都付一次推理开销。输出做一阶平滑后再交给 behaviors 做持续时间判定。

为什么还要模型：几何规则（手离耳近 + 抬到眼高）无法区分「贴耳打电话」与
「扶眼镜、挠头、手放耳后」，这些姿态在采集数据里作为负样本，由模型学习区分。
"""

import os
import time

import numpy as np
import onnxruntime as ort

from hand_crop import to_input
from phone_crop import crop_phone, phone_roi

# 现场排查用：置 DMS_PHONE_DEBUG=1 时逐帧打印几何门限与概率
DEBUG = os.environ.get("DMS_PHONE_DEBUG") == "1"

# 平滑系数：越大越灵敏。0.4 约等于 3~4 帧（0.4s）的等效窗口
EMA_ALPHA = 0.4
# 打电话判定阈值（与抽烟一致，验证后可按需微调）
PHONE_PROB = 0.6


class PhoneDetector:
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
        self._last = 0.0
        self._dump = 0

    def prob(self, frame, landmarks, hands, gate: bool = True) -> float:
        """返回平滑后的打电话概率；条件不满足时向 0 衰减。

        gate 由 behaviors.hand_near_ear（带高度约束）给出：只有手真的抬到
        耳/眼高度才跑模型，避免嘴边的抽烟手也进模型被误判成打电话。

        注意：这里不再重复做一次几何判断。之前引用了 phone_crop.hand_near_ear，
        那是采集脚本用的旧规则（只比距离、0.80 倍脸宽），比 behaviors 的规则
        更严且没有高度判据，导致右侧机位 gate 已经通过却被这里拦掉，模型不跑、
        概率一路衰减到 0，打电话从头到尾判不出来。
        """
        if not gate or landmarks is None or not hands:
            self._ema *= (1.0 - EMA_ALPHA)
            if DEBUG:
                print(f"[phone] gate={int(bool(gate))} hands={len(hands or [])} "
                      f"skip ema={self._ema:.3f}", flush=True)
            return self._ema
        h, w = frame.shape[:2]
        box = phone_roi(landmarks, hands, w, h)
        if box is None:
            self._ema *= (1.0 - EMA_ALPHA)
            return self._ema
        crop, _ok = crop_phone(frame, box)
        if crop is None:
            self._ema *= (1.0 - EMA_ALPHA)
            return self._ema
        logits = self.sess.run(None, {self.input: to_input(crop)})[0][0]
        logits = logits - logits.max()
        e = np.exp(logits)
        p = float(e[1] / e.sum())
        self._ema += EMA_ALPHA * (p - self._ema)
        self._last = p
        if DEBUG:
            print(f"[phone {time.strftime('%H:%M:%S')}] raw={p:.3f} ema={self._ema:.3f} "
                  f"roi={box[2] - box[0]}x{box[3] - box[1]}", flush=True)
            # 每 6 次落一张裁剪图 + 整帧，用来核对「模型看到的到底是不是手贴耳」
            self._dump += 1
            if self._dump % 6 == 0:
                import cv2
                os.makedirs("/tmp/phone_dbg", exist_ok=True)
                ts = int(time.time() * 1000)
                cv2.imwrite(f"/tmp/phone_dbg/{ts}_{p:.2f}_crop.png", crop)
                vis = frame.copy()
                cv2.rectangle(vis, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 3)
                for h in hands or []:
                    for px, py in np.asarray(h)[:, :2]:
                        cv2.circle(vis, (int(px), int(py)), 4, (255, 0, 0), -1)
                cv2.imwrite(f"/tmp/phone_dbg/{ts}_{p:.2f}_full.jpg", vis,
                            [cv2.IMWRITE_JPEG_QUALITY, 70])
        return self._ema
