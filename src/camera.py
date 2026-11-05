"""DMS 采集、分析线程与预览渲染。

线程划分（预览与分析解耦，推理不再阻塞预览）：
- 采集线程（本 QThread.run）：读帧 -> 红外增强 -> 缩放成预览 + 绘制标注 -> 发给 UI
- 人脸线程：BlazeFace / FaceMesh / 眼态 / 抽烟 / 打电话 / 行为状态机，约 12 次/秒
- 手部线程：BlazePalm + 手部 21 点，开销大，每 HAND_INTERVAL 秒一次

为什么这样拆：板端实测手掌检测 ~105ms、手部 21 点 ~75ms，原来和采集在同一个
循环里，每 0.4s 就有近 0.2s 完全不读摄像头；而且预览只在「分析帧」才发一次，
实际预览只有 5~7fps。拆开后采集线程只做轻量缩放和画线，预览稳定在摄像头帧率，
人脸关键点保持 12Hz 更新，手部维持原来的 2.5Hz（行为判定是秒级的，够用）。
"""

import os
import threading
import time

import cv2
import numpy as np
from PySide6.QtCore import QThread, Signal
from PySide6.QtGui import QImage

import models_mp
from behaviors import DmsAnalyzer, hand_near_ear
from capture import MODELS_DIR, all_video_nodes, ir_prepare, looks_gray, try_open_camera
from metrics import eye_aspect_ratio, head_pose, mouth_aspect_ratio
from overlay import draw_mini

# 红外摄像头适配开关：auto=按画面自动判断（三通道完全相同即视为红外灰度），
# 1=强制开启，0=强制关闭。红外/灰度画面统一做 CLAHE 增强，稳定关键点。
IR_MODE = os.environ.get("DMS_IR", "auto").lower()

# 预览小窗的图像宽度（像素）。面板里的视频区约 372px，取 400 基本 1:1，
# 再大只是白付缩放和拷贝的开销。
PREVIEW_W = int(os.environ.get("DMS_PREVIEW_W", "400"))

_EMPTY_STATE = {"level": 0, "messages": [], "perclos": 0.0, "ear": 0.0,
                "eye_prob": None, "mar": 0.0, "pitch": 0.0, "yaw": 0.0,
                "roll": 0.0, "smoke": 0.0, "phone_prob": None}


class DmsThread(QThread):
    frameReady = Signal(object, object)   # (QImage 预览图, 结果字典)
    failed = Signal(str)
    log = Signal(str)
    opened = Signal(int, str)

    MAX_READ_FAILS = 30
    # 人脸分析目标频率：12Hz 对秒级行为判定绰绰有余，同时给 Qt 渲染留出 CPU
    ANALYZE_HZ = float(os.environ.get("DMS_ANALYZE_HZ", "12"))
    # 手部检测（Palm+Hand 单次约 180ms CPU）开销大：按时间间隔抽帧并缓存，
    # 只在检出人脸时才跑。打电话判定是秒级行为，0.4s 一次的刷新率足够。
    HAND_INTERVAL = float(os.environ.get("DMS_HAND_INTERVAL", "0.4"))
    # 手持手机贴耳时手掌是握拳/侧掌，BlazePalm 的置信度经常掉到 0.5 以下，
    # 板端实测一半的帧检不到手。阈值放到 0.3 提高召回。
    PALM_THRESHOLD = float(os.environ.get("DMS_PALM_THRESHOLD", "0.3"))
    # 检不到手时不要立刻清空：手部结果本来就是 0.4s 缓存一次的，
    # 一次漏检清空会让打电话/抽烟的判定门限反复开关、计时永远攒不够。
    # 保留最近一次成功结果 HAND_GRACE 秒，超过才认为手真的离开了。
    HAND_GRACE = float(os.environ.get("DMS_HAND_GRACE", "1.0"))
    # 眼态只在「正脸对着摄像头」时判。低头看屏幕时刘海和镜框会挡住眼区，
    # 眼框裁到的其实是头发/皮肤，模型会把这种纹理稳定判成闭眼——板端实测
    # 低头 20° 睁眼被判 0.95 以上，连续一秒就误报 DROWSINESS。
    # 这里用原始俯仰角做门限（摄像头固定安装，正视时约 0~15°）。
    EYE_MAX_PITCH = float(os.environ.get("DMS_EYE_MAX_PITCH", "18"))
    EYE_MAX_YAW = float(os.environ.get("DMS_EYE_MAX_YAW", "45"))

    # ---- 人脸跟踪 ----
    # 检到一次人脸后，下一帧直接用上一帧的关键点框跑 FaceMesh，省掉 BlazeFace。
    # BlazeFace 在 ARM 上约 12~15ms，而且逐帧检测的框会抖，导致关键点跟着跳。
    TRACK_MAX_AGE = 0.5          # 超过这么久没更新跟踪框就重新检测
    TRACK_REFRESH_SEC = 1.0      # 跟踪期间每隔 1s 重跑一次 BlazeFace 校正漂移
    # FaceMesh 自带的人脸存在置信度：真脸实测 0.99~1.00（含手挡嘴/抽烟姿势）。
    # 用它做二次校验，可以把 BlazeFace 低阈值（0.3）放进来的误检挡掉。
    MESH_MIN_SCORE = float(os.environ.get("DMS_MESH_MIN_SCORE", "0.5"))
    # 关键点平滑系数：12Hz 下 0.7 约等于 80ms 时间常数，既压住抖动又不拖影
    SMOOTH_ALPHA = float(os.environ.get("DMS_SMOOTH_ALPHA", "0.7"))
    # 手部结果 0.4s 才更新一次，直接画会让骨架每 0.4s 跳一下；
    # 预览时向最新结果插值，让手部骨架连续滑动。
    HAND_DRAW_ALPHA = 0.35
    # 手掌检测逐次结果会有十几像素的跳变（握拳贴耳时手掌被手机/耳廓遮挡，
    # 每次检到的关键点略有不同）。手部点直接参与打电话/抽烟的几何门限和
    # ROI 裁剪，抖动会让 ROI 忽大忽小、模型输出在 0.2~0.8 之间乱跳。
    # 对关键点做一阶平滑后再进门限和裁剪。
    HAND_SMOOTH_ALPHA = float(os.environ.get("DMS_HAND_SMOOTH", "0.5"))

    def __init__(self, index=None, parent=None) -> None:
        super().__init__(parent)
        self.index = index
        self._running = True
        self._face = None
        self._mesh = None
        self._palm = None
        self._hand = None
        self.analyzer = DmsAnalyzer()
        # 跨线程共享：_frame 由采集线程写、分析线程读；_result/_last_hands 反之
        self._lock = threading.Lock()
        self._frame = None
        self._frame_id = 0
        self._result = {"face": False, "landmarks": None, "hands": None,
                        "bbox": None, "state": dict(_EMPTY_STATE)}
        self._last_hands = []
        self._hand_t = 0.0
        self._hand_ok_t = 0.0
        # 人脸跟踪状态（只在人脸线程里读写）
        self._track_bbox = None
        self._track_t = 0.0
        self._detect_t = 0.0
        self._smooth_lm = None
        self._smooth_n = 0
        self._draw_hands = []     # 预览用手部骨架（插值后的中间值）
        # IR 判定：env 显式指定 1/0 则锁定，auto 时在首帧按通道是否相同判定
        self._ir = {"1": True, "0": False}.get(IR_MODE)
        self._dump_noface = os.environ.get("DMS_DUMP_NOFACE") == "1"
        self._noface_t = 0.0

    def stop(self) -> None:
        self._running = False
        self.wait(3000)

    def _prepare(self, frame):
        """红外/灰度适配：统一转为 3 通道；灰度画面再做 CLAHE 增强。"""
        if frame.ndim == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        if self._ir is None:
            self._ir = looks_gray(frame)
            self.log.emit(f"Infrared/grayscale stream: {'yes' if self._ir else 'no'}")
        return ir_prepare(frame, self._ir)

    # ---------- 模型 ----------
    def _load_models(self) -> None:
        self.log.emit("Loading models ...")
        self._face = models_mp.BlazeFace(os.path.join(MODELS_DIR, "face_detection_short_range.onnx"))
        self._mesh = models_mp.FaceMesh(os.path.join(MODELS_DIR, "face_mesh_Nx3x192x192.onnx"))
        self._palm = models_mp.PalmDetection(os.path.join(MODELS_DIR, "palm_detection.onnx"))
        self._hand = models_mp.HandLandmark(os.path.join(MODELS_DIR, "hand_landmark.onnx"))
        # 眼态分类模型可选：缺失时 behaviors 自动回退到 EAR 阈值
        eye_path = os.path.join(MODELS_DIR, "eye_state.onnx")
        self._eye = None
        if os.path.exists(eye_path):
            from eye_state import EyeState
            self._eye = EyeState(eye_path)
            self.log.emit("Eye-state model loaded")
        else:
            self.log.emit("Eye-state model missing, fallback to EAR")
        # 抽烟分类模型可选：缺失时不做抽烟判定（不影响其他行为）
        smoke_path = os.path.join(MODELS_DIR, "smoke.onnx")
        self._smoke = None
        if os.path.exists(smoke_path):
            from smoke import SmokeDetector
            self._smoke = SmokeDetector(smoke_path)
            self.log.emit("Smoke model loaded")
        else:
            self.log.emit("Smoke model missing, smoking detection disabled")
        # 打电话分类模型可选：缺失时回退到纯几何判定（hand_near_ear）
        phone_path = os.path.join(MODELS_DIR, "phone.onnx")
        self._phone = None
        if os.path.exists(phone_path):
            from phone import PhoneDetector
            self._phone = PhoneDetector(phone_path)
            self.log.emit("Phone model loaded")
        else:
            self.log.emit("Phone model missing, fallback to geometric rule")
        self.log.emit("Models loaded")

    # ---------- 摄像头 ----------
    def _find_camera(self):
        for idx, name in all_video_nodes():
            if not self._running:
                return None, -1, ""
            self.log.emit(f"Trying /dev/video{idx} ({name}) ...")
            cap = try_open_camera(idx)
            if cap is not None:
                return cap, idx, name
            self.log.emit(f"  /dev/video{idx} not usable")
        return None, -1, ""

    # ---------- 人脸与关键点 ----------
    def _mesh_on(self, frame, bbox):
        """在给定框上跑 FaceMesh，并用模型自带的人脸置信度过滤误检。"""
        landmarks, scores = self._mesh.predict(frame, [bbox], margin=0.35)
        if len(landmarks) and float(scores[0]) >= self.MESH_MIN_SCORE:
            return landmarks[0]
        return None

    def _set_track(self, lm, now: float) -> None:
        """用关键点外接框更新跟踪框（比检测框更贴合人脸，下一帧 ROI 更准）。"""
        x, y = lm[:, 0], lm[:, 1]
        self._track_bbox = np.array([x.min(), y.min(), x.max(), y.max()], np.float32)
        self._track_t = now

    def _face_landmarks(self, frame):
        """返回 (468 点或 None, 人脸框或 None)。先跟踪、必要时再检测。"""
        now = time.monotonic()
        tracking = (self._track_bbox is not None
                    and now - self._track_t < self.TRACK_MAX_AGE)
        need_detect = not tracking or now - self._detect_t >= self.TRACK_REFRESH_SEC

        # 快路径：人脸还在上一帧的位置，直接跑网格
        if tracking and not need_detect:
            lm = self._mesh_on(frame, self._track_bbox)
            if lm is not None:
                self._set_track(lm, now)
                return lm, self._track_bbox

        # 人脸检测：先按 0.5 找，找不到再降到 0.3（手/烟挡下半张脸时置信度会掉）
        bboxes, kps, _scores = self._face.detect(frame, threshold=0.5)
        low_conf = False
        if len(bboxes) == 0:
            bboxes, kps, _scores = self._face.detect(frame, threshold=0.3)
            low_conf = True

        if len(bboxes) > 0:
            # 取面积最大的人脸作为驾驶员
            areas = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
            i = int(areas.argmax())
            landmarks, mesh_scores = self._mesh.predict(frame, [bboxes[i]],
                                                        keypoints=[kps[i]])
            if len(landmarks) and float(mesh_scores[0]) >= self.MESH_MIN_SCORE:
                lm = landmarks[0]
                self._detect_t = now
                self._set_track(lm, now)
                self._smooth_lm = None    # 重新检测到人脸，平滑器重新起步
                return lm, bboxes[i]
            if low_conf:
                # 低阈值检测 + 网格置信度不足 = 大概率是误检（家具纹理、反光）
                self._track_bbox = None
                return None, None

        # 检测失败但跟踪还没过期：用旧框再试一次（真丢脸时网格置信度会掉下去）
        if tracking and self._track_bbox is not None:
            lm = self._mesh_on(frame, self._track_bbox)
            if lm is not None:
                self._set_track(lm, now)
                return lm, self._track_bbox

        self._track_bbox = None
        return None, None

    def _smooth(self, lm):
        """关键点一阶平滑：压住网格输出的逐帧抖动，让绘制和指标都更稳。"""
        prev = self._smooth_lm
        if prev is None or prev.shape != lm.shape:
            self._smooth_lm = lm.copy()
            self._smooth_n = 0
            return lm
        alpha = min(0.95, self.SMOOTH_ALPHA + 0.25) if self._smooth_n < 3 \
            else self.SMOOTH_ALPHA
        out = prev + alpha * (lm - prev)
        self._smooth_lm = out
        self._smooth_n += 1
        return out

    # ---------- 单帧分析（人脸线程） ----------
    def _smooth_hands(self, hands):
        """手部关键点平滑：按手掌质心匹配上一轮结果，避免 ROI 逐次跳变。"""
        prev = self._last_hands
        if len(prev) != len(hands):
            return [np.asarray(h, np.float32).copy() for h in hands]
        out = []
        for hand in hands:
            cur = np.asarray(hand, np.float32)
            i = int(np.argmin([np.linalg.norm(p[:, :2].mean(axis=0)
                                              - cur[:, :2].mean(axis=0)) for p in prev]))
            base = prev[i]
            if base.shape != cur.shape:
                out.append(cur.copy())
            else:
                out.append(base + self.HAND_SMOOTH_ALPHA * (cur - base))
        return out

    def _analyze(self, frame):
        h, w = frame.shape[:2]
        result = {"face": False, "landmarks": None, "hands": None, "bbox": None}

        lm, bbox = self._face_landmarks(frame)
        if lm is not None:
            lm = self._smooth(lm)
            with self._lock:
                hands = self._last_hands
            ear = eye_aspect_ratio(lm)
            mar = mouth_aspect_ratio(lm)
            pitch, yaw, roll = head_pose(lm, w, h)
            phone = hand_near_ear(lm, hands)
            prob = None
            if self._eye is not None:
                if abs(pitch) <= self.EYE_MAX_PITCH and abs(yaw) <= self.EYE_MAX_YAW:
                    prob = self._eye.probs(frame, lm, hands, pose=(pitch, yaw, roll))[0]
                else:
                    # 低头/大角度转头：眼区被刘海或镜框遮挡，不判眼态。
                    # 姿态本身会由 behaviors 的分心逻辑给出告警。
                    prob = 0.0
                if prob is None:
                    # 侧脸/手挡眼时模型拒判，按「没有闭眼证据」处理，不要退回
                    # EAR（戴眼镜时 EAR 与睁闭眼的关系是反的）。
                    prob = 0.0
            smoke = self._smoke.prob(frame, lm, hands) if self._smoke is not None else 0.0
            phone_prob = None
            if self._phone is not None:
                phone_prob = self._phone.prob(frame, lm, hands, gate=phone)
            state = self.analyzer.update(ear, mar, pitch, yaw, roll, phone,
                                         closed_prob=prob, smoke_prob=smoke,
                                         phone_prob=phone_prob)
            result.update({"face": True, "landmarks": lm, "bbox": bbox,
                           "hands": hands, "state": state})
        else:
            self._smooth_lm = None
            self.analyzer.reset()
            result["state"] = self.analyzer.no_face()
        return result

    # ---------- 线程 ----------
    def _face_loop(self) -> None:
        """人脸/行为分析线程：始终处理最新帧，并按 ANALYZE_HZ 限速。"""
        last_id = -1
        while self._running:
            with self._lock:
                frame, fid = self._frame, self._frame_id
            if frame is None or fid == last_id:
                time.sleep(0.004)
                continue
            last_id = fid
            t0 = time.monotonic()
            try:
                result = self._analyze(frame)
            except Exception as e:  # 单帧异常不应终止线程
                self.log.emit(f"Analyze error: {e}")
                result = {"face": False, "landmarks": None, "hands": None,
                          "bbox": None, "state": dict(_EMPTY_STATE)}
            # 调试：DMS_DUMP_NOFACE=1 时把「没检出人脸」的帧每 2 秒存一张，
            # 用来区分是画面本身有问题还是检测器漏检。
            if self._dump_noface and not result.get("face"):
                now = time.monotonic()
                if now - self._noface_t > 2.0:
                    self._noface_t = now
                    os.makedirs("/tmp/noface", exist_ok=True)
                    cv2.imwrite(f"/tmp/noface/{int(time.time() * 1000)}.jpg", frame,
                                [cv2.IMWRITE_JPEG_QUALITY, 70])
            with self._lock:
                self._result = result
            delay = 1.0 / self.ANALYZE_HZ - (time.monotonic() - t0)
            if delay > 0:
                time.sleep(delay)

    def _hand_loop(self) -> None:
        """手部检测线程：低频、高开销，结果缓存给行为判定复用。"""
        while self._running:
            with self._lock:
                frame = self._frame
                face = self._result.get("face")
            if frame is None or not face:
                time.sleep(0.05)
                continue
            now = time.monotonic()
            if now - self._hand_t < self.HAND_INTERVAL:
                time.sleep(0.02)
                continue
            self._hand_t = now
            try:
                palms = self._palm.detect(frame, threshold=self.PALM_THRESHOLD)
                hand_lm, _presence, _handed = self._hand.predict(frame, palms)
            except Exception as e:
                self.log.emit(f"Hand error: {e}")
                continue
            if len(hand_lm):
                hands = self._smooth_hands([hand_lm[i] for i in range(len(hand_lm))])
                with self._lock:
                    self._last_hands = hands
                self._hand_ok_t = now
            elif now - self._hand_ok_t > self.HAND_GRACE:
                with self._lock:
                    self._last_hands = []

    # ---------- 采集线程主循环 ----------
    def run(self) -> None:
        try:
            self._load_models()
        except Exception as e:
            self.failed.emit(f"Model load failed: {e}")
            return

        if self.index is not None:
            self.log.emit(f"Opening /dev/video{self.index} ...")
            cap = try_open_camera(self.index)
            idx, name = self.index, f"video{self.index}"
            if cap is None:
                self.failed.emit(f"Cannot open /dev/video{self.index}")
                return
        else:
            cap, idx, name = self._find_camera()
            if cap is None:
                self.failed.emit("No working camera found")
                return

        self.opened.emit(idx, name)
        for target in (self._face_loop, self._hand_loop):
            threading.Thread(target=target, daemon=True).start()

        fails = 0
        fps_n, fps_t = 0, time.monotonic()
        fps_debug = os.environ.get("DMS_FPS_DEBUG") == "1"
        while self._running:
            ok, frame = cap.read()
            if not ok:
                fails += 1
                if fails >= self.MAX_READ_FAILS:
                    cap.release()
                    self.log.emit("Stream lost, searching for camera again ...")
                    cap, idx, name = self._find_camera()
                    if cap is None:
                        self.failed.emit("Camera disconnected")
                        return
                    self.log.emit(f"Reconnected: /dev/video{idx} ({name})")
                    self.opened.emit(idx, name)
                    fails = 0
                self.msleep(10)
                continue
            fails = 0
            frame = self._prepare(frame)
            with self._lock:
                self._frame = frame
                self._frame_id += 1
                result = self._result
            self.frameReady.emit(self._render_preview(frame, result), result)
            if fps_debug:
                fps_n += 1
                now = time.monotonic()
                if now - fps_t >= 2.0:
                    print(f"[fps] preview={fps_n / (now - fps_t):.1f}", flush=True)
                    fps_n, fps_t = 0, now

        cap.release()

    def _draw_hands_smooth(self, hands):
        """把手部结果插值成逐帧中间值，避免骨架以 2.5Hz 一卡一跳。"""
        target = hands or []
        if len(target) != len(self._draw_hands):
            self._draw_hands = [np.array(h, dtype=np.float32) for h in target]
            return self._draw_hands
        for i, h in enumerate(target):
            cur = self._draw_hands[i]
            new = np.asarray(h, dtype=np.float32)
            if cur.shape != new.shape:
                self._draw_hands[i] = new
            else:
                self._draw_hands[i] = cur + self.HAND_DRAW_ALPHA * (new - cur)
        return self._draw_hands

    def _render_preview(self, frame, result):
        """在采集线程里缩放并绘制标注，UI 线程只负责显示。"""
        h, w = frame.shape[:2]
        scale = PREVIEW_W / float(w)
        small = cv2.resize(frame, (PREVIEW_W, max(1, int(round(h * scale)))),
                           interpolation=cv2.INTER_AREA)
        view = result
        if result.get("hands"):
            view = dict(result)
            view["hands"] = self._draw_hands_smooth(result["hands"])
        else:
            self._draw_hands = []
        draw_mini(small, view, scale)
        # BGR888 直接复用 numpy 缓冲，省掉一次 BGR->RGB 转换
        return QImage(small.data, small.shape[1], small.shape[0],
                      small.strides[0], QImage.Format_BGR888).copy()
