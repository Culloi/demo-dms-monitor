"""MediaPipe 系列模型的 ONNX Runtime 封装。

由于本机 CPU（Cortex-A72/A53, ARMv8.0）不支持 LSE 原子指令，官方 mediapipe
运行时无法在此设备运行；这里改用等价思路：直接使用由 MediaPipe 权重导出的
ONNX 模型，配合 onnxruntime 推理，锚点解码 / ROI / NMS 等后处理在 Python 中完成。

包含：
- BlazeFace        人脸检测（short-range，128x128，6 关键点）
- FaceMesh         468 点人脸网格（192x192，输出 3D 关键点）
- PalmDetection    手掌检测（BlazePalm，192x192）
- HandLandmark     手部 21 点关键点（224x224）

本文件中的网络解码/ROI 逻辑移植自 github.com/yakhyo 的
mediapipe-face-mesh-onnx 与 mediapipe-hand-landmark-onnx（Apache-2.0）。
"""

import os
import sys

# 工程根目录（src 的上一级），models/ 与 ortlibs/ 都放在这里
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 工程自带 ortlibs/（onnxruntime + onnx + numpy 等）优先加入搜索路径，
# 使整个目录拷到同平台机器即可直接运行，无需手动设置 PYTHONPATH；
# 若不存在则回退到系统已安装的包。
_ORTLIBS = os.path.join(_ROOT, "ortlibs")
if os.path.isdir(_ORTLIBS) and _ORTLIBS not in sys.path:
    sys.path.insert(0, _ORTLIBS)

import cv2
import numpy as np
import onnxruntime as ort

FACE_INPUT_SIZE = 128
NUM_FACE_KEYPOINTS = 6
PALM_INPUT_SIZE = 192
HAND_INPUT_SIZE = 224
NUM_HAND_LANDMARKS = 21

# MediaPipe palm_detection_detection_to_roi：缩放 2.6、中心朝手指方向偏移半框
ROI_SCALE = 2.6
ROI_SHIFT = 0.5

# 手部 21 点骨架连线（MediaPipe HAND_CONNECTIONS）
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),           # 拇指
    (0, 5), (5, 6), (6, 7), (7, 8),           # 食指
    (5, 9), (9, 10), (10, 11), (11, 12),      # 中指
    (9, 13), (13, 14), (14, 15), (15, 16),    # 无名指
    (13, 17), (17, 18), (18, 19), (19, 20),   # 小指
    (0, 17),                                   # 掌缘
]

_PROVIDERS = ["CPUExecutionProvider"]
# 推理线程数：默认 2。实测（aarch64）4 线程相对 2 线程只快约 25%，
# 但 CPU 占用高约 25%，且会与 Qt 渲染抢核导致界面卡顿；
# 小输入模型（128~224）并行收益有限。可用环境变量 DMS_THREADS 覆盖。
ORT_THREADS = int(os.environ.get("DMS_THREADS", "2"))


def _session(model_path: str) -> ort.InferenceSession:
    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    opts.intra_op_num_threads = ORT_THREADS
    opts.inter_op_num_threads = 1
    # 关闭线程自旋：离开推理调用后线程立刻休眠，避免空转白烧 CPU
    opts.add_session_config_entry("session.intra_op.allow_spinning", "0")
    return ort.InferenceSession(model_path, sess_options=opts, providers=_PROVIDERS)


def _anchors(input_size: int, strides) -> np.ndarray:
    """SSD 锚点中心（固定的 anchor size，只需 x/y 中心）。"""
    out = []
    idx = 0
    while idx < len(strides):
        last = idx
        while last < len(strides) and strides[last] == strides[idx]:
            last += 1
        repeats = 2 * (last - idx)
        cells = input_size // strides[idx]
        for y in range(cells):
            for x in range(cells):
                out.extend([((x + 0.5) / cells, (y + 0.5) / cells)] * repeats)
        idx = last
    return np.array(out)


def _weighted_nms(detections: np.ndarray, iou_threshold: float = 0.3) -> np.ndarray:
    """MediaPipe 的加权 NMS：重叠候选按分数加权平均，而非直接丢弃。"""
    remaining = detections[np.argsort(-detections[:, 0])]
    output = []
    while len(remaining):
        top = remaining[0]
        x1 = remaining[:, 1] - remaining[:, 3] / 2
        y1 = remaining[:, 2] - remaining[:, 4] / 2
        x2 = remaining[:, 1] + remaining[:, 3] / 2
        y2 = remaining[:, 2] + remaining[:, 4] / 2
        tx1, ty1 = top[1] - top[3] / 2, top[2] - top[4] / 2
        tx2, ty2 = top[1] + top[3] / 2, top[2] + top[4] / 2
        inter = np.maximum(0, np.minimum(x2, tx2) - np.maximum(x1, tx1)) * \
            np.maximum(0, np.minimum(y2, ty2) - np.maximum(y1, ty1))
        union = (x2 - x1) * (y2 - y1) + (tx2 - tx1) * (ty2 - ty1) - inter
        iou = inter / np.maximum(union, 1e-9)

        overlapping = remaining[iou > iou_threshold]
        weights = overlapping[:, :1]
        blended = top.copy()
        blended[1:] = (overlapping[:, 1:] * weights).sum(axis=0) / weights.sum()
        output.append(blended)
        remaining = remaining[iou <= iou_threshold]
    return np.array(output)


class BlazeFace:
    """BlazeFace short-range 人脸检测（输入 128x128 RGB，letterbox，[-1,1]）。"""

    def __init__(self, model_path: str) -> None:
        self.session = _session(model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.anchors = _anchors(FACE_INPUT_SIZE, [8, 16, 16, 16])

    @staticmethod
    def _preprocess(crop: np.ndarray) -> np.ndarray:
        rgb = crop[:, :, ::-1].astype(np.float32)
        return np.transpose((rgb - 127.5) / 127.5, (2, 0, 1))

    def detect(self, image: np.ndarray, threshold: float = 0.5):
        """返回 (bboxes(N,4) xyxy, keypoints(N,6,2), scores(N,))，全图像素坐标。"""
        h, w = image.shape[:2]
        scale = FACE_INPUT_SIZE / max(h, w)
        pad_x = (FACE_INPUT_SIZE - w * scale) / 2.0
        pad_y = (FACE_INPUT_SIZE - h * scale) / 2.0
        matrix = np.array([[scale, 0.0, pad_x], [0.0, scale, pad_y]])
        canvas = cv2.warpAffine(image, matrix, (FACE_INPUT_SIZE, FACE_INPUT_SIZE))

        blob = self._preprocess(canvas)[np.newaxis]
        regressors, logits = self.session.run(None, {self.input_name: blob})

        scores = 1.0 / (1.0 + np.exp(-logits[0].ravel().astype(np.float64)))
        keep = scores >= threshold
        if not keep.any():
            return (np.empty((0, 4), np.float32),
                    np.empty((0, NUM_FACE_KEYPOINTS, 2), np.float32),
                    np.empty(0, np.float32))

        reg, anchor = regressors[0][keep].astype(np.float64), self.anchors[keep]
        rows = np.empty((int(keep.sum()), 5 + 2 * NUM_FACE_KEYPOINTS))
        rows[:, 0] = scores[keep]
        rows[:, 1:3] = reg[:, 0:2] / FACE_INPUT_SIZE + anchor
        rows[:, 3:5] = reg[:, 2:4] / FACE_INPUT_SIZE
        for k in range(NUM_FACE_KEYPOINTS):
            rows[:, 5 + 2 * k: 7 + 2 * k] = reg[:, 4 + 2 * k: 6 + 2 * k] / FACE_INPUT_SIZE + anchor
        rows = _weighted_nms(rows)

        def unletterbox(xy):
            return (xy * FACE_INPUT_SIZE - (pad_x, pad_y)) / scale

        centers = unletterbox(rows[:, 1:3])
        halves = rows[:, 3:5] * FACE_INPUT_SIZE / scale / 2.0
        bboxes = np.concatenate([centers - halves, centers + halves], axis=1)
        keypoints = unletterbox(rows[:, 5:].reshape(-1, NUM_FACE_KEYPOINTS, 2))
        return bboxes.astype(np.float32), keypoints.astype(np.float32), rows[:, 0].astype(np.float32)


def roi_from_box(bbox, keypoints=None, margin: float = 0.25):
    """由检测框构造方形（可旋转）ROI：(cx, cy, side, angle_deg)。"""
    x1, y1, x2, y2 = (float(v) for v in bbox[:4])
    side = (1.0 + 2.0 * margin) * max(x2 - x1, y2 - y1)
    angle = 0.0
    if keypoints is not None:
        dx, dy = (float(v) for v in np.asarray(keypoints)[1] - np.asarray(keypoints)[0])
        angle = float(np.degrees(np.arctan2(dy, dx)))
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0, side, angle


def warp_roi(image: np.ndarray, roi, out_size: int):
    """直接采样旋转方形 ROI 到模型输入尺寸；返回 (crop, 逆仿射矩阵)。"""
    cx, cy, side, angle = roi
    matrix = cv2.getRotationMatrix2D((cx, cy), angle, out_size / side)
    matrix[0, 2] += out_size / 2.0 - cx
    matrix[1, 2] += out_size / 2.0 - cy
    crop = cv2.warpAffine(image, matrix, (out_size, out_size))
    return crop, cv2.invertAffineTransform(matrix)


class FaceMesh:
    """MediaPipe Face Mesh（468 点）ONNX 推理。"""

    def __init__(self, model_path: str) -> None:
        self.session = _session(model_path)
        meta = self.session.get_inputs()[0]
        self.input_name = meta.name
        self.input_size = int(meta.shape[2])
        self.num_landmarks = int(self.session.get_outputs()[0].shape[1])

    def _preprocess(self, crop: np.ndarray) -> np.ndarray:
        if crop.shape[:2] != (self.input_size, self.input_size):
            crop = cv2.resize(crop, (self.input_size, self.input_size))
        rgb = crop[:, :, ::-1].astype(np.float32) / 255.0
        return np.transpose(rgb, (2, 0, 1))

    def predict(self, image, bboxes, margin: float = 0.25, keypoints=None):
        """对每个人脸框预测 468 点。返回 landmarks(N,468,3)、scores(N,)。"""
        if len(bboxes) == 0:
            return (np.empty((0, self.num_landmarks, 3), np.float32), np.empty(0, np.float32))

        blobs, inverses, rois = [], [], []
        for idx, bbox in enumerate(bboxes):
            kp = None if keypoints is None else keypoints[idx]
            roi = roi_from_box(bbox, kp, margin)
            crop, inv = warp_roi(image, roi, self.input_size)
            blobs.append(self._preprocess(crop))
            inverses.append(inv)
            rois.append(roi)

        landmarks, logits = self.session.run(None, {self.input_name: np.stack(blobs)})
        scores = 1.0 / (1.0 + np.exp(-logits.ravel().astype(np.float64)))
        landmarks = landmarks.astype(np.float64)
        for i, (roi, inv) in enumerate(zip(rois, inverses)):
            landmarks[i, :, :2] = landmarks[i, :, :2] @ inv[:, :2].T + inv[:, 2]
            landmarks[i, :, 2] *= roi[2] / self.input_size
        return landmarks.astype(np.float32), scores.astype(np.float32)


def palm_roi(palm):
    """由一掌检测行构造手部 ROI：(cx, cy, side, angle_deg)。"""
    _, cx, cy, width, wrist_x, wrist_y, mcp_x, mcp_y = (float(v) for v in palm[:8])
    direction = np.array([mcp_x - wrist_x, mcp_y - wrist_y])
    norm = float(np.linalg.norm(direction))
    unit = direction / norm if norm > 0 else np.array([0.0, -1.0])
    center_x = cx + ROI_SHIFT * width * unit[0]
    center_y = cy + ROI_SHIFT * width * unit[1]
    angle = float(np.degrees(np.arctan2(unit[1], unit[0]))) + 90.0
    return center_x, center_y, ROI_SCALE * width, angle


class PalmDetection:
    """BlazePalm 手掌检测（192x192，letterbox）。"""

    def __init__(self, model_path: str) -> None:
        self.session = _session(model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.anchors = _anchors(PALM_INPUT_SIZE, [8, 16, 16, 16])

    @staticmethod
    def _preprocess(crop: np.ndarray) -> np.ndarray:
        rgb = crop[:, :, ::-1].astype(np.float32) / 255.0
        return np.transpose(rgb, (2, 0, 1))

    def detect(self, image: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """返回 (N,8)：score, cx, cy, width, wrist_x, wrist_y, mcp_x, mcp_y。"""
        h, w = image.shape[:2]
        scale = PALM_INPUT_SIZE / max(h, w)
        pad_x = (PALM_INPUT_SIZE - w * scale) / 2.0
        pad_y = (PALM_INPUT_SIZE - h * scale) / 2.0
        matrix = np.array([[scale, 0.0, pad_x], [0.0, scale, pad_y]])
        canvas = cv2.warpAffine(image, matrix, (PALM_INPUT_SIZE, PALM_INPUT_SIZE))

        blob = self._preprocess(canvas)[np.newaxis]
        regressors, classificators = self.session.run(None, {self.input_name: blob})

        scores = 1.0 / (1.0 + np.exp(-classificators[0].ravel().astype(np.float64)))
        keep = scores >= threshold
        if not keep.any():
            return np.empty((0, 8), dtype=np.float32)

        reg, anchor = regressors[0][keep].astype(np.float64), self.anchors[keep]
        rows = np.empty((int(keep.sum()), 9))
        rows[:, 0] = scores[keep]
        rows[:, 1:3] = reg[:, 0:2] / PALM_INPUT_SIZE + anchor
        rows[:, 3:5] = reg[:, 2:4] / PALM_INPUT_SIZE
        rows[:, 5:7] = reg[:, 4:6] / PALM_INPUT_SIZE + anchor
        rows[:, 7:9] = reg[:, 8:10] / PALM_INPUT_SIZE + anchor
        rows = _weighted_nms(rows)

        def unletterbox(xy):
            return (xy * PALM_INPUT_SIZE - (pad_x, pad_y)) / scale

        palms = np.empty((len(rows), 8))
        palms[:, 0] = rows[:, 0]
        palms[:, 1:3] = unletterbox(rows[:, 1:3])
        palms[:, 3] = rows[:, 3:5].max(axis=1) * PALM_INPUT_SIZE / scale
        palms[:, 4:6] = unletterbox(rows[:, 5:7])
        palms[:, 6:8] = unletterbox(rows[:, 7:9])
        return palms.astype(np.float32)


class HandLandmark:
    """MediaPipe 手部 21 点关键点模型（224x224）。"""

    def __init__(self, model_path: str) -> None:
        self.session = _session(model_path)
        self.input_name = self.session.get_inputs()[0].name

    @staticmethod
    def _preprocess(crop: np.ndarray) -> np.ndarray:
        rgb = crop[:, :, ::-1].astype(np.float32) / 255.0
        return np.transpose(rgb, (2, 0, 1))

    def predict(self, image: np.ndarray, palms: np.ndarray):
        """返回 landmarks(N,21,3), presence(N,), handedness(N,)。"""
        if len(palms) == 0:
            return (np.empty((0, NUM_HAND_LANDMARKS, 3), np.float32),
                    np.empty(0, np.float32), np.empty(0, np.float32))

        blobs, inverses, z_scales = [], [], []
        for palm in palms:
            roi = palm_roi(palm)
            crop, inv = warp_roi(image, roi, HAND_INPUT_SIZE)
            blobs.append(self._preprocess(crop))
            inverses.append(inv)
            z_scales.append(roi[2] / HAND_INPUT_SIZE)

        screen, presence, handedness, _world = self.session.run(
            None, {self.input_name: np.stack(blobs)})
        landmarks = screen.reshape(-1, NUM_HAND_LANDMARKS, 3).astype(np.float64)
        for i, inv in enumerate(inverses):
            landmarks[i, :, :2] = landmarks[i, :, :2] @ inv[:, :2].T + inv[:, 2]
            landmarks[i, :, 2] *= z_scales[i]
        return (landmarks.astype(np.float32),
                presence.ravel().astype(np.float32),
                handedness.ravel().astype(np.float32))