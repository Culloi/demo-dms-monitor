"""眼睛状态（睁/闭）数据采集工具 —— 板端全屏运行，自动打标，全程免键盘。

四段自动流程：
  1) 不戴眼镜 · 睁眼    2) 不戴眼镜 · 闭眼
  3) 戴眼镜  · 睁眼     4) 戴眼镜  · 闭眼

每段开始前有倒计时（用于摘/戴眼镜、坐正），采集窗口内持续保存眼睛裁剪图，
并按阶段自动打标，无需人工标注。结束后生成拼图，便于抽检样本质量。

用法：
  python3 src/collect_eyes.py                 # 自动搜索摄像头
  python3 src/collect_eyes.py --camera 73     # 指定 /dev/video73
  python3 src/collect_eyes.py --headless      # 无界面自检（只统计、仍会存图）
  python3 src/collect_eyes.py --stages noglass_open,noglass_closed

快捷键：Esc / q 退出。
"""

import argparse
import csv
import os
import sys
import time

import cv2
import numpy as np

import models_mp
from capture import MODELS_DIR, all_video_nodes, ir_prepare, looks_gray, try_open_camera
from eye_crop import EYE_SIZE, crop_eye, eye_boxes
from metrics import eye_aspect_ratio

WIN = "DMS Eye Collection"
FONT_CANDIDATES = (
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
)

# key 里同时编码了标签(open/closed)与是否戴眼镜，训练脚本直接按 key 分组
STAGES = [
    {"key": "noglass_open", "label": 0, "glasses": 0,
     "title": "阶段 1/4　不戴眼镜 · 睁眼", "hint": "自然注视镜头，保持睁眼（可正常眨眼）"},
    {"key": "noglass_closed", "label": 1, "glasses": 0,
     "title": "阶段 2/4　不戴眼镜 · 闭眼", "hint": "自然闭合眼睑，不要用力挤眼"},
    {"key": "glass_open", "label": 0, "glasses": 1,
     "title": "阶段 3/4　戴眼镜 · 睁眼", "hint": "保持平时坐姿看屏幕，正常睁眼（可自然眨眼）"},
    {"key": "glass_closed", "label": 1, "glasses": 1,
     "title": "阶段 4/4　戴眼镜 · 闭眼", "hint": "保持平时坐姿，自然闭合眼睑，不要用力挤眼"},
]

SAVE_FPS = 6.0        # 每秒最多保存的帧数（每帧存左右眼两张）
MIN_SIDE = 20         # 眼框短边下限（像素），太小则丢弃
DUP_DIFF = 1.5        # 与上一张样本的 16x16 指纹差异小于此值视为重复帧
FULL_EVERY = 40       # 每 N 帧保存一张整帧，便于回看采集场景


class Canvas:
    """自适应分辨率的绘图与中文文字渲染。"""

    def __init__(self, w: int, h: int) -> None:
        self.w, self.h = w, h
        self.k = h / 720.0
        self.ft = None
        for path in FONT_CANDIDATES:
            if not os.path.exists(path):
                continue
            try:
                ft = cv2.freetype.createFreeType2()
                ft.loadFontData(path, 0)
                self.ft = ft
                break
            except Exception:
                continue

    def text(self, img, s: str, xy, size: float, color=(255, 255, 255), thick: int = 1) -> None:
        px = max(12, int(size * self.k))
        x, y = int(xy[0] * self.k), int(xy[1] * self.k)
        if self.ft is not None:
            self.ft.putText(img, s, (x, y), px, color, thick, cv2.LINE_AA, True)
        else:
            cv2.putText(img, s.encode("ascii", "ignore").decode() or "?",
                        (x, y), cv2.FONT_HERSHEY_SIMPLEX, px / 38.0,
                        color, thick, cv2.LINE_AA)


def open_screen() -> Canvas:
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(WIN, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow(WIN, np.zeros((720, 1280, 3), np.uint8))
    cv2.waitKey(1)
    try:
        _x, _y, w, h = cv2.getWindowImageRect(WIN)
    except Exception:
        w, h = 1280, 720
    if w < 320 or h < 240:
        w, h = 1280, 720
    return Canvas(w, h)


def render(cv: Canvas, frame, stage: dict, phase: str, remaining: float,
           counts: dict, quality: str, sharp: float, side: int) -> np.ndarray:
    """把预览帧 + 提示信息画到全屏画布上。"""
    canvas = np.zeros((cv.h, cv.w, 3), np.uint8)
    fh, fw = frame.shape[:2]
    s = min(cv.w / fw, cv.h / fh)
    nw, nh = int(fw * s), int(fh * s)
    ox, oy = (cv.w - nw) // 2, (cv.h - nh) // 2
    canvas[oy:oy + nh, ox:ox + nw] = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)

    dim = canvas.copy()
    k = cv.k
    cv2.rectangle(dim, (0, 0), (cv.w, int(150 * k)), (0, 0, 0), -1)
    cv2.rectangle(dim, (0, cv.h - int(120 * k)), (cv.w, cv.h), (0, 0, 0), -1)
    canvas = cv2.addWeighted(dim, 0.55, canvas, 0.45, 0)

    cv.text(canvas, stage["title"], (40, 62), 38, (255, 255, 255), 2)
    cv.text(canvas, stage["hint"], (40, 108), 24, (200, 220, 255), 1)

    secs = max(0, int(round(remaining)))
    total_all = sum(counts.values())
    if phase == "prepare":
        cv.text(canvas, "准备中，请按提示调整", (40, cv.h - 76), 30, (0, 255, 255), 2)
        cv.text(canvas, f"{secs}", (cv.w - 150, 118), 64, (0, 255, 255), 3)
    else:
        total = max(1.0, stage["_seconds"])
        ratio = 1.0 - max(0.0, remaining) / total
        x1, y1 = cv.w - int(40 * k) - int(360 * k), int(20 * k)
        x2, y2 = cv.w - int(40 * k), int(36 * k)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (70, 70, 70), -1)
        cv2.rectangle(canvas, (x1, y1), (x1 + int((x2 - x1) * ratio), y2), (0, 200, 0), -1)
        cv.text(canvas, f"剩余 {secs}s", (cv.w / k - 400, 82), 26, (255, 255, 255), 1)
        cv.text(canvas, stage["key"], (cv.w / k - 400, 116), 22, (180, 180, 180), 1)
        cv.text(canvas, f"已采集 {counts[stage['key']]} 张   人脸 {quality}   "
                        f"眼框 {side}px   清晰度 {sharp:.0f}",
                (40, cv.h - 66), 24, (255, 255, 255), 1)

    cv.text(canvas, f"总计 {total_all} 张", (40, cv.h - 26), 22, (160, 255, 160), 1)
    cv.text(canvas, "Esc 退出", (cv.w / k - 120, cv.h - 26), 20, (140, 140, 140), 1)
    return canvas


def grab_face(face, mesh, frame):
    """检测最大人脸并返回 (landmarks, bbox, score)；无人脸返回 None。"""
    bboxes, kps, scores = face.detect(frame, threshold=0.6)
    if len(bboxes) == 0:
        return None
    areas = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
    i = int(areas.argmax())
    lms, _ = mesh.predict(frame, [bboxes[i]], keypoints=[kps[i]])
    if len(lms) == 0:
        return None
    return lms[0], bboxes[i], float(scores[i])


def save_samples(writer, csv_file, out_dir: str, stage: dict, crops, boxes,
                 ear: float, score: float) -> int:
    """保存左右眼裁剪图并写 CSV，返回保存张数。"""
    ts = time.time()
    sid = f"{int(ts * 1000) % 10 ** 9:09d}"
    n = 0
    for tag, crop, box in zip(("R", "L"), crops, boxes):
        if crop is None:
            continue
        rel = os.path.join("crops", stage["key"], f"{sid}_{tag}.png")
        cv2.imwrite(os.path.join(out_dir, rel), crop)
        writer.writerow([stage["key"], stage["label"], stage["glasses"], tag, rel,
                         f"{ear:.4f}", f"{score:.3f}", box[2] - box[0], f"{ts:.3f}"])
        n += 1
    csv_file.flush()
    return n


def make_montage(out_dir: str, key: str, cols: int = 10, rows: int = 12) -> None:
    """把某阶段的样本拼成一张图，方便肉眼抽检标签是否正确。"""
    files = sorted(f for f in os.listdir(os.path.join(out_dir, "crops", key))
                   if f.endswith(".png"))[: cols * rows]
    if not files:
        return
    tile, pad = EYE_SIZE, 6
    sheet = np.full((rows * (tile + pad) + pad, cols * (tile + pad) + pad, 3), 30, np.uint8)
    for i, name in enumerate(files):
        img = cv2.imread(os.path.join(out_dir, "crops", key, name), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        r, c = divmod(i, cols)
        y, x = pad + r * (tile + pad), pad + c * (tile + pad)
        sheet[y:y + tile, x:x + tile] = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.imwrite(os.path.join(out_dir, f"montage_{key}.jpg"), sheet)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=None, help="指定 /dev/videoN")
    ap.add_argument("--seconds", type=float, default=20.0, help="每段采集时长")
    ap.add_argument("--gap", type=float, default=10.0, help="每段前准备倒计时")
    ap.add_argument("--out", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dataset", "eyes"))
    ap.add_argument("--headless", action="store_true", help="不开窗口，仅采集+打印统计")
    ap.add_argument("--snapshot", default="", help="调试用：保存第一张合成界面到指定路径")
    ap.add_argument("--stages", default="", help="只跑指定阶段，逗号分隔的 key")
    args = ap.parse_args()

    stages = STAGES
    if args.stages:
        want = {s.strip() for s in args.stages.split(",") if s.strip()}
        stages = [s for s in STAGES if s["key"] in want]
        if not stages:
            sys.exit(f"未知阶段：{args.stages}")
    for s in stages:
        s["_seconds"] = args.seconds

    os.makedirs(os.path.join(args.out, "crops"), exist_ok=True)
    for s in stages:
        os.makedirs(os.path.join(args.out, "crops", s["key"]), exist_ok=True)
    csv_path = os.path.join(args.out, "labels.csv")
    need_header = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0
    csv_file = open(csv_path, "a", newline="")
    writer = csv.writer(csv_file)
    if need_header:
        writer.writerow(["key", "label", "glasses", "eye", "path", "ear",
                         "score", "side_px", "ts"])

    print("loading models ...", flush=True)
    face = models_mp.BlazeFace(os.path.join(MODELS_DIR, "face_detection_short_range.onnx"))
    mesh = models_mp.FaceMesh(os.path.join(MODELS_DIR, "face_mesh_Nx3x192x192.onnx"))

    if args.camera is not None:
        cap, idx, name = try_open_camera(args.camera), args.camera, ""
    else:
        cap, idx, name = None, -1, ""
        for i, n in all_video_nodes():
            cap = try_open_camera(i)
            if cap is not None:
                idx, name = i, n
                break
    if cap is None:
        sys.exit("未找到可用摄像头")
    print(f"camera /dev/video{idx} {name}", flush=True)

    cv_ui = None if args.headless else open_screen()
    snap_done = False
    counts = {s["key"]: 0 for s in stages}
    ir = None
    last_save = 0.0
    quality, sharp, side = "-", 0.0, 0

    try:
        for stage in stages:
            # ---------- 准备倒计时 ----------
            t_end = time.time() + args.gap
            while time.time() < t_end:
                ok, frame = cap.read()
                if not ok:
                    continue
                if ir is None:
                    ir = looks_gray(frame)
                    print(f"infrared stream: {'yes' if ir else 'no'}", flush=True)
                frame = ir_prepare(frame, ir)
                if cv_ui is None:
                    continue
                cv2.imshow(WIN, render(cv_ui, frame, stage, "prepare",
                                       t_end - time.time(), counts, "-", 0.0, 0))
                if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                    return

            # ---------- 采集 ----------
            last_sig, frames = None, 0
            mark = 0
            t_end = time.time() + args.seconds
            while time.time() < t_end:
                ok, frame = cap.read()
                if not ok:
                    continue
                frame = ir_prepare(frame, ir)
                now = time.time()
                if now - last_save >= 1.0 / SAVE_FPS:
                    last_save = now
                    got = grab_face(face, mesh, frame)
                    if got is None:
                        quality = "未检出"
                    else:
                        lm, _bbox, score = got
                        h, w = frame.shape[:2]
                        boxes = eye_boxes(lm, w, h)
                        side = min(b[2] - b[0] for b in boxes)
                        crops = [crop_eye(frame, b)[0] for b in boxes]
                        if any(c is None for c in crops):
                            quality = "裁剪失败"
                        else:
                            sharp = min(float(cv2.Laplacian(c, cv2.CV_64F).var())
                                        for c in crops)
                            bright = min(float(c.mean()) for c in crops)
                            sig = np.concatenate([cv2.resize(c, (16, 16)).astype(np.float32)
                                                  for c in crops])
                            if side < MIN_SIDE:
                                quality = f"眼框过小 {side}px"
                            elif bright < 20 or bright > 240:
                                quality = "亮度过低/过高"
                            elif sharp < 15:
                                quality = "画面模糊"
                            elif last_sig is not None and \
                                    float(np.abs(sig - last_sig).mean()) < DUP_DIFF:
                                quality = "重复帧"
                            else:
                                counts[stage["key"]] += save_samples(
                                    writer, csv_file, args.out, stage, crops, boxes,
                                    eye_aspect_ratio(lm), score)
                                frames += 1
                                last_sig = sig
                                quality = "已保存"
                                if frames - mark >= FULL_EVERY:
                                    mark = frames
                                    full_dir = os.path.join(args.out, "full", stage["key"])
                                    os.makedirs(full_dir, exist_ok=True)
                                    cv2.imwrite(os.path.join(
                                        full_dir, f"{int(now)}_{frames:04d}.jpg"), frame)
                    if args.headless:
                        print(f"[{stage['key']}] {quality} side={side} "
                              f"sharp={sharp:.0f} n={counts[stage['key']]}", flush=True)
                if cv_ui is None:
                    continue
                canvas = render(cv_ui, frame, stage, "run",
                                t_end - time.time(), counts, quality, sharp, side)
                if args.snapshot and not snap_done:
                    cv2.imwrite(args.snapshot, canvas)
                    snap_done = True
                cv2.imshow(WIN, canvas)
                if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                    return
    finally:
        cap.release()
        csv_file.close()
        if cv_ui is not None:
            cv2.destroyAllWindows()

    for s in stages:
        make_montage(args.out, s["key"])
    print("\n==== 采集完成 ====")
    for s in stages:
        print(f"  {s['key']:<18} {counts[s['key']]:>5} 张")
    print(f"  输出目录: {args.out}")


if __name__ == "__main__":
    main()
