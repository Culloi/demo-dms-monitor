"""打电话数据采集工具 —— 板端全屏运行，自动打标，全程免键盘。

三段自动流程（每段前有倒计时，用来看提示、摆姿势）：
  1) 打电话     手机贴耳、手持手机（正样本）
  2) 手到头部   托腮、挠头、扶眼镜、撩头发、手放耳后（易混淆负样本）
  3) 正常       正常驾驶姿势，手放下或离开画面（负样本）

每段只保存「手-耳 ROI」（phone_crop.phone_roi），并按阶段自动打标。

用法：
  python3 src/collect_phone.py                 # 自动搜索摄像头
  python3 src/collect_phone.py --camera 73     # 指定 /dev/video73
  python3 src/collect_phone.py --headless      # 无界面自检（仍会存图）
  python3 src/collect_phone.py --stages phone,hand_head

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
from collect_eyes import Canvas
from phone_crop import PHONE_SIZE, crop_phone, hand_near_ear, phone_roi

WIN = "DMS Phone Collection"

STAGES = [
    {"key": "phone", "label": 1, "need_hand": True,
     "title": "阶段 1/4　打电话", "hint": "手机贴住耳侧、手指扶住手机；保持 3 秒再换个角度"},
    {"key": "hand_head", "label": 0, "need_hand": True,
     "title": "阶段 2/4　手到头部（负样本）", "hint": "托腮、挠头、扶眼镜、撩头发、手放耳后；不要拿手机"},
    {"key": "hand_ear", "label": 0, "need_hand": True,
     "title": "阶段 3/4　空手贴耳（负样本）",
     "hint": "空手做出打电话的动作：手掌、手指或拳面贴住耳侧，可压住脸颊；不要拿手机"},
    {"key": "normal", "label": 0, "need_hand": False,
     "title": "阶段 4/4　正常驾驶", "hint": "保持正常坐姿，手放方向盘/腿上或离开画面"},
]

SAVE_FPS = 6.0        # 每秒最多保存帧数
HAND_INTERVAL = 0.35  # 手部检测间隔（Palm+Hand 较慢）
DUP_DIFF = 1.2        # 16x16 指纹差异小于此值视为重复帧
FULL_EVERY = 40       # 每 N 帧存一张整帧，便于回看


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
           counts: dict, quality: str, roi_px: int, near: bool) -> np.ndarray:
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
        cv.text(canvas, f"已采集 {counts[stage['key']]} 张   状态 {quality}   "
                        f"ROI {roi_px}px   手在耳侧 {'是' if near else '否'}",
                (40, cv.h - 66), 24, (255, 255, 255), 1)

    cv.text(canvas, f"总计 {sum(counts.values())} 张", (40, cv.h - 26), 22, (160, 255, 160), 1)
    cv.text(canvas, "Esc 退出", (cv.w / k - 120, cv.h - 26), 20, (140, 140, 140), 1)
    return canvas


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=None, help="指定 /dev/videoN")
    ap.add_argument("--seconds", type=float, default=20.0, help="每段采集时长")
    ap.add_argument("--gap", type=float, default=10.0, help="每段前准备倒计时")
    ap.add_argument("--out", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dataset", "phone"))
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

    for s in stages:
        os.makedirs(os.path.join(args.out, "rois", s["key"]), exist_ok=True)
    csv_path = os.path.join(args.out, "labels.csv")
    need_header = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0
    csv_file = open(csv_path, "a", newline="")
    writer = csv.writer(csv_file)
    if need_header:
        writer.writerow(["key", "label", "path", "roi_px", "sharp", "near_ear", "ts"])

    print("loading models ...", flush=True)
    face = models_mp.BlazeFace(os.path.join(MODELS_DIR, "face_detection_short_range.onnx"))
    mesh = models_mp.FaceMesh(os.path.join(MODELS_DIR, "face_mesh_Nx3x192x192.onnx"))
    palm = models_mp.PalmDetection(os.path.join(MODELS_DIR, "palm_detection.onnx"))
    hand = models_mp.HandLandmark(os.path.join(MODELS_DIR, "hand_landmark.onnx"))

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
    last_hand_t = 0.0
    hands = []
    quality, roi_px, near = "-", 0, False

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
                                       t_end - time.time(), counts, "-", 0, False))
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
                    h, w = frame.shape[:2]
                    bboxes, kps, _scores = face.detect(frame, threshold=0.6)
                    lm = None
                    if len(bboxes) > 0:
                        areas = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
                        i = int(areas.argmax())
                        lms, _ = mesh.predict(frame, [bboxes[i]], keypoints=[kps[i]])
                        lm = lms[0] if len(lms) else None
                    if lm is None:
                        quality = "未检出人脸"
                    else:
                        if now - last_hand_t >= HAND_INTERVAL:
                            palms = palm.detect(frame, threshold=0.5)
                            hl, _p, _h = hand.predict(frame, palms)
                            hands = [hl[i] for i in range(len(hl))]
                            last_hand_t = now
                        near = hand_near_ear(lm, hands)
                        box = phone_roi(lm, hands, w, h)
                        crop = None
                        if box is not None:
                            crop, _ok = crop_phone(frame, box)
                            roi_px = min(box[2] - box[0], box[3] - box[1])
                        if crop is None:
                            quality = "ROI 裁剪失败"
                        elif stage["need_hand"] and not near:
                            quality = "手未靠近耳侧"
                        else:
                            sharp = float(cv2.Laplacian(crop, cv2.CV_64F).var())
                            bright = float(crop.mean())
                            sig = cv2.resize(crop, (16, 16)).astype(np.float32)
                            if roi_px < 40:
                                quality = f"ROI 过小 {roi_px}px"
                            elif bright < 20 or bright > 240:
                                quality = "亮度过低/过高"
                            elif sharp < 15:
                                quality = "画面模糊"
                            elif last_sig is not None and \
                                    float(np.abs(sig - last_sig).mean()) < DUP_DIFF:
                                quality = "重复帧"
                            else:
                                sid = f"{int(now * 1000) % 10 ** 9:09d}"
                                rel = os.path.join("rois", stage["key"], f"{sid}.png")
                                cv2.imwrite(os.path.join(args.out, rel), crop)
                                writer.writerow([stage["key"], stage["label"], rel,
                                                 roi_px, f"{sharp:.0f}",
                                                 int(near), f"{now:.3f}"])
                                csv_file.flush()
                                counts[stage["key"]] += 1
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
                        print(f"[{stage['key']}] {quality} roi={roi_px} "
                              f"near={int(near)} n={counts[stage['key']]}", flush=True)
                if cv_ui is None:
                    continue
                canvas = render(cv_ui, frame, stage, "run",
                                t_end - time.time(), counts, quality, roi_px, near)
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

    print("\n==== 采集完成 ====")
    for s in stages:
        print(f"  {s['key']:<12} {counts[s['key']]:>5} 张")
    print(f"  输出目录: {args.out}")
    print(f"  样本尺寸: {PHONE_SIZE}x{PHONE_SIZE}")


if __name__ == "__main__":
    main()
