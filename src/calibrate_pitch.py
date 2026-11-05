"""头部俯仰（pitch）零位与低头阈值标定工具 —— 板端全屏运行。

五段流程：
  1) 正视前方（基准零位）
  2) 中度低头（视线落在仪表台 / 中控）
  3) 重度低头（视线落在腿部 / 手机）
  4) 恢复正视（回位检查，验证阈值不会卡死）
  5) 连续低头 30 秒（检验 behaviors.py 的滑动零位会不会被带跑）

每段输出 raw pitch 的中位数与分位数，扣除正视基准后给出低头偏移量，
并按「正视抖动」与「低头幅度」的间隔推荐 PITCH_DISTRACT 阈值。
第 5 段还会按 DmsAnalyzer 的 300 帧滑动中位数离线复算残差，
用数据确认长时间低头是否会被零位漂移吃掉（漏报）。

用法：
  python3 src/calibrate_pitch.py --camera 73
  python3 src/calibrate_pitch.py --headless --seconds 6 --hold 20
  python3 src/calibrate_pitch.py --snapshot /tmp/pitch_ui.png

快捷键：Esc / q 退出。
"""

import argparse
import json
import os
import sys
import time
from collections import deque

import cv2
import numpy as np

import models_mp
from behaviors import PITCH_DISTRACT
from capture import MODELS_DIR, all_video_nodes, ir_prepare, looks_gray, try_open_camera
from collect_eyes import Canvas, grab_face
from metrics import head_pose

WIN = "DMS Pitch Calibration"
POSE_TRACK_FRAMES = 300   # 与 behaviors.py 的滑动零位窗口保持一致

STAGES = [
    {"key": "fwd", "title": "阶段 1/5　正视前方（基准）",
     "hint": "坐正，视线平视镜头或屏幕中央，保持 8 秒"},
    {"key": "down_mid", "title": "阶段 2/5　中度低头",
     "hint": "低头看仪表台 / 中控，像瞥一眼车速，保持 8 秒"},
    {"key": "down_deep", "title": "阶段 3/5　重度低头",
     "hint": "低头看腿部或手里的手机，保持 8 秒"},
    {"key": "fwd2", "title": "阶段 4/5　恢复正视",
     "hint": "重新抬头平视镜头，保持 8 秒（验证能回到正常态）"},
    {"key": "hold", "title": "阶段 5/5　连续低头 30 秒",
     "hint": "保持低头不要抬起来，用来检验滑动零位是否漂移", "_seconds": 30.0},
]


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
           cur, n: int, face_ok: bool) -> np.ndarray:
    """全屏画布：预览 + 阶段提示 + 实时 pitch 读数 + 进度条。"""
    canvas = np.zeros((cv.h, cv.w, 3), np.uint8)
    fh, fw = frame.shape[:2]
    s = min(cv.w / fw, cv.h / fh)
    nw, nh = int(fw * s), int(fh * s)
    ox, oy = (cv.w - nw) // 2, (cv.h - nh) // 2
    canvas[oy:oy + nh, ox:ox + nw] = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)

    dim = canvas.copy()
    k = cv.k
    cv2.rectangle(dim, (0, 0), (cv.w, int(150 * k)), (0, 0, 0), -1)
    cv2.rectangle(dim, (0, cv.h - int(130 * k)), (cv.w, cv.h), (0, 0, 0), -1)
    canvas = cv2.addWeighted(dim, 0.55, canvas, 0.45, 0)

    cv.text(canvas, stage["title"], (40, 62), 38, (255, 255, 255), 2)
    cv.text(canvas, stage["hint"], (40, 108), 24, (200, 220, 255), 1)

    if cur is None:
        cv.text(canvas, "-- 未检出人脸 --", (40, cv.h - 80), 30, (0, 120, 255), 2)
    else:
        p, y, r = cur
        cv.text(canvas, f"pitch {p:+6.1f}°    yaw {y:+6.1f}°    roll {r:+6.1f}°",
                (40, cv.h - 80), 34, (0, 255, 180), 2)

    secs = max(0, int(round(remaining)))
    if phase == "prepare":
        cv.text(canvas, "准备中，请按提示调整", (40, cv.h - 36), 26, (0, 255, 255), 2)
        cv.text(canvas, f"{secs}", (cv.w - 150, 118), 60, (0, 255, 255), 3)
    else:
        total = max(1.0, stage["_seconds"])
        ratio = 1.0 - max(0.0, remaining) / total
        x1, y1 = cv.w - int(40 * k) - int(360 * k), int(20 * k)
        x2, y2 = cv.w - int(40 * k), int(36 * k)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (70, 70, 70), -1)
        cv2.rectangle(canvas, (x1, y1), (x1 + int((x2 - x1) * ratio), y2), (0, 200, 0), -1)
        cv.text(canvas, f"剩余 {secs}s", (cv.w / k - 400, 82), 26, (255, 255, 255), 1)
        cv.text(canvas, stage["key"], (cv.w / k - 400, 116), 22, (180, 180, 180), 1)
        cv.text(canvas, f"有效样本 {n}   人脸 {'OK' if face_ok else '丢失'}",
                (40, cv.h - 36), 24, (255, 255, 255), 1)
    cv.text(canvas, "Esc 退出", (cv.w / k - 120, cv.h - 12), 20, (140, 140, 140), 1)
    return canvas


def summarize(vals) -> dict:
    a = np.asarray(vals, dtype=np.float64)
    return {
        "n": int(a.size),
        "median": float(np.median(a)),
        "p05": float(np.percentile(a, 5)),
        "p95": float(np.percentile(a, 95)),
        "min": float(a.min()),
        "max": float(a.max()),
        "std": float(a.std()),
    }


def sliding_residual(rows, win: int = POSE_TRACK_FRAMES):
    """按 behaviors.py 的滑动中位数零位，离线复算扣零后的 pitch 残差。"""
    buf = deque(maxlen=win)
    out = []
    for p, y, r in rows:
        buf.append((p, y, r))
        base = np.median(np.array(buf), axis=0)
        out.append(float(p - base[0]))
    return out


def analyze(rows_by_stage: dict, session_rows: list) -> dict:
    """打印统计并返回推荐结果。rows_* 元素为 (t, pitch, yaw, roll)。"""
    raw = {k: [r[1] for r in v] for k, v in rows_by_stage.items()}
    miss = {k: len(v) for k, v in raw.items() if not v}
    for k in miss:
        print(f"  !! {k} 段没有有效样本人脸，标定结果不可靠")

    if not raw.get("fwd") or not raw.get("down_mid"):
        print("缺少 fwd 或 down_mid 样本，无法给出建议阈值")
        return {}

    base = float(np.median(raw["fwd"]))
    print("\n==== raw pitch 统计（度）====")
    print(f"{'阶段':<10}{'n':>6}{'中位数':>9}{'p05':>8}{'p95':>8}{'范围':>16}{'std':>7}")
    for key in (s["key"] for s in STAGES):
        if key not in raw or not raw[key]:
            continue
        s = summarize(raw[key])
        span = "[{:.0f}, {:.0f}]".format(s["min"], s["max"])
        print(f"{key:<10}{s['n']:>6}{s['median']:>9.1f}{s['p05']:>8.1f}{s['p95']:>8.1f}"
              f"{span:>16}{s['std']:>7.2f}")

    print(f"\n正视基准零位（fwd 中位数）: {base:+.1f}°")
    print("（运行时由 behaviors.py 的 300 帧滑动中位数自动扣除，无需写死）")

    rel = {k: [v - base for v in vals] for k, vals in raw.items() if k not in ("fwd",)}
    for key in ("down_mid", "down_deep", "fwd2", "hold"):
        if rel.get(key):
            s = summarize(rel[key])
            print(f"  扣零后 {key:<9} 中位数 {s['median']:+6.1f}°  "
                  f"p05 {s['p05']:+6.1f}°  p95 {s['p95']:+6.1f}°")

    fwd_all = [v - base for v in raw["fwd"]] + \
              [v - base for v in raw.get("fwd2") or []]
    fs = summarize(fwd_all)
    noise = max(abs(fs["p05"]), abs(fs["p95"]), abs(fs["median"]))
    down = [summarize(rel[k])["median"] for k in ("down_mid", "down_deep") if rel.get(k)]
    down_min = min(down, key=abs) if down else None
    print(f"\n正视抖动幅度: ±{noise:.1f}°（p05~p95 = {fs['p05']:+.1f} ~ {fs['p95']:+.1f}）")
    if down_min is not None:
        print(f"轻度低头幅度: {down_min:+.1f}°；"
              f"判定符号：低头时 pitch {'增大' if down_min > 0 else '减小'}")

    result = {"baseline": base, "noise": noise, "down_mid": rel.get("down_mid") and
              summarize(rel["down_mid"]), "down_deep": rel.get("down_deep") and
              summarize(rel["down_deep"])}

    if down_min is None:
        return result

    # 阈值取「正视抖动上沿」与「轻度低头下沿」之间
    down_p = summarize([v for k in ("down_mid", "down_deep") if rel.get(k)
                        for v in rel[k]])
    lo = abs(down_p["p05"])
    hi = noise
    if lo <= hi + 2:
        print(f"\n[警告] 低头下沿 {lo:.1f}° 与正视抖动 {hi:.1f}° 太近，"
              "无法兼顾灵敏度与误报，建议调整摄像头安装角（略抬高、加大俯角）后重标定")
        suggest = round(max(lo, hi + 4), 1)
    else:
        suggest = round((hi + lo) / 2, 1)
    result["suggest"] = suggest
    print(f"\n建议 PITCH_DISTRACT = {suggest}°（当前 behaviors.py 为 {PITCH_DISTRACT}°）")

    # 第 5 段：滑动零位漂移检查
    if session_rows:
        span = max(1e-6, session_rows[-1][0] - session_rows[0][0])
        fps = len(session_rows) / span
        # behaviors.py 的 300 帧在分析频率约 15fps 下约等于 20 秒，这里按实际帧率换算
        win = max(60, int(round(20.0 * fps)))
        res = sliding_residual([(p, y, r) for _t, p, y, r, _k in session_rows], win)
        hold_idx = [i for i, r in enumerate(session_rows) if r[4] == "hold"]
        print(f"\n（会话 {len(session_rows)} 帧 / {span:.0f}s，滑动窗口按 {win} 帧复算）")
        if len(hold_idx) >= 10:
            h = hold_idx
            third = max(1, len(h) // 3)
            head = float(np.mean([res[i] for i in h[:third]]))
            tail = float(np.mean([res[i] for i in h[-third:]]))
            raw_tail = float(np.mean([session_rows[i][1] - base for i in h[-third:]]))
            result["hold"] = {"residual_head": head, "residual_tail": tail,
                              "raw_tail": raw_tail}
            print("\n==== 滑动零位漂移检查（连续低头段）====")
            print(f"  扣零残差  段首 {head:+.1f}°   段尾 {tail:+.1f}°")
            print(f"  段尾原始相对基准 {raw_tail:+.1f}°")
            if abs(tail) < suggest and abs(raw_tail) > suggest + 5:
                print("  [问题] 零位被低头样本带跑，段尾几乎判不出低头"
                      "（长时间低头会漏报）")
                result["drift"] = True
            else:
                print("  零位稳定，低头期间残差仍超过阈值")
                result["drift"] = False
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=None, help="指定 /dev/videoN")
    ap.add_argument("--seconds", type=float, default=8.0, help="常规阶段时长")
    ap.add_argument("--hold", type=float, default=30.0, help="连续低头阶段时长")
    ap.add_argument("--gap", type=float, default=6.0, help="每段前准备倒计时")
    ap.add_argument("--headless", action="store_true", help="不开窗口，只打印统计")
    ap.add_argument("--snapshot", default="", help="调试用：保存一张合成界面")
    ap.add_argument("--json", default="", help="把标定结果写到指定 JSON")
    ap.add_argument("--stages", default="", help="只跑指定阶段，逗号分隔的 key")
    args = ap.parse_args()

    stages = STAGES
    if args.stages:
        want = {s.strip() for s in args.stages.split(",") if s.strip()}
        stages = [s for s in STAGES if s["key"] in want]
        if not stages:
            sys.exit(f"未知阶段：{args.stages}")
    for s in stages:
        s["_seconds"] = s.get("_seconds", args.seconds)
        if s["key"] == "hold":
            s["_seconds"] = args.hold

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
    ir = None
    rows_by_stage = {s["key"]: [] for s in stages}
    session_rows = []          # (t, pitch, yaw, roll, stage_key) 按时间顺序
    cur, n_valid, face_ok = None, 0, False
    t0 = time.time()

    try:
        for stage in stages:
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
                                       t_end - time.time(), None, 0, True))
                if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                    return

            n_valid, face_ok = 0, False
            t_end = time.time() + stage["_seconds"]
            while time.time() < t_end:
                ok, frame = cap.read()
                if not ok:
                    continue
                frame = ir_prepare(frame, ir)
                now = time.time()
                got = grab_face(face, mesh, frame)
                if got is not None:
                    lm = got[0]
                    h, w = frame.shape[:2]
                    p, y, r = head_pose(lm, w, h)
                    cur = (p, y, r)
                    rows_by_stage[stage["key"]].append((now - t0, p, y, r))
                    session_rows.append((now - t0, p, y, r, stage["key"]))
                    n_valid += 1
                    face_ok = True
                else:
                    face_ok = False
                if cv_ui is None:
                    continue
                canvas = render(cv_ui, frame, stage, "run", t_end - time.time(),
                                cur, n_valid, face_ok)
                if args.snapshot and not snap_done:
                    cv2.imwrite(args.snapshot, canvas)
                    snap_done = True
                cv2.imshow(WIN, canvas)
                if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                    return
            print(f"[{stage['key']}] n={len(rows_by_stage[stage['key']])}", flush=True)
    finally:
        cap.release()
        if cv_ui is not None:
            cv2.destroyAllWindows()

    result = analyze(rows_by_stage, session_rows)
    if args.json and result:
        result["raw"] = {k: summarize([r[1] for r in v]) for k, v in rows_by_stage.items()
                         if v}
        result["stage_seconds"] = {s["key"]: s["_seconds"] for s in stages}
        with open(args.json, "w") as fh:
            json.dump(result, fh, ensure_ascii=False, indent=2)
        print(f"\n结果已写入 {args.json}")


if __name__ == "__main__":
    main()
