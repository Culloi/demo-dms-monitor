"""行为判定与分级告警状态机。

把每帧的指标（EAR/MAR/姿态/手部）转成具体的驾驶行为状态，并按持续时长
给出 绿(0)/黄(1)/红(2) 三级告警。
"""

import os
import time
from collections import deque

import numpy as np

from hand_crop import EAR_MIN, NEAR_MOUTH
from metrics import Perclos

GATE_DEBUG = os.environ.get("DMS_GATE_DEBUG") == "1"

# ---- 阈值（可调）----
EAR_CLOSED = 0.12          # EAR 兜底阈值（仅在眼态模型缺失时使用）
EYE_CLOSED_PROB = 0.5      # 眼态模型输出的闭眼概率阈值
                           # 红外实测：戴眼镜时闭眼 EAR 反而高于睁眼（0.226 vs 0.20），
                           # EAR 不可靠，改用 src/eye_state.py 的 ONNX 分类模型
EYE_CLOSED_PROB_TILT = 0.85  # 头部偏离正视时的收紧阈值（透视压扁的睁眼会误判）
PITCH_EYE_LIMIT = 20.0     # 明显低头时眼态模型不可信（上眼睑盖住虹膜）
MAR_YAWN = 0.45            # 高于此值视为张嘴（哈欠）。红外实测：闭嘴 0.05~0.07，
                           # 张大嘴 0.55~0.77（均值 0.66），取 0.45 留出部分哈欠的余量
YAW_DISTRACT = 30.0        # 头部左右转动超过此角度视为分心（实测刻意右转 yaw≈+33°，正视基线≈0）
PITCH_DISTRACT = 20.0      # 头部俯仰超过此角度视为分心。实测：正视抖动 ±5°，
                           # 瞥一眼仪表台 +11°，真正低头看腿/手机 +37~46°，取 20 兼顾两侧
ROLL_DISTRACT = 30.0       # 画面内倾斜

EYE_CLOSED_RED_SEC = 1.0   # 连续闭眼超过此秒数 -> 红（疑似瞌睡）
YAWN_YELLOW_SEC = 0.8      # 连续哈欠时长 -> 黄
DISTRACT_YELLOW_SEC = 1.5  # 连续分心 -> 黄
DISTRACT_RED_SEC = 3.0     # 连续分心 -> 红
PHONE_YELLOW_SEC = 1.2     # 手在耳侧持续 -> 黄
PHONE_RED_SEC = 2.5        # 持续更久 -> 红
# 打电话二分类概率阈值。板端实测：真打电话（含右耳被手机挡住、手部点只露出一半）
# 的 EMA 均值约 0.65，个别帧会掉到 0.45~0.55；空手贴耳负样本中位数只有 0.19。
# 取 0.5 既不误报空手贴耳，也不会因为个别帧掉点就清零计时。
PHONE_PROB = float(os.environ.get("DMS_PHONE_PROB", "0.5"))
PHONE_HOLD_SEC = 1.0       # 手短暂离开耳侧时保持打电话状态，避免计时被清零

SMOKE_PROB = 0.6           # 抽烟二分类概率阈值（与 src/smoke.py 保持一致）
SMOKE_YELLOW_SEC = 1.2     # 手嘴边连续检出抽烟 -> 黄
SMOKE_RED_SEC = 2.5        # 持续更久 -> 红

NO_FACE_YELLOW_SEC = 2.5   # 连续检不到人脸 -> 黄（低头出画面、手挡脸、镜头被碰歪）
NO_FACE_RED_SEC = 5.0      # 持续更久 -> 红。阈值比闭眼/分心宽，避免偶发丢帧误报

PERCLOS_YELLOW = 0.35
PERCLOS_RED = 0.50

# 头部姿态零位校准：取最近若干帧的中位数作为零点，消除通用 3D 模型零点偏置
# 与摄像头安装角带来的固定偏差。不能用「启动时定一次就锁死」的做法——启动那
# 一两秒如果没坐正，零位就被固定成歪的，之后无论怎么调整都会一直报分心。
POSE_TRACK_FRAMES = 300     # 滑动窗口长度（约 20 秒）
POSE_REANCHOR_SEC = 20.0    # 姿态持续超限这么久就重新锚定零位（见 update 里的说明）

GREEN, YELLOW, RED = 0, 1, 2

# FaceMesh 中耳朵附近的锚点索引
EAR_RIGHT_IDX, EAR_LEFT_IDX = 234, 454
# 双眼外眼角（用来取眼线高度）
EYE_RIGHT_IDX, EYE_LEFT_IDX = 33, 263


def hand_near_ear(face_landmarks: np.ndarray, hands: list) -> bool:
    """判断是否有手靠近耳侧、且没垂到下巴以下（打电话姿态）。

    只看「离耳近」会把抽烟误判成打电话：手把烟送到嘴边时，指尖到耳侧锚点
    （234/454）的距离只有 0.27~0.5 倍脸宽，落在阈值内，与打电话无法区分。
    真正的区别在高度——打电话的手会抬到耳/眼高度，抽烟的手停在嘴/下巴附近。

    高度改用「手部质心」而不是「最高点」：戴右耳时手掌常被耳廓挡住，21 点里
    只露出一两根手指，用最高点判会时有时无，右侧机位因此更容易漏判成「非打电话」，
    进而让抽烟模型接管、误报抽烟。质心实测：打电话低于眼线 0.33 倍脸宽，
    抽烟 0.77~0.91 倍脸宽。

    门限取 1.00 而不是 0.55/0.75：板端实测右耳打电话时质心在 0.39~0.70 之间，
    但肘搭在桌上、手机压在颈侧时质心会掉到 0.8 以上，卡 0.75 就会判成「不在
    耳侧」，打电话模型根本不跑，抽烟模型接管、把贴着脸的手机当成烟（实测
    smoke 0.98）。抽烟姿势实测质心 1.19 以上，取 1.00 两边都留得住余量。

    但高度门限放宽后，抽烟的手（质心 0.83~0.97）也会进来，打电话模型对它会
    给 0.6~0.8，phone_held 一成立就把抽烟计时压掉。此时唯一稳定的区别是
    「离嘴近且离耳远」：抽烟 dm 0.04~0.09 / d 0.47~0.59，打电话 d 0.10~0.31
    或 0.33~0.64。所以这里显式排除抽烟几何（与 hand_crop.hand_near_mouth
    用同一组常量，保证两个动作互斥）。
    """
    if not hands or face_landmarks is None or len(face_landmarks) == 0:
        if GATE_DEBUG:
            print("[gate] no hand", flush=True)
        return False
    ear_r = np.array(face_landmarks[EAR_RIGHT_IDX][:2])
    ear_l = np.array(face_landmarks[EAR_LEFT_IDX][:2])
    face_w = float(np.linalg.norm(ear_r - ear_l))
    if face_w < 1e-6:
        return False
    eye_y = float((face_landmarks[EYE_RIGHT_IDX][1] + face_landmarks[EYE_LEFT_IDX][1]) / 2.0)
    mouth = (np.array(face_landmarks[13][:2]) + np.array(face_landmarks[14][:2])) / 2.0
    face_cx = float((ear_r[0] + ear_l[0]) / 2.0)
    drop_limit = eye_y + 1.00 * face_w   # 手部质心要高于这条线（图像 y 越小越高）
    limit = 1.00 * face_w
    for hand in hands:              # hand: (21,3)
        pts = np.asarray(hand)[:, :2]
        d = min(float(np.linalg.norm(pts - ear_r, axis=1).min()),
                float(np.linalg.norm(pts - ear_l, axis=1).min()))
        dm = float(np.linalg.norm(pts - mouth, axis=1).min())
        if GATE_DEBUG:
            print(f"[gate] fw={face_w:.0f} eye_y={eye_y:.0f} "
                  f"top={(pts[:, 1].min() - eye_y) / face_w:+.2f} "
                  f"cen={(pts[:, 1].mean() - eye_y) / face_w:+.2f} "
                  f"p25={(np.percentile(pts[:, 1], 25) - eye_y) / face_w:+.2f} "
                  f"d={d / face_w:.2f} dx={(pts[:, 0].mean() - face_cx) / face_w:+.2f} "
                  f"dm={dm / face_w:.2f} -> "
                  f"{pts[:, 1].mean() <= drop_limit and d < limit}", flush=True)
        # 手在嘴前、又不在耳侧 = 抽烟的手。实测抽烟 dm 0.04~0.09 / d 0.47~0.59，
        # 打电话 d 0.10~0.31（贴耳）或 0.33~0.64（指节掠过嘴角），两者只有「离耳
        # 距离」能把它们分开：贴上耳的手 dm 也可能只有 0.07。不排掉这几帧，抽烟
        # 会被送进打电话模型（实测 raw 0.6~0.8），phone_held 把抽烟计时清零。
        if dm < NEAR_MOUTH * face_w and d > EAR_MIN * face_w:
            continue
        if pts[:, 1].mean() > drop_limit:   # 手垂在嘴/下巴以下，不是打电话
            continue
        if d < limit:
            return True
    return False


class _Duration:
    """跟踪某个布尔条件是否持续满足。"""

    def __init__(self) -> None:
        self._since = None

    def duration(self, active: bool, now: float) -> float:
        if not active:
            self._since = None
            return 0.0
        if self._since is None:
            self._since = now
        return now - self._since


class DmsAnalyzer:
    """将逐帧指标聚合为行为状态与告警等级。"""

    def __init__(self) -> None:
        self.perclos = Perclos(window_sec=30.0, min_samples=15)
        self._eye = _Duration()
        self._yawn = _Duration()
        self._distract = _Duration()
        self._phone = _Duration()
        self._phone_hold = 0.0
        self._smoke = _Duration()
        self._no_face = _Duration()      # 人脸丢失时长，不随 reset() 清零
        self._stuck = _Duration()        # 姿态持续超限时长（零位自救用）
        self._cal_buf = deque(maxlen=POSE_TRACK_FRAMES)   # 姿态样本滑动窗口

    def reset(self) -> None:
        self.perclos.reset()
        self._eye = _Duration()
        self._yawn = _Duration()
        self._distract = _Duration()
        self._phone = _Duration()
        self._phone_hold = 0.0
        self._smoke = _Duration()
        # 注意：不清除姿态零位校准（会话级基准）。

    def _calibrate(self, pitch: float, yaw: float, roll: float):
        """返回用于扣除的零位基准。

        用最近 POSE_TRACK_FRAMES 帧的中位数持续更新：正常驾驶时头部大部分时间
        朝前，中位数就稳定在「正视前方」；万一启动时没坐正或摄像头被碰过，
        零位也会在十几秒内自己纠正回来，不会把错误状态永久锁死。

        但样本不能无条件进窗：实测持续低头 25 秒后，滑动中位数被低头姿态带跑，
        扣零后的残差从 +30° 衰减到 +0.4°，告警在 20 秒左右自动消失（漏报）。
        因此这里只学习「不比自己更低头」的姿态（实测低头时 pitch 增大），
        持续低头永远进不了窗；反方向放宽，保证启动时坐姿偏低也能靠抬头修正。
        """
        if self._cal_buf:
            base = np.median(np.array(self._cal_buf), axis=0)
        else:
            base = np.array([pitch, yaw, roll], dtype=np.float64)
        if (pitch <= base[0] + PITCH_DISTRACT
                and abs(yaw - base[1]) <= YAW_DISTRACT
                and abs(roll - base[2]) <= ROLL_DISTRACT):
            self._cal_buf.append((pitch, yaw, roll))
            base = np.median(np.array(self._cal_buf), axis=0)
        return float(base[0]), float(base[1]), float(base[2])

    def recalibrate(self) -> None:
        """清空姿态零位样本，按接下来几秒的姿态重新学习（摄像头被移动后使用）。"""
        self._cal_buf.clear()

    def no_face(self, now: float = None) -> dict:
        """连续检不到人脸时的状态：低头到画面外、手挡脸、镜头被碰歪都算危险。

        计时不能放进 reset()：camera.py 在丢脸期间每帧都会 reset() 复位
        眼/嘴/分心的连续计时，人脸丢失时长必须独立累计。
        """
        now = time.time() if now is None else now
        dur = self._no_face.duration(True, now)
        level = GREEN
        if dur >= NO_FACE_RED_SEC:
            level = RED
            messages = [f"DRIVER NOT VISIBLE {dur:.0f}s"]
        elif dur >= NO_FACE_YELLOW_SEC:
            level = YELLOW
            messages = ["DRIVER NOT VISIBLE"]
        else:
            messages = ["No face detected"]
        return {
            "ear": 0.0, "eye_prob": None, "mar": 0.0, "pitch": 0.0, "yaw": 0.0,
            "roll": 0.0, "eye_closed": False, "yawning": False, "distracted": False,
            "phone": False, "phone_prob": None, "perclos": 0.0, "no_face_sec": dur,
            "level": level, "messages": messages,
        }

    def update(self, ear: float, mar: float, pitch: float, yaw: float, roll: float,
               phone_visible: bool, closed_prob: float = None,
               smoke_prob: float = 0.0, phone_prob: float = None,
               now: float = None) -> dict:
        now = time.time() if now is None else now
        self._no_face.duration(False, now)   # 重新检到人脸，丢失计时归零

        raw = (pitch, yaw, roll)
        base = self._calibrate(*raw)
        pitch, yaw, roll = raw[0] - base[0], raw[1] - base[1], raw[2] - base[2]

        # 零位自救：启动时如果人没对着摄像头（比如正对着桌上的笔记本），零位会被
        # 定成当时的姿态；而 _calibrate 的过滤条件又会把「能纠正它」的样本挡在窗
        # 外，于是之后无论怎么坐正都一直报分心（板端实测：实际 yaw +35°，扣零后
        # 显示 +105°，DISTRACTION 每秒刷一次）。所以持续超限 20 秒就清窗重新锚定
        # 到当前姿态——此时黄色/红色分心告警早已触发过，不会漏掉真实的分心。
        out_of_range = (abs(yaw) > YAW_DISTRACT or abs(pitch) > PITCH_DISTRACT
                        or abs(roll) > ROLL_DISTRACT)
        if self._stuck.duration(out_of_range, now) >= POSE_REANCHOR_SEC:
            self._cal_buf.clear()
            self._stuck.duration(False, now)
            base = self._calibrate(*raw)      # 窗口已空，基准直接取当前姿态
            pitch, yaw, roll = raw[0] - base[0], raw[1] - base[1], raw[2] - base[2]

        # 优先用眼态模型；模型缺失/裁剪失败时回退到 EAR
        distracted = (abs(yaw) > YAW_DISTRACT or abs(pitch) > PITCH_DISTRACT
                      or abs(roll) > ROLL_DISTRACT)

        # 头部偏离正视时（转头、歪头、低头看手机），两只眼在画面里都被透视压扁，
        # 眼态模型会把「被压扁的睁眼」判成闭眼。板端实测：正视睁眼 0.02~0.2，
        # 真正闭眼 0.93~0.99，而低头/歪头时睁眼会落到 0.5~0.85 这条带子里，
        # 连续半秒到一秒就误报 DROWSINESS。此时把阈值提到 0.85，只认高置信度
        # 的闭眼；真打瞌睡时模型输出仍在 0.93 以上，不会漏。
        # 低头看仪表/手机时上眼睑会盖住虹膜，眼态模型看到的纹理和真闭眼几乎
        # 一样（板端实测 0.9~0.99 与真闭眼完全重叠），这种情况不再用眼态模型的
        # 结论，否则低头一眼就会报「闭眼」。低头本身由姿态判定给出分心告警。
        if pitch > PITCH_EYE_LIMIT:
            eye_closed = False
        elif closed_prob is not None:
            thr = EYE_CLOSED_PROB_TILT if distracted else EYE_CLOSED_PROB
            eye_closed = closed_prob > thr
        else:
            eye_closed = ear < EAR_CLOSED
        yawning = mar > MAR_YAWN

        pclos = self.perclos.update(eye_closed)

        eye_dur = self._eye.duration(eye_closed, now)
        yawn_dur = self._yawn.duration(yawning, now)
        dist_dur = self._distract.duration(distracted, now)
        # 有打电话模型时以模型结果为准（几何规则区分不了扶眼镜/挠头），
        # 模型缺失时回退到纯几何判定
        phone = phone_visible if phone_prob is None else (phone_prob > PHONE_PROB)

        # 手部关键点逐帧抖动会让 phone 掉一两帧，计时被清零就永远到不了 1.2s。
        # 手短暂离开耳侧后保持 PHONE_HOLD_SEC，让持续时间能连续累计。
        if phone:
            self._phone_hold = now
        phone_held = phone or (now - self._phone_hold <= PHONE_HOLD_SEC)

        # 打电话与抽烟互斥：贴耳的姿态手里不会有烟。判定顺序上打电话优先——
        # 之前反过来（抽烟成立就压掉打电话），结果右侧机位打电话先被抽烟模型
        # 误判成抽烟，真打电话反而被永久压住。现在只要模型判为打电话，抽烟
        # 计时立即清零；模型不认为在打电话时，才允许抽烟累计。
        phone_dur = self._phone.duration(phone_held, now)
        smoke_dur = self._smoke.duration(smoke_prob > SMOKE_PROB and not phone_held, now)

        level = GREEN
        messages = []

        if eye_dur >= EYE_CLOSED_RED_SEC:
            level = max(level, RED)
            messages.append("DROWSINESS: eyes closed")
        if pclos >= PERCLOS_RED:
            level = max(level, RED)
            messages.append(f"PERCLOS {pclos:.2f} critical")
        elif pclos >= PERCLOS_YELLOW:
            level = max(level, YELLOW)
            messages.append(f"PERCLOS {pclos:.2f} high")

        if yawn_dur >= YAWN_YELLOW_SEC:
            level = max(level, YELLOW)
            messages.append("YAWNING")

        # 打电话/抽烟是主行为，优先级高于分心：司机打电话时眼神本来就会乱飘，
        # 若让分心抢先成立，提示会被「DISTRACTION」盖掉，看不出真正危险的动作。
        # 因此只要主行为成立，分心的提示与等级都不再参与。
        smoking = smoke_dur >= SMOKE_YELLOW_SEC
        phone_active = phone_dur >= PHONE_YELLOW_SEC
        action = smoking or phone_active

        if smoking:
            level = max(level, RED if smoke_dur >= SMOKE_RED_SEC else YELLOW)
            messages.append("SMOKING" if smoke_dur >= SMOKE_RED_SEC else "Possible smoking")
        elif phone_dur >= PHONE_RED_SEC:
            level = max(level, RED)
            messages.append("PHONE CALL")
        elif phone_active:
            level = max(level, YELLOW)
            messages.append("Possible phone call")

        if not action and not phone_held:
            if dist_dur >= DISTRACT_RED_SEC:
                level = max(level, RED)
                messages.append("DISTRACTION: sustained")
            elif dist_dur >= DISTRACT_YELLOW_SEC:
                level = max(level, YELLOW)
                messages.append("DISTRACTION")

        return {
            "ear": ear,
            "eye_prob": closed_prob,
            "mar": mar,
            "pitch": pitch,
            "yaw": yaw,
            "roll": roll,
            "eye_closed": eye_closed,
            "yawning": yawning,
            "distracted": distracted,
            "phone": phone,
            "phone_prob": phone_prob,
            "smoke": smoke_prob,
            "perclos": pclos,
            "level": level,
            "messages": messages,
        }