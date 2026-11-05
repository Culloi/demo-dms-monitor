"""告警语音播报：按当前告警类型播放对应提示音。

规则：
- 同一类型 cooldown 秒内不重复播报，避免连续帧反复触发把喇叭占满
- 类型切换（抽烟 -> 打电话）时立即播报
- 只在告警等级 > 0 时播报；等级归零后复位，下次触发重新播报

播放入口复用 audio.play_async（ALSA 直连、只读系统音量、不修改用户设置）。
"""

import os
import time

import audio

ASSETS = {
    "smoke": "smoke.wav",        # 请勿在车内抽烟
    "phone": "phone.wav",        # 请不要接打电话
    "no_face": "no_face.wav",    # 未检测到人脸，请正视摄像头
    "focus": "focus_drive.wav",  # 请保持专注驾驶（疲劳/分心）
}

COOLDOWN = 8.0


def _category(messages) -> str:
    """从告警文案里取出播放类别；无对应提示音时返回空串。"""
    text = " ".join(str(m) for m in messages).lower()
    if "smok" in text:
        return "smoke"
    if "phone" in text:
        return "phone"
    if "visible" in text or "no face" in text:
        return "no_face"
    if any(k in text for k in ("drows", "perclos", "yawn", "distraction", "eyes")):
        return "focus"
    return ""


class VoiceAlerter:
    def __init__(self, assets_dir: str, cooldown: float = COOLDOWN) -> None:
        self.dir = assets_dir
        self.cooldown = cooldown
        self._last = {}
        self._current = ""

    def update(self, messages, level: int) -> str:
        """按当前告警状态决定是否播报，返回本次播放的类别（未播放返回空串）。"""
        cat = _category(messages) if level > 0 else ""
        if not cat:
            self._current = ""
            return ""
        now = time.time()
        if cat == self._current and now - self._last.get(cat, 0.0) < self.cooldown:
            return ""
        path = os.path.join(self.dir, ASSETS[cat])
        if not os.path.exists(path):
            return ""
        audio.play_async(path)
        self._last[cat] = now
        self._current = cat
        return cat
