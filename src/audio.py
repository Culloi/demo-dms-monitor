import os
import shutil
import struct
import subprocess
import threading
import wave

ALSA_DEVICE = "hw:0,0"
PEAK = 0.45          # 提示音文件归一化峰值（-7dB），对应系统音量 100%
_LOCK = threading.Lock()


def _run(cmd: list, timeout: float = 30.0):
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return out.returncode, (out.stdout or "") + (out.stderr or "")
    except (OSError, subprocess.SubprocessError):
        return -1, ""


def _sink_volume():
    """返回 (系统音量 0~1, 是否静音)；读不到时按满音量处理。"""
    if shutil.which("wpctl") is None:
        return 1.0, False
    code, out = _run(["wpctl", "get-volume", "@DEFAULT_AUDIO_SINK@"], timeout=5.0)
    if code != 0 or "Volume:" not in out:
        return 1.0, False
    try:
        vol = float(out.split("Volume:")[1].split()[0])
    except (IndexError, ValueError):
        return 1.0, False
    return max(0.0, min(1.0, vol)), "[MUTED]" in out


def _read_wav(path: str):
    with wave.open(path, "rb") as w:
        if w.getsampwidth() != 2:
            return None
        return (w.getframerate(), w.getnchannels(), w.getnframes(),
                w.readframes(w.getnframes()))


def _scale(raw: bytes, gain: float) -> bytes:
    n = len(raw) // 2
    samples = struct.unpack("<%dh" % n, raw)
    return struct.pack("<%dh" % n,
                       *[max(-32768, min(32767, int(v * gain))) for v in samples])


def _play_alsa(wav, gain: float) -> bool:
    if shutil.which("aplay") is None:
        return False
    rate, channels, _, raw = wav
    data = raw if gain >= 0.999 else _scale(raw, gain)
    cmd = ["aplay", "-q", "-D", ALSA_DEVICE, "-t", "raw",
           "-f", "S16_LE", "-r", str(rate), "-c", str(channels)]
    try:
        out = subprocess.run(cmd, input=data, capture_output=True, timeout=30.0)
        return out.returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def _play_pipewire(path: str) -> bool:
    """aplay 不可用时的兜底，同样不碰系统音量。"""
    player = next((c for c in ("pw-play", "paplay") if shutil.which(c)), None)
    if player is None:
        return False
    code, _ = _run([player, path])
    return code == 0


def play(path: str) -> bool:
    """按当前系统音量播放提示音；静音时直接跳过。返回是否真的播了。"""
    if not os.path.exists(path):
        return False
    wav = _read_wav(path)
    if wav is None:
        return False
    with _LOCK:
        vol, muted = _sink_volume()
        if muted or vol <= 0.001:
            return False
        if _play_alsa(wav, vol):
            return True
        return _play_pipewire(path)


def play_async(path: str) -> threading.Thread:
    """在后台线程播放，避免阻塞 Qt 界面。"""
    t = threading.Thread(target=play, args=(path,), daemon=True)
    t.start()
    return t


if __name__ == "__main__":
    import sys

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    target = sys.argv[1] if len(sys.argv) > 1 else "assets/focus_drive.wav"
    if not os.path.isabs(target):
        target = os.path.join(root, target)
    vol, muted = _sink_volume()
    print(f"system volume {vol:.2f}{' (muted)' if muted else ''} -> gain {vol:.3f}")
    print("played" if play(target) else "skipped")
