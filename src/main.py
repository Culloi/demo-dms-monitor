"""DMS 驾驶员监控 Demo —— 座舱式主界面。

布局：整个窗口是第一人称驾驶画面，右上角浮着驾驶员摄像头小窗与指标面板，
左下角是控制条与日志，检测到危险行为时整屏闪红光。界面文案均为英文。

按键：← / → 转向，Space 开关摄像头，D 开关日志，R 重标定姿态零位，Esc 退出。
控制条里的 LOG 按钮（可触摸）与 D 键等效。
"""

import os
import sys
import time

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QApplication, QHBoxLayout, QLabel, QPlainTextEdit,
                               QPushButton, QVBoxLayout, QWidget)

from alert import VoiceAlerter
from camera import DmsThread
from drive import DriveScene
from panel import AlertOverlay, DmsPanel

ASSETS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "assets")

PANEL_W, PANEL_H = 400, 400

QSS = """
QWidget#ctrlCard {
    background: rgba(6, 12, 20, 220);
    border: 1px solid rgba(70, 110, 140, 90);
    border-radius: 14px;
}
QLabel#ctrlHint {
    color: #5d7488; font-family: monospace; font-size: 10px;
}
QLabel#ctrlStatus {
    color: #cfe0f0; font-family: monospace; font-size: 10px;
}
QPushButton#primaryBtn {
    background: rgba(56, 189, 248, 235); color: #04121c; border: none;
    border-radius: 9px; padding: 9px 18px;
    font-family: monospace; font-size: 11px; font-weight: 700;
}
QPushButton#primaryBtn:hover { background: #64c8f8; }
QPushButton#primaryBtn:disabled {
    background: rgba(56, 189, 248, 40); color: #4a6b80;
}
QPushButton#ghostBtn {
    background: rgba(255, 255, 255, 14); color: #cfe0f0;
    border: 1px solid rgba(120, 160, 190, 80); border-radius: 9px;
    padding: 9px 16px; font-family: monospace; font-size: 11px; font-weight: 600;
}
QPushButton#ghostBtn:hover { background: rgba(255, 255, 255, 26); }
QPushButton#ghostBtn:disabled {
    color: #44576a; border-color: rgba(120, 160, 190, 40); background: rgba(255, 255, 255, 6);
}
QPlainTextEdit#logView {
    background: rgba(4, 8, 13, 200);
    border: 1px solid rgba(70, 110, 140, 70);
    border-radius: 12px; padding: 6px;
    color: #7fd6a8; font-family: monospace; font-size: 10px;
}
"""

LEVEL_NAMES = {0: "NORMAL", 1: "WARNING", 2: "ALERT"}


class MainWindow(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("DMS Cockpit - Driver Monitoring Demo")
        self.resize(1280, 800)
        self.setStyleSheet(QSS)
        self.setFocusPolicy(Qt.StrongFocus)

        self.thread: DmsThread | None = None
        self._last_level = 0
        self._last_msg = ""
        self._tick_t = 0.0
        self.voice = VoiceAlerter(ASSETS_DIR)

        self.scene = DriveScene(self)
        self.panel = DmsPanel(self)
        self.alert = AlertOverlay(self)

        self.ctrl = QWidget(self)
        self.ctrl.setObjectName("ctrlCard")
        self.start_btn = QPushButton("Start Camera")
        self.start_btn.setObjectName("primaryBtn")
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setObjectName("ghostBtn")
        self.stop_btn.setEnabled(False)
        self.log_btn = QPushButton("LOG")
        self.log_btn.setObjectName("ghostBtn")
        self.start_btn.clicked.connect(self._start)
        self.stop_btn.clicked.connect(self._stop)
        self.log_btn.clicked.connect(self._toggle_log)
        for btn in (self.start_btn, self.stop_btn, self.log_btn):
            btn.setFocusPolicy(Qt.NoFocus)

        self.status = QLabel("CAMERA OFFLINE")
        self.status.setObjectName("ctrlStatus")
        hint = QLabel("ARROWS / DRAG WHEEL   GAS / BRAKE   SPACE   R = RECAL")
        hint.setObjectName("ctrlHint")

        row = QHBoxLayout(self.ctrl)
        row.setContentsMargins(12, 8, 14, 8)
        row.setSpacing(10)
        row.addWidget(self.start_btn)
        row.addWidget(self.stop_btn)
        row.addWidget(self.log_btn)
        row.addSpacing(4)
        column = QVBoxLayout()
        column.setSpacing(1)
        column.addWidget(self.status)
        column.addWidget(hint)
        row.addLayout(column, 1)

        self.log = QPlainTextEdit(self)
        self.log.setObjectName("logView")
        self.log.setReadOnly(True)
        self.log.setFocusPolicy(Qt.NoFocus)
        self.log.setVisible(False)   # 默认收起，按 D 展开，避免遮挡驾驶画面

        self._append_log("Cockpit ready")
        self._layout_children()

    # ---------------- 布局 ----------------

    def _layout_children(self) -> None:
        w, h = self.width(), self.height()
        self.scene.setGeometry(0, 0, w, h)
        self.panel.setGeometry(w - PANEL_W - 16, 16, PANEL_W, PANEL_H)
        ctrl_h = 62
        self.ctrl.setGeometry(16, h - ctrl_h - 16, 600, ctrl_h)
        log_h = 118
        self.log.setGeometry(16, h - ctrl_h - 16 - 12 - log_h, 600, log_h)
        self.alert.setGeometry(0, 0, w, h)

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._layout_children()

    # ---------------- 日志 ----------------

    def _append_log(self, text: str) -> None:
        self.log.appendPlainText(text)

    def _toggle_log(self) -> None:
        """LOG 按钮 / D 键都走这里，触摸屏也能展开或收起日志。"""
        show = not self.log.isVisible()
        self.log.setVisible(show)
        self.log_btn.setText("HIDE" if show else "LOG")
        if show:
            self.log.verticalScrollBar().setValue(
                self.log.verticalScrollBar().maximum())

    # ---------------- 摄像头控制 ----------------

    def _start(self) -> None:
        if self.thread is not None:
            return
        self.start_btn.setEnabled(False)
        self._append_log("Starting ...")
        index = int(sys.argv[1]) if len(sys.argv) > 1 else None
        self.thread = DmsThread(index=index)
        self.thread.frameReady.connect(self._on_frame)
        self.thread.log.connect(self._append_log)
        self.thread.opened.connect(self._on_opened)
        self.thread.failed.connect(self._on_failed)
        self.thread.start()
        self.stop_btn.setEnabled(True)

    def _stop(self) -> None:
        if self.thread is None:
            return
        self.thread.stop()
        self.thread = None
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.status.setText("CAMERA OFFLINE")
        self.panel.clear_frame()
        self.alert.set_level(0)
        self._append_log("Stopped")

    def _toggle(self) -> None:
        if self.thread is None:
            self._start()
        else:
            self._stop()

    def _recalibrate(self) -> None:
        """摄像头被碰过或刚上车姿势不对时，清空姿态零位重新学习。"""
        if self.thread is None:
            return
        self.thread.analyzer.recalibrate()
        self._append_log("Posture baseline reset - sit straight for a few seconds")

    def _on_opened(self, index: int, name: str) -> None:
        self.status.setText(f"/dev/video{index}")
        self._append_log(f"Camera open: /dev/video{index} ({name})")

    def _on_failed(self, msg: str) -> None:
        self._append_log(f"ERROR: {msg}")
        self._stop()

    # ---------------- 每帧 ----------------

    def _on_frame(self, image, result) -> None:
        # 预览缩放与标注已在采集线程完成，这里只交给面板显示
        self.panel.set_frame(image)

        state = result.get("state") or {}
        self.panel.set_state(state, bool(result.get("face")))
        level = int(state.get("level", 0))
        self.alert.set_level(level)
        self.voice.update(state.get("messages", []), level)

        # 按「等级或文案变化」打印：只看等级变化会漏掉同级文案切换
        # （例如等级已是 2 的 PERCLOS，之后又追加 SMOKING 就看不到）。
        messages = ", ".join(state.get("messages", [])) or "normal"
        if level != self._last_level or messages != self._last_msg:
            self._last_level = level
            self._last_msg = messages
            eye = state.get("eye_prob")
            eye_txt = "n/a" if eye is None else f"{eye:.2f}"
            # 附带原始指标，方便判断是「误报」还是真的触发
            self._append_log(
                f"[{LEVEL_NAMES.get(level, 'NORMAL')}] {messages}  "
                f"(ear {state.get('ear', 0.0):.2f} eye {eye_txt} "
                f"mar {state.get('mar', 0.0):.2f} "
                f"yaw {state.get('yaw', 0.0):+.0f} pitch {state.get('pitch', 0.0):+.0f} "
                f"perclos {state.get('perclos', 0.0):.2f} "
                f"smoke {state.get('smoke', 0.0):.2f} "
                f"phone {float(state.get('phone_prob') or 0.0):.2f})")
            # 同步打到标准输出：界面日志在控件里，事后排查只能靠这份 stdout
            print(f"[state {time.strftime('%H:%M:%S')}] level={level} msg={messages} "
                  f"eye={eye_txt} mar={state.get('mar', 0.0):.2f} "
                  f"smoke={state.get('smoke', 0.0):.2f} "
                  f"phone={float(state.get('phone_prob') or 0.0):.2f} "
                  f"pitch={state.get('pitch', 0.0):+.0f} yaw={state.get('yaw', 0.0):+.0f}",
                  flush=True)

        # 每秒打一行指标快照：只看等级变化的话，某个行为「一直正常」就无从核对；
        # 有了这行就能从日志逐项对照测试动作（眼态模型被姿态门限跳过时 eye 为 0.00）。
        now = time.monotonic()
        if now - self._tick_t >= 1.0:
            self._tick_t = now
            eye = state.get("eye_prob")
            print(f"[tick {time.strftime('%H:%M:%S')}] "
                  f"eye={'n/a' if eye is None else f'{eye:.2f}'} "
                  f"smoke={state.get('smoke', 0.0):.2f} "
                  f"phone={float(state.get('phone_prob') or 0.0):.2f} "
                  f"pitch={state.get('pitch', 0.0):+.0f} yaw={state.get('yaw', 0.0):+.0f} "
                  f"level={level} face={bool(result.get('face'))}", flush=True)

    # ---------------- 键盘 ----------------

    def keyPressEvent(self, event) -> None:  # noqa: N802
        if event.isAutoRepeat():
            return
        key = event.key()
        if key == Qt.Key_Left:
            self.scene.set_steer(-1)
        elif key == Qt.Key_Right:
            self.scene.set_steer(1)
        elif key == Qt.Key_Space:
            self._toggle()
        elif key == Qt.Key_D:
            self._toggle_log()
        elif key == Qt.Key_R:
            self._recalibrate()
        elif key == Qt.Key_Escape:
            self.close()
        else:
            super().keyPressEvent(event)

    def keyReleaseEvent(self, event) -> None:  # noqa: N802
        if event.isAutoRepeat():
            return
        if event.key() in (Qt.Key_Left, Qt.Key_Right):
            self.scene.set_steer(0)
        else:
            super().keyReleaseEvent(event)

    def closeEvent(self, event) -> None:  # noqa: N802
        if self.thread is not None:
            self.thread.stop()
            self.thread = None
        super().closeEvent(event)


def main() -> int:
    app = QApplication(sys.argv)
    win = MainWindow()
    win.showMaximized()
    win.raise_()
    win.activateWindow()
    win.setFocus()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
