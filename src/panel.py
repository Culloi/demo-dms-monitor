"""座舱右侧的驾驶员监控面板，以及全屏告警闪烁层。

DmsPanel：右上角小窗视频 + 指标条 + 告警条，所有绘制用 QPainter 完成。
AlertOverlay：危险行为时整屏闪红光，警告级为琥珀色边缘辉光。
"""

import math
import time

from PySide6.QtCore import QPointF, QRectF, Qt, QTimer
from PySide6.QtGui import (
    QColor,
    QFont,
    QImage,
    QPainter,
    QPainterPath,
    QPen,
    QFontMetrics,
    QRadialGradient,
)
from PySide6.QtWidgets import QWidget

LEVEL_TEXT = {0: "NORMAL", 1: "WARNING", 2: "ALERT"}
LEVEL_COLOR = {0: QColor("#22c55e"), 1: QColor("#f59e0b"), 2: QColor("#ef4444")}
ACCENT = QColor("#38bdf8")
DIM = QColor("#7b93a8")


def _mono(size: int, bold: bool = False) -> QFont:
    font = QFont("DejaVu Sans Mono")
    font.setStyleHint(QFont.Monospace)
    font.setPixelSize(size)
    font.setBold(bold)
    return font


class DmsPanel(QWidget):
    """驾驶员摄像头小窗 + 实时指标 + 行为告警。"""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setAttribute(Qt.WA_StyledBackground, False)
        self._image: QImage | None = None
        self._state: dict = {}
        self._face = False
        self._level = 0
        self._pulse = 0.0

        self._timer = QTimer(self)
        self._timer.setInterval(50)
        self._timer.timeout.connect(self._tick)
        self._timer.start()

    # ---------------- 数据入口 ----------------

    def set_frame(self, image: QImage) -> None:
        self._image = image
        self.update()

    def clear_frame(self) -> None:
        self._image = None
        self._state = {}
        self._level = 0
        self.update()

    def set_state(self, state: dict, face: bool) -> None:
        self._state = state or {}
        self._face = face
        self._level = int(self._state.get("level", 0))
        self.update()

    def _tick(self) -> None:
        self._pulse = (self._pulse + 0.09) % (2 * math.pi)
        if self._level:
            self.update()

    # ---------------- 绘制 ----------------

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)

        rect = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        color = LEVEL_COLOR.get(self._level, LEVEL_COLOR[0])
        border = QColor(color)
        border.setAlpha(120 + int(90 * (0.5 + 0.5 * math.sin(self._pulse))) if self._level else 70)

        painter.setPen(QPen(border, 1.4))
        painter.setBrush(QColor(6, 12, 20, 216))
        painter.drawRoundedRect(rect, 14, 14)

        self._paint_header(painter, rect, color)
        preview = self._paint_preview(painter, rect, color)
        self._paint_metrics(painter, rect, preview)
        self._paint_alerts(painter, rect, color)

    def _paint_header(self, painter: QPainter, rect: QRectF, color: QColor) -> None:
        painter.setFont(_mono(10, True))
        painter.setPen(DIM)
        painter.drawText(QRectF(rect.left() + 16, rect.top() + 10, rect.width() - 32, 16),
                         Qt.AlignLeft | Qt.AlignVCenter, "DRIVER CAM")

        painter.setFont(_mono(9, True))
        badge = QRectF(rect.right() - 96, rect.top() + 8, 82, 18)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(color.red(), color.green(), color.blue(), 40))
        painter.drawRoundedRect(badge, 9, 9)
        painter.setPen(color)
        painter.drawText(badge, Qt.AlignCenter, LEVEL_TEXT.get(self._level, "NORMAL"))

    def _paint_preview(self, painter: QPainter, rect: QRectF, color: QColor) -> QRectF:
        pad = 14
        area = QRectF(rect.left() + pad, rect.top() + 32,
                      rect.width() - pad * 2, rect.width() * 9 / 16)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor("#04070c"))
        painter.drawRoundedRect(area, 10, 10)

        if self._image is None:
            painter.setFont(_mono(11))
            painter.setPen(QColor(90, 112, 132))
            painter.drawText(area, Qt.AlignCenter, "CAMERA OFFLINE")
            return area

        painter.save()
        # 缩放到面板尺寸（400 -> 372）时开启平滑采样，否则边缘会出现锯齿和摩尔纹
        painter.setRenderHint(QPainter.SmoothPixmapTransform, True)
        path = _rounded_path(area, 10)
        painter.setClipPath(path)
        painter.drawImage(area, self._image)
        painter.restore()

        # 取景角标
        painter.setPen(QPen(QColor(255, 255, 255, 90), 1.2))
        painter.setBrush(Qt.NoBrush)
        for dx, dy in ((1, 1), (-1, 1), (1, -1), (-1, -1)):
            x = area.left() + 6 if dx > 0 else area.right() - 6
            y = area.top() + 6 if dy > 0 else area.bottom() - 6
            painter.drawLine(QPointF(x, y), QPointF(x + dx * 14, y))
            painter.drawLine(QPointF(x, y), QPointF(x, y + dy * 14))

        painter.setPen(QPen(QColor(color), 1.4))
        painter.setBrush(Qt.NoBrush)
        painter.drawRoundedRect(area.adjusted(0.5, 0.5, -0.5, -0.5), 10, 10)

        if not self._face:
            label = QRectF(area.left(), area.bottom() - 24, area.width(), 20)
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor(239, 68, 68, 60))
            painter.drawRoundedRect(label.adjusted(8, 0, -8, 0), 6, 6)
            painter.setFont(_mono(10, True))
            painter.setPen(QColor("#fca5a5"))
            painter.drawText(label, Qt.AlignCenter, "NO FACE")
        return area

    def _paint_metrics(self, painter: QPainter, rect: QRectF, preview: QRectF) -> None:
        top = preview.bottom() + 10
        left = rect.left() + 14
        width = rect.width() - 28
        state = self._state

        # 有眼态模型时直接显示闭眼概率（0~1），否则回退显示 EAR（0~0.4 归一化）
        prob = state.get("eye_prob")
        if prob is None:
            eye_row = ("EYE", state.get("ear", 0.0), 0.40, "#38bdf8")
        else:
            tint = "#ef4444" if float(prob or 0.0) > 0.5 else "#38bdf8"
            eye_row = ("EYE CLOSE", prob, 1.00, tint)
        rows = [
            eye_row,
            ("MOUTH", state.get("mar", 0.0), 1.00, "#a78bfa"),
            ("PERCLOS", state.get("perclos", 0.0), 1.00, "#f59e0b"),
            ("SMOKE", state.get("smoke", 0.0), 1.00, "#f472b6"),
            ("PHONE", float(state.get("phone_prob") or 0.0), 1.00, "#22d3ee"),
        ]
        painter.setFont(_mono(9, True))
        for i, (name, value, full, tint) in enumerate(rows):
            y = top + i * 20
            painter.setPen(QColor(90, 112, 132))
            painter.drawText(QRectF(left, y, 62, 14), Qt.AlignLeft | Qt.AlignVCenter, name)

            bar = QRectF(left + 64, y + 3, width - 128, 8)
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor(255, 255, 255, 20))
            painter.drawRoundedRect(bar, 4, 4)
            ratio = max(0.0, min(1.0, float(value or 0.0) / full))
            if ratio > 0:
                painter.setBrush(QColor(tint))
                painter.drawRoundedRect(QRectF(bar.left(), bar.top(), bar.width() * ratio,
                                               bar.height()), 4, 4)

            painter.setPen(QColor("#e6f1ff"))
            painter.drawText(QRectF(bar.right() + 6, y, 54, 14),
                             Qt.AlignRight | Qt.AlignVCenter, f"{float(value or 0.0):.2f}")

        # 头部姿态
        y = top + 4 * 20
        painter.setPen(QColor(90, 112, 132))
        painter.drawText(QRectF(left, y, 62, 14), Qt.AlignLeft | Qt.AlignVCenter, "HEAD")
        painter.setFont(_mono(9, True))
        painter.setPen(QColor("#cfe0f0"))
        painter.drawText(
            QRectF(left + 64, y, width - 64, 14), Qt.AlignLeft | Qt.AlignVCenter,
            f"yaw {state.get('yaw', 0.0):+5.1f}   pitch {state.get('pitch', 0.0):+5.1f}"
            f"   roll {state.get('roll', 0.0):+5.1f}")

    def _paint_alerts(self, painter: QPainter, rect: QRectF, color: QColor) -> None:
        top = rect.bottom() - 34
        messages = self._state.get("messages") or []
        text = " · ".join(messages[:2]) if messages else "All clear"
        box = QRectF(rect.left() + 14, top, rect.width() - 28, 22)

        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(color.red(), color.green(), color.blue(), 30))
        painter.drawRoundedRect(box, 6, 6)
        font = _mono(10, True)
        painter.setFont(font)
        painter.setPen(color if messages else QColor("#4d6478"))
        elided = QFontMetrics(font).elidedText(text, Qt.ElideRight, int(box.width()) - 20)
        painter.drawText(box.adjusted(10, 0, -10, 0), Qt.AlignLeft | Qt.AlignVCenter, elided)


def _rounded_path(rect: QRectF, radius: float) -> QPainterPath:
    path = QPainterPath()
    path.addRoundedRect(rect, radius, radius)
    return path


class AlertOverlay(QWidget):
    """全屏告警层：ALERT 时整屏脉冲红光，WARNING 时琥珀色边缘辉光。"""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.setAttribute(Qt.WA_NoSystemBackground, True)
        self._level = 0
        self._phase = 0.0
        self._alpha = 0.0
        self._last = time.time()

        self._timer = QTimer(self)
        self._timer.setInterval(33)
        self._timer.timeout.connect(self._tick)
        self._timer.start()

    def set_level(self, level: int) -> None:
        if level != self._level:
            self._level = level
            if level:
                self._phase = 0.0
            self.update()

    def _tick(self) -> None:
        now = time.time()
        dt = min(0.1, now - self._last)
        self._last = now
        if not self._level and self._alpha <= 0.01:
            return
        self._phase = (self._phase + dt * 2.4) % (2 * math.pi)
        target = 1.0 if self._level else 0.0
        self._alpha += (target - self._alpha) * min(1.0, dt * 6)
        self._alpha = max(0.0, min(1.0, self._alpha))
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802
        if self._alpha <= 0.01:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        w, h = float(self.width()), float(self.height())
        pulse = 0.5 + 0.5 * math.sin(self._phase)
        alert = self._level >= 2

        if alert:
            base = QColor(239, 38, 38)
            fill = QColor(base)
            fill.setAlpha(int((26 + 34 * pulse) * self._alpha))
            painter.fillRect(QRectF(0, 0, w, h), fill)

            vig = QRadialGradient(QPointF(w / 2, h / 2), max(w, h) * 0.72)
            vig.setColorAt(0.45, QColor(200, 20, 20, 0))
            vig.setColorAt(1.0, QColor(200, 20, 20, int((120 + 70 * pulse) * self._alpha)))
            painter.fillRect(QRectF(0, 0, w, h), vig)
            return

        edge = QColor(245, 158, 11, int((70 + 50 * pulse) * self._alpha))
        vig = QRadialGradient(QPointF(w / 2, h / 2), max(w, h) * 0.7)
        vig.setColorAt(0.5, QColor(245, 158, 11, 0))
        vig.setColorAt(1.0, edge)
        painter.fillRect(QRectF(0, 0, w, h), vig)
