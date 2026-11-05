"""第一人称驾驶场景。

纯 QPainter 绘制的透视路面：天空 / 远山 / 路面 / 车道线 / 路侧灯柱 / 驾驶舱
方向盘与双手。←/→ 键控制转向，也可用鼠标/触摸拖动方向盘转动；右下角是
油门与刹车踏板，按住即可加减速。方向盘与路面消失点同步摆动，制造「在开车」
的第一人称感受。该控件不参与任何检测逻辑，只做视觉呈现。
"""

import math
import time

from PySide6.QtCore import QPointF, QRectF, Qt, QTimer
from PySide6.QtGui import (
    QColor,
    QEventPoint,
    QFont,
    QLinearGradient,
    QPainter,
    QPainterPath,
    QPen,
    QPolygonF,
    QRadialGradient,
)
from PySide6.QtWidgets import QWidget

ACCENT = QColor("#38bdf8")

HORIZON = 0.40          # 地平线位置（占高度比例）
ROAD_BOTTOM = 0.88      # 路面在画面底部的延伸位置
ROAD_HALF = 0.46        # 路面在画面底部的半宽（占宽度比例）
SAMPLES = 56            # 路面采样段数


def _mono(size: int, bold: bool = False) -> QFont:
    font = QFont("DejaVu Sans Mono")
    font.setStyleHint(QFont.Monospace)
    font.setPixelSize(size)
    font.setBold(bold)
    return font


def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v


def _dist(a: QPointF, b: QPointF) -> float:
    return math.hypot(a.x() - b.x(), a.y() - b.y())


def _angle_delta(rad: float) -> float:
    """把角度差归一到 (-pi, pi]，避免跨越 ±180° 时跳动。"""
    while rad > math.pi:
        rad -= 2 * math.pi
    while rad <= -math.pi:
        rad += 2 * math.pi
    return rad


class DriveScene(QWidget):
    """第一人称驾驶画面（含转向交互与仪表 HUD）。"""

    STEER_RATE = 1.5        # 每秒转向量（heading 单位）
    RECENTER = 0.86         # 松手后回到中位的速度

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("driveScene")
        self.setFocusPolicy(Qt.StrongFocus)
        self.setAttribute(Qt.WA_OpaquePaintEvent, True)
        self.setAttribute(Qt.WA_AcceptTouchEvents, True)   # 触摸屏可拖动方向盘/踏板

        self._heading = 0.0      # -1..1，正为向右
        self._steer = 0          # -1 / 0 / +1
        self._wheel = 0.0        # 方向盘当前角度（度）
        self._drift = 0.0        # 车相对路中心的横向位置，正=车偏左（路面在画面里右移）
        self._phase = 0.0        # 路面滚动相位
        self._kmh = 62.0
        self._cruise = 70.0      # 巡航速度（松开油门/刹车后回到这里）

        # 指针（鼠标 / 触摸）状态：key -> "wheel" / "gas" / "brake"
        self._pointers: dict = {}
        self._anchors: dict = {}     # key -> (起始角度, 起始 heading)
        self._gas = False
        self._brake = False
        self._t0 = time.time()
        self._last = time.time()

        self._timer = QTimer(self)
        # 全屏渐变+多边形在软件渲染下开销大（实测 30fps 约占满 1 个核），
        # 20fps 视觉上仍连贯，CPU 明显下降。
        self._timer.setInterval(50)
        self._timer.timeout.connect(self._tick)
        self._timer.start()

    # ---------------- 交互 ----------------

    def set_steer(self, direction: int) -> None:
        """direction: -1 左, 0 回正, +1 右。"""
        self._steer = _clamp(direction, -1, 1)

    @property
    def heading(self) -> float:
        return self._heading

    def _tick(self) -> None:
        now = time.time()
        dt = _clamp(now - self._last, 0.0, 0.1)
        self._last = now

        if self._steer:
            self._heading += self._steer * self.STEER_RATE * dt
        elif not self._dragging_wheel():
            # 松手慢慢回中，避免一直贴着边缘走（触摸拖住方向盘时不回中）
            self._heading *= self.RECENTER ** (dt * 60)
        self._heading = _clamp(self._heading, -1.0, 1.0)

        # 车往打方向的一侧平移：方向盘左打，车往左开（画面里路面整体右移）
        self._drift = _clamp(self._drift - self._heading * (0.7 + self._kmh / 160.0) * dt,
                             -0.95, 0.95)

        self._wheel += (self._heading * 34.0 - self._wheel) * min(1.0, dt * 10)
        # 按住油门加速 / 按住刹车减速，松手后保持当前车速
        if self._gas:
            self._cruise = min(124.0, self._cruise + 34.0 * dt)
        elif self._brake:
            self._cruise = max(0.0, self._cruise - 58.0 * dt)
        # 车速随画面轻微起伏，并受转向影响，读数不至于死板
        target = self._cruise - abs(self._heading) * 8.0 + math.sin(now * 0.6) * 2.0
        self._kmh += (target - self._kmh) * min(1.0, dt * 2.0)
        self._phase += dt * (self._kmh * 0.055)
        self.update()

    # ---------------- 触摸 / 鼠标交互 ----------------

    def _wheel_geometry(self, w: float, h: float):
        """方向盘圆心与半径（圆心在屏幕下方，只露出上半圈）。"""
        return w * 0.5, h * 1.10, h * 0.42

    def _pedal_geometry(self, w: float, h: float):
        """右下角两个踏板：(油门圆心, 刹车圆心, 半径)。"""
        r = h * 0.046
        y = h * 0.872
        gas = QPointF(w - h * 0.075, y)
        return gas, QPointF(gas.x() - r * 2.7, y), r

    def _dragging_wheel(self) -> bool:
        return "wheel" in self._pointers.values()

    def _begin(self, key, pos: QPointF) -> None:
        """按下：落在踏板上就加速/减速，落在盘圈上就开始拖动方向盘。"""
        w, h = float(self.width()), float(self.height())
        gas, brake, r = self._pedal_geometry(w, h)
        if _dist(pos, gas) <= r * 1.2:
            self._pointers[key] = "gas"
            self._gas = True
        elif _dist(pos, brake) <= r * 1.2:
            self._pointers[key] = "brake"
            self._brake = True
        else:
            cx, cy, wr = self._wheel_geometry(w, h)
            if wr * 0.52 <= _dist(pos, QPointF(cx, cy)) <= wr * 1.22:
                self._pointers[key] = "wheel"
                self._anchors[key] = (math.atan2(pos.y() - cy, pos.x() - cx),
                                      self._heading)
        self.update()

    def _drag(self, key, pos: QPointF) -> None:
        if self._pointers.get(key) != "wheel":
            return
        angle0, heading0 = self._anchors[key]
        cx, cy, _ = self._wheel_geometry(float(self.width()), float(self.height()))
        angle = math.atan2(pos.y() - cy, pos.x() - cx)
        # 方向盘最大摆幅是 34°，按同样比例换算，让盘面跟着手指走
        delta = math.degrees(_angle_delta(angle - angle0))
        self._heading = _clamp(heading0 + delta / 34.0, -1.0, 1.0)

    def _end(self, key) -> None:
        mode = self._pointers.pop(key, None)
        self._anchors.pop(key, None)
        if mode == "gas":
            self._gas = False
        elif mode == "brake":
            self._brake = False
        self.update()

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.LeftButton:
            self._begin("mouse", event.position())
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        if event.buttons() & Qt.LeftButton:
            self._drag("mouse", event.position())
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.LeftButton:
            self._end("mouse")
        super().mouseReleaseEvent(event)

    def touchEvent(self, event) -> None:  # noqa: N802
        """多点触摸：一根手指拖方向盘，另一根手指可以同时踩油门/刹车。"""
        for pt in event.points():
            state = pt.state()
            if state == QEventPoint.State.Pressed:
                self._begin(pt.id(), pt.position())
            elif state == QEventPoint.State.Updated:
                self._drag(pt.id(), pt.position())
            elif state == QEventPoint.State.Released:
                self._end(pt.id())
        event.accept()

    # ---------------- 几何 ----------------

    def _vanishing_x(self, w: float) -> float:
        """消失点位置：左打方向时远处路面略微右移（和近处同向），幅度很小，
        所以看起来是车在横移，而不是整屏镜头在摇头。"""
        return w * 0.5 - self._heading * w * 0.16

    def _center_x(self, s: float, w: float) -> float:
        """s=0 在地平线（消失点），s=1 在画面底部（车头）。

        近处按车的横向位置整体平移（车往左开 -> 路面往右挪），远处几乎不动，
        这样看到的是「车在往左/右开」，而不是「镜头在摇头」。
        """
        near = w * 0.5 + self._drift * w * 0.42
        far = self._vanishing_x(w)
        return far + (near - far) * s

    def _geom(self, h: float):
        horizon = h * HORIZON
        bottom = h * ROAD_BOTTOM
        rows = []
        for i in range(SAMPLES + 1):
            s = (i / SAMPLES) ** 1.7
            rows.append((s, horizon + (bottom - horizon) * s))
        return horizon, rows

    # ---------------- 绘制 ----------------

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        w, h = float(self.width()), float(self.height())
        horizon, rows = self._geom(h)

        self._paint_sky(painter, w, h, horizon)
        self._paint_ground(painter, w, h, horizon)
        road = self._paint_road(painter, w, rows)
        self._paint_posts(painter, w, rows)
        self._paint_headlight(painter, w, h, road)
        self._paint_vignette(painter, w, h)
        self._paint_dashboard(painter, w, h)

    def _paint_sky(self, painter: QPainter, w: float, h: float, horizon: float) -> None:
        sky = QLinearGradient(0, 0, 0, horizon)
        sky.setColorAt(0.0, QColor("#04070d"))
        sky.setColorAt(0.55, QColor("#0a1726"))
        sky.setColorAt(1.0, QColor("#16405f"))
        painter.fillRect(QRectF(0, 0, w, horizon), sky)

        van = self._vanishing_x(w)
        glow = QRadialGradient(QPointF(van, horizon), w * 0.42)
        glow.setColorAt(0.0, QColor(90, 190, 255, 80))
        glow.setColorAt(0.45, QColor(60, 140, 210, 30))
        glow.setColorAt(1.0, QColor(0, 0, 0, 0))
        painter.fillRect(QRectF(0, 0, w, horizon), glow)

        # 远山剪影
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor("#081019"))
        ridge = QPainterPath()
        ridge.moveTo(0, horizon)
        pts = [0.0, 0.10, 0.22, 0.31, 0.44, 0.52, 0.64, 0.75, 0.86, 1.0]
        heights = [0.0, 0.42, 0.18, 0.55, 0.30, 0.62, 0.24, 0.50, 0.15, 0.0]
        for fx, fy in zip(pts, heights):
            ridge.lineTo(w * fx, horizon - h * 0.075 * fy)
        ridge.lineTo(w, horizon)
        ridge.closeSubpath()
        painter.drawPath(ridge)

        # 地平线辉光
        line = QLinearGradient(0, horizon - 4, 0, horizon + 70)
        line.setColorAt(0.0, QColor(90, 190, 255, 55))
        line.setColorAt(1.0, QColor(0, 0, 0, 0))
        painter.fillRect(QRectF(0, horizon - 4, w, 74), line)

    def _paint_ground(self, painter: QPainter, w: float, h: float, horizon: float) -> None:
        ground = QLinearGradient(0, horizon, 0, h)
        ground.setColorAt(0.0, QColor("#0b1a1c"))
        ground.setColorAt(0.55, QColor("#071013"))
        ground.setColorAt(1.0, QColor("#030608"))
        painter.fillRect(QRectF(0, horizon, w, h - horizon), ground)

    def _paint_road(self, painter: QPainter, w: float, rows) -> QPolygonF:
        left, right = [], []
        for s, y in rows:
            c = self._center_x(s, w)
            half = w * ROAD_HALF * s + 1.5
            left.append(QPointF(c - half, y))
            right.append(QPointF(c + half, y))

        road = QPolygonF(left + list(reversed(right)))
        paint = QLinearGradient(0, rows[0][1], 0, rows[-1][1])
        paint.setColorAt(0.0, QColor("#1c2836"))
        paint.setColorAt(0.45, QColor("#2a3441"))
        paint.setColorAt(1.0, QColor("#343f4d"))
        painter.setPen(Qt.NoPen)
        painter.setBrush(paint)
        painter.drawPolygon(road)

        # 车道虚线
        painter.setBrush(QColor(232, 244, 255, 215))
        for i in range(SAMPLES):
            s0, y0 = rows[i]
            s1, y1 = rows[i + 1]
            u = 1.0 / max(s1, 0.014)
            if (u * 0.62 + self._phase) % 1.7 >= 0.95:
                continue
            c0 = self._center_x(s0, w)
            c1 = self._center_x(s1, w)
            dw0 = 1.0 + w * 0.005 * s0
            dw1 = 1.0 + w * 0.005 * s1
            painter.drawPolygon(QPolygonF([
                QPointF(c0 - dw0, y0), QPointF(c0 + dw0, y0),
                QPointF(c1 + dw1, y1), QPointF(c1 - dw1, y1),
            ]))

        # 两侧实线
        painter.setBrush(QColor(226, 240, 252, 170))
        for side in (-1, 1):
            outer, inner = [], []
            for s, y in rows:
                c = self._center_x(s, w)
                half = w * ROAD_HALF * s + 1.5
                t = 3.2 * s + 0.6
                outer.append(QPointF(c + side * half, y))
                inner.append(QPointF(c + side * (half - t), y))
            painter.drawPolygon(QPolygonF(outer + list(reversed(inner))))
        return road

    def _paint_posts(self, painter: QPainter, w: float, rows) -> None:
        """路侧灯柱：随路面一起向观察者移动，越近越大。"""
        painter.setPen(Qt.NoPen)
        for i in range(SAMPLES):
            s, y = rows[i]
            if s < 0.05:
                continue
            u = 1.0 / s
            if (u * 0.45 + self._phase) % 3.4 >= 0.55:
                continue
            c = self._center_x(s, w)
            half = w * ROAD_HALF * s + 1.5
            post_h = 66.0 * s + 1.0
            for side in (-1, 1):
                x = c + side * (half + 14.0 * s)
                glow = QRadialGradient(QPointF(x, y - post_h), 18.0 * s + 3)
                glow.setColorAt(0.0, QColor(120, 220, 255, 140))
                glow.setColorAt(1.0, QColor(56, 189, 248, 0))
                painter.setBrush(glow)
                painter.drawEllipse(QPointF(x, y - post_h), 18.0 * s + 3, 18.0 * s + 3)
                painter.setBrush(QColor(148, 200, 232, 190))
                painter.drawRect(QRectF(x - max(0.6, 1.6 * s), y - post_h,
                                        max(1.2, 3.2 * s), post_h))

    def _paint_headlight(self, painter: QPainter, w: float, h: float,
                         road: QPolygonF) -> None:
        """车灯在路面上的投光，让画面读起来像坐在车里。"""
        painter.save()
        path = QPainterPath()
        path.addPolygon(road)
        path.closeSubpath()
        painter.setClipPath(path)
        cy, rx, ry = h * 0.70, w * 0.44, h * 0.22
        glow = QRadialGradient(QPointF(w * 0.5, cy), rx)
        glow.setColorAt(0.0, QColor(255, 250, 225, 26))
        glow.setColorAt(0.55, QColor(255, 250, 225, 12))
        glow.setColorAt(1.0, QColor(255, 250, 225, 0))
        painter.setBrush(glow)
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(QPointF(w * 0.5, cy), rx, ry)
        painter.restore()

    def _paint_vignette(self, painter: QPainter, w: float, h: float) -> None:
        vig = QRadialGradient(QPointF(w * 0.5, h * 0.46), max(w, h) * 0.75)
        vig.setColorAt(0.55, QColor(0, 0, 0, 0))
        vig.setColorAt(1.0, QColor(0, 0, 0, 150))
        painter.fillRect(QRectF(0, 0, w, h), vig)

    def _paint_dashboard(self, painter: QPainter, w: float, h: float) -> None:
        top = h * 0.80
        panel = QPainterPath()
        panel.moveTo(0, top + h * 0.06)
        panel.quadTo(w * 0.5, top - h * 0.075, w, top + h * 0.06)
        panel.lineTo(w, h)
        panel.lineTo(0, h)
        panel.closeSubpath()

        paint = QLinearGradient(0, top - h * 0.06, 0, h)
        paint.setColorAt(0.0, QColor("#111a24"))
        paint.setColorAt(0.35, QColor("#0a1119"))
        paint.setColorAt(1.0, QColor("#04070b"))
        painter.setPen(QPen(QColor(56, 189, 248, 55), 1.5))
        painter.setBrush(paint)
        painter.drawPath(panel)

        self._paint_wheel(painter, w, h)
        self._paint_pedals(painter, w, h)
        self._paint_cluster(painter, w, h)

    def _paint_wheel(self, painter: QPainter, w: float, h: float) -> None:
        cx, cy, r = self._wheel_geometry(w, h)
        painter.save()
        painter.translate(cx, cy)
        painter.rotate(self._wheel)

        rim = QPen(QColor("#1f2a36"), max(8.0, r * 0.075))
        rim.setCapStyle(Qt.RoundCap)
        painter.setPen(rim)
        painter.setBrush(Qt.NoBrush)
        painter.drawEllipse(QPointF(0, 0), r, r)

        inner = QPen(QColor(56, 189, 248, 110), max(2.0, r * 0.012))
        painter.setPen(inner)
        painter.drawEllipse(QPointF(0, 0), r * 0.965, r * 0.965)

        spoke = QPen(QColor("#232f3d"), max(6.0, r * 0.05))
        spoke.setCapStyle(Qt.RoundCap)
        painter.setPen(spoke)
        for angle in (-90.0 - 62.0, -90.0 + 62.0, 90.0):
            rad = math.radians(angle)
            painter.drawLine(QPointF(0, 0), QPointF(math.cos(rad) * r * 0.94,
                                                    math.sin(rad) * r * 0.94))
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor("#1b2632"))
        painter.drawEllipse(QPointF(0, 0), r * 0.20, r * 0.20)
        painter.setPen(QPen(QColor(56, 189, 248, 150), 1.5))
        painter.setBrush(Qt.NoBrush)
        painter.drawEllipse(QPointF(0, 0), r * 0.20, r * 0.20)

        for side in (-1, 1):
            self._paint_hand(painter, r, side)
        painter.restore()

    def _paint_hand(self, painter: QPainter, r: float, side: int) -> None:
        """握在方向盘上的手（简化的驾驶手套造型）。"""
        angle = math.radians(-90 + side * 52)
        gx, gy = math.cos(angle) * r, math.sin(angle) * r

        painter.save()
        painter.translate(gx, gy)
        painter.rotate(math.degrees(angle) + 90)
        palm_w, palm_h = r * 0.26, r * 0.17
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor("#2f3d4d"))
        painter.drawRoundedRect(QRectF(-palm_w / 2, -palm_h / 2, palm_w, palm_h),
                                palm_h * 0.45, palm_h * 0.45)

        # 四指跨过盘幅
        finger = QPen(QColor("#3d4d5f"), max(3.0, r * 0.024))
        finger.setCapStyle(Qt.RoundCap)
        painter.setPen(finger)
        for k in range(4):
            x = -palm_w * 0.36 + k * palm_w * 0.24
            painter.drawLine(QPointF(x, -palm_h * 0.30), QPointF(x, palm_h * 0.55))

        painter.setPen(QPen(QColor(184, 208, 226, 70), 1.2))
        painter.drawRoundedRect(QRectF(-palm_w / 2, -palm_h / 2, palm_w, palm_h),
                                palm_h * 0.45, palm_h * 0.45)
        painter.restore()

    def _paint_pedals(self, painter: QPainter, w: float, h: float) -> None:
        """右下角的油门 / 刹车踏板，按住（触摸或鼠标）即可加减速。"""
        gas, brake, r = self._pedal_geometry(w, h)
        self._draw_pedal(painter, brake, r, QColor("#ef4444"), self._brake, "BRAKE", up=False)
        self._draw_pedal(painter, gas, r, QColor("#22c55e"), self._gas, "GAS", up=True)

    def _draw_pedal(self, painter: QPainter, c: QPointF, r: float, color: QColor,
                    active: bool, label: str, up: bool) -> None:
        if active:
            glow = QRadialGradient(c, r * 2.4)
            glow.setColorAt(0.0, QColor(color.red(), color.green(), color.blue(), 90))
            glow.setColorAt(1.0, QColor(color.red(), color.green(), color.blue(), 0))
            painter.setPen(Qt.NoPen)
            painter.setBrush(glow)
            painter.drawEllipse(c, r * 2.4, r * 2.4)

        painter.setPen(QPen(QColor(color.red(), color.green(), color.blue(),
                                   235 if active else 110), max(1.4, r * 0.07)))
        painter.setBrush(QColor(6, 12, 20, 205 if active else 150))
        painter.drawEllipse(c, r, r)

        # 箭头：油门朝上、刹车朝下
        pen = QPen(QColor(color.red(), color.green(), color.blue(), 255 if active else 175),
                   max(2.0, r * 0.16))
        pen.setCapStyle(Qt.RoundCap)
        pen.setJoinStyle(Qt.RoundJoin)
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)
        sign = 1.0 if up else -1.0
        chevron = QPainterPath()
        chevron.moveTo(c.x() - r * 0.34, c.y() + sign * r * 0.16)
        chevron.lineTo(c.x(), c.y() - sign * r * 0.24)
        chevron.lineTo(c.x() + r * 0.34, c.y() + sign * r * 0.16)
        painter.drawPath(chevron)

        painter.setFont(_mono(max(7, int(r * 0.40)), True))
        painter.setPen(QColor(color.red(), color.green(), color.blue(),
                              255 if active else 150))
        painter.drawText(QRectF(c.x() - r, c.y() + r * 0.28, r * 2, r * 0.55),
                         Qt.AlignHCenter | Qt.AlignTop, label)

    def _paint_cluster(self, painter: QPainter, w: float, h: float) -> None:
        """仪表：车速与转向量，浮在画面左上角的天空区域。"""
        x0 = w * 0.042
        y0 = h * 0.075
        card = QRectF(x0 - 14, y0 - 12, w * 0.175, h * 0.165)
        painter.setPen(QPen(QColor(70, 110, 140, 80), 1))
        painter.setBrush(QColor(6, 12, 20, 150))
        painter.drawRoundedRect(card, 12, 12)

        painter.setFont(_mono(int(h * 0.013), True))
        painter.setPen(QColor(123, 147, 168, 180))
        painter.drawText(QRectF(x0, y0, 140, 16), Qt.AlignLeft | Qt.AlignTop, "SPEED")

        painter.setFont(_mono(int(h * 0.048), True))
        painter.setPen(QColor("#eaf6ff"))
        painter.drawText(QRectF(x0, y0 + h * 0.014, 200, h * 0.055),
                         Qt.AlignLeft | Qt.AlignTop, f"{int(round(self._kmh))}")

        painter.setFont(_mono(int(h * 0.013), True))
        painter.setPen(QColor(123, 147, 168, 190))
        painter.drawText(QRectF(x0 + w * 0.052, y0 + h * 0.048, 80, 16),
                         Qt.AlignLeft | Qt.AlignTop, "KM/H")

        # 转向量刻度条
        bar = QRectF(x0, y0 + h * 0.082, w * 0.105, 5)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(255, 255, 255, 26))
        painter.drawRoundedRect(bar, 3, 3)
        knob_x = bar.center().x() + self._heading * bar.width() * 0.42
        painter.setBrush(ACCENT if abs(self._heading) < 0.75 else QColor("#f59e0b"))
        painter.drawRoundedRect(QRectF(knob_x - 10, bar.top() - 2, 20, 9), 4, 4)
