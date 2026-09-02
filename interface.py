import sys
import math
import random
from PySide6.QtCore import Qt, QTimer, QPointF, Signal
from PySide6.QtGui import (
    QPainter,
    QColor,
    QPen,
    QBrush,
    QRadialGradient,
    QLinearGradient,
    QFont,
)
from PySide6.QtWidgets import QApplication, QWidget


class DummyInterface(QWidget):
    close_requested = Signal()

    def __init__(self):
        super().__init__()

        self.setWindowTitle("DUMMY")
        self.resize(1000, 700)
        self.setMinimumSize(800, 600)

        self.state = "STARTING"
        self.time = 0
        self.pulse = 0
        self.rotation = 0

        self.particles = []

        for _ in range(180):
            self.particles.append({
                "angle": random.uniform(0, math.pi * 2),
                "radius": random.uniform(170, 340),
                "speed": random.uniform(0.0005, 0.002),
                "size": random.uniform(0.5, 2.2),
                "alpha": random.randint(40, 150),
            })

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.animate)
        self.timer.start(16)

        self.setAttribute(Qt.WA_TranslucentBackground)
        self._close_allowed = False

    def allow_close(self):
        self._close_allowed = True

    def closeEvent(self, event):
        if self._close_allowed:
            event.accept()
        else:
            event.ignore()
            self.close_requested.emit()

    # ---------------------------------------------------------
    # STATE
    # ---------------------------------------------------------

    def set_state(self, state):
        self.state = state.upper()
        self.update()

    # ---------------------------------------------------------
    # ANIMATION
    # ---------------------------------------------------------

    def animate(self):
        self.time += 1
        self.rotation += 0.25

        if self.state == "LISTENING":
            self.pulse += 0.09
        elif self.state == "THINKING":
            self.pulse += 0.045
        elif self.state == "SPEAKING":
            self.pulse += 0.075
        else:
            self.pulse += 0.025

        for p in self.particles:
            p["angle"] += p["speed"]

        self.update()

    # ---------------------------------------------------------
    # PAINT
    # ---------------------------------------------------------

    def paintEvent(self, event):
        painter = QPainter(self)

        painter.setRenderHint(QPainter.Antialiasing)

        w = self.width()
        h = self.height()

        cx = w / 2
        cy = h / 2 - 20

        # -----------------------------------------------------
        # BACKGROUND
        # -----------------------------------------------------

        painter.fillRect(
            0,
            0,
            w,
            h,
            QColor(3, 3, 6)
        )

        # subtle red atmospheric glow
        gradient = QRadialGradient(
            QPointF(cx, cy),
            420
        )

        gradient.setColorAt(
            0,
            QColor(80, 0, 12, 40)
        )

        gradient.setColorAt(
            0.45,
            QColor(40, 0, 8, 20)
        )

        gradient.setColorAt(
            1,
            QColor(0, 0, 0, 0)
        )

        painter.setBrush(gradient)
        painter.setPen(Qt.NoPen)

        painter.drawEllipse(
            QPointF(cx, cy),
            420,
            420
        )

        # -----------------------------------------------------
        # TOP TITLE
        # -----------------------------------------------------

        painter.setPen(QColor(255, 255, 255, 210))

        painter.setFont(
            QFont(
                "SF Pro Display",
                15,
                QFont.Medium
            )
        )

        painter.drawText(
            0,
            38,
            w,
            30,
            Qt.AlignCenter,
            "D U M M Y"
        )

        painter.setPen(QColor(255, 255, 255, 55))

        painter.setFont(
            QFont(
                "SF Pro Display",
                8
            )
        )

        painter.drawText(
            0,
            60,
            w,
            20,
            Qt.AlignCenter,
            "LOCAL INTELLIGENCE SYSTEM"
        )

        # -----------------------------------------------------
        # PARTICLES
        # -----------------------------------------------------

        for p in self.particles:

            angle = p["angle"]

            radius = (
                p["radius"]
                + math.sin(
                    self.time * 0.01 + angle * 3
                ) * 10
            )

            x = cx + math.cos(angle) * radius
            y = cy + math.sin(angle) * radius * 0.62

            alpha = int(
                p["alpha"]
                * (
                    0.65
                    + 0.35
                    * math.sin(
                        self.time * 0.02 + angle
                    )
                )
            )

            painter.setPen(Qt.NoPen)

            painter.setBrush(
                QColor(
                    220,
                    35,
                    55,
                    max(10, alpha)
                )
            )

            painter.drawEllipse(
                QPointF(x, y),
                p["size"],
                p["size"]
            )

        # -----------------------------------------------------
        # OUTER RINGS
        # -----------------------------------------------------

        self.draw_ring(
            painter,
            cx,
            cy,
            260,
            1,
            90
        )

        self.draw_ring(
            painter,
            cx,
            cy,
            230,
            -0.7,
            120
        )

        self.draw_ring(
            painter,
            cx,
            cy,
            195,
            0.5,
            80
        )

        # -----------------------------------------------------
        # AUDIO REACTIVE RING
        # -----------------------------------------------------

        pulse = math.sin(self.pulse) * 8

        if self.state == "LISTENING":
            pulse += math.sin(self.time * 0.25) * 12

        if self.state == "SPEAKING":
            pulse += math.sin(self.time * 0.35) * 15

        self.draw_ring(
            painter,
            cx,
            cy,
            150 + pulse,
            1,
            180
        )

        # -----------------------------------------------------
        # CENTRAL ORB
        # -----------------------------------------------------

        self.draw_orb(
            painter,
            cx,
            cy,
            105 + pulse * 0.35
        )

        # -----------------------------------------------------
        # WAVEFORM
        # -----------------------------------------------------

        self.draw_waveform(
            painter,
            cx,
            cy + 150
        )

        # -----------------------------------------------------
        # STATE
        # -----------------------------------------------------

        painter.setFont(
            QFont(
                "SF Pro Display",
                11,
                QFont.Medium
            )
        )

        painter.setPen(
            QColor(
                255,
                255,
                255,
                190
            )
        )

        painter.drawText(
            0,
            h - 90,
            w,
            25,
            Qt.AlignCenter,
            self.state
        )

        # -----------------------------------------------------
        # STATUS DOT
        # -----------------------------------------------------

        painter.setBrush(
            QColor(230, 35, 55, 220)
        )

        painter.setPen(Qt.NoPen)

        painter.drawEllipse(
            QPointF(
                cx - 4,
                h - 52
            ),
            3,
            3
        )

        painter.setPen(
            QColor(
                255,
                255,
                255,
                70
            )
        )

        painter.setFont(
            QFont(
                "SF Pro Display",
                8
            )
        )

        painter.drawText(
            0,
            h - 55,
            w,
            20,
            Qt.AlignCenter,
            "SYSTEM ONLINE  •  GEMMA 3  •  LOCAL"
        )

    # ---------------------------------------------------------
    # ORB
    # ---------------------------------------------------------

    def draw_orb(self, painter, cx, cy, radius):

        # outer glow
        glow = QRadialGradient(
            QPointF(cx, cy),
            radius * 1.7
        )

        glow.setColorAt(
            0,
            QColor(255, 25, 50, 100)
        )

        glow.setColorAt(
            0.3,
            QColor(150, 5, 25, 55)
        )

        glow.setColorAt(
            1,
            QColor(0, 0, 0, 0)
        )

        painter.setBrush(glow)
        painter.setPen(Qt.NoPen)

        painter.drawEllipse(
            QPointF(cx, cy),
            radius * 1.7,
            radius * 1.7
        )

        # core
        core = QRadialGradient(
            QPointF(
                cx - radius * 0.25,
                cy - radius * 0.25
            ),
            radius
        )

        core.setColorAt(
            0,
            QColor(255, 245, 245)
        )

        core.setColorAt(
            0.12,
            QColor(255, 90, 105)
        )

        core.setColorAt(
            0.45,
            QColor(130, 5, 25)
        )

        core.setColorAt(
            0.8,
            QColor(35, 0, 8)
        )

        core.setColorAt(
            1,
            QColor(5, 0, 2)
        )

        painter.setBrush(core)

        painter.drawEllipse(
            QPointF(cx, cy),
            radius,
            radius
        )

        # bright center
        painter.setBrush(
            QColor(255, 230, 230, 210)
        )

        painter.drawEllipse(
            QPointF(
                cx - radius * 0.12,
                cy - radius * 0.12
            ),
            radius * 0.12,
            radius * 0.12
        )

    # ---------------------------------------------------------
    # RING
    # ---------------------------------------------------------

    def draw_ring(
        self,
        painter,
        cx,
        cy,
        radius,
        direction,
        alpha
    ):

        painter.save()

        painter.translate(cx, cy)

        painter.rotate(
            self.rotation * direction
        )

        pen = QPen(
            QColor(
                220,
                30,
                50,
                alpha
            )
        )

        pen.setWidthF(1.2)

        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)

        painter.drawArc(
            -radius,
            -radius,
            radius * 2,
            radius * 2,
            0,
            250 * 16
        )

        painter.drawArc(
            -radius,
            -radius,
            radius * 2,
            radius * 2,
            290 * 16,
            55 * 16
        )

        painter.restore()

    # ---------------------------------------------------------
    # WAVEFORM
    # ---------------------------------------------------------

    def draw_waveform(
        self,
        painter,
        cx,
        cy
    ):

        pen = QPen(
            QColor(
                220,
                35,
                55,
                130
            )
        )

        pen.setWidthF(1.2)

        painter.setPen(pen)

        points = []

        for i in range(180):

            x = cx - 180 + i * 2

            distance = abs(i - 90) / 90

            amplitude = (
                18
                * (1 - distance)
            )

            if self.state == "LISTENING":
                amplitude *= 1.8

            elif self.state == "SPEAKING":
                amplitude *= 2.2

            elif self.state == "THINKING":
                amplitude *= 0.7

            y = (
                cy
                + math.sin(
                    i * 0.22
                    + self.time * 0.18
                )
                * amplitude
            )

            points.append(
                QPointF(x, y)
            )

        for i in range(
            len(points) - 1
        ):
            painter.drawLine(
                points[i],
                points[i + 1]
            )


# -------------------------------------------------------------
# MAIN
# -------------------------------------------------------------

if __name__ == "__main__":

    app = QApplication(sys.argv)

    window = DummyInterface()

    window.show()

    sys.exit(
        app.exec()
    )
