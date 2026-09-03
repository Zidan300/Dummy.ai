"""Premium, lightweight Qt interface for Dummy's local voice assistant.

Phase 10: makes the interface feel alive and intent-aware while keeping the
paint loop lightweight (targets ~60 FPS on an M1, no expensive blur, no
per-frame allocations beyond a few QPen/QColor objects, no worker-thread
widget access).
"""

from __future__ import annotations

import math
import random
import sys
import time

from PySide6.QtCore import QEvent, QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import (
    QColor,
    QFont,
    QLinearGradient,
    QPainter,
    QPainterPath,
    QPen,
    QRadialGradient,
)
from PySide6.QtWidgets import QApplication, QWidget

from animations import VisualAnimator


class DummyInterface(QWidget):
    """A single QWidget owned by Qt's GUI thread."""

    close_requested = Signal()
    interrupt_requested = Signal()

    BACKGROUND = QColor(4, 7, 14)
    TEXT = QColor(232, 239, 251)
    MUTED_TEXT = QColor(147, 163, 187)

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("DUMMY")
        self.resize(1000, 700)
        self.setMinimumSize(720, 560)
        self.setFocusPolicy(Qt.StrongFocus)
        QApplication.instance().installEventFilter(self)
        self.setAttribute(Qt.WA_OpaquePaintEvent, True)
        self.setAutoFillBackground(False)

        self.state = "STARTING"
        self._animator = VisualAnimator()
        self._last_frame_at = time.monotonic()
        self._close_allowed = False
        self._particles = self._make_particles()

        self._timer = QTimer(self)
        self._timer.setInterval(16)
        self._timer.timeout.connect(self.animate)
        self._timer.start()

    def _make_particles(self) -> list[dict[str, float]]:
        rng = random.Random(17)
        return [
            {
                "angle": rng.uniform(0.0, math.tau),
                "radius": rng.uniform(0.82, 1.42),
                "speed": rng.uniform(-0.035, 0.055),
                "size": rng.uniform(0.7, 1.8),
                "alpha": rng.uniform(0.16, 0.55),
                "phase": rng.uniform(0.0, math.tau),
            }
            for _ in range(72)
        ]

    def allow_close(self) -> None:
        self._close_allowed = True

    def closeEvent(self, event) -> None:
        if self._close_allowed:
            event.accept()
        else:
            event.ignore()
            self.close_requested.emit()

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key_S and not event.isAutoRepeat():
            self.interrupt_requested.emit()
            event.accept()
            return
        super().keyPressEvent(event)

    def eventFilter(self, watched, event) -> bool:
        if (
            event.type() == QEvent.KeyPress
            and event.key() == Qt.Key_S
            and not event.isAutoRepeat()
            and self.isActiveWindow()
        ):
            self.interrupt_requested.emit()
            event.accept()
            return True
        return super().eventFilter(watched, event)

    # -- Qt-thread-safe signal slots (GUI thread only) ----------------------
    def set_state(self, state: str) -> None:
        """Called through the queued controller.state_changed connection."""
        self.state = state.upper()
        self._animator.set_state(self.state)
        self.update()

    def set_audio_level(self, level: float) -> None:
        """GUI-thread slot for microphone or TTS RMS levels."""
        self._animator.set_audio_level(level)

    def set_question_category(self, category: str) -> None:
        """Set a subtle visual mood from deterministic controller metadata."""
        self._animator.set_category(category)

    def set_intent(self, intent: str) -> None:
        """Set the coarse intent mood (QUESTION/COMMAND/CONVERSATION/...)."""
        self._animator.set_intent(intent)

    def note_activity(self, amount: float = 0.35) -> None:
        """Receive progress pulses through a queued Qt connection."""
        self._animator.note_activity(amount)

    def note_thinking(self) -> None:
        self._animator.note_activity(0.30)

    def note_first_token(self) -> None:
        self._animator.note_generation_milestone(0.50)

    def note_first_sentence(self) -> None:
        self._animator.note_generation_milestone(0.70)

    def note_whisper_first_result(self) -> None:
        self._animator.note_activity(0.40)

    def note_speech_started(self) -> None:
        self._animator.note_speech_started()

    def note_speech_ended(self) -> None:
        self._animator.note_speech_ended()

    def note_interrupted(self) -> None:
        self._animator.note_activity(0.90)

    def animate(self) -> None:
        now = time.monotonic()
        self._animator.tick(now - self._last_frame_at)
        self._last_frame_at = now
        self.update()

    def _accent(self) -> tuple[int, int, int]:
        return self._animator.accent

    @staticmethod
    def _with_alpha(color: tuple[int, int, int], alpha: int) -> QColor:
        return QColor(color[0], color[1], color[2], max(0, min(255, alpha)))

    def paintEvent(self, event) -> None:
        del event
        painter = QPainter(self)
        if not painter.isActive():
            return
        painter.setRenderHint(QPainter.Antialiasing, True)
        try:
            width = float(self.width())
            height = float(self.height())
            center = QPointF(width * 0.5, height * 0.47)
            base_radius = min(width, height) * 0.145
            accent = self._accent()

            self._draw_background(painter, width, height, center, accent)
            self._draw_voice_rings(painter, center, base_radius, accent)
            self._draw_particles(painter, center, base_radius, accent)
            self._draw_hud_arcs(painter, center, base_radius, accent)
            self._draw_core(painter, center, base_radius, accent)
            self._draw_waveform(painter, center, base_radius, accent)
            self._draw_header(painter, width)
            self._draw_status(painter, width, height, accent)
        finally:
            painter.end()

    def _draw_background(
        self,
        painter: QPainter,
        width: float,
        height: float,
        center: QPointF,
        accent: tuple[int, int, int],
    ) -> None:
        painter.fillRect(0, 0, int(width), int(height), self.BACKGROUND)

        atmosphere = QRadialGradient(center, max(width, height) * 0.62)
        atmosphere.setColorAt(0.0, self._with_alpha(accent, 22))
        atmosphere.setColorAt(0.34, QColor(22, 31, 58, 18))
        atmosphere.setColorAt(0.72, QColor(4, 7, 14, 8))
        atmosphere.setColorAt(1.0, QColor(4, 7, 14, 0))
        painter.setPen(Qt.NoPen)
        painter.setBrush(atmosphere)
        painter.drawEllipse(center, max(width, height) * 0.62, max(width, height) * 0.62)

        horizon = QLinearGradient(0, 0, 0, height)
        horizon.setColorAt(0.0, QColor(18, 26, 46, 34))
        horizon.setColorAt(0.42, QColor(5, 8, 16, 0))
        horizon.setColorAt(1.0, QColor(0, 2, 7, 78))
        painter.setBrush(horizon)
        painter.drawRect(QRectF(0, 0, width, height))

    def _draw_voice_rings(
        self,
        painter: QPainter,
        center: QPointF,
        base_radius: float,
        accent: tuple[int, int, int],
    ) -> None:
        """State-specific expanding rings driven by real voice amplitude."""
        ring = self._animator.listening_ring
        speech = self._animator.speech_ring
        painter.setBrush(Qt.NoBrush)
        if ring > 0.02:
            pen = QPen(self._with_alpha(accent, int(38 * ring)))
            pen.setWidthF(1.0 + ring * 1.4)
            painter.setPen(pen)
            painter.drawEllipse(
                center,
                base_radius * (1.7 + ring * 0.9),
                base_radius * (1.7 + ring * 0.9),
            )
        if speech > 0.02:
            pen = QPen(self._with_alpha(accent, int(60 * speech)))
            pen.setWidthF(1.2 + speech * 1.8)
            painter.setPen(pen)
            painter.drawEllipse(
                center,
                base_radius * (1.5 + speech * 1.1),
                base_radius * (1.5 + speech * 1.1),
            )
        scan = self._animator.scan_ring
        if scan > 0.02:
            intensity = self._animator.state_intensity
            pen = QPen(self._with_alpha(accent, int((40 + 30 * scan) * intensity)))
            pen.setWidthF(0.8 + scan)
            painter.setPen(pen)
            painter.drawEllipse(
                center,
                base_radius * (1.3 + scan * 1.6),
                base_radius * (1.3 + scan * 1.6),
            )

    def _draw_particles(
        self,
        painter: QPainter,
        center: QPointF,
        base_radius: float,
        accent: tuple[int, int, int],
    ) -> None:
        energy = self._animator.spark_energy
        intensity = self._animator.state_intensity
        painter.setPen(Qt.NoPen)
        for particle in self._particles:
            angle = particle["angle"] + self._animator.time * particle["speed"] * (0.2 + energy * 0.6)
            radius = base_radius * (3.7 + particle["radius"] * 2.2)
            radius += math.sin(self._animator.time * 0.75 + particle["phase"]) * base_radius * 0.16
            x = center.x() + math.cos(angle) * radius
            y = center.y() + math.sin(angle) * radius * 0.57
            shimmer = 0.65 + 0.35 * math.sin(self._animator.time * 1.4 + particle["phase"])
            alpha = int(255 * particle["alpha"] * shimmer * (0.22 + energy * 0.62) * (0.5 + intensity * 0.5))
            painter.setBrush(self._with_alpha(accent, alpha))
            size = particle["size"] * (0.75 + energy * 0.35)
            painter.drawEllipse(QPointF(x, y), size, size)

        for index in range(9):
            phase = index * 0.71 + 1.3
            angle = phase + self._animator.time * (0.16 + index * 0.012)
            radius = base_radius * (2.3 + (index % 3) * 0.36)
            radius += math.sin(self._animator.time * 1.6 + phase) * base_radius * 0.12
            point = QPointF(
                center.x() + math.cos(angle) * radius,
                center.y() + math.sin(angle) * radius * 0.62,
            )
            alpha = int(150 * self._animator.spark_energy * (0.5 + 0.45 * math.sin(self._animator.time * 2.0 + phase)))
            painter.setBrush(self._with_alpha(accent, max(0, alpha)))
            painter.drawEllipse(point, 1.0 + energy * 0.8, 1.0 + energy * 0.8)

    def _draw_hud_arcs(
        self,
        painter: QPainter,
        center: QPointF,
        base_radius: float,
        accent: tuple[int, int, int],
    ) -> None:
        painter.save()
        painter.translate(center)
        painter.rotate(self._animator.rotation)
        energy = self._animator.energy
        pen = QPen(self._with_alpha(accent, int(42 + energy * 54)))
        pen.setWidthF(1.0)
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)
        sweep = 112 + self._animator.state_intensity * 26
        for radius, start, span in (
            (base_radius * 2.02, 18, sweep),
            (base_radius * 2.18, 198, 62),
            (base_radius * 2.40, 284, 34),
        ):
            painter.drawArc(
                QRectF(-radius, -radius * 0.68, radius * 2, radius * 1.36),
                int(start * 16),
                int(span * 16),
            )
        painter.restore()

        painter.save()
        painter.translate(center)
        painter.rotate(-self._animator.rotation * 0.45 + 16)
        pen = QPen(self._with_alpha(accent, int(24 + energy * 32)))
        pen.setWidthF(0.8)
        painter.setPen(pen)
        painter.drawArc(
            QRectF(-base_radius * 2.58, -base_radius * 0.79, base_radius * 5.16, base_radius * 1.58),
            -35 * 16,
            58 * 16,
        )
        painter.restore()

    def _draw_core(
        self,
        painter: QPainter,
        center: QPointF,
        base_radius: float,
        accent: tuple[int, int, int],
    ) -> None:
        factor = self._animator.core_radius_factor
        radius = base_radius * (1.0 + self._animator.scale * 0.055) * factor
        glow_radius = radius * (2.25 + self._animator.energy * 0.32)

        glow = QRadialGradient(center, glow_radius)
        glow.setColorAt(0.0, self._with_alpha(accent, int(70 + self._animator.energy * 55)))
        glow.setColorAt(0.22, self._with_alpha(accent, int(32 + self._animator.energy * 26)))
        glow.setColorAt(0.62, self._with_alpha(accent, 9))
        glow.setColorAt(1.0, self._with_alpha(accent, 0))
        painter.setPen(Qt.NoPen)
        painter.setBrush(glow)
        painter.drawEllipse(center, glow_radius, glow_radius)

        painter.save()
        painter.setClipPath(self._ellipse_path(center, radius * 0.98))
        inner = QRadialGradient(
            QPointF(center.x() - radius * 0.30, center.y() - radius * 0.34),
            radius * 1.35,
        )
        inner.setColorAt(0.0, QColor(246, 252, 255, 245))
        inner.setColorAt(0.10, self._with_alpha(accent, 235))
        inner.setColorAt(0.38, self._with_alpha(accent, 150))
        inner.setColorAt(0.73, QColor(17, 27, 52, 210))
        inner.setColorAt(1.0, QColor(5, 8, 17, 250))
        painter.setBrush(inner)
        painter.drawEllipse(center, radius, radius)

        currents = 2 if self.state in {"IDLE", "STARTING"} else 3
        flow = self._animator.flow
        for lane in range(currents):
            path = QPainterPath()
            for index in range(34):
                fraction = index / 33.0
                x = center.x() - radius * 1.18 + fraction * radius * 2.36
                y = center.y() + math.sin(fraction * 7.2 + flow + lane * 2.0) * radius * (0.16 + lane * 0.035)
                y += (lane - 1) * radius * 0.22
                if index == 0:
                    path.moveTo(x, y)
                else:
                    path.lineTo(x, y)
            pen = QPen(self._with_alpha(accent, 56 - lane * 11))
            pen.setWidthF(1.2 + lane * 0.45)
            painter.setPen(pen)
            painter.setBrush(Qt.NoBrush)
            painter.drawPath(path)
        painter.restore()

        self._draw_energy_orbits(painter, center, radius, accent)

        edge = QPen(self._with_alpha(accent, int(98 + self._animator.energy * 62)))
        edge.setWidthF(1.1)
        painter.setPen(edge)
        painter.setBrush(Qt.NoBrush)
        painter.drawEllipse(center, radius, radius)

        highlight = QRadialGradient(
            QPointF(center.x() - radius * 0.34, center.y() - radius * 0.38),
            radius * 0.42,
        )
        highlight.setColorAt(0.0, QColor(255, 255, 255, 150))
        highlight.setColorAt(0.36, QColor(255, 255, 255, 34))
        highlight.setColorAt(1.0, QColor(255, 255, 255, 0))
        painter.setPen(Qt.NoPen)
        painter.setBrush(highlight)
        painter.drawEllipse(
            QPointF(center.x() - radius * 0.25, center.y() - radius * 0.30),
            radius * 0.30,
            radius * 0.24,
        )

        nucleus = radius * (0.12 + self._animator.energy * 0.025)
        nucleus_gradient = QRadialGradient(center, nucleus * 1.8)
        nucleus_gradient.setColorAt(0.0, QColor(255, 255, 255, 235))
        nucleus_gradient.setColorAt(0.34, self._with_alpha(accent, 180))
        nucleus_gradient.setColorAt(1.0, self._with_alpha(accent, 0))
        painter.setBrush(nucleus_gradient)
        painter.drawEllipse(center, nucleus * 1.8, nucleus * 1.8)
        painter.setBrush(QColor(255, 255, 255, 225))
        painter.drawEllipse(center, nucleus * 0.32, nucleus * 0.32)

    def _draw_energy_orbits(
        self,
        painter: QPainter,
        center: QPointF,
        radius: float,
        accent: tuple[int, int, int],
    ) -> None:
        energy = self._animator.orbit_energy
        painter.save()
        painter.translate(center)
        painter.rotate(self._animator.rotation * 0.62)
        painter.setBrush(Qt.NoBrush)
        for index, (x_radius, y_radius, start, span) in enumerate(
            (
                (radius * 1.28, radius * 0.48, 206, 96),
                (radius * 1.42, radius * 0.60, 28, 72),
                (radius * 1.16, radius * 0.74, 132, 54),
            )
        ):
            pen = QPen(self._with_alpha(accent, int((24 + energy * 52) / (index + 1))))
            pen.setWidthF(0.8 + energy * 0.45)
            painter.setPen(pen)
            painter.drawArc(
                QRectF(-x_radius, -y_radius, x_radius * 2, y_radius * 2),
                int(start * 16),
                int(span * 16),
            )
        painter.restore()

    @staticmethod
    def _ellipse_path(center: QPointF, radius: float) -> QPainterPath:
        path = QPainterPath()
        path.addEllipse(center, radius, radius)
        return path

    def _draw_waveform(
        self,
        painter: QPainter,
        center: QPointF,
        base_radius: float,
        accent: tuple[int, int, int],
    ) -> None:
        width = base_radius * 3.15
        baseline = center.y() + base_radius * 2.08
        amplitude = base_radius * 0.11 * self._animator.wave_factor
        if self.state == "IDLE":
            amplitude *= 0.40
        if self.state == "INTERRUPTED":
            amplitude *= max(0.18, self._animator.collapse)
        voice = self._animator.voice_level * self._animator.audio_coupling
        phase_speed = 2.2 + voice * 2.4
        path = QPainterPath()
        samples = 96
        for index in range(samples):
            fraction = index / (samples - 1)
            x = center.x() - width * 0.5 + fraction * width
            envelope = math.sin(math.pi * fraction) ** 0.72
            carrier = math.sin(fraction * 18.0 - self._animator.flow * phase_speed)
            y = baseline + carrier * amplitude * envelope
            if index == 0:
                path.moveTo(x, y)
            else:
                path.lineTo(x, y)
        pen = QPen(self._with_alpha(accent, int(54 + self._animator.energy * 50)))
        pen.setWidthF(1.1)
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)
        painter.drawPath(path)

    def _draw_header(self, painter: QPainter, width: float) -> None:
        painter.setPen(self.TEXT)
        painter.setFont(QFont("Avenir Next", 15, QFont.Medium))
        painter.drawText(QRectF(0, 31, width, 24), Qt.AlignCenter, "D U M M Y")

        painter.setPen(self.MUTED_TEXT)
        painter.setFont(QFont("Avenir Next", 8, QFont.Normal))
        painter.drawText(QRectF(0, 56, width, 18), Qt.AlignCenter, "PRIVATE  ·  LOCAL  ·  PRESENT")

    def _draw_status(
        self,
        painter: QPainter,
        width: float,
        height: float,
        accent: tuple[int, int, int],
    ) -> None:
        status_y = height - 78
        painter.setPen(self.TEXT)
        painter.setFont(QFont("Avenir Next", 11, QFont.Medium))
        painter.drawText(QRectF(0, status_y, width, 24), Qt.AlignCenter, self.state)

        dot_x = width * 0.5 - 61
        painter.setPen(Qt.NoPen)
        painter.setBrush(self._with_alpha(accent, 220))
        painter.drawEllipse(QPointF(dot_x, status_y + 12), 3.2, 3.2)

        painter.setPen(self.MUTED_TEXT)
        painter.setFont(QFont("Avenir Next", 8, QFont.Normal))
        painter.drawText(
            QRectF(0, status_y + 29, width, 18),
            Qt.AlignCenter,
            "VOICE CORE  ·  READY",
        )


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = DummyInterface()
    window.show()
    sys.exit(app.exec())
