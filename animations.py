"""Small, GUI-thread-owned animation model for Dummy's visual core."""

from __future__ import annotations

import math


STATE_PROFILES = {
    "STARTING": (0.20, 0.45, 0.35, 0.45),
    "IDLE": (0.18, 0.35, 0.25, 0.38),
    "LISTENING": (0.72, 1.15, 0.92, 0.95),
    "PROCESSING": (0.50, 0.75, 0.58, 0.72),
    "THINKING": (0.64, 1.35, 0.70, 1.15),
    "SPEAKING": (0.84, 1.55, 1.00, 1.30),
    "INTERRUPTED": (0.10, 2.20, 0.20, 0.30),
    "ERROR": (0.42, 0.55, 0.52, 0.60),
    "SHUTTING_DOWN": (0.12, 0.25, 0.18, 0.25),
    "STOPPED": (0.05, 0.15, 0.10, 0.15),
}

STATE_COLORS = {
    "STARTING": (129.0, 163.0, 239.0),
    "IDLE": (129.0, 163.0, 239.0),
    "LISTENING": (109.0, 224.0, 216.0),
    "PROCESSING": (145.0, 153.0, 255.0),
    "THINKING": (164.0, 142.0, 255.0),
    "SPEAKING": (255.0, 193.0, 139.0),
    "INTERRUPTED": (255.0, 194.0, 124.0),
    "ERROR": (255.0, 116.0, 132.0),
    "SHUTTING_DOWN": (86.0, 111.0, 164.0),
    "STOPPED": (49.0, 65.0, 100.0),
}


class VisualAnimator:
    """Frame-rate independent visual state used only by the Qt paint timer."""

    def __init__(self) -> None:
        self.state = "STARTING"
        self.time = 0.0
        self.energy = 0.20
        self.speed = 0.45
        self.scale = 0.35
        self.wave = 0.45
        self.audio_level = 0.0
        self._audio_target = 0.0
        self._accent = STATE_COLORS[self.state]
        self._accent_target = self._accent
        self.state_age = 0.0
        self._interrupt_impulse = 0.0

    def set_state(self, state: str) -> None:
        normalized = state.upper()
        if normalized not in STATE_PROFILES:
            normalized = "IDLE"
        if normalized == "INTERRUPTED" and self.state != normalized:
            self._interrupt_impulse = 1.0
        self.state = normalized
        self._accent_target = STATE_COLORS[normalized]
        self.state_age = 0.0

    def set_audio_level(self, level: float) -> None:
        """Accept a UI-safe level from the existing microphone signal."""
        self._audio_target = max(0.0, min(1.0, float(level)))

    def tick(self, elapsed: float) -> None:
        elapsed = max(0.0, min(0.05, float(elapsed)))
        self.time += elapsed
        self.state_age += elapsed
        target_energy, target_speed, target_scale, target_wave = STATE_PROFILES[self.state]
        smoothing = 1.0 - math.exp(-elapsed * 8.0)
        self.energy += (target_energy - self.energy) * smoothing
        self.speed += (target_speed - self.speed) * smoothing
        self.scale += (target_scale - self.scale) * smoothing
        self.wave += (target_wave - self.wave) * smoothing
        audio_rate = 18.0 if self._audio_target > self.audio_level else 5.0
        audio_smoothing = 1.0 - math.exp(-elapsed * audio_rate)
        self.audio_level += (self._audio_target - self.audio_level) * audio_smoothing
        accent_smoothing = 1.0 - math.exp(-elapsed * 5.5)
        self._accent = tuple(
            current + (target - current) * accent_smoothing
            for current, target in zip(self._accent, self._accent_target)
        )
        self._interrupt_impulse *= math.exp(-elapsed * 7.0)

    @property
    def pulse(self) -> float:
        idle_breath = math.sin(self.time * 1.35) * 0.035
        active_breath = math.sin(self.time * (2.0 + self.speed)) * 0.055 * self.energy
        audio_reaction = self.audio_level * (0.20 if self.state == "LISTENING" else 0.12)
        return idle_breath + active_breath + audio_reaction + self._interrupt_impulse * 0.10

    @property
    def rotation(self) -> float:
        return self.time * self.speed * 18.0

    @property
    def flow(self) -> float:
        return self.time * (0.7 + self.speed * 1.8)

    @property
    def collapse(self) -> float:
        return self._interrupt_impulse

    @property
    def core_radius_factor(self) -> float:
        return 1.0 + self.pulse + self.audio_level * 0.08

    @property
    def wave_factor(self) -> float:
        return self.wave + self.audio_level * 0.55

    @property
    def accent(self) -> tuple[int, int, int]:
        return tuple(int(round(channel)) for channel in self._accent)

    @property
    def orbit_energy(self) -> float:
        """State-aware orbital detail, kept bounded for inexpensive painting."""
        if self.state in {"SHUTTING_DOWN", "STOPPED"}:
            return self.energy * 0.35
        return min(1.0, self.energy + self.audio_level * 0.25)

    @property
    def spark_energy(self) -> float:
        return min(1.0, self.energy * 0.72 + self.audio_level * 0.38)
