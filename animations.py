"""Small, GUI-thread-owned animation model for Dummy's visual core.

Phase 10: keeps the frame-rate independent, allocation-light model used only by
the Qt paint timer, but gives each state a more distinct and believable visual
personality, couples motion to real voice amplitude (fast attack / slow decay),
and reacts to the deterministic intent of the current turn.
"""

from __future__ import annotations

import math


# (energy, speed, scale, wave) tuned per state so each has a distinct feel.
STATE_PROFILES = {
    "STARTING": (0.20, 0.45, 0.35, 0.45),
    "IDLE": (0.18, 0.35, 0.25, 0.38),
    "PROCESSING": (0.50, 0.75, 0.58, 0.72),
}

# Phase 10: the "active listening / thinking / speaking" trio share a vibrant
# baseline but each is tuned further via dedicated intensity + audio coupling.
LISTENING_PROFILE = (0.60, 1.05, 0.92, 0.90)
THINKING_PROFILE = (0.66, 1.40, 0.70, 1.10)
SPEAKING_PROFILE = (0.90, 1.65, 1.05, 1.40)

STATE_ORDER = ("STARTING", "IDLE", "LISTENING", "PROCESSING", "THINKING",
               "SPEAKING", "INTERRUPTED", "ERROR", "SHUTTING_DOWN", "STOPPED")

# Derived-intensity and audio-coupling factors per state (0..1).
# state_intensity feeds scanning/listening-rings visibility; audio_coupling
# controls how strongly mic RMS / TTS amplitude drive the motion.
STATE_INTENSITY = {
    "STARTING": 0.35,
    "IDLE": 0.20,
    "LISTENING": 0.62,
    "PROCESSING": 0.45,
    "THINKING": 0.85,
    "SPEAKING": 1.00,
    "INTERRUPTED": 0.28,
    "ERROR": 0.50,
    "SHUTTING_DOWN": 0.30,
    "STOPPED": 0.10,
}

# 0 = mic RMS ignored, 1 = fully voice-driven.
AUDIO_COUPLING = {
    "STARTING": 0.0,
    "IDLE": 0.10,
    "LISTENING": 0.90,
    "PROCESSING": 0.10,
    "THINKING": 0.25,
    "SPEAKING": 1.00,
    "INTERRUPTED": 0.0,
    "ERROR": 0.20,
    "SHUTTING_DOWN": 0.0,
    "STOPPED": 0.0,
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

# Question-category profiles: (energy_scale, speed_scale, hue_shift).
# These come from classify_question_category deterministically.
CATEGORY_PROFILES = {
    "TECHNICAL": (1.08, 1.08, 0.0),
    "COMPLEX": (1.16, 1.18, -8.0),
    "FACTUAL": (1.00, 0.96, 6.0),
    "CREATIVE": (1.04, 0.90, 14.0),
    "COMMAND": (1.12, 1.24, -18.0),
    "CASUAL": (0.94, 0.88, 10.0),
}

# Coarse intent drives the broad visual mood on top of state.
INTENT_MOOD = {
    "QUESTION": 1.0,      # analytical/focused
    "COMMAND": 1.14,      # precise/direct
    "CONVERSATION": 0.86, # softer/relaxed
    "INTERRUPTION": 0.5,  # sharp (handled by impulse)
    "EXIT": 0.6,
}

_FAST_ATTACK = 20.0
_SLOW_DECAY = 4.5


class VisualAnimator:
    """Frame-rate independent visual state used only by the Qt paint timer."""

    def __init__(self) -> None:
        self.state = "STARTING"
        self.time = 0.0
        self.energy = 0.18
        self.speed = 0.35
        self.scale = 0.25
        self.wave = 0.38
        self.audio_level = 0.0
        self._audio_target = 0.0
        self._accent = STATE_COLORS[self.state]
        self._accent_target = self._accent
        self.state_age = 0.0
        self._interrupt_impulse = 0.0
        self._conclusion_impulse = 0.0
        self.category = "CASUAL"
        self.intent = "QUESTION"
        self._category_energy = 0.94
        self._category_speed = 0.88
        self._category_hue = 0.0
        self._intent_scale = 1.0
        self._activity_impulse = 0.0

    # -- state profile selection -------------------------------------------
    def _profile(self):
        if self.state == "LISTENING":
            return LISTENING_PROFILE
        if self.state == "THINKING":
            return THINKING_PROFILE
        if self.state == "SPEAKING":
            return SPEAKING_PROFILE
        return STATE_PROFILES.get(self.state, STATE_PROFILES["IDLE"])

    # -- public API (called from interface, all GUI thread) ----------------
    def set_state(self, state: str) -> None:
        normalized = state.upper()
        if normalized not in STATE_ORDER:
            normalized = "IDLE"
        if normalized == "INTERRUPTED" and self.state != normalized:
            self._interrupt_impulse = 1.0
        elif normalized == "SPEAKING" and self.state == "THINKING":
            # Computation reached a conclusion: focus energy before voice.
            self._conclusion_impulse = 1.0
        self.state = normalized
        self._accent_target = STATE_COLORS[normalized]
        self.state_age = 0.0

    def set_audio_level(self, level: float) -> None:
        """Accept a UI-safe level (mic RMS or TTS amplitude)."""
        self._audio_target = max(0.0, min(1.0, float(level)))

    def set_category(self, category: str) -> None:
        normalized = category.upper()
        if normalized not in CATEGORY_PROFILES:
            normalized = "CASUAL"
        self.category = normalized
        _, _, hue = CATEGORY_PROFILES[normalized]
        self._category_hue = float(hue)

    def set_intent(self, intent: str) -> None:
        normalized = intent.upper()
        if normalized not in INTENT_MOOD:
            normalized = "QUESTION"
        self._intent_scale = INTENT_MOOD[normalized]
        self.intent = normalized

    def note_activity(self, amount: float = 0.35) -> None:
        self._activity_impulse = max(self._activity_impulse, min(1.0, float(amount)))

    def note_speech_started(self) -> None:
        self._activity_impulse = max(self._activity_impulse, 0.30)

    def note_speech_ended(self) -> None:
        self._activity_impulse = max(self._activity_impulse, 0.35)

    def note_generation_milestone(self, amount: float) -> None:
        """Thinking milestones (first token / first sentence) pulse the core."""
        self._activity_impulse = max(self._activity_impulse, min(1.0, float(amount)))

    # -- per-frame advance --------------------------------------------------
    def tick(self, elapsed: float) -> None:
        elapsed = max(0.0, min(0.05, float(elapsed)))
        self.time += elapsed
        self.state_age += elapsed
        target_energy, target_speed, target_scale, target_wave = self._profile()
        cat_energy, cat_speed, _ = CATEGORY_PROFILES[self.category]
        target_energy *= self._intent_scale
        target_speed *= self._intent_scale
        smoothing = 1.0 - math.exp(-elapsed * 8.0)
        self.energy += (target_energy - self.energy) * smoothing
        self.speed += (target_speed - self.speed) * smoothing
        self.scale += (target_scale - self.scale) * smoothing
        self.wave += (target_wave - self.wave) * smoothing
        self._category_energy += (cat_energy * self._intent_scale - self._category_energy) * smoothing
        self._category_speed += (cat_speed * self._intent_scale - self._category_speed) * smoothing
        # fast attack / slow decay on voice level
        rate = _FAST_ATTACK if self._audio_target > self.audio_level else _SLOW_DECAY
        audio_smoothing = 1.0 - math.exp(-elapsed * rate)
        self.audio_level += (self._audio_target - self.audio_level) * audio_smoothing
        accent_smoothing = 1.0 - math.exp(-elapsed * 5.5)
        self._accent = tuple(
            current + (target - current) * accent_smoothing
            for current, target in zip(self._accent, self._accent_target)
        )
        self._interrupt_impulse *= math.exp(-elapsed * 7.0)
        self._conclusion_impulse *= math.exp(-elapsed * 6.0)
        self._activity_impulse *= math.exp(-elapsed * 5.5)

    # -- derived visual properties ------------------------------------------
    @property
    def pulse(self) -> float:
        idle_breath = math.sin(self.time * 1.35) * 0.035
        active_breath = math.sin(self.time * (2.0 + self.speed)) * 0.055 * self.energy * self._category_energy
        audio_reaction = self.voice_level * 0.55 * self.audio_coupling
        voice_flutter = math.sin(self.time * 17.0 + 0.7) * self.voice_level * 0.018 * self.audio_coupling
        thought_shift = math.sin(self.time * 0.73 + 1.8) * 0.025 * (self.state == "THINKING")
        collapse = -self._interrupt_impulse * 0.22
        return (idle_breath + active_breath + audio_reaction + voice_flutter + thought_shift
                + collapse
                + self._interrupt_impulse * 0.10 + self._activity_impulse * 0.08)

    @property
    def rotation(self) -> float:
        drift = math.sin(self.time * 0.41) * 3.0
        return self.time * self.speed * self._category_speed * 18.0 + drift

    @property
    def flow(self) -> float:
        voice_flow = self.voice_level * 0.8 * self.audio_coupling
        return self.time * (0.7 + self.speed * 1.8 * self._category_speed + voice_flow)

    @property
    def collapse(self) -> float:
        return self._interrupt_impulse

    @property
    def core_radius_factor(self) -> float:
        vibration = math.sin(self.time * 21.0) * self.voice_level * 0.012 * self.audio_coupling
        return 1.0 + self.pulse + vibration + self.voice_level * 0.10 * self.audio_coupling + self._conclusion_impulse * 0.06

    @property
    def wave_factor(self) -> float:
        return self.wave + self.voice_level * (0.90 + 0.08 * math.sin(self.time * 11.0)) * self.audio_coupling

    @property
    def accent(self) -> tuple[int, int, int]:
        # Apply a subtle category-based warmth shift toward the accent baseline,
        # clamped to keep colors premium and never overwhelming.
        shift = self._category_hue * 0.012
        r, g, b = self._accent
        blend = (255.0 * shift, 196.0 * shift, 120.0 * abs(shift))
        mixed = (
            r + (blend[0] - r) * abs(shift),
            g + (blend[1] - g) * abs(shift),
            b + (blend[2] - b) * abs(shift),
        )
        return tuple(int(round(max(0.0, min(255.0, c)))) for c in mixed)

    @property
    def orbit_energy(self) -> float:
        if self.state in {"SHUTTING_DOWN", "STOPPED"}:
            return self.energy * 0.35
        return min(1.0, self.energy * self._category_energy + self.voice_level * 0.28 * self.audio_coupling + self._activity_impulse * 0.18)

    @property
    def spark_energy(self) -> float:
        return min(1.0, self.energy * 0.72 * self._category_energy + self.voice_level * 0.40 * self.audio_coupling + self._activity_impulse * 0.12)

    # -- Phase 10 additions --------------------------------------------------
    @property
    def state_intensity(self) -> float:
        return STATE_INTENSITY.get(self.state, 0.0) * self._intent_scale

    @property
    def audio_coupling(self) -> float:
        return AUDIO_COUPLING.get(self.state, 0.0)

    @property
    def voice_level(self) -> float:
        """The smoothed voice level (mic RMS / TTS amplitude)."""
        return self.audio_level

    @property
    def listening_ring(self) -> float:
        """An expanding ring that breathes with the mic while listening."""
        if self.state != "LISTENING":
            return 0.0
        return (0.35 + 0.65 * self.audio_level) * (0.75 + 0.25 * math.sin(self.time * 3.0))

    @property
    def speech_ring(self) -> float:
        """Voice-reactive expansion while speaking; collapses on interruption."""
        if self.state == "SPEAKING":
            return 0.45 + 0.55 * self.audio_level
        if self.state == "INTERRUPTED":
            return max(0.0, self._interrupt_impulse * 0.8)
        return 0.0

    @property
    def scan_ring(self) -> float:
        """A focused scanning ring while thinking."""
        if self.state != "THINKING":
            return 0.0
        return 0.5 + 0.5 * math.sin(self.time * 3.4)

    @property
    def conclusion(self) -> float:
        return self._interrupt_impulse if self.state == "INTERRUPTED" else self._conclusion_impulse
