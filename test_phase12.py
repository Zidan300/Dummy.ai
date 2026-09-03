"""Phase 12 natural barge-in regression tests."""

from __future__ import annotations

import threading
import unittest
from unittest.mock import patch

import numpy as np

from audio import BARGE_MIN_AUDIO_SECONDS, BargeInDetector, FRAME_SIZE
from main import ResponsePipeline, VoiceController


class FakePlayer:
    def __init__(self) -> None:
        self.cancelled = False

    def speak(self, text, cancel_event=None, **callbacks):
        del text, callbacks
        return not self.cancelled and not (cancel_event and cancel_event.is_set())

    def cancel(self):
        self.cancelled = True
        return True

    def reset_cancellation(self):
        self.cancelled = False


class NaturalBargeInTests(unittest.TestCase):
    def test_speech_onset_carries_preroll_to_callback(self):
        class FakeVad:
            def __init__(self, aggressiveness):
                del aggressiveness
                self.calls = 0

            def is_speech(self, raw, sample_rate):
                del raw, sample_rate
                self.calls += 1
                return self.calls >= 3

        class Capture:
            def __init__(self):
                self.frames = [np.full(FRAME_SIZE, 1024, dtype=np.int16) for _ in range(14)]

            def drain_events(self):
                return []

            def is_active(self):
                return True

            def is_healthy(self):
                return True

            def read_frame(self, timeout=0.2):
                del timeout
                return self.frames.pop(0)

        detected = []
        with patch("audio.webrtcvad.Vad", FakeVad):
            BargeInDetector().listen_for_segments(
                Capture(),
                threading.Event(),
                on_segment=lambda audio, paused: detected.append((audio, paused)) or True,
                on_speech_detected=lambda audio: detected.append(("onset", audio)),
            )

        self.assertEqual(detected[0][0], "onset")
        self.assertGreaterEqual(len(detected[0][1]), FRAME_SIZE * 4)

    def test_natural_barge_in_cancels_and_restarts_normal_listener(self):
        controller = VoiceController()
        player = FakePlayer()
        pipeline = ResponsePipeline(
            "old request",
            controller.stop_event,
            player,
            lambda: None,
            lambda: None,
            lambda: None,
            lambda exc: None,
        )
        seed = np.ones(FRAME_SIZE * 3, dtype=np.float32) * 0.1
        started = []

        class FakeNormalMonitor:
            def reset(self):
                pass

            def start(self, initial_audio=None, detected_at=None):
                started.append((initial_audio, detected_at))

        controller.player = player
        controller._normal_monitor = FakeNormalMonitor()
        with controller._pipeline_lock:
            controller._pipeline = pipeline
            controller._active_session_id = 12

        with patch.object(controller, "interrupt_tts", wraps=controller.interrupt_tts) as interrupt:
            controller._handle_natural_barge_in(12, seed)

        self.assertTrue(interrupt.called)
        self.assertTrue(pipeline.is_cancelled())
        self.assertTrue(player.cancelled)
        self.assertIsNone(controller._pipeline)
        self.assertIsNone(controller._active_session_id)
        self.assertEqual(controller._state, "LISTENING")
        self.assertEqual(len(started), 1)
        np.testing.assert_array_equal(started[0][0], seed)
        self.assertIsNotNone(started[0][1])

    def test_barge_threshold_is_short_but_not_a_single_frame(self):
        self.assertGreaterEqual(BARGE_MIN_AUDIO_SECONDS, 0.09)
        self.assertGreaterEqual(BARGE_MIN_AUDIO_SECONDS, 3 * 0.03)


if __name__ == "__main__":
    unittest.main()
