"""Deterministic Phase 9 self-talk / infinite-loop prevention tests."""

from __future__ import annotations

import threading
import time
import unittest
from unittest.mock import patch

import numpy as np

from main import PerformanceTimeline, ResponsePipeline, VoiceController
from context import is_exit_command, is_interruption_command


class FakePlayer:
    def __init__(self) -> None:
        self.cancelled = False
        self.played: list[str] = []

    def speak(
        self,
        text,
        cancel_event=None,
        on_piper_start=None,
        on_audio_ready=None,
        on_playback_start=None,
        on_playback_level=None,
    ):
        del on_playback_level
        if self.cancelled or (cancel_event and cancel_event.is_set()):
            return False
        if on_piper_start:
            on_piper_start()
        if on_audio_ready:
            on_audio_ready()
        self.played.append(text)
        if on_playback_start:
            on_playback_start()
        return True

    def cancel(self):
        self.cancelled = True
        return bool(self.played)

    def reset_cancellation(self):
        self.cancelled = False


class SelfTalkPreventionTests(unittest.TestCase):
    """Tests proving TTS output cannot become a new user question."""

    def test_tts_playback_cannot_enter_normal_pipeline_during_cooldown(self):
        controller = VoiceController()
        controller._set_state("LISTENING")
        self.assertGreaterEqual(controller._tts_cooldown_until, 0.0)
        controller._tts_cooldown_until = time.monotonic() + 1.0

        controller._queue_normal_utterance(
            np.ones(480, dtype=np.float32),
            PerformanceTimeline(),
        )
        self.assertEqual(controller._utterances.qsize(), 0,
                         "Normal utterance during TTS cooldown must be discarded")

    def test_tts_playback_cannot_trigger_another_gemma_response(self):
        controller = VoiceController()
        controller._set_state("LISTENING")
        controller._tts_cooldown_until = time.monotonic() + 1.0
        controller._queue_normal_utterance(
            np.ones(480, dtype=np.float32),
            PerformanceTimeline(),
        )
        self.assertEqual(controller._utterances.qsize(), 0)
        self.assertEqual(controller._pipeline, None)

    def test_no_endless_self_response_loop(self):
        controller = VoiceController()
        controller._set_state("LISTENING")
        controller._tts_cooldown_until = time.monotonic() + 1.0
        for _ in range(20):
            controller._queue_normal_utterance(
                np.ones(480, dtype=np.float32),
                PerformanceTimeline(),
            )
        self.assertEqual(controller._utterances.qsize(), 0)
        self.assertEqual(controller._pipeline, None)

    def test_new_question_allowed_after_cooldown_expires(self):
        controller = VoiceController()
        controller._set_state("LISTENING")
        controller._tts_cooldown_until = time.monotonic() + 0.05
        time.sleep(0.07)
        controller._queue_normal_utterance(
            np.ones(480, dtype=np.float32),
            PerformanceTimeline(),
        )
        self.assertEqual(controller._utterances.qsize(), 1)

    def test_microphone_remains_active_during_speaking(self):
        controller = VoiceController()
        controller._set_state("SPEAKING")
        self.assertIsNotNone(controller.audio)

    def test_microphone_remains_active_during_thinking(self):
        controller = VoiceController()
        controller._set_state("THINKING")
        self.assertIsNotNone(controller.audio)

    def test_stop_interrupts_speaking(self):
        controller = VoiceController()
        pipeline = ResponsePipeline(
            "test",
            controller.stop_event,
            FakePlayer(),
            lambda: None,
            lambda: None,
            lambda: None,
            lambda exc: None,
        )
        with controller._pipeline_lock:
            controller._pipeline = pipeline
            controller._active_session_id = 1
        controller._set_state("SPEAKING")
        controller.keyboard_interrupt()
        self.assertTrue(pipeline.is_cancelled())
        self.assertEqual(controller._tts_cooldown_until, 0.0)
        self.assertEqual(controller._state, "LISTENING")

    def test_stop_interrupts_thinking(self):
        controller = VoiceController()
        pipeline = ResponsePipeline(
            "test",
            controller.stop_event,
            FakePlayer(),
            lambda: None,
            lambda: None,
            lambda: None,
            lambda exc: None,
        )
        with controller._pipeline_lock:
            controller._pipeline = pipeline
            controller._active_session_id = 2
        controller._set_state("THINKING")
        controller.keyboard_interrupt()
        self.assertTrue(pipeline.is_cancelled())

    def test_stop_never_enters_gemma(self):
        controller = VoiceController()
        controller._set_state("LISTENING")
        seen = []

        def fake_stream(prompt, on_token, cancel_event):
            seen.append(prompt)
            return ""

        controller.transcriber = type(
            "FakeTranscriber",
            (),
            {"transcribe": lambda self, audio, stop_event, on_first_result=None: "stop"},
        )()
        controller.player = FakePlayer()
        timeline = PerformanceTimeline()
        timeline.mark("speech_finished")

        with patch("main.stream_dummy", side_effect=fake_stream):
            controller._process_utterance(np.ones(480, dtype=np.float32), timeline)
            deadline = time.monotonic() + 1.0
            while controller._pipeline is not None and time.monotonic() < deadline:
                time.sleep(0.01)

        self.assertEqual(seen, ["stop"], "Voice STOP is normal speech; S key is the interrupt")

    def test_cancel_interrupts(self):
        controller = VoiceController()
        pipeline = ResponsePipeline(
            "test",
            controller.stop_event,
            FakePlayer(),
            lambda: None,
            lambda: None,
            lambda: None,
            lambda exc: None,
        )
        with controller._pipeline_lock:
            controller._pipeline = pipeline
            controller._active_session_id = 3
        controller._set_state("SPEAKING")
        controller.keyboard_interrupt()
        self.assertTrue(pipeline.is_cancelled())

    def test_quiet_interrupts(self):
        controller = VoiceController()
        pipeline = ResponsePipeline(
            "test",
            controller.stop_event,
            FakePlayer(),
            lambda: None,
            lambda: None,
            lambda: None,
            lambda exc: None,
        )
        with controller._pipeline_lock:
            controller._pipeline = pipeline
            controller._active_session_id = 4
        controller._set_state("SPEAKING")
        controller.keyboard_interrupt()
        self.assertTrue(pipeline.is_cancelled())

    def test_false_positive_interruption_phrases_remain_safe(self):
        self.assertFalse(is_interruption_command("How do I stop a server?"))
        self.assertFalse(is_interruption_command("How do I exit Vim?"))
        self.assertFalse(is_interruption_command("Why should I cancel this?"))
        self.assertFalse(is_interruption_command("How do I shut down Linux?"))
        self.assertFalse(is_interruption_command("Why is the program quiet?"))

    def test_false_positive_phrases_are_not_interrupted(self):
        controller = VoiceController()
        pipeline = ResponsePipeline(
            "test",
            controller.stop_event,
            FakePlayer(),
            lambda: None,
            lambda: None,
            lambda: None,
            lambda exc: None,
        )
        with controller._pipeline_lock:
            controller._pipeline = pipeline
            controller._active_session_id = 5
        controller.keyboard_interrupt()
        self.assertFalse(pipeline.is_cancelled())

    def test_session_invalidation_works(self):
        controller = VoiceController()
        pipeline1 = ResponsePipeline(
            "test1",
            controller.stop_event,
            FakePlayer(),
            lambda: None,
            lambda: None,
            lambda: None,
            lambda exc: None,
        )
        with controller._pipeline_lock:
            controller._pipeline = pipeline1
            controller._active_session_id = 1

        controller._set_state("SPEAKING")
        controller.keyboard_interrupt()
        self.assertTrue(pipeline1.is_cancelled())

        controller._next_session_id += 2
        session2 = controller._next_session_id
        pipeline2 = ResponsePipeline(
            "test2",
            controller.stop_event,
            FakePlayer(),
            lambda: None,
            lambda: None,
            lambda: None,
            lambda exc: None,
        )
        with controller._pipeline_lock:
            controller._pipeline = pipeline2
            controller._active_session_id = session2
            controller._tts_cooldown_until = 0.0

        controller.keyboard_interrupt()
        self.assertFalse(pipeline2.is_cancelled(),
                         "Stale session interruption must not affect new session")

    def test_exit_still_performs_full_shutdown(self):
        controller = VoiceController()
        pipeline = ResponsePipeline(
            "test",
            controller.stop_event,
            FakePlayer(),
            lambda: None,
            lambda: None,
            lambda: None,
            lambda exc: None,
        )
        with controller._pipeline_lock:
            controller._pipeline = pipeline
            controller._active_session_id = 7
        controller.request_shutdown("test exit")
        self.assertTrue(controller.stop_event.is_set())
        self.assertTrue(pipeline.is_cancelled())
        self.assertEqual(controller._state, "SHUTTING_DOWN")

    def test_cooldown_is_cleared_after_interruption(self):
        controller = VoiceController()
        controller._set_state("LISTENING")
        controller._tts_cooldown_until = time.monotonic() + 1.0
        pipeline = ResponsePipeline(
            "test",
            controller.stop_event,
            FakePlayer(),
            lambda: None,
            lambda: None,
            lambda: None,
            lambda exc: None,
        )
        with controller._pipeline_lock:
            controller._pipeline = pipeline
            controller._active_session_id = 8
        controller._set_state("SPEAKING")
        controller.keyboard_interrupt()
        self.assertEqual(controller._tts_cooldown_until, 0.0)
        controller._queue_normal_utterance(
            np.ones(480, dtype=np.float32),
            PerformanceTimeline(),
        )
        self.assertEqual(controller._utterances.qsize(), 1,
                         "New question after STOP must be accepted immediately")


if __name__ == "__main__":
    unittest.main()
