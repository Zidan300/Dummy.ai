"""Deterministic Phase 5.7 command, pipeline, and animation regressions."""

from __future__ import annotations

import threading
import time
import unittest
from unittest.mock import patch

import numpy as np

from animations import VisualAnimator
from audio import AudioCapture, FRAME_SIZE
from context import classify_intent, is_exit_command, is_interruption_command
from main import ResponsePipeline


class FakePlayer:
    def __init__(self) -> None:
        self.cancelled = False
        self.played: list[str] = []

    def speak(self, text, cancel_event=None, on_playback_start=None):
        if self.cancelled or (cancel_event and cancel_event.is_set()):
            return False
        self.played.append(text)
        if on_playback_start:
            on_playback_start()
        return True

    def cancel(self):
        self.cancelled = True
        return bool(self.played)


class BlockingPlayer(FakePlayer):
    def __init__(self) -> None:
        super().__init__()
        self.started = threading.Event()

    def speak(self, text, cancel_event=None, on_playback_start=None):
        if self.cancelled or (cancel_event and cancel_event.is_set()):
            return False
        self.played.append(text)
        self.started.set()
        if on_playback_start:
            on_playback_start()
        while not self.cancelled and not (cancel_event and cancel_event.is_set()):
            time.sleep(0.005)
        return False


class CommandRegressionTests(unittest.TestCase):
    def test_stop_and_exit_are_distinct_exact_commands(self):
        self.assertEqual(classify_intent("stop"), "INTERRUPTION")
        self.assertEqual(classify_intent("turn off"), "EXIT")
        self.assertTrue(is_interruption_command("stop talking."))
        self.assertTrue(is_exit_command("goodbye!"))
        self.assertFalse(is_exit_command("How do I exit Vim?"))
        self.assertFalse(is_interruption_command("How do I stop a server?"))
        self.assertFalse(is_exit_command("Why should I stop using Python?"))

    def test_existing_microphone_callback_exposes_only_rms_level(self):
        levels = []
        capture = AudioCapture(on_level=levels.append)
        capture._accepting = True
        capture._callback(
            np.full((FRAME_SIZE, 1), 8192, dtype=np.int16),
            FRAME_SIZE,
            None,
            None,
        )
        self.assertEqual(len(levels), 1)
        self.assertGreater(levels[0], 0.0)
        self.assertLessEqual(levels[0], 1.0)
        self.assertIsNotNone(capture.read_frame(timeout=0.01))


class ResponsePipelineRegressionTests(unittest.TestCase):
    def _pipeline(self, stream, player=None):
        player = player or FakePlayer()
        callbacks = {"token": 0, "sentence": 0, "speaking": 0}
        pipeline = ResponsePipeline(
            "What is Docker?",
            threading.Event(),
            player,
            lambda: callbacks.__setitem__("token", callbacks["token"] + 1),
            lambda: callbacks.__setitem__("sentence", callbacks["sentence"] + 1),
            lambda: callbacks.__setitem__("speaking", callbacks["speaking"] + 1),
            lambda exc: self.fail(f"pipeline error: {exc}"),
        )
        return pipeline, player, callbacks

    def test_generation_produces_sentences_and_waits_for_tts(self):
        player = FakePlayer()

        def fake_stream(prompt, on_token, cancel_event):
            on_token("Docker packages software with its dependencies. ")
            on_token("That makes deployments more consistent.")
            return "Docker packages software with its dependencies. That makes deployments more consistent."

        pipeline, player, callbacks = self._pipeline(fake_stream, player)
        with patch("main.stream_dummy", side_effect=fake_stream):
            pipeline.start()
            pipeline.wait()

        self.assertTrue(pipeline.generation_succeeded)
        self.assertTrue(pipeline.tts_succeeded)
        self.assertEqual(len(player.played), 2)
        self.assertEqual(callbacks["token"], 1)
        self.assertEqual(callbacks["sentence"], 1)
        self.assertEqual(callbacks["speaking"], 1)
        self.assertTrue(pipeline.sentences.empty())
        self.assertEqual(pipeline.sentences.unfinished_tasks, 0)

    def test_cancelled_generation_cannot_publish_late_audio(self):
        started = threading.Event()
        player = FakePlayer()

        def slow_stream(prompt, on_token, cancel_event):
            started.set()
            while not cancel_event.is_set():
                time.sleep(0.01)
            return "stale response."

        pipeline, player, callbacks = self._pipeline(slow_stream, player)
        with patch("main.stream_dummy", side_effect=slow_stream):
            pipeline.start()
            self.assertTrue(started.wait(1.0))
            pipeline.cancel()
            player.cancel()
            pipeline.wait()

        self.assertTrue(pipeline.is_cancelled())
        self.assertEqual(player.played, [])
        self.assertEqual(callbacks["speaking"], 0)
        self.assertTrue(pipeline.sentences.empty())

    def test_cancellation_discards_sentences_waiting_behind_active_audio(self):
        player = BlockingPlayer()

        def fake_stream(prompt, on_token, cancel_event):
            on_token("First sentence is already playing. ")
            on_token("Second sentence must be discarded.")
            return "First sentence is already playing. Second sentence must be discarded."

        pipeline, player, _ = self._pipeline(fake_stream, player)
        with patch("main.stream_dummy", side_effect=fake_stream):
            pipeline.start()
            self.assertTrue(player.started.wait(1.0))
            pipeline.cancel()
            player.cancel()
            pipeline.wait()

        self.assertEqual(player.played, ["First sentence is already playing."])
        self.assertTrue(pipeline.sentences.empty())


class AnimationRegressionTests(unittest.TestCase):
    def test_state_and_audio_transitions_are_smoothed(self):
        animator = VisualAnimator()
        animator.set_state("LISTENING")
        animator.set_audio_level(0.8)
        animator.tick(0.016)
        self.assertGreater(animator.audio_level, 0.0)
        self.assertLess(animator.audio_level, 0.8)
        self.assertNotEqual(animator.accent, (129, 163, 239))

        animator.set_audio_level(0.0)
        previous = animator.audio_level
        animator.tick(0.016)
        self.assertLess(animator.audio_level, previous)
        animator.set_state("INTERRUPTED")
        self.assertGreater(animator.collapse, 0.0)


if __name__ == "__main__":
    unittest.main()
