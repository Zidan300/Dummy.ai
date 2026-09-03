"""Deterministic Phase 5.7 command, pipeline, and animation regressions."""

from __future__ import annotations

import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from animations import VisualAnimator
from audio import AudioCapture, BargeInDetector, FRAME_SIZE
from context import classify_intent, is_exit_command, is_interruption_command
from main import ResponsePipeline, VoiceController
import tts as tts_module


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

    def test_barge_command_requires_pause_confirmation(self):
        class FakeTranscriber:
            def transcribe(self, audio, cancel):
                return "stop"

        from main import BargeInMonitor

        commands = []
        monitor = BargeInMonitor(
            object(),
            FakeTranscriber(),
            threading.Event(),
            lambda session_id, command: commands.append((session_id, command)),
        )
        audio = np.zeros(FRAME_SIZE, dtype=np.float32)
        self.assertFalse(monitor._recognize_segment(7, audio, threading.Event(), False))
        self.assertEqual(commands, [])
        self.assertTrue(monitor._recognize_segment(7, audio, threading.Event(), True))
        self.assertEqual(commands, [(7, "stop")])

    def test_barge_detector_can_offer_audio_before_normal_silence_window(self):
        class FakeVad:
            def __init__(self, aggressiveness):
                self.calls = 0

            def is_speech(self, raw, sample_rate):
                self.calls += 1
                return self.calls > 10

        class FakeCapture:
            def __init__(self):
                self.frames = [np.zeros(FRAME_SIZE, dtype=np.int16) for _ in range(16)]
                self.frames.extend(np.zeros(FRAME_SIZE, dtype=np.int16) for _ in range(2))

            def drain_events(self):
                return []

            def is_active(self):
                return True

            def is_healthy(self):
                return True

            def read_frame(self, timeout=0.2):
                if self.frames:
                    time.sleep(0.03)
                    return self.frames.pop(0)
                return None

        capture = FakeCapture()
        stop_event = threading.Event()
        segments = []
        with patch("audio.webrtcvad.Vad", FakeVad):
            detector = BargeInDetector()
            detector.listen_for_segments(
                capture,
                stop_event,
                lambda audio, paused: segments.append((audio, paused)) or True,
            )
        self.assertEqual(len(segments), 1)
        self.assertFalse(segments[0][1])
        self.assertGreaterEqual(len(segments[0][0]), FRAME_SIZE)


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

    def test_controller_invalidates_active_session_on_stop(self):
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
            controller._active_session_id = 41

        controller._handle_barge_in(41, "stop")

        self.assertTrue(pipeline.is_cancelled())
        self.assertEqual(controller._active_session_id, None)
        self.assertEqual(controller._interrupted_session_id, 41)
        self.assertEqual(controller._state, "INTERRUPTED")

    def test_controller_keeps_exit_as_shutdown(self):
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
            controller._active_session_id = 42

        controller._handle_barge_in(42, "exit")

        self.assertTrue(controller.stop_event.is_set())
        self.assertTrue(pipeline.is_cancelled())
        self.assertEqual(controller._state, "SHUTTING_DOWN")

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


class TTSRegressionTests(unittest.TestCase):
    class FakeStdin:
        def __init__(self, process):
            self.process = process
            self.closed = False

        def write(self, text):
            del text

        def close(self):
            self.closed = True
            self.process.returncode = 0
            self.process.alive = False

    class FakeProcess:
        instances = []
        ffplay_started = threading.Event()
        block_ffplay = False

        def __init__(self, args, stdin, **kwargs):
            del kwargs
            self.args = args
            self.stdin = TTSRegressionTests.FakeStdin(self) if stdin is not None else None
            self.stderr = None
            self.returncode = None
            self.alive = True
            self.terminated = False
            self.__class__.instances.append(self)
            if args[0] == "fake-ffplay":
                self.ffplay_started.set()
                if not self.block_ffplay:
                    self.returncode = 0
                    self.alive = False

        def poll(self):
            return None if self.alive else self.returncode

        def terminate(self):
            self.terminated = True
            self.returncode = -15
            self.alive = False

        def wait(self, timeout=None):
            del timeout
            return self.returncode

    def setUp(self):
        self.FakeProcess.instances = []
        self.FakeProcess.ffplay_started.clear()
        self.FakeProcess.block_ffplay = False

    def test_ffplay_uses_known_working_arguments_and_cleans_wav(self):
        player = tts_module.SpeechPlayer()
        with patch.object(tts_module, "PIPER", "fake-piper"), patch.object(
            tts_module, "FFPLAY", "fake-ffplay"
        ), patch.object(tts_module.subprocess, "Popen", self.FakeProcess):
            self.assertTrue(player.speak("Hello."))

        piper, ffplay = self.FakeProcess.instances
        self.assertEqual(piper.args[:3], ["fake-piper", "-m", tts_module.MODEL])
        self.assertEqual(ffplay.args[:3], ["fake-ffplay", "-autoexit", "-nodisp"])
        self.assertTrue(ffplay.args[3].endswith(".wav"))
        self.assertFalse(Path(ffplay.args[3]).exists())

    def test_ffplay_cancellation_terminates_process_and_cleans_wav(self):
        self.FakeProcess.block_ffplay = True
        player = tts_module.SpeechPlayer()
        result = []
        with patch.object(tts_module, "PIPER", "fake-piper"), patch.object(
            tts_module, "FFPLAY", "fake-ffplay"
        ), patch.object(tts_module.subprocess, "Popen", self.FakeProcess):
            worker = threading.Thread(target=lambda: result.append(player.speak("Hello.")))
            worker.start()
            self.assertTrue(self.FakeProcess.ffplay_started.wait(1.0))
            player.cancel()
            worker.join(1.0)

        self.assertEqual(result, [False])
        ffplay = self.FakeProcess.instances[-1]
        self.assertTrue(ffplay.terminated)
        self.assertFalse(Path(ffplay.args[3]).exists())


if __name__ == "__main__":
    unittest.main()
