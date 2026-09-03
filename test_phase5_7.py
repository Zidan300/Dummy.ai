"""Deterministic Phase 5.7 command, pipeline, and animation regressions."""

from __future__ import annotations

import threading
import time
import unittest
import wave
from pathlib import Path
from tempfile import NamedTemporaryFile
from unittest.mock import patch

import numpy as np

from animations import VisualAnimator
from audio import AudioCapture, BargeInDetector, FRAME_SIZE, UtteranceDetector
from context import (
    classify_intent,
    classify_question_category,
    is_exit_command,
    is_interruption_command,
)
from main import NormalSpeechMonitor, PerformanceTimeline, ResponsePipeline, VoiceController
import tts as tts_module


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


class BlockingPlayer(FakePlayer):
    def __init__(self) -> None:
        super().__init__()
        self.started = threading.Event()

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
        self.started.set()
        if on_playback_start:
            on_playback_start()
        while not self.cancelled and not (cancel_event and cancel_event.is_set()):
            time.sleep(0.005)
        return False


class CommandRegressionTests(unittest.TestCase):
    def test_stop_and_exit_are_distinct_exact_commands(self):
        self.assertEqual(classify_intent("stop"), "CONVERSATION")
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

    def test_one_microphone_frame_is_broadcast_to_normal_and_barge_consumers(self):
        capture = AudioCapture()
        capture.register_consumer("normal")
        capture.register_consumer("barge")
        capture._accepting = True
        frame = np.full((FRAME_SIZE, 1), 1024, dtype=np.int16)
        capture._callback(frame, FRAME_SIZE, None, None)

        normal = capture.read_frame(timeout=0.01, consumer="normal")
        barge = capture.read_frame(timeout=0.01, consumer="barge")
        self.assertIsNotNone(normal)
        self.assertIsNotNone(barge)
        np.testing.assert_array_equal(normal, barge)

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
                self.frames = [np.full(FRAME_SIZE, 1024, dtype=np.int16) for _ in range(16)]
                self.frames.extend(np.full(FRAME_SIZE, 1024, dtype=np.int16) for _ in range(2))

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

    def test_question_categories_are_deterministic_and_lightweight(self):
        self.assertEqual(classify_question_category("What is Docker?"), "TECHNICAL")
        self.assertEqual(classify_question_category("Explain this in detail"), "COMPLEX")
        self.assertEqual(classify_question_category("Write a short poem"), "CREATIVE")
        self.assertEqual(classify_question_category("What time is it?"), "FACTUAL")
        self.assertEqual(classify_question_category("How are you?"), "CASUAL")

    def test_normal_monitor_stays_alive_until_shutdown_and_uses_named_queue(self):
        class FakeCapture:
            def register_consumer(self, name):
                self.consumer = name

            def clear_pending_frames(self):
                pass

        class FakeDetector:
            def listen(self, capture, stop_event, **kwargs):
                self.consumer = kwargs["consumer"]
                started.set()
                while not stop_event.is_set():
                    time.sleep(0.005)
                return None

        started = threading.Event()
        app_stop = threading.Event()
        capture = FakeCapture()
        detector = FakeDetector()
        monitor = NormalSpeechMonitor(
            capture,
            detector,
            app_stop,
            lambda utterance, timeline: None,
            lambda exc: self.fail(f"normal monitor error: {exc}"),
            lambda timeline: None,
            lambda timeline: None,
            lambda: True,
            lambda info: None,
        )
        monitor.start()
        self.assertTrue(started.wait(1.0))
        self.assertEqual(detector.consumer, "normal")
        self.assertTrue(monitor.is_alive())
        monitor.stop()
        self.assertFalse(monitor.is_alive())

    def test_normal_utterance_is_rejected_during_speaking(self):
        controller = VoiceController()
        controller._set_state("SPEAKING")
        controller._queue_normal_utterance(
            np.ones(480, dtype=np.float32),
            PerformanceTimeline(),
        )
        self.assertEqual(controller._utterances.qsize(), 0)

    def test_silence_does_not_create_a_normal_utterance(self):
        class SilentVad:
            def is_speech(self, raw, sample_rate):
                del raw, sample_rate
                return False

        class Capture:
            def __init__(self, stop_event):
                self.stop_event = stop_event
                self.frames = 0

            def drain_events(self):
                return []

            def is_active(self):
                return True

            def is_healthy(self):
                return True

            def read_frame(self, timeout=0.2):
                del timeout
                self.frames += 1
                if self.frames >= 8:
                    self.stop_event.set()
                return np.zeros(FRAME_SIZE, dtype=np.int16)

        stop_event = threading.Event()
        detector = UtteranceDetector.__new__(UtteranceDetector)
        detector._vad = SilentVad()
        self.assertIsNone(detector.listen(Capture(stop_event), stop_event))


class PerformanceRegressionTests(unittest.TestCase):
    def test_timeline_uses_monotonic_marks_and_reports_stage_durations(self):
        values = iter((10.0, 10.1, 10.4, 10.6, 11.0, 11.2))
        timeline = PerformanceTimeline(clock=lambda: next(values))
        timeline.mark("speech_finished")
        timeline.mark("whisper_finished")
        timeline.mark("gemma_started")
        timeline.mark("first_token")
        timeline.mark("first_sentence")
        timeline.mark("playback_started")

        self.assertAlmostEqual(timeline.elapsed("speech_finished", "whisper_finished"), 0.1)
        self.assertAlmostEqual(timeline.elapsed("gemma_started", "first_token"), 0.2)
        self.assertAlmostEqual(timeline.elapsed("speech_finished", "playback_started"), 1.2)

    def test_pipeline_exposes_generation_to_audio_stage_callbacks(self):
        events = []
        piper_started = threading.Event()

        def fake_stream(prompt, on_token, cancel_event):
            del prompt, cancel_event
            events.append("generation_started")
            on_token("The answer is concise. ")
            self.assertTrue(piper_started.wait(1.0))
            events.append("generation_continues")
            on_token("The second sentence follows.")
            events.append("generation_finished")
            return "The answer is concise. The second sentence follows."

        player = FakePlayer()
        pipeline = ResponsePipeline(
            "test",
            threading.Event(),
            player,
            lambda: events.append("first_token"),
            lambda: events.append("first_sentence"),
            lambda: events.append("playback_started"),
            lambda exc: self.fail(f"pipeline error: {exc}"),
            on_generation_started=lambda: events.append("gemma_started"),
            on_piper_started=lambda: (events.append("piper_started"), piper_started.set()),
            on_audio_ready=lambda: events.append("audio_ready"),
        )
        with patch("main.stream_dummy", side_effect=fake_stream):
            pipeline.start()
            pipeline.wait()

        self.assertTrue(pipeline.generation_succeeded)
        self.assertTrue(pipeline.tts_succeeded)
        self.assertLess(events.index("first_sentence"), events.index("generation_finished"))
        self.assertLess(events.index("piper_started"), events.index("generation_finished"))
        self.assertLess(events.index("audio_ready"), events.index("generation_finished"))
        self.assertEqual(events.count("playback_started"), 1)


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
        states = []
        controller.state_changed.connect(states.append)
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
        self.assertEqual(states[-2:], ["INTERRUPTED", "LISTENING"])
        self.assertEqual(controller._state, "LISTENING")
        controller._queue_normal_utterance(np.ones(480, dtype=np.float32), PerformanceTimeline())
        self.assertEqual(controller._utterances.qsize(), 1)

    def test_controller_does_not_block_on_response_playback(self):
        controller = VoiceController()
        class NoopBargeMonitor:
            def start(self, session_id):
                del session_id

            def stop(self, session_id=None):
                del session_id

            def set_playback_guard(self, session_id):
                del session_id

            def request_stop(self, session_id):
                del session_id

        controller._barge_monitor = NoopBargeMonitor()
        controller.transcriber = type(
            "FakeTranscriber",
            (),
            {"transcribe": lambda self, audio, stop_event, on_first_result=None: "What is Python?"},
        )()
        controller.player = FakePlayer()
        timeline = PerformanceTimeline()
        timeline.mark("speech_finished")

        with patch(
            "main.stream_dummy",
            side_effect=lambda prompt, on_token, cancel_event, history=None: (
                on_token("Python is a programming language."),
                "Python is a programming language.",
            )[1],
        ):
            started_at = time.perf_counter()
            controller._process_utterance(np.ones(480, dtype=np.float32), timeline)
            return_latency = time.perf_counter() - started_at

            self.assertLess(return_latency, 0.20)
            deadline = time.monotonic() + 1.0
            while controller._pipeline is not None and time.monotonic() < deadline:
                time.sleep(0.01)

        self.assertIsNone(controller._pipeline)
        self.assertEqual(controller._state, "LISTENING")
        self.assertEqual(controller.player.played, ["Python is a programming language."])

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

    def test_playback_meter_reports_real_wav_level(self):
        with NamedTemporaryFile(suffix=".wav") as wav_file:
            with wave.open(wav_file.name, "wb") as output:
                output.setnchannels(1)
                output.setsampwidth(2)
                output.setframerate(1000)
                output.writeframes((10000).to_bytes(2, "little", signed=True) * 50)
            meter = tts_module._WaveLevelMeter(wav_file.name)
            try:
                self.assertGreater(meter.level_at(0.05), 0.0)
            finally:
                meter.close()


if __name__ == "__main__":
    unittest.main()
