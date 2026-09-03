"""Dummy Phase 3 application: streamed responses with natural interruption."""

from __future__ import annotations

import logging
import queue
import signal
import sys
import threading
import time

from PySide6.QtCore import QObject, QThread, Qt, QtMsgType, Signal, qInstallMessageHandler
from PySide6.QtWidgets import QApplication

from ai import (
    AIError,
    SentenceBuffer,
    clean_for_speech,
    local_reference_response,
    stream_dummy,
    unsupported_command_response,
)
from audio import (
    AudioCapture,
    AudioError,
    BargeInDetector,
    TranscriptionError,
    UtteranceDetector,
    VadError,
    WhisperTranscriber,
)
from context import (
    ConversationContext,
    EXIT_COMMANDS,
    INTERRUPTION_COMMANDS,
    classify_intent,
    classify_question_category,
    is_exit_command,
    normalize_spoken_text,
)
from interface import DummyInterface
from tts import SpeechPlayer, TTSError


logger = logging.getLogger("dummy")

STATES = {
    "STARTING",
    "IDLE",
    "LISTENING",
    "PROCESSING",
    "THINKING",
    "SPEAKING",
    "INTERRUPTED",
    "ERROR",
    "SHUTTING_DOWN",
    "STOPPED",
}

class CombinedCancellation:
    """Event-like view that cancels on app shutdown or response interruption."""

    def __init__(self, *events: threading.Event) -> None:
        self._events = events

    def is_set(self) -> bool:
        return any(event.is_set() for event in self._events)


class PerformanceTimeline:
    """Monotonic, per-utterance timing marks for the voice pipeline."""

    def __init__(self, clock=time.perf_counter) -> None:
        self._clock = clock
        self.started_at: float | None = None
        self.marks: dict[str, float] = {}

    def mark(self, name: str) -> float:
        timestamp = self._clock()
        if self.started_at is None:
            self.started_at = timestamp
        self.marks[name] = timestamp
        logger.info("[PERF] %s: %.0f ms", name, (timestamp - self.started_at) * 1000.0)
        return timestamp

    def elapsed(self, start: str, end: str) -> float | None:
        first = self.marks.get(start)
        last = self.marks.get(end)
        if first is None or last is None:
            return None
        return max(0.0, last - first)

    def report(self) -> None:
        token_start = "gemma_started" if "gemma_started" in self.marks else "whisper_finished"
        pairs = (
            ("time_to_understand", "speech_finished", "whisper_finished"),
            ("time_to_first_token", token_start, "first_token"),
            ("time_to_first_sentence", "first_token", "first_sentence"),
            ("time_to_first_audio", "speech_finished", "playback_started"),
        )
        for label, start, end in pairs:
            duration = self.elapsed(start, end)
            if duration is not None:
                logger.info("[PERF] %s: %.0f ms", label, duration * 1000.0)


class ResponsePipeline:
    """Generate streamed text and synthesize sentences concurrently."""

    QUEUE_SIZE = 8

    def __init__(
        self,
        prompt: str,
        stop_event: threading.Event,
        player: SpeechPlayer,
        on_first_token,
        on_first_sentence,
        on_playback_start,
        on_error,
        on_tts_finished=None,
        history=None,
        response_override: str | None = None,
        on_generation_started=None,
        on_piper_started=None,
        on_audio_ready=None,
        on_audio_level=None,
    ) -> None:
        self.prompt = prompt
        self.stop_event = stop_event
        self._cancel_event = threading.Event()
        self._cancel_token = CombinedCancellation(stop_event, self._cancel_event)
        self.player = player
        self.on_first_token = on_first_token
        self.on_first_sentence = on_first_sentence
        self.on_playback_start = on_playback_start
        self.on_error = on_error
        self.on_tts_finished = on_tts_finished
        self.history = list(history or ())
        self.response_override = response_override
        self.on_generation_started = on_generation_started
        self.on_piper_started = on_piper_started
        self.on_audio_ready = on_audio_ready
        self.on_audio_level = on_audio_level
        self.sentences: queue.Queue[str | None] = queue.Queue(maxsize=self.QUEUE_SIZE)
        self.done = threading.Event()
        self.response = ""
        self.generation_finished = threading.Event()
        self.tts_finished = threading.Event()
        self.generation_succeeded = False
        self.tts_succeeded = False
        self.generation_error: Exception | None = None
        self.tts_error: Exception | None = None
        self._first_token_seen = False
        self._first_sentence_seen = False
        self._first_audio_started = False
        self._spoken_sentence_count = 0
        self._generation_thread = threading.Thread(
            target=self._generate,
            name="dummy-gemma",
            daemon=False,
        )
        self._tts_thread = threading.Thread(
            target=self._speak_sentences,
            name="dummy-tts",
            daemon=False,
        )

    def start(self) -> None:
        # Start the consumer first so the first generated sentence has no
        # extra handoff delay.
        self._tts_thread.start()
        self._generation_thread.start()

    def wait(self) -> None:
        """Wait normally, but return promptly once a turn is cancelled.

        Ollama's HTTP iterator may not unblock immediately after cancellation.
        TTS is stopped synchronously, while the cancelled generation worker
        observes the event and is prevented from publishing further output.
        """
        self._tts_thread.join()
        if self.is_cancelled():
            self._generation_thread.join(timeout=0.15)
            self._clear_pending_sentences()
            self.done.set()
            return

        self._generation_thread.join()
        # join() is the authoritative completion check. The events make the
        # lifecycle observable in tests and prevent a consumer-only event from
        # being mistaken for full pipeline completion.
        self.generation_finished.wait()
        self.tts_finished.wait()
        self.sentences.join()
        self.done.set()

    def cancel(self) -> int:
        """Cancel this response without shutting down the application."""
        self._cancel_event.set()
        return self._clear_pending_sentences()

    def _clear_pending_sentences(self) -> int:
        cleared = 0
        while True:
            try:
                self.sentences.get_nowait()
                self.sentences.task_done()
                cleared += 1
            except queue.Empty:
                return cleared

    def is_cancelled(self) -> bool:
        return self._cancel_token.is_set()

    def _generate(self) -> None:
        buffer = SentenceBuffer()
        if self.response_override is None:
            logger.info("Gemma generation started")
        else:
            logger.info("Local command response prepared")
        if self.response_override is None and self.on_generation_started:
            self.on_generation_started()

        def receive_token(token: str) -> None:
            if self.is_cancelled():
                return
            if not self._first_token_seen:
                self._first_token_seen = True
                logger.info("First Gemma token received")
                self.on_first_token()
            for sentence in buffer.add(token):
                if not self._first_sentence_seen:
                    self._first_sentence_seen = True
                    logger.info("First sentence ready")
                    self.on_first_sentence()
                self._put_sentence(sentence)

        try:
            if self.response_override is not None:
                self.response = clean_for_speech(self.response_override)
                if not self.is_cancelled() and self.response:
                    receive_token(self.response)
            else:
                stream_kwargs = {
                    "on_token": receive_token,
                    "cancel_event": self._cancel_token,
                }
                if self.history:
                    stream_kwargs["history"] = self.history
                self.response = stream_dummy(self.prompt, **stream_kwargs)
                self.response = clean_for_speech(self.response)
            if not self.is_cancelled():
                for sentence in buffer.finish():
                    if not self._first_sentence_seen:
                        self._first_sentence_seen = True
                        logger.info("First sentence ready")
                        self.on_first_sentence()
                    self._put_sentence(sentence)
            if not self.is_cancelled():
                self.generation_succeeded = True
                if self.response_override is None:
                    logger.info("Gemma generation finished")
        except AIError as exc:
            self.generation_error = exc
            if not self.is_cancelled():
                self.on_error(exc)
        except Exception as exc:
            self.generation_error = AIError(f"Gemma worker failed: {exc}")
            if not self.is_cancelled():
                self.on_error(self.generation_error)
        finally:
            self._put_sentinel()
            self.generation_finished.set()

    def _put_sentence(self, sentence: str) -> bool:
        sentence = clean_for_speech(sentence)
        if not sentence:
            return False
        while not self.is_cancelled():
            try:
                self.sentences.put(sentence, timeout=0.1)
                return True
            except queue.Full:
                if not self._tts_thread.is_alive():
                    raise TTSError("TTS worker stopped before consuming the sentence queue")
                continue
        return False

    def _put_sentinel(self) -> None:
        while not self.is_cancelled():
            try:
                self.sentences.put(None, timeout=0.1)
                return
            except queue.Full:
                if not self._tts_thread.is_alive():
                    return
                continue

    def _speak_sentences(self) -> None:
        try:
            while not self.is_cancelled():
                try:
                    sentence = self.sentences.get(timeout=0.1)
                except queue.Empty:
                    continue
                try:
                    if sentence is None:
                        self.tts_succeeded = (
                            not self.is_cancelled()
                            and self.generation_error is None
                            and self.tts_error is None
                            and self._spoken_sentence_count > 0
                        )
                        if self.tts_succeeded:
                            logger.info("TTS finished")
                        return
                    played = self.player.speak(
                        sentence,
                        self._cancel_token,
                        on_piper_start=self._piper_started,
                        on_audio_ready=self._audio_ready,
                        on_playback_start=self._playback_started,
                        on_playback_level=self._playback_level,
                    )
                    if not played and not self.is_cancelled():
                        raise TTSError("TTS returned without playing the sentence")
                    if played:
                        self._spoken_sentence_count += 1
                except TTSError as exc:
                    self.tts_error = exc
                    if not self.is_cancelled():
                        self.on_error(exc)
                        # Do not allow later queued sentences to play after
                        # the player has reported a real failure.
                        self.cancel()
                finally:
                    self.sentences.task_done()
        except Exception as exc:
            self.tts_error = exc
            if not self.is_cancelled():
                self.on_error(exc)
        finally:
            if self.on_tts_finished:
                self.on_tts_finished()
            self.tts_finished.set()

    def _playback_started(self) -> None:
        if self._first_audio_started or self.is_cancelled():
            return
        self._first_audio_started = True
        logger.info("TTS started")
        self.on_playback_start()

    def _piper_started(self) -> None:
        if not self.is_cancelled() and self.on_piper_started:
            self.on_piper_started()

    def _audio_ready(self) -> None:
        if not self.is_cancelled() and self.on_audio_ready:
            self.on_audio_ready()

    def _playback_level(self, level: float) -> None:
        if not self.is_cancelled() and self.on_audio_level:
            self.on_audio_level(level)


class NormalSpeechMonitor:
    """Continuously detect normal utterances from a broadcast mic queue."""

    CONSUMER = "normal"

    def __init__(self, capture, detector, app_stop_event, on_utterance, on_error, on_speech_started, on_speech_ended):
        self.capture = capture
        self.detector = detector
        self.app_stop_event = app_stop_event
        self.on_utterance = on_utterance
        self.on_error = on_error
        self.on_speech_started = on_speech_started
        self.on_speech_ended = on_speech_ended
        self._lock = threading.RLock()
        self._thread: threading.Thread | None = None
        self._stop_event: threading.Event | None = None
        self._reset_event = threading.Event()
        register = getattr(self.capture, "register_consumer", None)
        if register is not None:
            register(self.CONSUMER)

    def start(self) -> None:
        self.stop()
        self.capture.clear_pending_frames()
        local_stop = threading.Event()
        with self._lock:
            self._stop_event = local_stop
            self._thread = threading.Thread(
                target=self._run,
                args=(local_stop,),
                name="dummy-normal-listener",
                daemon=False,
            )
            thread = self._thread
        thread.start()

    def reset(self) -> None:
        self._reset_event.set()

    def stop(self) -> None:
        with self._lock:
            stop_event = self._stop_event
            thread = self._thread
        if stop_event is not None:
            stop_event.set()
        self._reset_event.set()
        if thread is not None and thread is not threading.current_thread():
            thread.join()
        with self._lock:
            if self._thread is thread:
                self._thread = None
                self._stop_event = None

    def is_alive(self) -> bool:
        with self._lock:
            return bool(self._thread and self._thread.is_alive())

    def _run(self, local_stop: threading.Event) -> None:
        try:
            while not self.app_stop_event.is_set() and not local_stop.is_set():
                self._reset_event.clear()
                cancel = CombinedCancellation(self.app_stop_event, local_stop, self._reset_event)
                timeline = PerformanceTimeline()

                def speech_started() -> None:
                    logger.info("Speech detected")
                    timeline.mark("speech_detected")
                    self.on_speech_started(timeline)

                try:
                    utterance = self.detector.listen(
                        self.capture,
                        cancel,
                        on_speech_detected=speech_started,
                        consumer=self.CONSUMER,
                    )
                except (AudioError, VadError) as exc:
                    if cancel.is_set():
                        continue
                    self.on_error(exc)
                    return

                if cancel.is_set():
                    continue
                if utterance is not None:
                    timeline.mark("speech_finished")
                    self.on_speech_ended(timeline)
                    self.on_utterance(utterance, timeline)
        except Exception as exc:
            if not self.app_stop_event.is_set() and not local_stop.is_set():
                self.on_error(exc)


class BargeInMonitor:
    """Recognize explicit control phrases on a short rolling VAD path."""

    SPEAKER_START_GUARD_SECONDS = 0.45
    CONSUMER = "barge"

    def __init__(self, capture, transcriber, app_stop_event, on_command) -> None:
        self.capture = capture
        self.transcriber = transcriber
        self.app_stop_event = app_stop_event
        self.on_command = on_command
        self._lock = threading.RLock()
        self._thread: threading.Thread | None = None
        self._stop_event: threading.Event | None = None
        self._session_id: int | None = None
        self._guard_until = 0.0
        register = getattr(self.capture, "register_consumer", None)
        if register is not None:
            register(self.CONSUMER)

    def start(self, session_id: int) -> None:
        self.stop()
        self.capture.clear_pending_frames()
        local_stop = threading.Event()
        with self._lock:
            self._session_id = session_id
            self._stop_event = local_stop
            self._guard_until = 0.0
            self._thread = threading.Thread(
                target=self._run,
                args=(session_id, local_stop),
                name=f"dummy-barge-in-{session_id}",
                daemon=False,
            )
            thread = self._thread
        thread.start()

    def set_playback_guard(self, session_id: int) -> None:
        with self._lock:
            if self._session_id != session_id:
                return
            self._guard_until = time.monotonic() + self.SPEAKER_START_GUARD_SECONDS
        self.capture.clear_pending_frames()

    def request_stop(self, session_id: int) -> None:
        with self._lock:
            if self._session_id == session_id and self._stop_event is not None:
                self._stop_event.set()

    def stop(self, session_id: int | None = None) -> None:
        with self._lock:
            if session_id is not None and self._session_id != session_id:
                return
            stop_event = self._stop_event
            thread = self._thread
        if stop_event is not None:
            stop_event.set()
        if thread is not None and thread is not threading.current_thread():
            thread.join()
        with self._lock:
            if self._thread is thread:
                self._thread = None
                self._stop_event = None
                self._session_id = None

    def _run(self, session_id: int, local_stop: threading.Event) -> None:
        cancel = CombinedCancellation(self.app_stop_event, local_stop)
        try:
            detector = BargeInDetector()
            while not cancel.is_set():
                try:
                    detector.listen_for_segments(
                        self.capture,
                        cancel,
                        on_speech_detected=lambda: logger.info("[BARGE] speech detected"),
                        on_segment=lambda audio, pause_confirmed: self._recognize_segment(
                            session_id,
                            audio,
                            cancel,
                            pause_confirmed=pause_confirmed,
                        ),
                        ignore_until=lambda: self._guard_until_for(session_id),
                        consumer=self.CONSUMER,
                    )
                except (AudioError, VadError) as exc:
                    if not cancel.is_set():
                        logger.error("[BARGE] monitor failed: %s", exc)
                    return
                if cancel.is_set():
                    return
        except Exception as exc:
            if not cancel.is_set():
                logger.exception("[BARGE] worker failed: %s", exc)

    def _recognize_segment(
        self,
        session_id: int,
        audio,
        cancel,
        pause_confirmed: bool = False,
    ) -> bool:
        if cancel.is_set():
            return True
        logger.info("[BARGE] recognition started")
        try:
            text = self.transcriber.transcribe(audio, cancel)
        except TranscriptionError as exc:
            if not cancel.is_set():
                logger.error("[BARGE] transcription failed: %s", exc)
            return False

        command = normalize_spoken_text(text)
        is_control = command in INTERRUPTION_COMMANDS or command in EXIT_COMMANDS
        if is_control and pause_confirmed:
            self.on_command(session_id, command)
            return True
        return False

    def _guard_until_for(self, session_id: int) -> float:
        with self._lock:
            if self._session_id != session_id:
                return time.monotonic()
            return self._guard_until


class VoiceController(QObject):
    """The Qt worker owning the continuous voice loop."""

    state_changed = Signal(str)
    audio_level = Signal(float)
    question_category = Signal(str)
    speech_started = Signal()
    speech_ended = Signal()
    thinking_started = Signal()
    whisper_first_result = Signal()
    first_token = Signal()
    first_sentence = Signal()
    tts_started = Signal()
    tts_finished = Signal()
    response_finished = Signal()
    interrupted = Signal()
    error = Signal(str)
    finished = Signal()

    def __init__(self) -> None:
        super().__init__()
        self._stop_event = threading.Event()
        self._state_lock = threading.Lock()
        self._state = "STARTING"
        self.context = ConversationContext(max_turns=10)
        self.audio = AudioCapture(on_level=self.audio_level.emit)
        self.transcriber = WhisperTranscriber()
        self.detector: UtteranceDetector | None = None
        self.player = SpeechPlayer()
        self._pipeline: ResponsePipeline | None = None
        self._pipeline_lock = threading.RLock()
        self._processing = False
        self._utterances: queue.Queue[tuple[object, PerformanceTimeline]] = queue.Queue(maxsize=2)
        self._next_session_id = 0
        self._active_session_id: int | None = None
        self._interrupted_session_id: int | None = None
        self._interruption_started_at: float | None = None
        self._barge_monitor = BargeInMonitor(
            self.audio,
            self.transcriber,
            self._stop_event,
            self._handle_barge_in,
        )
        self._normal_monitor: NormalSpeechMonitor | None = None

    @property
    def stop_event(self) -> threading.Event:
        return self._stop_event

    def request_shutdown(self, reason: str = "shutdown requested") -> None:
        """Thread-safe, idempotent entry point for every shutdown path."""
        first_request = not self._stop_event.is_set()
        self._stop_event.set()
        self.context.clear()
        self._set_state("SHUTTING_DOWN")
        if first_request:
            logger.info("Shutdown requested: %s", reason)
        self.interrupt_tts()
        self._barge_monitor.stop()
        self.audio.stop()

    def interrupt_tts(
        self,
        session_id: int | None = None,
        detected_at: float | None = None,
    ) -> None:
        """Stop current speech and generation without killing the app thread."""
        with self._pipeline_lock:
            pipeline = self._pipeline
            active_session_id = self._active_session_id
        if session_id is not None and active_session_id != session_id:
            return

        barge = session_id is not None
        if pipeline is not None:
            pipeline.cancel()
            if not pipeline.generation_finished.is_set():
                logger.info("%sGemma cancellation requested", "[BARGE] " if barge else "")
            logger.info("%sTTS queue cleared", "[BARGE] " if barge else "")
        if self.player.cancel():
            logger.info("%sffplay terminated", "[BARGE] " if barge else "")
        self.audio.clear_pending_frames()
        if barge:
            if self._normal_monitor is not None:
                self._normal_monitor.reset()
            self._clear_queued_utterances()
            logger.info("[BARGE] microphone buffer cleared")
        if detected_at is not None:
            logger.info(
                "[BARGE] interruption latency: %.0f ms",
                (time.perf_counter() - detected_at) * 1000.0,
            )
            logger.info(
                "[PERF] Barge-in response: %.2fs",
                time.perf_counter() - detected_at,
            )

    def run(self) -> None:
        try:
            logger.info("Dummy starting")
            self._set_state("STARTING")
            if not self._initialize():
                return

            if self._normal_monitor is None:
                raise VadError("normal speech monitor is not initialized")
            self._normal_monitor.start()
            self._set_state("IDLE")
            self._set_state("LISTENING")
            while not self._stop_event.is_set():
                if not self._normal_monitor.is_alive():
                    if not self._recover_microphone():
                        break
                    self._normal_monitor.start()
                try:
                    utterance, timeline = self._utterances.get(timeout=0.1)
                except queue.Empty:
                    continue
                try:
                    self._process_utterance(utterance, timeline)
                finally:
                    self._utterances.task_done()
        except Exception as exc:
            logger.exception("Voice worker failed")
            self.error.emit(str(exc))
            self.request_shutdown("fatal voice worker error")
        finally:
            self.context.clear()
            self._set_state("SHUTTING_DOWN")
            self.interrupt_tts()
            if self._normal_monitor is not None:
                self._normal_monitor.stop()
            self._barge_monitor.stop()
            with self._pipeline_lock:
                active_pipeline = self._pipeline
            if active_pipeline is not None:
                active_pipeline.cancel()
                active_pipeline.wait()
            self.audio.stop()
            self.player.cancel()
            self._set_state("STOPPED")
            self.finished.emit()

    def _initialize(self) -> bool:
        while not self._stop_event.is_set():
            try:
                if not self.audio.is_active():
                    microphone_started_at = time.perf_counter()
                    self.audio.start()
                    logger.info("Microphone ready")
                    logger.info(
                        "[PERF] microphone_startup: %.0f ms",
                        (time.perf_counter() - microphone_started_at) * 1000.0,
                    )
                if self.detector is None:
                    self.detector = UtteranceDetector()
                if self._normal_monitor is None:
                    self._normal_monitor = NormalSpeechMonitor(
                        self.audio,
                        self.detector,
                        self._stop_event,
                        self._queue_normal_utterance,
                        self._normal_monitor_error,
                        self._normal_speech_started,
                        self._normal_speech_ended,
                    )
                if not self.transcriber.ready:
                    self.transcriber.load()
                return True
            except (AudioError, TranscriptionError, VadError) as exc:
                self._report_recoverable_error(exc)
                self.audio.stop()
                if self._wait_or_stop(2.0):
                    return False
        return False

    def _normal_speech_started(self, timeline: PerformanceTimeline) -> None:
        del timeline
        self.speech_started.emit()

    def _normal_speech_ended(self, timeline: PerformanceTimeline) -> None:
        del timeline
        self.speech_ended.emit()

    def _normal_monitor_error(self, exc: Exception) -> None:
        if self._stop_event.is_set():
            return
        logger.error("Normal speech monitor failed: %s", exc)
        self._report_recoverable_error(exc)

    def _queue_normal_utterance(self, utterance, timeline: PerformanceTimeline) -> None:
        with self._pipeline_lock:
            accepting = (
                not self._stop_event.is_set()
                and self._pipeline is None
                and self._active_session_id is None
                and not self._processing
            )
        if not accepting:
            logger.debug("Discarding speech while a response is active")
            return
        try:
            self._utterances.put_nowait((utterance, timeline))
        except queue.Full:
            logger.warning("Normal utterance queue full; discarding oldest input")
            try:
                self._utterances.get_nowait()
                self._utterances.task_done()
            except queue.Empty:
                pass
            try:
                self._utterances.put_nowait((utterance, timeline))
            except queue.Full:
                pass

    def _clear_queued_utterances(self) -> None:
        while True:
            try:
                self._utterances.get_nowait()
                self._utterances.task_done()
            except queue.Empty:
                return

    def _processing_done(self) -> None:
        with self._pipeline_lock:
            self._processing = False

    def _process_utterance(self, utterance, timeline: PerformanceTimeline) -> None:
        with self._pipeline_lock:
            if self._processing or self._pipeline is not None or self._stop_event.is_set():
                return
            self._processing = True

        self._set_state("PROCESSING")

        def whisper_first_result() -> None:
            timeline.mark("whisper_first_result")
            self.whisper_first_result.emit()

        timeline.mark("whisper_started")
        try:
            text = self.transcriber.transcribe(
                utterance,
                self._stop_event,
                on_first_result=whisper_first_result,
            )
        except TranscriptionError as exc:
            self._report_recoverable_error(exc)
            self._wait_or_stop(0.75)
            self._processing_done()
            return

        timeline.mark("whisper_finished")
        logger.info("Transcription complete")
        if self._stop_event.is_set():
            self._processing_done()
            return
        if not text:
            logger.warning("Empty transcription; returning to LISTENING")
            self._set_state("LISTENING")
            self._processing_done()
            return
        logger.info("User: %s", text)

        intent = classify_intent(text)
        if intent == "EXIT":
            logger.info("Voice exit command detected")
            self.request_shutdown("voice exit command")
            self._processing_done()
            return
        if intent == "INTERRUPTION":
            logger.info("Interruption command ignored while listening")
            self._set_state("LISTENING")
            self._processing_done()
            return

        history = self.context.snapshot()
        response_override = local_reference_response(text, history)
        if response_override is None and intent == "COMMAND":
            response_override = unsupported_command_response(text)

        self.question_category.emit(classify_question_category(text))
        self._set_state("THINKING")
        self.thinking_started.emit()
        with self._pipeline_lock:
            self._next_session_id += 1
            session_id = self._next_session_id
            self._interrupted_session_id = None
            self._interruption_started_at = None

        def first_token() -> None:
            if "first_token" not in timeline.marks:
                timeline.mark("first_token")
                self.first_token.emit()

        def first_sentence() -> None:
            if "first_sentence" not in timeline.marks:
                timeline.mark("first_sentence")
                self.first_sentence.emit()

        def gemma_started() -> None:
            timeline.mark("gemma_started")

        def playback_start() -> None:
            with self._pipeline_lock:
                pipeline = self._pipeline
                if (
                    self._active_session_id != session_id
                    or pipeline is None
                    or pipeline.is_cancelled()
                    or self._stop_event.is_set()
                    or self._interrupted_session_id == session_id
                ):
                    return
                if "playback_started" not in timeline.marks:
                    timeline.mark("playback_started")
                    self._barge_monitor.set_playback_guard(session_id)
                    self._set_state("SPEAKING")
                    self.tts_started.emit()

        def piper_started() -> None:
            timeline.mark("piper_started")

        def audio_ready() -> None:
            timeline.mark("audio_ready")

        def playback_level(level: float) -> None:
            with self._pipeline_lock:
                if (
                    self._active_session_id == session_id
                    and self._interrupted_session_id != session_id
                    and not self._stop_event.is_set()
                ):
                    self.audio_level.emit(level)

        def tts_finished() -> None:
            with self._pipeline_lock:
                current = self._pipeline is pipeline and self._active_session_id == session_id
            if current:
                self._barge_monitor.request_stop(session_id)
                self.tts_finished.emit()

        def pipeline_error(exc: Exception) -> None:
            with self._pipeline_lock:
                current = self._pipeline is pipeline and self._active_session_id == session_id
            if current:
                self._report_recoverable_error(exc)

        try:
            self.player.reset_cancellation()
        except TTSError as exc:
            self._report_recoverable_error(exc)
            self._wait_or_stop(0.25)
            self._processing_done()
            return

        pipeline = ResponsePipeline(
            text,
            self._stop_event,
            self.player,
            first_token,
            first_sentence,
            playback_start,
            pipeline_error,
            on_tts_finished=tts_finished,
            history=history,
            response_override=response_override,
            on_generation_started=gemma_started,
            on_piper_started=piper_started,
            on_audio_ready=audio_ready,
            on_audio_level=playback_level,
        )
        with self._pipeline_lock:
            self._pipeline = pipeline
            self._active_session_id = session_id
        try:
            self._barge_monitor.start(session_id)
            pipeline.start()
            self._processing_done()
        except Exception:
            self._processing_done()
            raise
        threading.Thread(
            target=self._finish_pipeline,
            args=(session_id, pipeline, timeline, text),
            name=f"dummy-response-finish-{session_id}",
            daemon=False,
        ).start()

    def _finish_pipeline(
        self,
        session_id: int,
        pipeline: ResponsePipeline,
        timeline: PerformanceTimeline,
        text: str,
    ) -> None:
        pipeline.wait()
        with self._pipeline_lock:
            owns_session = self._pipeline is pipeline and self._active_session_id == session_id
            interrupted = self._interrupted_session_id == session_id
        if owns_session and not self._stop_event.is_set() and not interrupted:
            if pipeline.generation_error is not None:
                self._wait_or_stop(0.75)
            elif not pipeline.response.strip():
                logger.error("Gemma returned empty response")
                self._set_state("ERROR")
                self._wait_or_stop(0.75)
            elif (
                not pipeline.generation_finished.is_set()
                or not pipeline.tts_finished.is_set()
                or not pipeline.generation_succeeded
                or not pipeline.tts_succeeded
                or not pipeline.sentences.empty()
                or pipeline.sentences.unfinished_tasks != 0
            ):
                logger.error(
                    "Response pipeline incomplete: generation_finished=%s "
                    "tts_finished=%s generation_ok=%s tts_ok=%s queue_empty=%s",
                    pipeline.generation_finished.is_set(),
                    pipeline.tts_finished.is_set(),
                    pipeline.generation_succeeded,
                    pipeline.tts_succeeded,
                    pipeline.sentences.empty(),
                )
                self._set_state("ERROR")
                self._wait_or_stop(0.75)
            else:
                self.context.append(text, pipeline.response)
                timeline.mark("response_finished")
                self.response_finished.emit()
                timeline.report()
                logger.info(
                    "[PERF] Total response: %.2fs",
                    timeline.elapsed("speech_finished", "response_finished") or 0.0,
                )

        self._barge_monitor.stop(session_id)
        with self._pipeline_lock:
            owns_session = self._pipeline is pipeline
            if owns_session:
                self._pipeline = None
                self._active_session_id = None
        if owns_session:
            self.audio.clear_pending_frames()
            if not self._stop_event.is_set():
                try:
                    self.player.reset_cancellation()
                except TTSError:
                    logger.exception("Could not reset TTS cancellation state")
                self._set_state("LISTENING")
            if interrupted:
                logger.info("[BARGE] LISTENING")

    def _handle_barge_in(self, session_id: int, command: str) -> None:
        detected_at = time.perf_counter()
        with self._pipeline_lock:
            pipeline = self._pipeline
            if (
                self._active_session_id != session_id
                or pipeline is None
                or pipeline.is_cancelled()
                or self._interrupted_session_id == session_id
            ):
                return
            self._interrupted_session_id = session_id
            self._interruption_started_at = detected_at
            logger.info("[BARGE] session invalidated")

        logger.info('[BARGE] recognized: "%s"', command)
        if command in EXIT_COMMANDS:
            self.request_shutdown("voice exit command during response")
            return
        logger.info("[BARGE] interruption confirmed")
        self.interrupt_tts(session_id, detected_at)
        self._barge_monitor.request_stop(session_id)
        with self._pipeline_lock:
            if self._active_session_id == session_id:
                self._active_session_id = None
            if self._pipeline is pipeline:
                self._pipeline = None
        self.interrupted.emit()
        self._set_state("INTERRUPTED")
        self._set_state("LISTENING")

    def _recover_microphone(self) -> bool:
        self._set_state("ERROR")
        self._report_recoverable_error(AudioError("microphone disconnected"))
        self.audio.stop()
        while not self._stop_event.is_set():
            try:
                microphone_started_at = time.perf_counter()
                self.audio.start()
                logger.info("Microphone recovered")
                logger.info(
                    "[PERF] microphone_recovery: %.0f ms",
                    (time.perf_counter() - microphone_started_at) * 1000.0,
                )
                with self._pipeline_lock:
                    active_session_id = self._active_session_id
                if active_session_id is not None and not self._stop_event.is_set():
                    self._barge_monitor.start(active_session_id)
                self._set_state("LISTENING")
                return True
            except AudioError as exc:
                self._report_recoverable_error(exc)
                if self._wait_or_stop(2.0):
                    return False
        return False

    def _report_recoverable_error(self, exc: Exception) -> None:
        logger.error("%s", exc)
        self.error.emit(str(exc))
        self._set_state("ERROR")

    def _wait_or_stop(self, seconds: float) -> bool:
        return self._stop_event.wait(seconds)

    @staticmethod
    def _is_exit_command(text: str) -> bool:
        return is_exit_command(text)

    def _set_state(self, state: str) -> None:
        state = state.upper()
        if state not in STATES:
            raise ValueError(f"unknown state: {state}")
        with self._state_lock:
            if self._state == state:
                return
            self._state = state
        logger.info("State: %s", state)
        self.state_changed.emit(state)


class ShutdownCoordinator(QObject):
    """The single application-level shutdown coordinator."""

    def __init__(self, app: QApplication, ui: DummyInterface, controller: VoiceController):
        super().__init__()
        self.app = app
        self.ui = ui
        self.controller = controller
        self._lock = threading.Lock()
        self._started = False

    def request(self, reason: str) -> None:
        with self._lock:
            if self._started:
                return
            self._started = True
        self.controller.request_shutdown(reason)

    def finish(self) -> None:
        self.ui.allow_close()
        self.ui.close()
        self.app.quit()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)

    ui = DummyInterface()
    controller = VoiceController()
    thread = QThread()
    controller.moveToThread(thread)
    coordinator = ShutdownCoordinator(app, ui, controller)

    def handle_qt_message(mode, context, message) -> None:
        if mode == QtMsgType.QtFatalMsg:
            logger.critical("Qt fatal error: %s", message)
            coordinator.request("fatal Qt error")
        elif mode == QtMsgType.QtCriticalMsg:
            logger.error("Qt error: %s", message)
        elif mode == QtMsgType.QtWarningMsg:
            logger.warning("Qt warning: %s", message)

    previous_qt_handler = qInstallMessageHandler(handle_qt_message)
    controller.state_changed.connect(ui.set_state, Qt.QueuedConnection)
    controller.audio_level.connect(ui.set_audio_level, Qt.QueuedConnection)
    controller.question_category.connect(ui.set_question_category, Qt.QueuedConnection)
    controller.thinking_started.connect(ui.note_thinking, Qt.QueuedConnection)
    controller.first_token.connect(ui.note_first_token, Qt.QueuedConnection)
    controller.first_sentence.connect(ui.note_first_sentence, Qt.QueuedConnection)
    controller.interrupted.connect(ui.note_interrupted, Qt.QueuedConnection)
    thread.started.connect(controller.run, Qt.QueuedConnection)
    controller.finished.connect(thread.quit, Qt.DirectConnection)
    controller.finished.connect(controller.deleteLater)
    thread.finished.connect(coordinator.finish)
    ui.close_requested.connect(lambda: coordinator.request("window close"))

    def handle_sigint(signum, frame) -> None:
        coordinator.request("Ctrl+C")

    previous_sigint = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, handle_sigint)
    previous_excepthook = sys.excepthook

    def handle_uncaught_exception(exc_type, exc_value, exc_traceback) -> None:
        if issubclass(exc_type, KeyboardInterrupt):
            coordinator.request("Ctrl+C")
            return
        logger.critical("Uncaught application error", exc_info=(exc_type, exc_value, exc_traceback))
        coordinator.request("uncaught application error")

    sys.excepthook = handle_uncaught_exception
    app.aboutToQuit.connect(lambda: coordinator.request("Qt application quit"))

    ui.show()
    thread.start()
    try:
        return app.exec()
    finally:
        coordinator.request("application exit")
        thread.quit()
        thread.wait(3000)
        signal.signal(signal.SIGINT, previous_sigint)
        sys.excepthook = previous_excepthook
        qInstallMessageHandler(previous_qt_handler)


if __name__ == "__main__":
    sys.exit(main())
