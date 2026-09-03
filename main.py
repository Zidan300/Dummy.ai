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
        """Wait until both worker threads have stopped."""
        self._generation_thread.join()
        self._tts_thread.join()
        # join() is the authoritative completion check. The events make the
        # lifecycle observable in tests and prevent a consumer-only event from
        # being mistaken for full pipeline completion.
        self.generation_finished.wait()
        self.tts_finished.wait()
        if not self.is_cancelled():
            self.sentences.join()
        self.done.set()

    def cancel(self) -> int:
        """Cancel this response without shutting down the application."""
        self._cancel_event.set()
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
                        on_playback_start=self._playback_started,
                    )
                    if not played and not self.is_cancelled():
                        raise TTSError("TTS returned without playing the sentence")
                    if played:
                        self._spoken_sentence_count += 1
                except TTSError as exc:
                    self.tts_error = exc
                    if not self.is_cancelled():
                        self.on_error(exc)
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


class BargeInMonitor:
    """Listen for an exact interruption phrase during one response session."""

    SPEAKER_START_GUARD_SECONDS = 0.35

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
            detector = UtteranceDetector()
            while not cancel.is_set():
                with self._lock:
                    guard_until = self._guard_until
                try:
                    audio = detector.listen(
                        self.capture,
                        cancel,
                        ignore_until=guard_until,
                        max_utterance_seconds=3.0,
                    )
                except (AudioError, VadError) as exc:
                    if not cancel.is_set():
                        logger.error("Barge-in monitor failed: %s", exc)
                    return
                if audio is None or cancel.is_set():
                    return
                # Playback can begin while listen() is finishing a VAD
                # segment. Discard that segment if the speaker-start guard
                # was raised during the read, rather than transcribing
                # Dummy's own audio as a possible command.
                with self._lock:
                    guard_until = self._guard_until
                if time.monotonic() < guard_until:
                    continue

                try:
                    text = self.transcriber.transcribe(audio, cancel)
                except TranscriptionError as exc:
                    if not cancel.is_set():
                        logger.error("Barge-in transcription failed: %s", exc)
                    continue

                command = normalize_spoken_text(text)
                if command in INTERRUPTION_COMMANDS or command in EXIT_COMMANDS:
                    self.on_command(session_id, command)
                    return
        except Exception as exc:
            if not cancel.is_set():
                logger.exception("Barge-in worker failed: %s", exc)


class VoiceController(QObject):
    """The Qt worker owning the continuous voice loop."""

    state_changed = Signal(str)
    error = Signal(str)
    finished = Signal()

    def __init__(self) -> None:
        super().__init__()
        self._stop_event = threading.Event()
        self._state_lock = threading.Lock()
        self._state = "STARTING"
        self.context = ConversationContext(max_turns=10)
        self.audio = AudioCapture()
        self.transcriber = WhisperTranscriber()
        self.detector: UtteranceDetector | None = None
        self.player = SpeechPlayer()
        self._pipeline: ResponsePipeline | None = None
        self._pipeline_lock = threading.RLock()
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

        if pipeline is not None:
            pipeline.cancel()
            if not pipeline.generation_finished.is_set():
                logger.info("Gemma generation cancellation requested")
            logger.info("TTS queue cleared")
        if self.player.cancel():
            logger.info("Active ffplay terminated")
        if detected_at is not None:
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

            self._set_state("IDLE")
            self._set_state("LISTENING")
            while not self._stop_event.is_set():
                self._listen_and_process()
        except Exception as exc:
            logger.exception("Voice worker failed")
            self.error.emit(str(exc))
            self.request_shutdown("fatal voice worker error")
        finally:
            self.context.clear()
            self._set_state("SHUTTING_DOWN")
            self.interrupt_tts()
            self._barge_monitor.stop()
            self.audio.stop()
            self.player.cancel()
            self._set_state("STOPPED")
            self.finished.emit()

    def _initialize(self) -> bool:
        while not self._stop_event.is_set():
            try:
                if not self.audio.is_active():
                    self.audio.start()
                    logger.info("Microphone ready")
                if self.detector is None:
                    self.detector = UtteranceDetector()
                if not self.transcriber.ready:
                    self.transcriber.load()
                return True
            except (AudioError, TranscriptionError, VadError) as exc:
                self._report_recoverable_error(exc)
                self.audio.stop()
                if self._wait_or_stop(2.0):
                    return False
        return False

    def _listen_and_process(self) -> None:
        if not self.audio.is_active():
            if not self._recover_microphone():
                return

        self._set_state("LISTENING")
        try:
            if self.detector is None:
                raise VadError("VAD is not initialized")
            utterance = self.detector.listen(
                self.audio,
                self._stop_event,
                on_speech_detected=lambda: logger.info("Speech detected"),
            )
        except (AudioError, VadError) as exc:
            if self._stop_event.is_set():
                return
            self._report_recoverable_error(exc)
            self.audio.stop()
            self._wait_or_stop(0.5)
            return

        if utterance is None or self._stop_event.is_set():
            return

        speech_end_at = time.perf_counter()
        self._set_state("PROCESSING")
        try:
            text = self.transcriber.transcribe(utterance, self._stop_event)
        except TranscriptionError as exc:
            self._report_recoverable_error(exc)
            self._wait_or_stop(0.75)
            return

        transcription_complete_at = time.perf_counter()
        logger.info("Transcription complete")
        logger.info("[PERF] Transcription: %.2fs", transcription_complete_at - speech_end_at)
        if self._stop_event.is_set():
            return
        if not text:
            logger.warning("Empty transcription; returning to LISTENING")
            self._set_state("LISTENING")
            return
        logger.info("User: %s", text)

        intent = classify_intent(text)
        if intent == "EXIT":
            logger.info("Voice exit command detected")
            self.request_shutdown("voice exit command")
            return
        if intent == "INTERRUPTION":
            logger.info("Interruption command ignored while listening")
            self._set_state("LISTENING")
            return

        history = self.context.snapshot()
        response_override = local_reference_response(text, history)
        if response_override is None and intent == "COMMAND":
            response_override = unsupported_command_response(text)

        self._set_state("THINKING")
        with self._pipeline_lock:
            self._next_session_id += 1
            session_id = self._next_session_id
            self._interrupted_session_id = None
            self._interruption_started_at = None

        first_token_at: float | None = None
        first_sentence_at: float | None = None
        first_audio_at: float | None = None

        def first_token() -> None:
            nonlocal first_token_at
            if first_token_at is None:
                first_token_at = time.perf_counter()
                logger.info(
                    "[PERF] First token: %.2fs",
                    first_token_at - transcription_complete_at,
                )

        def first_sentence() -> None:
            nonlocal first_sentence_at
            if first_sentence_at is None:
                first_sentence_at = time.perf_counter()
                if first_token_at is not None:
                    logger.info(
                        "[PERF] First sentence: %.2fs",
                        first_sentence_at - first_token_at,
                    )

        def playback_start() -> None:
            nonlocal first_audio_at
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
                if first_audio_at is None:
                    first_audio_at = time.perf_counter()
                    self._barge_monitor.set_playback_guard(session_id)
                    self._set_state("SPEAKING")
                    if first_sentence_at is not None:
                        logger.info(
                            "[PERF] TTS start: %.2fs",
                            first_audio_at - first_sentence_at,
                        )

        def tts_finished() -> None:
            self._barge_monitor.request_stop(session_id)

        try:
            self.player.reset_cancellation()
        except TTSError as exc:
            self._report_recoverable_error(exc)
            self._wait_or_stop(0.25)
            return

        pipeline = ResponsePipeline(
            text,
            self._stop_event,
            self.player,
            first_token,
            first_sentence,
            playback_start,
            self._report_recoverable_error,
            on_tts_finished=tts_finished,
            history=history,
            response_override=response_override,
        )
        with self._pipeline_lock:
            self._pipeline = pipeline
            self._active_session_id = session_id
        try:
            self._barge_monitor.start(session_id)
            pipeline.start()
            pipeline.wait()
            with self._pipeline_lock:
                interrupted = self._interrupted_session_id == session_id
            if not self._stop_event.is_set() and not interrupted:
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
                    logger.info(
                        "[PERF] Total response: %.2fs",
                        time.perf_counter() - transcription_complete_at,
                    )
        finally:
            self._barge_monitor.stop(session_id)
            self.audio.clear_pending_frames()
            with self._pipeline_lock:
                if self._pipeline is pipeline:
                    self._pipeline = None
                    self._active_session_id = None
            if not self._stop_event.is_set():
                try:
                    self.player.reset_cancellation()
                except TTSError:
                    logger.exception("Could not reset TTS cancellation state")
                self._set_state("LISTENING")

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

        logger.info('Barge-in detected: "%s"', command)
        if command in EXIT_COMMANDS:
            self.request_shutdown("voice exit command during response")
            return
        self.interrupt_tts(session_id, detected_at)
        self._barge_monitor.request_stop(session_id)
        self._set_state("INTERRUPTED")

    def _recover_microphone(self) -> bool:
        self._set_state("ERROR")
        self._report_recoverable_error(AudioError("microphone disconnected"))
        self.audio.stop()
        while not self._stop_event.is_set():
            try:
                self.audio.start()
                logger.info("Microphone recovered")
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
