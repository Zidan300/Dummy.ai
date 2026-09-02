"""Dummy Phase 1 application and cancellable voice controller."""

from __future__ import annotations

import logging
import re
import signal
import sys
import threading

from PySide6.QtCore import QObject, QThread, Qt, QtMsgType, Signal, qInstallMessageHandler
from PySide6.QtWidgets import QApplication

from ai import AIError, ask_dummy
from audio import (
    AudioCapture,
    AudioError,
    TranscriptionError,
    UtteranceDetector,
    VadError,
    WhisperTranscriber,
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

EXIT_COMMANDS = {
    "exit",
    "quit",
    "shutdown",
    "shut down",
    "goodbye",
    "good bye",
    "stop",
    "terminate",
}


class VoiceController(QObject):
    """The only owner of the voice pipeline, running outside Qt's UI thread."""

    state_changed = Signal(str)
    error = Signal(str)
    finished = Signal()

    def __init__(self) -> None:
        super().__init__()
        self._stop_event = threading.Event()
        self._state_lock = threading.Lock()
        self._state = "STARTING"
        self.audio = AudioCapture()
        self.transcriber = WhisperTranscriber()
        self.detector: UtteranceDetector | None = None
        self.player = SpeechPlayer()

    @property
    def stop_event(self) -> threading.Event:
        return self._stop_event

    def request_shutdown(self, reason: str = "shutdown requested") -> None:
        """Thread-safe and idempotent shutdown entry point.

        This method intentionally does not wait for the worker. It wakes the
        audio loop and cancels any active playback; the worker performs the
        ordered cleanup and emits ``finished`` when it is actually stopped.
        """
        first_request = not self._stop_event.is_set()
        self._stop_event.set()
        self._set_state("SHUTTING_DOWN")
        if first_request:
            logger.info("Shutdown requested: %s", reason)
        self.audio.stop()
        self.player.cancel()

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
            self._set_state("SHUTTING_DOWN")
            # Ordered cleanup: stop listening, release microphone, cancel
            # processing/playback, then let the QThread finish.
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

        self._set_state("PROCESSING")
        try:
            text = self.transcriber.transcribe(utterance, self._stop_event)
        except TranscriptionError as exc:
            self._report_recoverable_error(exc)
            self._wait_or_stop(0.75)
            return

        if self._stop_event.is_set():
            return
        logger.info("Transcription complete")
        if not text:
            self._set_state("LISTENING")
            return
        print(f"You: {text}", flush=True)

        if self._is_exit_command(text):
            logger.info("Voice exit command detected")
            self.request_shutdown("voice exit command")
            return

        self._set_state("THINKING")
        try:
            response = ask_dummy(text, cancel_event=self._stop_event)
        except AIError as exc:
            self._report_recoverable_error(exc)
            self._wait_or_stop(0.75)
            return

        if self._stop_event.is_set():
            return
        if not response:
            self._set_state("LISTENING")
            return
        print(f"Dummy: {response}", flush=True)

        self._set_state("SPEAKING")
        try:
            self.player.speak(response, self._stop_event)
        except TTSError as exc:
            self._report_recoverable_error(exc)
            self._wait_or_stop(0.75)
        finally:
            if not self._stop_event.is_set():
                self._set_state("LISTENING")

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
        self.error.emit(str(exc))
        self._set_state("ERROR")

    def _wait_or_stop(self, seconds: float) -> bool:
        return self._stop_event.wait(seconds)

    @staticmethod
    def _is_exit_command(text: str) -> bool:
        normalized = re.sub(r"[.!?,;:]+$", "", text.lower().strip())
        return normalized in EXIT_COMMANDS

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
    """The single application-level shutdown entry point."""

    def __init__(self, app: QApplication, ui: DummyInterface, controller: VoiceController, thread: QThread):
        super().__init__()
        self.app = app
        self.ui = ui
        self.controller = controller
        self.thread = thread
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
    coordinator = ShutdownCoordinator(app, ui, controller, thread)

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
    controller.error.connect(lambda message: logger.error("Worker: %s", message), Qt.QueuedConnection)
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

    # If another Qt path requests application exit, use the same cancellation
    # path. Normal shutdown calls app.quit only after the worker is finished.
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
