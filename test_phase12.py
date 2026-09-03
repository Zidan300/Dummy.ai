"""Phase 13 keyboard-only interruption regression tests."""

from __future__ import annotations

import unittest

from PySide6.QtCore import QEvent, QCoreApplication, Qt
from PySide6.QtGui import QKeyEvent
from PySide6.QtWidgets import QApplication

from interface import DummyInterface
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


class KeyboardInterruptionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def _controller_with_pipeline(self, state):
        controller = VoiceController()
        controller._set_state(state)
        pipeline = ResponsePipeline(
            "old request",
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
        return controller, pipeline

    def test_s_key_interrupts_speaking(self):
        controller, pipeline = self._controller_with_pipeline("SPEAKING")
        controller.keyboard_interrupt()
        self.assertTrue(pipeline.is_cancelled())
        self.assertEqual(controller._state, "LISTENING")
        self.assertIsNone(controller._active_session_id)

    def test_s_key_interrupts_thinking(self):
        controller, pipeline = self._controller_with_pipeline("THINKING")
        controller.keyboard_interrupt()
        self.assertTrue(pipeline.is_cancelled())

    def test_s_key_does_nothing_when_not_processing(self):
        for state in ("STARTING", "IDLE", "LISTENING", "SHUTTING_DOWN"):
            controller = VoiceController()
            controller._set_state(state)
            controller.keyboard_interrupt()
            self.assertEqual(controller._state, state)

    def test_repeated_s_key_is_safe(self):
        controller, pipeline = self._controller_with_pipeline("SPEAKING")
        controller.keyboard_interrupt()
        controller.keyboard_interrupt()
        self.assertTrue(pipeline.is_cancelled())
        self.assertEqual(controller._state, "LISTENING")

    def test_shutdown_is_idempotent_after_interrupt(self):
        controller, pipeline = self._controller_with_pipeline("SPEAKING")
        controller.keyboard_interrupt()
        controller.request_shutdown("test")
        controller.request_shutdown("test again")
        self.assertTrue(pipeline.is_cancelled())
        self.assertTrue(controller.stop_event.is_set())
        self.assertEqual(controller._state, "SHUTTING_DOWN")

    def test_s_key_emits_only_for_physical_non_repeat_key(self):
        widget = DummyInterface()
        widget.show()
        widget.activateWindow()
        widget.setFocus()
        events = []
        widget.interrupt_requested.connect(lambda: events.append(True))
        QCoreApplication.sendEvent(
            widget,
            QKeyEvent(QEvent.KeyPress, Qt.Key_S, Qt.NoModifier, "s", False),
        )
        QCoreApplication.sendEvent(
            widget,
            QKeyEvent(QEvent.KeyPress, Qt.Key_S, Qt.NoModifier, "S", True),
        )
        QCoreApplication.sendEvent(
            widget,
            QKeyEvent(QEvent.KeyPress, Qt.Key_A, Qt.NoModifier, "a", False),
        )
        self.assertEqual(events, [True])
        widget.close()

    def test_s_key_signal_reaches_controller(self):
        widget = DummyInterface()
        controller, pipeline = self._controller_with_pipeline("SPEAKING")
        widget.interrupt_requested.connect(controller.keyboard_interrupt, Qt.QueuedConnection)
        widget.show()
        widget.activateWindow()
        widget.setFocus()
        QCoreApplication.sendEvent(
            widget,
            QKeyEvent(QEvent.KeyPress, Qt.Key_S, Qt.NoModifier, "s", False),
        )
        self.app.processEvents()
        self.assertTrue(pipeline.is_cancelled())
        self.assertEqual(controller._state, "LISTENING")
        widget.close()

    def test_controller_keeps_one_persistent_microphone(self):
        controller = VoiceController()
        self.assertIsNotNone(controller.audio)


if __name__ == "__main__":
    unittest.main()
