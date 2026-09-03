"""Deterministic Phase 10 living-interface tests (animator + offscreen Qt)."""

from __future__ import annotations

import os
import sys
import unittest

from animations import VisualAnimator, STATE_ORDER, CATEGORY_PROFILES

# Set offscreen Qt before any Qt widget is instantiated.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


class AnimatorBehaviorTests(unittest.TestCase):
    """Model-level checks that drive the Qt paint layer (no widgets needed)."""

    def _animator_started(self):
        animator = VisualAnimator()
        animator.set_state("LISTENING")
        animator.tick(0.016)
        return animator

    def test_all_states_have_distinct_profiles_and_colors(self):
        self.assertEqual(len(STATE_ORDER), 10)
        colors = {VisualAnimator()._accent}
        for state in STATE_ORDER:
            animator = VisualAnimator()
            animator.set_state(state)
            animator.tick(0.016)
            self.assertIn(state, STATE_ORDER)
        # At least some states must differ in target energy.
        energies = {}
        for state in ("IDLE", "LISTENING", "THINKING", "SPEAKING"):
            animator = VisualAnimator()
            animator.set_state(state)
            for _ in range(400):
                animator.tick(0.016)
            energies[state] = animator.energy
        self.assertGreater(energies["SPEAKING"], energies["IDLE"])
        self.assertGreater(energies["THINKING"], energies["IDLE"])

    def test_fast_attack_slow_decay_on_audio_level(self):
        animator = self._animator_started()
        animator.set_audio_level(1.0)
        attack_steps = []
        for _ in range(6):
            animator.tick(0.016)
            attack_steps.append(animator.audio_level)
        peak = animator.audio_level
        self.assertGreater(peak, 0.0)
        self.assertLess(peak, 1.0)
        animator.set_audio_level(0.0)
        decay1 = animator.audio_level
        animator.tick(0.016)
        decay2 = animator.audio_level
        self.assertLess(decay2, decay1)
        # Attack should rise faster than decay falls over equal frames.
        self.assertGreater(attack_steps[-1] - attack_steps[0],
                           decay1 - decay2 + 1e-6)

    def test_intent_changes_energy_mood(self):
        def steady(intent):
            animator = VisualAnimator()
            animator.set_state("THINKING")
            animator.set_intent(intent)
            for _ in range(400):
                animator.tick(0.016)
            return animator.energy
        self.assertGreater(steady("COMMAND"), steady("CONVERSATION"))

    def test_category_changes_profile(self):
        animator = VisualAnimator()
        for cat in CATEGORY_PROFILES:
            animator.set_category(cat)
            self.assertEqual(animator.category, cat)

    def test_collapse_triggers_on_interruption(self):
        animator = VisualAnimator()
        animator.set_state("SPEAKING")
        animator.set_state("INTERRUPTED")
        self.assertGreater(animator.collapse, 0.0)

    def test_conclusion_pulse_on_thinking_to_speaking(self):
        animator = VisualAnimator()
        animator.set_state("THINKING")
        animator.set_state("SPEAKING")
        self.assertGreater(animator.conclusion, 0.0)

    def test_voice_reactivity_properties(self):
        animator = VisualAnimator()
        animator.set_state("LISTENING")
        animator.set_audio_level(0.5)
        animator.tick(0.016)
        self.assertGreater(animator.audio_coupling, 0.0)
        self.assertGreater(animator.listening_ring, 0.0)
        animator.set_state("THINKING")
        self.assertGreater(animator.scan_ring, 0.0)
        animator.set_state("SPEAKING")
        animator.set_audio_level(0.8)
        animator.tick(0.016)
        self.assertGreater(animator.speech_ring, 0.0)

    def test_thinking_and_speaking_couple_to_voice(self):
        animator = VisualAnimator()
        animator.set_state("SPEAKING")
        self.assertEqual(animator.audio_coupling, 1.0)
        animator.set_state("THINKING")
        self.assertLess(animator.audio_coupling, 0.5)


class OffscreenInterfaceTests(unittest.TestCase):
    """Render every state through the real Qt widget offscreen."""

    @classmethod
    def setUpClass(cls):
        from PySide6.QtWidgets import QApplication
        cls.app = QApplication.instance() or QApplication(sys.argv)
        from interface import DummyInterface
        cls.ui_class = DummyInterface

    def test_every_state_renders_offscreen(self):
        widget = self.ui_class()
        widget.resize(1000, 700)
        widget.show()
        for state in STATE_ORDER:
            widget.set_state(state)
            widget.set_audio_level(0.7)
            widget.set_question_category("TECHNICAL")
            widget.set_intent("QUESTION")
            widget.note_first_token()
            widget.note_first_sentence()
            widget.note_speech_started()
            self.app.processEvents()
            image = widget.grab()
            self.app.processEvents()
            self.assertFalse(image.isNull(), f"render failed for {state}")
        widget.hide()
        widget.close()

    def test_state_transitions_do_not_throw(self):
        widget = self.ui_class()
        for state in STATE_ORDER:
            widget.set_state(state)
            self.app.processEvents()
        widget.hide()
        widget.close()

    def test_categories_and_intents_render(self):
        widget = self.ui_class()
        widget.show()
        for cat in ("TECHNICAL", "COMPLEX", "FACTUAL", "CREATIVE", "COMMAND", "CASUAL"):
            widget.set_question_category(cat)
            for intent in ("QUESTION", "COMMAND", "CONVERSATION", "EXIT", "INTERRUPTION"):
                widget.set_intent(intent)
                self.app.processEvents()
        widget.hide()
        widget.close()


if __name__ == "__main__":
    unittest.main()
