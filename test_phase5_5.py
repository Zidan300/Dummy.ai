"""Lightweight Phase 5.5 response-quality and context tests."""

from __future__ import annotations

import threading
import unittest
from unittest.mock import patch

import ai
from context import ConversationContext, classify_intent, is_exit_command, is_interruption_command


class FakeStream:
    def __init__(self, chunks):
        self.chunks = chunks
        self.closed = False

    def __iter__(self):
        return iter(self.chunks)

    def close(self):
        self.closed = True


class ContextTests(unittest.TestCase):
    def test_history_is_bounded_and_references_are_available(self):
        context = ConversationContext(max_turns=2)
        context.append("What is Docker?", "Docker packages apps into containers.")
        context.append("Why use it?", "It makes deployments consistent.")
        context.append("Is it free?", "The core tools are available for free.")

        self.assertEqual(len(context), 2)
        self.assertEqual(context.latest_user(), "Is it free?")
        self.assertEqual(context.latest_assistant(), "The core tools are available for free.")
        self.assertEqual([turn.user for turn in context.snapshot()], ["Why use it?", "Is it free?"])

    def test_context_clears(self):
        context = ConversationContext()
        context.append("Hello", "Hi.")
        context.clear()
        self.assertEqual(len(context), 0)
        self.assertIsNone(context.latest_user())


class IntentTests(unittest.TestCase):
    def test_requested_intent_examples(self):
        self.assertEqual(classify_intent("what's the weather like?"), "QUESTION")
        self.assertEqual(classify_intent("tell me about Docker"), "QUESTION")
        self.assertEqual(classify_intent("how are you?"), "CONVERSATION")
        self.assertEqual(classify_intent("open Safari"), "COMMAND")
        self.assertEqual(classify_intent("launch Spotify"), "COMMAND")
        self.assertEqual(classify_intent("stop"), "INTERRUPTION")
        self.assertEqual(classify_intent("cancel that"), "INTERRUPTION")
        self.assertEqual(classify_intent("goodbye"), "EXIT")

    def test_command_like_words_inside_questions_are_safe(self):
        self.assertEqual(classify_intent("How do I stop a running server?"), "QUESTION")
        self.assertEqual(classify_intent("I want to stop for lunch"), "CONVERSATION")
        self.assertFalse(is_interruption_command("How do I stop the service?"))
        self.assertTrue(is_exit_command("shut down."))


class ResponseQualityTests(unittest.TestCase):
    def test_prompt_contains_voice_quality_and_factuality_rules(self):
        self.assertIn("most useful information first", ai.SYSTEM_PROMPT)
        self.assertIn("Never invent facts", ai.SYSTEM_PROMPT)
        self.assertIn("what did I just", ai.SYSTEM_PROMPT)
        self.assertIn("step-by-step", ai.SYSTEM_PROMPT)
        self.assertIn("primary request", ai.SYSTEM_PROMPT)

    def test_reliable_references_can_be_answered_locally(self):
        context = ConversationContext()
        context.append("What is Docker?", "Docker packages apps into containers.")
        self.assertEqual(
            ai.local_reference_response("What did I just ask?", context.snapshot()),
            "You just said, What is Docker?",
        )
        self.assertEqual(
            ai.local_reference_response("What did you say?", context.snapshot()),
            "I said, Docker packages apps into containers.",
        )
        self.assertEqual(
            ai.local_reference_response("What did I just say?", None),
            "I don't have an earlier user message in this session.",
        )

    def test_context_is_sent_before_current_message(self):
        context = ConversationContext()
        context.append("What is Docker?", "Docker packages apps into containers.")
        messages = ai.build_messages("Why would I use it?", context.snapshot())
        self.assertEqual(messages[1], {"role": "user", "content": "What is Docker?"})
        self.assertEqual(messages[2], {"role": "assistant", "content": "Docker packages apps into containers."})
        self.assertEqual(messages[-1], {"role": "user", "content": "Why would I use it?"})

    def test_speech_cleanup_removes_formatting_without_rewriting_content(self):
        result = ai.clean_for_speech("## **Docker**\n1. Packages apps.\n[Docs](https://example.com) 🙂")
        self.assertEqual(result, "Docker Packages apps. Docs")
        self.assertEqual(ai.clean_for_speech("```python\nprint('hi')\n```"), "print('hi')")

    def test_streaming_and_generation_parameters(self):
        stream = FakeStream([
            {"message": {"content": "Docker is useful. "}},
            {"message": {"content": "It keeps deployments consistent."}},
        ])
        call = {}

        def fake_chat(**kwargs):
            call.update(kwargs)
            return stream

        with patch.object(ai.ollama, "chat", side_effect=fake_chat):
            tokens = []
            result = ai.stream_dummy("Why use it?", on_token=tokens.append)

        self.assertEqual(result, "Docker is useful. It keeps deployments consistent.")
        self.assertEqual(tokens, ["Docker is useful. ", "It keeps deployments consistent."])
        self.assertTrue(call["stream"])
        self.assertEqual(call["options"]["temperature"], 0.18)
        self.assertEqual(call["options"]["top_p"], 0.8)
        self.assertEqual(call["options"]["top_k"], 20)
        self.assertEqual(call["options"]["num_predict"], 128)
        self.assertTrue(stream.closed)

    def test_empty_stream_is_honest_and_closes(self):
        stream = FakeStream([])
        with patch.object(ai.ollama, "chat", return_value=stream):
            self.assertEqual(ai.stream_dummy("empty", cancel_event=threading.Event()), "")
        self.assertTrue(stream.closed)


if __name__ == "__main__":
    unittest.main()
