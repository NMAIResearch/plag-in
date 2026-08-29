"""Unit tests for the typed request-validation gates of both protocols.

`_validate_text_only_chat_body` and `_validate_responses_body` run before
any backend (worker or embedded) is called, so these regressions are
tested directly against the pure functions rather than through a live
HTTP round trip. The Responses document and stream builders are pure too
and are tested the same way.
"""
from __future__ import annotations

import json
import unittest

from plag_in.errors import (
    BackendResponseError,
    InvalidAliasError,
    InvalidRequestError,
    UnsupportedCapabilityError,
)
from plag_in.gateway import (
    CHAT_COMPLETIONS_SUBSET,
    RESPONSES_SUBSET,
    _buffered_responses_stream,
    _responses_document,
    _validate_responses_body,
    _validate_text_only_chat_body,
)


def _body(**overrides) -> dict:
    body = {"model": "fixture", "messages": [{"role": "user", "content": "hi"}]}
    body.update(overrides)
    return body


class MessageShapeTests(unittest.TestCase):
    def test_empty_messages_list_is_rejected(self):
        with self.assertRaises(InvalidRequestError):
            _validate_text_only_chat_body(_body(messages=[]))

    def test_non_list_messages_is_rejected(self):
        with self.assertRaises(InvalidRequestError):
            _validate_text_only_chat_body(_body(messages="hi"))

    def test_non_object_message_entry_is_rejected(self):
        with self.assertRaises(InvalidRequestError):
            _validate_text_only_chat_body(_body(messages=["hi"]))

    def test_unsupported_role_is_rejected(self):
        with self.assertRaises(InvalidRequestError):
            _validate_text_only_chat_body(
                _body(messages=[{"role": "admin", "content": "hi"}])
            )

    def test_missing_role_is_rejected(self):
        with self.assertRaises(InvalidRequestError):
            _validate_text_only_chat_body(_body(messages=[{"content": "hi"}]))

    def test_non_string_content_is_rejected(self):
        with self.assertRaises(InvalidRequestError):
            _validate_text_only_chat_body(
                _body(messages=[{"role": "user", "content": 5}])
            )

    def test_structured_media_content_is_unsupported_not_invalid(self):
        with self.assertRaises(UnsupportedCapabilityError):
            _validate_text_only_chat_body(
                _body(
                    messages=[
                        {
                            "role": "user",
                            "content": [{"type": "image_url", "image_url": {"url": "https://x/y.png"}}],
                        }
                    ]
                )
            )

    def test_media_field_on_message_is_unsupported(self):
        with self.assertRaises(UnsupportedCapabilityError):
            _validate_text_only_chat_body(
                _body(messages=[{"role": "user", "content": "hi", "image_url": "https://x/y.png"}])
            )

    def test_well_formed_body_is_accepted(self):
        _validate_text_only_chat_body(_body())


class StreamingTests(unittest.TestCase):
    def test_stream_true_is_accepted_for_buffered_sse(self):
        _validate_text_only_chat_body(_body(stream=True))

    def test_stream_must_be_boolean(self):
        with self.assertRaises(InvalidRequestError):
            _validate_text_only_chat_body(_body(stream="yes"))


class MaxTokensTests(unittest.TestCase):
    def test_true_is_rejected(self):
        with self.assertRaises(InvalidRequestError):
            _validate_text_only_chat_body(_body(max_tokens=True))

    def test_zero_is_rejected(self):
        with self.assertRaises(InvalidRequestError):
            _validate_text_only_chat_body(_body(max_tokens=0))

    def test_negative_is_rejected(self):
        with self.assertRaises(InvalidRequestError):
            _validate_text_only_chat_body(_body(max_tokens=-1))

    def test_float_is_rejected(self):
        with self.assertRaises(InvalidRequestError):
            _validate_text_only_chat_body(_body(max_tokens=4.5))

    def test_beyond_generic_bound_is_rejected(self):
        with self.assertRaises(InvalidRequestError):
            _validate_text_only_chat_body(_body(max_tokens=10_000_000))

    def test_positive_integer_is_accepted(self):
        _validate_text_only_chat_body(_body(max_tokens=64))


class SamplingValidationTests(unittest.TestCase):
    def test_non_finite_temperature_is_rejected(self):
        with self.assertRaises(InvalidRequestError):
            _validate_text_only_chat_body(_body(temperature=float("inf")))

    def test_nan_top_p_is_rejected(self):
        with self.assertRaises(InvalidRequestError):
            _validate_text_only_chat_body(_body(top_p=float("nan")))

    def test_negative_temperature_is_rejected(self):
        with self.assertRaises(InvalidRequestError):
            _validate_text_only_chat_body(_body(temperature=-0.01))

    def test_top_p_above_one_is_rejected(self):
        with self.assertRaises(InvalidRequestError):
            _validate_text_only_chat_body(_body(top_p=1.01))

    def test_top_p_below_zero_is_rejected(self):
        with self.assertRaises(InvalidRequestError):
            _validate_text_only_chat_body(_body(top_p=-0.01))

    def test_bool_temperature_is_rejected(self):
        with self.assertRaises(InvalidRequestError):
            _validate_text_only_chat_body(_body(temperature=True))

    def test_non_integer_seed_is_rejected(self):
        with self.assertRaises(InvalidRequestError):
            _validate_text_only_chat_body(_body(seed=1.5))

    def test_bool_seed_is_rejected(self):
        with self.assertRaises(InvalidRequestError):
            _validate_text_only_chat_body(_body(seed=True))

    def test_valid_sampling_values_are_accepted(self):
        _validate_text_only_chat_body(_body(temperature=0.2, top_p=0.9, seed=42))


def _chat_response(content: str = "hello", finish: str = "stop", usage: dict | None = None) -> dict:
    response = {
        "id": "chatcmpl-fixture",
        "object": "chat.completion",
        "created": 1_700_000_000,
        "model": "fixture",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": finish,
            }
        ],
    }
    if usage is not None:
        response["usage"] = usage
    return response


class ResponsesInputFormTests(unittest.TestCase):
    """Requirement 1: text input as a string and as message-style items."""

    def test_string_input_becomes_one_user_message(self):
        result = _validate_responses_body({"model": "fixture", "input": "hello"})
        self.assertEqual(result["messages"], [{"role": "user", "content": "hello"}])
        self.assertFalse(result["stream"])

    def test_message_items_preserve_role_and_order(self):
        result = _validate_responses_body(
            {
                "model": "fixture",
                "input": [
                    {"role": "user", "content": [{"type": "input_text", "text": "one"}]},
                    {"role": "assistant", "content": [{"type": "output_text", "text": "two"}]},
                    {"role": "user", "content": "three"},
                ],
            }
        )
        self.assertEqual(
            result["messages"],
            [
                {"role": "user", "content": "one"},
                {"role": "assistant", "content": "two"},
                {"role": "user", "content": "three"},
            ],
        )

    def test_instructions_lead_the_conversation_as_a_system_message(self):
        result = _validate_responses_body(
            {"model": "fixture", "instructions": "be brief", "input": "hello"}
        )
        self.assertEqual(result["messages"][0], {"role": "system", "content": "be brief"})

    def test_developer_role_is_carried_as_system_not_dropped(self):
        result = _validate_responses_body(
            {"model": "fixture", "input": [{"role": "developer", "content": "policy"}]}
        )
        self.assertEqual(result["messages"], [{"role": "system", "content": "policy"}])

    def test_multiple_text_parts_are_concatenated(self):
        result = _validate_responses_body(
            {
                "model": "fixture",
                "input": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": "a"},
                            {"type": "input_text", "text": "b"},
                        ],
                    }
                ],
            }
        )
        self.assertEqual(result["messages"][0]["content"], "ab")

    def test_missing_model_is_rejected(self):
        with self.assertRaises(InvalidRequestError):
            _validate_responses_body({"input": "hello"})

    def test_invalid_alias_is_rejected(self):
        with self.assertRaises(InvalidAliasError):
            _validate_responses_body({"model": "../escape", "input": "hello"})

    def test_missing_input_is_rejected(self):
        with self.assertRaises(InvalidRequestError):
            _validate_responses_body({"model": "fixture"})

    def test_empty_input_array_is_rejected(self):
        with self.assertRaises(InvalidRequestError):
            _validate_responses_body({"model": "fixture", "input": []})

    def test_unsupported_input_role_is_rejected(self):
        with self.assertRaises(InvalidRequestError):
            _validate_responses_body(
                {"model": "fixture", "input": [{"role": "admin", "content": "x"}]}
            )

    def test_non_string_instructions_are_rejected(self):
        with self.assertRaises(InvalidRequestError):
            _validate_responses_body({"model": "fixture", "instructions": 5, "input": "hi"})

    def test_non_boolean_stream_is_rejected(self):
        with self.assertRaises(InvalidRequestError):
            _validate_responses_body({"model": "fixture", "input": "hi", "stream": "yes"})


class ResponsesUnsupportedFeatureTests(unittest.TestCase):
    """Requirement 5: unsupported features are refused truthfully, not ignored."""

    def _refused(self, **overrides):
        body = {"model": "fixture", "input": "hello"}
        body.update(overrides)
        with self.assertRaises(UnsupportedCapabilityError) as caught:
            _validate_responses_body(body)
        return caught.exception

    def test_tools_are_refused(self):
        error = self._refused(tools=[{"type": "function", "name": "f"}])
        self.assertEqual(error.fields["field"], "tools")
        self.assertEqual(error.fields["supported_subset"]["tool_execution"], "absent")

    def test_forced_tool_choice_is_refused(self):
        self._refused(tool_choice="required")

    def test_tool_choice_none_is_accepted(self):
        result = _validate_responses_body(
            {"model": "fixture", "input": "hello", "tool_choice": "none"}
        )
        self.assertEqual(result["messages"], [{"role": "user", "content": "hello"}])

    def test_background_mode_is_refused(self):
        self.assertEqual(self._refused(background=True).fields["field"], "background")

    def test_stored_responses_are_refused(self):
        self.assertEqual(self._refused(store=True).fields["field"], "store")

    def test_conversation_state_is_refused(self):
        self.assertEqual(
            self._refused(previous_response_id="resp_1").fields["field"], "previous_response_id"
        )

    def test_conversation_object_is_refused(self):
        self.assertEqual(self._refused(conversation="conv_1").fields["field"], "conversation")

    def test_reasoning_configuration_is_refused(self):
        self.assertEqual(self._refused(reasoning={"effort": "low"}).fields["field"], "reasoning")

    def test_include_is_refused(self):
        self.assertEqual(self._refused(include=["message.output_text"]).fields["field"], "include")

    def test_image_input_is_refused(self):
        error = self._refused(
            input=[{"role": "user", "content": [{"type": "input_image", "image_url": "u"}]}]
        )
        self.assertEqual(error.fields["content_type"], "input_image")

    def test_file_input_is_refused(self):
        error = self._refused(
            input=[{"role": "user", "content": [{"type": "input_file", "file_id": "f"}]}]
        )
        self.assertEqual(error.fields["content_type"], "input_file")

    def test_non_message_input_items_are_refused(self):
        error = self._refused(input=[{"type": "function_call", "name": "f", "arguments": "{}"}])
        self.assertEqual(error.fields["item_type"], "function_call")

    def test_absent_and_empty_unsupported_fields_do_not_refuse(self):
        result = _validate_responses_body(
            {
                "model": "fixture",
                "input": "hello",
                "tools": [],
                "include": [],
                "store": False,
                "background": False,
                "previous_response_id": None,
            }
        )
        self.assertEqual(result["model"], "fixture")


class ResponsesBoundedFieldTests(unittest.TestCase):
    def test_max_output_tokens_must_be_a_positive_bounded_integer(self):
        for value in (0, -1, True, 4.5, 10_000_000):
            with self.subTest(value=value):
                with self.assertRaises(InvalidRequestError):
                    _validate_responses_body(
                        {"model": "fixture", "input": "hi", "max_output_tokens": value}
                    )

    def test_max_output_tokens_maps_to_the_generation_bound(self):
        result = _validate_responses_body(
            {"model": "fixture", "input": "hi", "max_output_tokens": 64}
        )
        self.assertEqual(result["max_output_tokens"], 64)

    def test_sampling_values_are_carried_for_the_shared_chat_gate(self):
        result = _validate_responses_body(
            {"model": "fixture", "input": "hi", "temperature": 0.2, "top_p": 0.9, "seed": 7}
        )
        self.assertEqual((result["temperature"], result["top_p"], result["seed"]), (0.2, 0.9, 7))


class ResponsesDocumentTests(unittest.TestCase):
    """Requirements 2 and 4: output construction and usage mapping."""

    def test_output_item_and_text_are_constructed(self):
        document = _responses_document("fixture", _chat_response("the answer"))
        self.assertEqual(document["object"], "response")
        self.assertEqual(document["status"], "completed")
        self.assertEqual(document["model"], "fixture")
        item = document["output"][0]
        self.assertEqual(item["type"], "message")
        self.assertEqual(item["role"], "assistant")
        self.assertEqual(item["content"][0]["type"], "output_text")
        self.assertEqual(item["content"][0]["text"], "the answer")
        self.assertEqual(document["output_text"], "the answer")

    def test_absent_capability_is_stated_on_every_response(self):
        document = _responses_document("fixture", _chat_response())
        self.assertEqual(document["tools"], [])
        self.assertEqual(document["tool_choice"], "none")
        self.assertFalse(document["store"])
        self.assertFalse(document["parallel_tool_calls"])

    def test_usage_is_mapped_when_the_backend_supplies_it(self):
        document = _responses_document(
            "fixture",
            _chat_response(
                usage={"prompt_tokens": 11, "completion_tokens": 5, "total_tokens": 16}
            ),
        )
        self.assertEqual(
            document["usage"],
            {"input_tokens": 11, "output_tokens": 5, "total_tokens": 16},
        )

    def test_usage_is_absent_when_the_backend_supplies_none(self):
        document = _responses_document("fixture", _chat_response())
        self.assertNotIn("usage", document)

    def test_a_length_stop_is_reported_as_incomplete_details(self):
        document = _responses_document("fixture", _chat_response(finish="length"))
        self.assertEqual(document["incomplete_details"], {"reason": "max_output_tokens"})

    def test_a_normal_stop_carries_no_incomplete_details(self):
        document = _responses_document("fixture", _chat_response(finish="stop"))
        self.assertIsNone(document["incomplete_details"])

    def test_a_backend_without_choices_is_a_typed_backend_error(self):
        with self.assertRaises(BackendResponseError):
            _responses_document("fixture", {"choices": []})

    def test_non_text_backend_content_is_a_typed_backend_error(self):
        response = _chat_response()
        response["choices"][0]["message"]["content"] = {"parts": []}
        with self.assertRaises(BackendResponseError):
            _responses_document("fixture", response)


class ResponsesStreamTests(unittest.TestCase):
    """Requirement 3: event order, terminal state and reconstructed text."""

    def _events(self, text: str = "streamed answer"):
        document = _responses_document("fixture", _chat_response(text))
        raw = _buffered_responses_stream(document).data.decode("utf-8")
        events = []
        for block in raw.split("\n\n"):
            if not block.strip():
                continue
            name, payload = block.split("\n", 1)
            events.append(
                (name[len("event: "):], json.loads(payload[len("data: "):]))
            )
        return events

    def test_event_order_is_the_documented_sequence(self):
        self.assertEqual(
            [name for name, _payload in self._events()],
            [
                "response.created",
                "response.in_progress",
                "response.output_item.added",
                "response.content_part.added",
                "response.output_text.delta",
                "response.output_text.done",
                "response.content_part.done",
                "response.output_item.done",
                "response.completed",
            ],
        )

    def test_sequence_numbers_are_contiguous_from_zero(self):
        events = self._events()
        self.assertEqual(
            [payload["sequence_number"] for _name, payload in events],
            list(range(len(events))),
        )

    def test_the_terminal_event_carries_the_completed_response(self):
        name, payload = self._events()[-1]
        self.assertEqual(name, "response.completed")
        self.assertEqual(payload["response"]["status"], "completed")
        self.assertEqual(payload["response"]["output"][0]["content"][0]["text"], "streamed answer")

    def test_reconstructed_text_equals_the_final_text(self):
        events = self._events()
        deltas = "".join(
            payload["delta"] for name, payload in events if name == "response.output_text.delta"
        )
        final = next(
            payload["response"]["output_text"]
            for name, payload in events
            if name == "response.completed"
        )
        self.assertEqual(deltas, final)

    def test_no_output_is_announced_before_it_exists(self):
        events = dict(self._events())
        self.assertEqual(events["response.created"]["response"]["output"], [])
        self.assertEqual(events["response.in_progress"]["response"]["output"], [])
        self.assertEqual(events["response.output_item.added"]["item"]["content"], [])

    def test_every_event_names_its_own_type(self):
        for name, payload in self._events():
            self.assertEqual(payload["type"], name)


class PublishedProtocolSubsetTests(unittest.TestCase):
    """Handoff 43: both protocols publish a subset and neither is ranked first."""

    def test_both_subsets_name_their_own_path(self):
        self.assertEqual(CHAT_COMPLETIONS_SUBSET["path"], "/v1/chat/completions")
        self.assertEqual(RESPONSES_SUBSET["path"], "/v1/responses")

    def test_neither_subset_names_a_harness_a_vendor_or_a_model(self):
        text = json.dumps([CHAT_COMPLETIONS_SUBSET, RESPONSES_SUBSET]).lower()
        for name in (
            "codex",
            "claude",
            "anthropic",
            "openai",
            "gemini",
            "google",
            "odysseus",
            "qwen",
            "ollama",
        ):
            self.assertNotIn(name, text, name)

    def test_both_subsets_declare_the_same_absent_capabilities(self):
        self.assertEqual(RESPONSES_SUBSET["tool_execution"], "absent")
        self.assertEqual(RESPONSES_SUBSET["conversation_state"], "absent")
        self.assertEqual(CHAT_COMPLETIONS_SUBSET["tool_call_execution"], "absent")
        self.assertEqual(CHAT_COMPLETIONS_SUBSET["conversation_state"], "absent")


if __name__ == "__main__":
    unittest.main()
