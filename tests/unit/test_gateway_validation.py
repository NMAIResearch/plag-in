"""Unit tests for the total, typed chat-request validation gate.

`_validate_text_only_chat_body` runs before any backend (worker or
embedded) is called, so these regressions are tested directly against the
pure function rather than through a live HTTP round trip.
"""
from __future__ import annotations

import unittest

from plag_in.errors import InvalidRequestError, UnsupportedCapabilityError
from plag_in.gateway import _validate_text_only_chat_body


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
    def test_stream_true_is_unsupported(self):
        with self.assertRaises(UnsupportedCapabilityError):
            _validate_text_only_chat_body(_body(stream=True))


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


if __name__ == "__main__":
    unittest.main()
