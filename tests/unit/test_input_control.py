import io
import unittest

from plag_in.input_control import confirm_action, select_number


class InputControlTests(unittest.TestCase):
    def test_escape_poisoned_confirmation_reprompts_then_accepts_clean_yes(self):
        answers = iter(["\x1b[Ay", "y"])
        output = io.StringIO()
        accepted = confirm_action(lambda _: next(answers), output, "Confirm? ")
        self.assertTrue(accepted)
        text = output.getvalue()
        self.assertIn("Input not recognised", text)
        self.assertIn("\\x1b", text)
        self.assertNotIn("\x1b", text)

    def test_confirmation_interrupt_is_clean_decline(self):
        output = io.StringIO()

        def interrupted(_):
            raise KeyboardInterrupt

        self.assertFalse(confirm_action(interrupted, output, "Confirm? "))
        self.assertIn("Action cancelled", output.getvalue())

    def test_confirmation_eof_is_clean_decline(self):
        output = io.StringIO()

        def ended(_):
            raise EOFError

        self.assertFalse(confirm_action(ended, output, "Confirm? "))
        self.assertIn("No change was made", output.getvalue())

    def test_repeated_invalid_confirmations_stop_at_attempt_limit(self):
        output = io.StringIO()
        accepted = confirm_action(lambda _: "perhaps", output, "Confirm? ", max_attempts=2)
        self.assertFalse(accepted)
        self.assertEqual(output.getvalue().count("Input not recognised"), 2)
        self.assertIn("attempt limit", output.getvalue())

    def test_numbered_selection_reprompts_and_bounds(self):
        answers = iter(["up", "2"])
        output = io.StringIO()
        selected = select_number(
            lambda _: next(answers), output, "Select: ", minimum=1, maximum=3
        )
        self.assertEqual(selected, 2)
        self.assertIn("Choose a number from 1 to 3", output.getvalue())


if __name__ == "__main__":
    unittest.main()
