from unittest.mock import patch
from spark_ai import AI


def test_summarize_uses_prebuilt_udf():
    fake_udf = lambda col: f"summary::{col}"  # noqa: E731
    with (
        patch("spark_ai.ai.build_summarize_udf", return_value=fake_udf),
        patch("spark_ai.ai.spark_col", side_effect=lambda x: x),
    ):
        ai = AI()
        result = ai.summarize("message")
    assert result == "summary::message"
