from unittest.mock import patch

import pytest

from spark_ai import AI


def test_classify_reuses_cached_udf_for_same_labels():
    fake_udf = lambda col: col  # noqa: E731
    with (
        patch(
            "spark_ai.ai.build_classify_udf", return_value=fake_udf
        ) as mocked_builder,
        patch("spark_ai.ai.spark_col", side_effect=lambda x: x),
    ):
        ai = AI()
        ai.classify("message", ["urgent", "spam", "normal"])
        ai.classify("message", ["urgent", "spam", "normal"])

    assert mocked_builder.call_count == 1


def test_classify_requires_labels():
    ai = AI()
    with pytest.raises(ValueError, match="labels must not be empty"):
        ai.classify("message", [])
