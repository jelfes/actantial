import pytest
import pandas as pd
from actantial.validation import compare_labels


class TestCompareLabels:

    def test_perfect_agreement_accuracy(self):
        df = pd.DataFrame({"id": [1, 2], "Subject": ["Alice", "Bob"], "Object": ["X", "Y"]})
        result = compare_labels(df, df, metric="accuracy", actant_columns=["Subject", "Object"])
        assert result["avg"] == pytest.approx(1.0)

    def test_no_agreement_accuracy(self):
        df1 = pd.DataFrame({"id": [1], "Subject": ["Alice"]})
        df2 = pd.DataFrame({"id": [1], "Subject": ["Bob"]})
        result = compare_labels(df1, df2, metric="accuracy", actant_columns=["Subject"])
        assert result["avg"] == pytest.approx(0.0)

    def test_partial_agreement(self):
        df1 = pd.DataFrame({"id": [1, 2], "Subject": ["Alice", "Alice"]})
        df2 = pd.DataFrame({"id": [1, 2], "Subject": ["Alice", "Bob"]})
        result = compare_labels(df1, df2, metric="accuracy", actant_columns=["Subject"])
        assert result["avg"] == pytest.approx(0.5)

    def test_missing_id_column_raises(self):
        df = pd.DataFrame({"Subject": ["Alice"]})
        with pytest.raises(ValueError, match="id_column"):
            compare_labels(df, df, actant_columns=["Subject"])

    def test_no_matching_ids_raises(self):
        df1 = pd.DataFrame({"id": [1], "Subject": ["Alice"]})
        df2 = pd.DataFrame({"id": [2], "Subject": ["Bob"]})
        with pytest.raises(ValueError, match="No matching IDs"):
            compare_labels(df1, df2, actant_columns=["Subject"])

    def test_invalid_metric_raises(self):
        df = pd.DataFrame({"id": [1], "Subject": ["Alice"]})
        with pytest.raises(ValueError, match="Unsupported metric"):
            compare_labels(df, df, metric="invalid", actant_columns=["Subject"])

    def test_f1_micro(self):
        df = pd.DataFrame({"id": [1, 2], "Subject": ["Alice", "Bob"]})
        result = compare_labels(df, df, metric="f1_micro", actant_columns=["Subject"])
        assert result["avg"] == pytest.approx(1.0)

    def test_per_actant_scores_present(self):
        df = pd.DataFrame({"id": [1], "Subject": ["Alice"], "Object": ["X"]})
        result = compare_labels(df, df, actant_columns=["Subject", "Object"])
        assert "Subject" in result["per_actant"]
        assert "Object" in result["per_actant"]
