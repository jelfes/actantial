import numpy as np
import pandas as pd
import pytest

from actantial.validation import compare_annotations

COLS = ["Subject", "Object"]


def make_df(ids, subjects, objects):
    return pd.DataFrame({"id": ids, "Subject": subjects, "Object": objects})


class TestCompareLabelsErrors:
    def test_single_df_raises(self):
        df = make_df([1], ["hero"], ["sword"])
        with pytest.raises(ValueError, match="At least two DataFrames"):
            compare_annotations([df])

    def test_names_length_mismatch_raises(self):
        df = make_df([1], ["hero"], ["sword"])
        with pytest.raises(ValueError, match=r"len\(names\)"):
            compare_annotations([df, df], names=["a", "b", "c"])

    def test_missing_id_column_raises(self):
        df = pd.DataFrame({"Subject": ["hero"]})
        with pytest.raises(ValueError, match="id_column"):
            compare_annotations([df, df], actant_columns=COLS)

    def test_invalid_metric_raises(self):
        df = make_df([1], ["hero"], ["sword"])
        with pytest.raises(ValueError, match="Unsupported metric"):
            compare_annotations([df, df], metric="jaccard", actant_columns=COLS)


class TestCompareLabelsStructure:
    def test_index_contains_actants_and_avg(self):
        df = make_df([1, 2], ["hero", "villain"], ["sword", "shield"])
        result = compare_annotations([df, df], metric="accuracy", actant_columns=COLS)
        assert list(result.index) == ["Subject", "Object", "avg"]

    def test_pairwise_columns_two_coders(self):
        df = make_df([1], ["hero"], ["sword"])
        result = compare_annotations(
            [df, df], names=["a", "b"], metric="accuracy", actant_columns=COLS
        )
        assert list(result.columns) == ["a_b", "avg", "N"]

    def test_pairwise_columns_three_coders(self):
        df = make_df([1], ["hero"], ["sword"])
        result = compare_annotations(
            [df, df, df], names=["a", "b", "c"], metric="accuracy", actant_columns=COLS
        )
        assert list(result.columns) == ["a_b", "a_c", "b_c", "avg", "N"]

    def test_krippendorff_has_alpha_column_only(self):
        df = make_df([1, 2], ["hero", "villain"], ["sword", "shield"])
        result = compare_annotations(
            [df, df], metric="krippendorff_alpha", actant_columns=COLS
        )
        assert list(result.columns) == ["alpha", "N"]

    def test_avg_column_is_mean_of_pair_columns(self):
        df1 = make_df([1, 2], ["hero", "hero"], ["sword", "sword"])
        df2 = make_df([1, 2], ["hero", "villain"], ["sword", "shield"])
        df3 = make_df([1, 2], ["hero", "villain"], ["sword", "shield"])
        result = compare_annotations(
            [df1, df2, df3],
            names=["a", "b", "c"],
            metric="accuracy",
            actant_columns=["Subject"],
        )
        # a_b: 0.5, a_c: 0.5, b_c: 1.0 → avg = 2/3
        assert result.loc["Subject", "avg"] == pytest.approx(
            result.loc["Subject", ["a_b", "a_c", "b_c"]].mean()
        )

    def test_avg_row_is_mean_of_actant_rows(self):
        df1 = make_df([1, 2], ["hero", "hero"], ["sword", "sword"])
        df2 = make_df([1, 2], ["hero", "villain"], ["sword", "shield"])
        result = compare_annotations(
            [df1, df2], names=["a", "b"], metric="accuracy", actant_columns=COLS
        )
        assert result.loc["avg", "a_b"] == pytest.approx(
            result.loc[["Subject", "Object"], "a_b"].mean()
        )


class TestCompareLabelsScores:
    def test_perfect_agreement_accuracy(self):
        df = make_df([1, 2], ["hero", "villain"], ["sword", "shield"])
        result = compare_annotations([df, df], metric="accuracy", actant_columns=COLS)
        assert result.loc["Subject", "avg"] == pytest.approx(1.0)
        assert result.loc["Object", "avg"] == pytest.approx(1.0)
        assert result.loc["avg", "avg"] == pytest.approx(1.0)

    def test_no_agreement_accuracy(self):
        df1 = make_df([1, 2], ["hero", "villain"], ["sword", "shield"])
        df2 = make_df([1, 2], ["villain", "hero"], ["shield", "sword"])
        result = compare_annotations([df1, df2], metric="accuracy", actant_columns=COLS)
        assert result.loc["Subject", "avg"] == pytest.approx(0.0)
        assert result.loc["Object", "avg"] == pytest.approx(0.0)

    def test_partial_agreement_accuracy(self):
        df1 = make_df([1, 2], ["hero", "villain"], ["sword", "shield"])
        df2 = make_df([1, 2], ["hero", "hero"], ["sword", "sword"])
        result = compare_annotations([df1, df2], metric="accuracy", actant_columns=COLS)
        assert result.loc["Subject", "avg"] == pytest.approx(0.5)
        assert result.loc["Object", "avg"] == pytest.approx(0.5)

    def test_perfect_agreement_krippendorff(self):
        df = make_df(
            [1, 2, 3], ["hero", "villain", "hero"], ["sword", "shield", "sword"]
        )
        result = compare_annotations(
            [df, df, df], metric="krippendorff_alpha", actant_columns=COLS
        )
        assert result.loc["Subject", "alpha"] == pytest.approx(1.0)
        assert result.loc["Object", "alpha"] == pytest.approx(1.0)

    def test_three_coders_all_pairs_computed(self):
        df = make_df([1, 2], ["hero", "villain"], ["sword", "shield"])
        result = compare_annotations(
            [df, df, df], names=["a", "b", "c"], metric="accuracy", actant_columns=COLS
        )
        assert result.loc["Subject", "a_b"] == pytest.approx(1.0)
        assert result.loc["Subject", "a_c"] == pytest.approx(1.0)
        assert result.loc["Subject", "b_c"] == pytest.approx(1.0)


class TestCompareLabelsMissingValues:
    def test_none_treated_as_nan(self):
        df1 = make_df([1, 2], ["hero", None], ["sword", "sword"])
        df2 = make_df([1, 2], ["hero", "villain"], ["sword", "shield"])
        result = compare_annotations(
            [df1, df2], names=["a", "b"], metric="accuracy", actant_columns=["Subject"]
        )
        # Row 2 has None for df1, so only row 1 is valid → 1/1 = 1.0
        assert result.loc["Subject", "a_b"] == pytest.approx(1.0)

    def test_missing_rows_dropped_per_pair(self):
        # df_c is missing row 2; a_b pair should still use all 3 rows
        df_a = make_df(
            [1, 2, 3], ["hero", "villain", "hero"], ["sword", "shield", "sword"]
        )
        df_b = make_df(
            [1, 2, 3], ["hero", "villain", "hero"], ["sword", "shield", "sword"]
        )
        df_c = make_df([1, 3], ["hero", "hero"], ["sword", "sword"])
        result = compare_annotations(
            [df_a, df_b, df_c],
            names=["a", "b", "c"],
            metric="accuracy",
            actant_columns=["Subject"],
        )
        # a_b: perfect across all 3 rows
        assert result.loc["Subject", "a_b"] == pytest.approx(1.0)
        # a_c and b_c: only rows 1 and 3 (row 2 is NaN for c), still perfect on those rows
        assert result.loc["Subject", "a_c"] == pytest.approx(1.0)
        assert result.loc["Subject", "b_c"] == pytest.approx(1.0)

    def test_missing_actant_column_gives_nan(self):
        df1 = pd.DataFrame({"id": [1], "Subject": ["hero"]})
        df2 = pd.DataFrame({"id": [1], "Subject": ["hero"]})
        result = compare_annotations(
            [df1, df2], names=["a", "b"], metric="accuracy", actant_columns=COLS
        )
        assert np.isnan(result.loc["Object", "a_b"])


class TestCompareLabelsN:
    def test_n_two_coders_no_missing(self):
        df = make_df([1, 2, 3], ["hero", "villain", "hero"], ["sword", "shield", None])
        result = compare_annotations([df, df], metric="accuracy", actant_columns=COLS)
        # 1 pair × 3 rows = 3
        assert result.loc["Subject", "N"] == 3
        assert result.loc["Object", "N"] == 2

    def test_n_three_coders(self):
        df_a = make_df(
            [1, 2, 3], ["hero", "villain", "hero"], ["sword", "shield", "sword"]
        )
        df_b = make_df(
            [1, 2, 3], ["hero", "villain", "hero"], ["sword", "shield", "sword"]
        )
        df_c = make_df([1, 3], ["hero", "hero"], ["sword", "sword"])
        result = compare_annotations(
            [df_a, df_b, df_c],
            names=["a", "b", "c"],
            metric="accuracy",
            actant_columns=["Subject"],
        )
        # a_b: 3 rows, a_c: 2 rows, b_c: 2 rows → N = 7
        assert result.loc["Subject", "N"] == 7

    def test_n_avg_row_is_nan(self):
        df = make_df([1, 2], ["hero", "villain"], ["sword", "shield"])
        result = compare_annotations([df, df], metric="accuracy", actant_columns=COLS)
        assert pd.isna(result.loc["avg", "N"])

    def test_n_missing_actant_is_zero(self):
        df1 = pd.DataFrame({"id": [1], "Subject": ["hero"]})
        df2 = pd.DataFrame({"id": [1], "Subject": ["hero"]})
        result = compare_annotations(
            [df1, df2], names=["a", "b"], metric="accuracy", actant_columns=COLS
        )
        assert result.loc["Object", "N"] == 0
