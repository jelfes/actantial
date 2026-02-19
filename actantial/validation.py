import pandas as pd

from krippendorff import alpha
from sklearn.metrics import accuracy_score, f1_score
from actantial.config import ACTANTS
import math


def compare_labels(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    metric: str = "accuracy",
    id_column: str = "id",
    actant_columns: list = ACTANTS,
    verbose: bool = True,
    # ignore_case: bool = True,
    # partial_match: bool = False
):
    """
    Compare two DataFrames containing actantial annotations and compute agreement metrics.

    Args:
        df1: First DataFrame with actantial annotations.
        df2: Second DataFrame with actantial annotations.
        metric: The agreement metric to compute (e.g., "accuracy", "f1_micro").
        id_column: The name of the column containing unique identifiers for matching rows.
        actant_columns: List of columns corresponding to actants to compare.
    """
    results = {}

    len1 = len(df1)
    len2 = len(df2)
    if len1 != len2:
        if verbose:
            print(f"DataFrame length mismatch: df1={len1}, df2={len2}")

    # ensure id_column exists in both inputs
    if id_column not in df1.columns or id_column not in df2.columns:
        raise ValueError(f"id_column '{id_column}' not found in both dataframes.")

    # Merge on id column to align rows
    merged = pd.merge(df1, df2, on=id_column, how="inner", suffixes=("_1", "_2"))

    if merged.shape[0] == 0:
        return {"error": "no matching ids between dataframes"}

    scores = {}
    dropped_per_actant = {}

    for actant in actant_columns:
        col1 = f"{actant}_1"
        col2 = f"{actant}_2"
        if col1 not in merged.columns or col2 not in merged.columns:
            scores[actant] = float("nan")
            dropped_per_actant[actant] = 0
            continue

        before = merged.loc[:, [col1, col2]]
        before_count = before.shape[0]
        after = before.dropna()
        after_count = after.shape[0]
        dropped = before_count - after_count
        dropped_per_actant[actant] = int(dropped)

        if dropped > 0 and verbose:
            print(
                f"Dropped {dropped} rows for actant '{actant}' (before={before_count}, after={after_count})"
            )

        if after_count == 0:
            scores[actant] = float("nan")
            continue

        y_true = after[col1].astype(str)
        y_pred = after[col2].astype(str)

        if metric == "accuracy":
            scores[actant] = float(accuracy_score(y_true, y_pred))
        elif metric == "f1_micro":
            scores[actant] = float(f1_score(y_true, y_pred, average="micro"))
        elif metric == "f1_macro":
            scores[actant] = float(f1_score(y_true, y_pred, average="macro"))
        elif metric == "f1_weighted":
            scores[actant] = float(f1_score(y_true, y_pred, average="weighted"))
        elif metric == "krippendorff_alpha":

            label_set = set(y_true.unique()).union(set(y_pred.unique()))
            label_dict = {label: idx for idx, label in enumerate(label_set)}
            y_true_mapped = y_true.map(label_dict)
            y_pred_mapped = y_pred.map(label_dict)

            data = [y_true_mapped.tolist(), y_pred_mapped.tolist()]

            try:
                scores[actant] = float(
                    alpha(reliability_data=data, level_of_measurement="nominal")
                )
            except Exception as e:
                if verbose:
                    print(
                        f"Error computing Krippendorff's alpha for actant '{actant}'. Setting score to NaN. Error: {e}"
                    )
                scores[actant] = float("nan")
        else:
            raise ValueError(
                f"Unsupported metric: {metric}, choose from 'accuracy', 'f1_micro', 'f1_macro', 'f1_weighted', 'krippendorff_alpha'."
            )
    # aggregate
    vals = [v for v in scores.values() if not (isinstance(v, float) and math.isnan(v))]
    try:
        avg_score = float(sum(vals) / len(vals)) if len(vals) > 0 else float("nan")
    except Exception:
        avg_score = float("nan")

    results["per_actant"] = scores
    results["avg"] = avg_score

    return results
