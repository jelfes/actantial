import itertools
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from krippendorff import alpha as krippendorff_alpha
from sklearn.metrics import accuracy_score, f1_score

from actantial.config import ACTANTS

SUPPORTED_METRICS = {
    "accuracy",
    "f1_micro",
    "f1_macro",
    "f1_weighted",
    "krippendorff_alpha",
}


def compare_annotations(
    dfs: List[pd.DataFrame],
    names: Optional[List[str]] = None,
    id_column: str = "id",
    metric: str = "krippendorff_alpha",
    actant_columns: list = ACTANTS,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Compare multiple sets of actantial annotations and compute an agreement metric.

    Aligns all DataFrames on ``id_column`` (outer join), then computes the
    chosen metric for each actant. Pairwise metrics (accuracy, F1) iterate over
    all annotator pairs and average the results. Krippendorff's alpha uses all
    coders simultaneously and handles missing values natively.

    Args:
        dfs: List of DataFrames with actantial annotations (one per annotator).
        names: Annotator names, one per DataFrame. Defaults to ``coder_0``,
            ``coder_1``, etc.
        id_column: Column name used to align rows across DataFrames.
        metric: Agreement metric. One of ``"accuracy"``, ``"f1_micro"``,
            ``"f1_macro"``, ``"f1_weighted"``, ``"krippendorff_alpha"``.
        actant_columns: Actant columns to compare. Defaults to the full set
            defined in the package config.
        verbose: If True, print warnings about dropped rows.

    Returns:
        DataFrame with actants as rows (plus an ``"avg"`` row). Columns are
            annotator pairs plus ``"avg"`` for pairwise metrics, or a single
            ``"alpha"`` column for Krippendorff.
    """
    if len(dfs) < 2:
        raise ValueError("At least two DataFrames are required.")

    if names is None:
        names = [f"coder_{i}" for i in range(len(dfs))]

    if len(names) != len(dfs):
        raise ValueError(f"len(names)={len(names)} must equal len(dfs)={len(dfs)}.")

    if metric not in SUPPORTED_METRICS:
        raise ValueError(
            f"Unsupported metric '{metric}'. Choose from {sorted(SUPPORTED_METRICS)}."
        )

    for i, df in enumerate(dfs):
        if id_column not in df.columns:
            raise ValueError(f"id_column '{id_column}' not found in dfs[{i}].")

    # Build annotations dictionary: actant -> DataFrame(index=id, columns=names)
    # Outer join preserves all IDs; missing annotations become NaN.
    annotations: Dict[str, pd.DataFrame] = {}
    for actant in actant_columns:
        series = []
        for name, df in zip(names, dfs):
            if actant not in df.columns:
                continue
            s = df.set_index(id_column)[actant].rename(name)
            series.append(s)
        if not series:
            continue
        annotations[actant] = pd.concat(series, axis=1, join="outer")

    if metric == "krippendorff_alpha":
        result = _compute_krippendorff(annotations, actant_columns, verbose)
    else:
        pairs = list(itertools.combinations(names, 2))
        pair_labels = [f"{a}_{b}" for a, b in pairs]
        result = _compute_pairwise(
            annotations, actant_columns, pairs, pair_labels, metric, verbose
        )
        result["avg"] = result.mean(axis=1, skipna=True)

    avg_row = result.mean(axis=0, skipna=True)
    avg_row.name = "avg"
    result = pd.concat([result, avg_row.to_frame().T])
    result["N"] = _count_pairs(annotations, actant_columns, names)

    return result


def _count_pairs(
    annotations: Dict[str, pd.DataFrame],
    actant_columns: list,
    names: list,
) -> pd.Series:
    pairs = list(itertools.combinations(names, 2))
    counts = {}
    for actant in actant_columns:
        if actant not in annotations:
            counts[actant] = 0
            continue
        df_actant = annotations[actant]
        counts[actant] = sum(
            len(df_actant[[a1, a2]].dropna())
            for a1, a2 in pairs
            if a1 in df_actant.columns and a2 in df_actant.columns
        )
    return pd.Series(counts, dtype="Int64")


def _compute_pairwise(
    annotations: Dict[str, pd.DataFrame],
    actant_columns: list,
    pairs: list,
    pair_labels: list,
    metric: str,
    verbose: bool,
) -> pd.DataFrame:
    rows: Dict[str, Dict[str, float]] = {}

    for actant in actant_columns:
        if actant not in annotations:
            rows[actant] = {pl: np.nan for pl in pair_labels}
            continue

        df_actant = annotations[actant]
        rows[actant] = {}

        for (a1, a2), pl in zip(pairs, pair_labels):
            if a1 not in df_actant.columns or a2 not in df_actant.columns:
                rows[actant][pl] = np.nan
                continue

            subset = df_actant[[a1, a2]].dropna()

            if verbose and len(subset) < len(df_actant):
                dropped = len(df_actant) - len(subset)
                print(
                    f"Dropped {dropped}/{len(df_actant)} rows because of missing annotations for actant '{actant}', pair '{pl}'"
                )

            if len(subset) == 0:
                rows[actant][pl] = np.nan
                continue

            y1 = subset[a1].astype(str)
            y2 = subset[a2].astype(str)

            if metric == "accuracy":
                rows[actant][pl] = float(accuracy_score(y1, y2))
            elif metric == "f1_micro":
                rows[actant][pl] = float(f1_score(y1, y2, average="micro"))
            elif metric == "f1_macro":
                rows[actant][pl] = float(f1_score(y1, y2, average="macro"))
            elif metric == "f1_weighted":
                rows[actant][pl] = float(f1_score(y1, y2, average="weighted"))

    return pd.DataFrame.from_dict(rows, orient="index")


def _compute_krippendorff(
    annotations: Dict[str, pd.DataFrame],
    actant_columns: list,
    verbose: bool,
) -> pd.DataFrame:
    rows: Dict[str, Dict[str, float]] = {}

    for actant in actant_columns:
        if actant not in annotations:
            rows[actant] = {"alpha": np.nan}
            continue

        df_actant = annotations[actant]

        all_labels = sorted(df_actant.stack().astype(str).unique())
        label_dict = {label: idx for idx, label in enumerate(all_labels)}
        data = (
            df_actant.astype(str)
            .apply(lambda col: col.map(label_dict))
            .T.values.tolist()
        )

        try:
            score = float(
                krippendorff_alpha(
                    reliability_data=data, level_of_measurement="nominal"
                )
            )
        except Exception as e:
            if verbose:
                print(f"Error computing Krippendorff's alpha for '{actant}': {e}")
            score = np.nan

        rows[actant] = {"alpha": score}

    return pd.DataFrame.from_dict(rows, orient="index")
