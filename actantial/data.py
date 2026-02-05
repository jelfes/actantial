import os
import json
import warnings

from typing import Literal, Optional, Dict, Any, Union
from actantial.config import ACTANTS
from pandas import DataFrame
from pathlib import Path


def _read_json_file(file_path: Union[str, Path]) -> Dict[str, Any]:
    """Read and parse a JSON file, returning a dict.

    Raises FileNotFoundError or ValueError on invalid JSON.
    """
    path = Path(file_path)
    try:
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)
    except FileNotFoundError:
        raise
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}: {exc}") from exc


def _create_file_path(base_folder: str, file_id: str) -> Optional[str]:
    """
    Create a file path combining the base folder and file ID. Return None if the file does not exist.

    Args:
        base_folder (str): The base directory path.
        file_id (str): The unique identifier for the file.

    Returns:
        str: The constructed file path.
    """

    file_path = Path(base_folder) / f"{file_id}.txt"

    if file_path.exists():
        return str(file_path)
    return None


def load_actors(
    data: DataFrame,
    column_name: str = "file_name",
    select_actor: Literal["first", "combine"] = "first",
) -> DataFrame:
    """
    Extract actors from JSON files. If multiple actors are present for an actant, only the first one is retained.

    Args:
        data (DataFrame): The input DataFrame containing file paths.
        column_name (str): The name of the column with file paths.
        select_actor (Literal["first", "combine"], optional): Strategy for handling multiple actors per actant. Options are:
            - "first": Use only the first actor (default)
            - "combine": Combine all actors for each actant

    Returns:
        DataFrame: The DataFrame with extracted actants.

    """

    data_out = data.copy()

    # initialize columns explicitly for clarity
    for a in ACTANTS:
        data_out[a] = None

    for index, row in data.iterrows():
        file_path = row.get(column_name)

        # skip non-existing files
        if not file_path:
            continue

        file_data = _read_json_file(file_path)

        # TODO improve robustness of extraction logic
        for actant in ACTANTS:
            value = file_data.get(actant)

            # skip empty values
            if value in (None, ""):
                continue

            # normalize to list
            if isinstance(value, str):
                value = [value]
            elif not isinstance(value, list):
                value = [value]

            # skip empty lists
            if len(value) == 0:
                continue

            if select_actor == "first":
                actor = value[0]
            elif select_actor == "combine":
                actor = ", ".join(map(str, value))
            else:
                raise ValueError(
                    f"select_actor must be 'first' or 'combine', got '{select_actor}'"
                )

            data_out.at[index, actant] = actor

    return data_out


def load_annotations(data: DataFrame, label_folder: str, **kwargs) -> DataFrame:
    """
    Load annotations from the specified label folder and integrate them into the DataFrame.
    The dataframe is expected to have an 'id' column that matches the annotation files.

    Args:
        data (DataFrame): The input DataFrame to annotate.
        label_folder (str): The path to the folder containing annotation files.
        kwargs: Additional arguments for annotation extraction.

    Returns:
        DataFrame: The DataFrame with integrated annotations.
    """

    # Check if label folder exists and is a directory
    label_path = Path(label_folder)
    if not label_path.exists() or not label_path.is_dir():
        raise KeyError(f"Label folder not found or not a directory: {label_folder}")

    # Create a copy of the data to avoid modifying the original DataFrame
    data_annot = data.copy()

    # validate input DataFrame
    if "id" not in data_annot.columns:
        raise KeyError("Input DataFrame must contain an 'id' column")

    # Map IDs to annotation file paths
    data_annot["file_name"] = data_annot.apply(
        lambda x: _create_file_path(label_folder, x.id), axis=1
    )

    n_missing_files = data_annot.file_name.isna().sum()

    if n_missing_files > 0:
        warnings.warn(
            f"Warning: {n_missing_files}/{len(data_annot)} annotation files are missing."
        )

    # Load annotations from files
    data_annot = load_actors(data_annot, column_name="file_name", **kwargs)

    return data_annot
