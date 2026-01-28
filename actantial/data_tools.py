import os
import json

from actantial.utils import parse_json
from actantial.config import ACTANTS
from pandas import DataFrame
from pathlib import Path


def create_file_path(base_folder: str, file_id: str) -> str:
    """
    Create a file path combining the base folder and file ID. Return None if the file does not exist.

    Args:
        base_folder (str): The base directory path.
        file_id (str): The unique identifier for the file.

    Returns:
        str: The constructed file path.
    """

    file_path = Path(base_folder) / f"{file_id}.txt"

    if os.path.exists(file_path):
        return str(file_path)
    else:
        return None


def read_json_file(file_path):
    with open(file_path, "r") as file:
        data = json.load(file)
    return data


def extract_first_actant(data, column_name="file_name"):
    """
    Extract actants from JSON files. If multiple actors are present for an actant, only the first one is retained.

    Args:
        data (DataFrame): The input DataFrame containing file paths.
        column_name (str): The name of the column with file paths.

    Returns:
        DataFrame: The DataFrame with extracted actants.

    """

    data_out = data.copy()
    data_out[ACTANTS] = None

    for index, row in data.iterrows():
        file_path = row[column_name]

        # skip non-existing files
        if not file_path:
            continue

        data = read_json_file(file_path)

        # TODO improve robustnes of extraction logic
        for actant in ACTANTS:
            value = data.get(actant)

            # skip empty values
            if not value:
                continue

            # skip empty lists
            if value == []:
                continue

            # convert string to list
            if isinstance(value, str):
                value = [value]

            first_actant = value[0]
            data_out.at[index, actant] = first_actant

    return data_out


def load_annotations(data: DataFrame, label_folder: str) -> DataFrame:
    """
    Load annotations from the specified label folder and integrate them into the DataFrame.
    The dataframe is expected to have an 'id' column that matches the annotation files.

    Args:
        data (DataFrame): The input DataFrame to annotate.
        label_folder (str): The path to the folder containing annotation files.

    Returns:
        DataFrame: The DataFrame with integrated annotations.
    """

    # Check if label folder exists
    if not os.path.exists(label_folder):
        KeyError(f"Label folder not found: {label_folder}")

    # Create a copy of the data to avoid modifying the original DataFrame
    data_annot = data.copy()

    # Map ID to annotation file paths
    data_annot["file_name"] = data_annot.apply(
        lambda x: create_file_path(label_folder, x.id), axis=1
    )

    n_missing_files = data_annot.file_name.isna().sum()

    if n_missing_files > 0:
        print(
            f"Warning: {n_missing_files}/{len(data_annot)} annotation files are missing."
        )

    # Load annotations from files
    data_annot = extract_first_actant(data_annot, column_name="file_name")

    return data_annot
