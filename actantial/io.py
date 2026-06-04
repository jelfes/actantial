import re
import json
import warnings
import logging

import yaml

from typing import Literal, Optional, Dict, Any, Union
from actantial.config import ACTANTS
from pandas import DataFrame
from pathlib import Path


def _ensure_directory(dir_path: Path | str) -> None:
    """Create the directory at ``dir_path`` if it does not already exist."""
    path = Path(dir_path)
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)


def _configure_logging(
    log_dir: Path | str, log_name: str, append: bool = False
) -> None:
    """
    Configure file-based logging for a pipeline run.

    Attaches a file handler to the root logger, writing to
    ``log_dir/{log_name}.log``. Any existing handlers are removed first.

    Args:
        log_dir: Directory where the log file will be created.
        log_name: Name of the log file, without extension.
        append: If ``True``, append to an existing log file; otherwise overwrite.
    """
    _ensure_directory(log_dir)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Clear existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Add file handler
    log_file = Path(log_dir) / f"{log_name}.log"
    file_mode = "a" if append else "w"
    handler = logging.FileHandler(log_file, mode=file_mode, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    logger.addHandler(handler)


def _read_json_file(file_path: Union[str, Path]) -> Dict[str, Any]:
    """Read and parse a JSON file, returning its contents as a dict."""
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
    Return the path to ``{base_folder}/{file_id}.txt``, or ``None`` if the file does not exist.

    Args:
        base_folder: Base directory path.
        file_id: Unique identifier used as the filename stem.

    Returns:
        The file path as a string, or ``None`` if the file does not exist.
    """
    file_path = Path(base_folder) / f"{file_id}.txt"

    if file_path.exists():
        return str(file_path)
    return None


def _parse_json(input_text: str) -> Dict:
    """
    Extract the first flat JSON object from a string.

    Used to parse raw backend output into a dict. Only matches flat
    (non-nested) JSON objects.

    Args:
        input_text: The raw string output from the backend.

    Returns:
        The parsed JSON object as a dict, or an empty dict if no valid
            JSON object is found or parsing fails.
    """
    # match only flat JSON objects
    json_pattern = re.compile(r"\{[^\{\}]*\}")
    matches = json_pattern.findall(input_text)

    if not matches:
        logging.warning("No JSON object found in the input.")
        return {}

    if len(matches) > 1:
        logging.warning(
            f"Multiple JSON objects found ({len(matches)}). Using the first one."
        )

    try:
        return json.loads(matches[0])
    except json.JSONDecodeError as e:
        logging.error(f"Error decoding JSON: {e}")
        return {}


def _load_allowed(path: str) -> set:
    with open(path) as f:
        labels = yaml.safe_load(f)
    if not labels:
        raise ValueError(f"Label file is empty or could not be parsed: {path}")
    return set(label.lower() for label in labels)


def load_actors(
    data: DataFrame,
    file_path_column: str = "file_name",
    actant_columns: Optional[list[str]] = ACTANTS,
    select_actor: Literal["first", "combine"] = "first",
    actor_labels_path: Optional[str] = None,
    object_labels_path: Optional[str] = None,
    verbose: bool = True,
    missing_actant_token: Optional[str] = "[UNK]",
) -> DataFrame:
    """
    Read per-text JSON annotation files and add actant columns to the DataFrame.

    Each file is expected to map actant role names to actor values. When
    multiple actors are listed for a role, ``select_actor`` controls whether
    to keep only the first or join them all.

    When label paths are provided, actor values not in the allowed set are
    replaced with ``None``.

    Args:
        data: DataFrame containing a column with file paths to JSON annotation files.
        file_path_column: Name of the column containing the file paths.
        actant_columns: List of actants to extract from the JSON files.
             Defaults to the global ACTANTS list.
        select_actor: Strategy for handling multiple actors per actant role.
            ``"first"`` keeps only the first actor; ``"combine"`` joins all
            actors into a comma-separated string.
        actor_labels_path: Path to a YAML file with allowed actor labels.
            If provided, actor values for non-Object actants not in the list
            are replaced with ``None``.
        object_labels_path: Path to a YAML file with allowed object labels.
            If provided, actor values for the Object actant not in the list
            are replaced with ``None``.
        verbose: If True, print a per-actant summary of dropped unknown actors.
        missing_actant_token: Token used in the data to denote a missing or
            unknown actant. Occurrences are replaced with ``None``. Set to
            ``None`` to disable. Defaults to ``"[UNK]"``.

    Returns:
        A copy of the input DataFrame with one column added per actant role.
    """
    actor_allowed = _load_allowed(actor_labels_path) if actor_labels_path else None
    object_allowed = _load_allowed(object_labels_path) if object_labels_path else None

    data_out = data.copy()

    # initialize columns explicitly for clarity
    for a in actant_columns:
        data_out[a] = None

    dropped = {a: 0 for a in actant_columns}
    total = {a: 0 for a in actant_columns}

    for index, row in data.iterrows():
        file_path = row.get(file_path_column)

        # skip non-existing files
        if not file_path:
            continue

        file_data = _read_json_file(file_path)

        for actant in actant_columns:
            value = file_data.get(actant)

            # skip empty values
            if value in (None, ""):
                continue

            # normalize to list
            if isinstance(value, str):
                value = [value]
            elif not isinstance(value, list):
                value = [value]

            # remove empty entries and skip if nothing remains
            value = [v for v in value if v not in (None, "")]
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

            actor = str(actor).lower()

            allowed = object_allowed if actant == "Object" else actor_allowed
            if allowed is not None:
                total[actant] += 1
                if actor not in allowed:
                    dropped[actant] += 1
                    continue

            data_out.at[index, actant] = actor

    if verbose and (actor_allowed is not None or object_allowed is not None):
        for actant in actant_columns:
            print(
                f"Dropped {dropped[actant]}/{total[actant]} unknown actors for actant '{actant}'"
            )

    if missing_actant_token is not None:
        data_out[actant_columns] = data_out[actant_columns].replace(
            missing_actant_token, None
        )

    return data_out


def load_annotations(
    data: DataFrame,
    label_folder: str,
    actor_labels_path: Optional[str] = None,
    object_labels_path: Optional[str] = None,
    verbose: bool = True,
    **kwargs,
) -> DataFrame:
    """
    Load actant annotations from a run output folder into a DataFrame.

    Matches each row to an annotation file in ``label_folder`` by its ``id``
    value, then extracts actant roles from each file. Rows without a matching
    file receive ``None`` for all actant columns.

    When label paths are provided, actor values not in the allowed set are
    replaced with ``None``. This is useful when using closed annotation,
    where the LLM may assign labels outside the predefined label set.

    Args:
        data: DataFrame with at least an ``id`` column.
        label_folder: Path to the folder containing per-text JSON annotation
            files, as produced by [`run_extract`][actantial.runner.run_extract].
        actor_labels_path: Path to a YAML file with allowed actor labels.
            If provided, values for non-Object actants not in the list are
            replaced with ``None``.
        object_labels_path: Path to a YAML file with allowed object labels.
            If provided, values for the Object actant not in the list are
            replaced with ``None``.
        verbose: If True, print warnings about missing annotation files and
            a per-actant summary of dropped unknown actors.

    Returns:
        A copy of the input DataFrame with actant columns added.
    """
    label_path = Path(label_folder)
    if not label_path.exists() or not label_path.is_dir():
        raise KeyError(f"Label folder not found or not a directory: {label_folder}")

    data_annot = data.copy()

    if "id" not in data_annot.columns:
        raise KeyError("Input DataFrame must contain an 'id' column")

    data_annot["file_name"] = data_annot.apply(
        lambda x: _create_file_path(label_folder, x.id), axis=1
    )

    n_missing_files = data_annot.file_name.isna().sum()

    if n_missing_files > 0:
        warnings.warn(
            f"Warning: {n_missing_files}/{len(data_annot)} annotation files are missing."
        )

    data_annot = load_actors(
        data_annot,
        file_path_column="file_name",
        actor_labels_path=actor_labels_path,
        object_labels_path=object_labels_path,
        verbose=verbose,
        **kwargs,
    )

    return data_annot
