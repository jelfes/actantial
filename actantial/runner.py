import os
import json
from typing import Optional
import pandas as pd
import logging


from actantial.extract import extract_actants
from actantial.backends.base import LLMBackend
from tqdm import tqdm
from jinja2 import Environment, FileSystemLoader, Template, meta as jinja2_meta
from pathlib import Path
from datetime import datetime
from actantial.io import _parse_json, _ensure_directory, _configure_logging

os.environ["TOKENIZERS_PARALLELISM"] = (
    "false"  # to suppress warnings from transformers (TODO: check if still needed)
)


def run_extract(
    data: pd.DataFrame,
    backend: LLMBackend,
    output_dir: Path,
    template: str,
    templates_dir: Path = Path(__file__).parent / "templates",
    actor_labels: Optional[list] = None,
    object_labels: Optional[list] = None,
    resume_timestamp: Optional[str] = None,
    template_columns: Optional[list[str]] = None,
):
    """
    Run the actantial extraction pipeline over a DataFrame of texts.

    Iterates over each row, renders the prompt template, calls the backend,
    parses the JSON output, and writes per-text result files to disk. Supports
    resuming an interrupted run by skipping texts that already have a saved
    result. Logs all activity to a timestamped log file under
    ``output_dir/logs/``.

    Results are saved under
    ``output_dir/actantial_models/{model_name}/{template}/{timestamp}/``,
    with one JSON file per text ID and one file containing the full raw
    backend response.

    Args:
        data: DataFrame with at least ``id`` and ``text`` columns.
        backend: An initialised [`LLMBackend`][actantial.backends.base.LLMBackend]
            instance.
        output_dir: Root directory for saving results and logs.
        template: Name of the prompt template to use. Must exist in
            ``templates_dir/{backend.model_name}/``.
        templates_dir: Root directory containing per-model template
            subdirectories. Defaults to the built-in ``templates/`` folder.
        actor_labels: List of actor labels for closed-set annotation. Only
            used if the template supports ``actor_labels``.
        object_labels: List of object labels for closed-set annotation. Only
            used if the template supports ``object_labels``.
        resume_timestamp: Timestamp of a previous run to resume, in
            ``YYYYMMDD_HHMMSS`` format. Texts already processed in that run
            are skipped. The model and template must match the original run.
        template_columns: Column names from ``data`` to pass as additional
            template variables. Each name maps directly to a Jinja2 variable
            of the same name (e.g. ``"parent_post"`` → ``{{ parent_post }}``).
            Columns must be string dtype; cast with ``data[col].astype(str)``
            before calling if needed.
    """
    print(templates_dir)
    template_name = template if template.endswith(".txt") else template + ".txt"

    if resume_timestamp is not None:
        RUN_DIR = Path(
            output_dir,
            "actantial_models",
            backend.model_name,
            template_name.removesuffix(".txt"),
            resume_timestamp,
        )
        if not RUN_DIR.exists():
            raise FileNotFoundError(
                f"No run found at {RUN_DIR}. "
                "Check that the timestamp, model, and template match the original run."
            )
        config_path = RUN_DIR / "run_config.json"
        if not config_path.exists():
            msg = "run_config.json not found in run directory — this may be an older run. Skipping quantisation check."
            print(f"Warning: {msg}")
            logging.warning(msg)
        else:
            with open(config_path) as f:
                saved_config = json.load(f)
            saved_q = saved_config.get("quantisation", False)
            current_q = getattr(backend, "quantisation", False)
            if saved_q != current_q:
                raise ValueError(
                    f"Quantisation mismatch: original run used quantisation={saved_q}, "
                    f"but current backend has quantisation={current_q}."
                )
        RUN_ID = f"{backend.model_name}_{template_name.removesuffix('.txt')}_{resume_timestamp}"
        resuming = True
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        RUN_ID = (
            f"{backend.model_name}_{template_name.removesuffix('.txt')}_{timestamp}"
        )
        RUN_DIR = Path(
            output_dir,
            "actantial_models",
            backend.model_name,
            template_name.removesuffix(".txt"),
            timestamp,
        )
        resuming = False

    LOG_DIR = Path(output_dir, "logs")
    log_file = LOG_DIR / f"{RUN_ID}.log"
    _configure_logging(LOG_DIR, RUN_ID, append=resuming)

    if resuming:
        logging.info(
            f"--- Resuming run at {datetime.now().strftime('%Y%m%d_%H%M%S')} ---"
        )
    else:
        logging.info(f"Run directory: {RUN_DIR}")

    if not resuming:
        print(f"Timestamp: \t{timestamp}")
    print(f"Log: \t\t{log_file}")
    print(f"Files: \t\t{RUN_DIR}")

    # get template
    environment = Environment(loader=FileSystemLoader(templates_dir))

    try:
        template = environment.get_template(
            str(Path(backend.model_name, template_name))
        )
    except Exception as e:
        error_message = f"Error loading template {e}. Please ensure that the template exists in the templates/{backend.model_name} directory and is named correctly."
        logging.error(error_message)
        raise FileNotFoundError(error_message)

    # validate template variables
    template_source = environment.loader.get_source(
        environment, str(Path(backend.model_name, template_name))
    )[0]
    template_vars = jinja2_meta.find_undeclared_variables(
        environment.parse(template_source)
    )

    if "text" not in template_vars:
        raise ValueError(
            f"Template '{template_name}' is missing the required variable '{{{{ text }}}}'. "
            "Please add '{{ text }}' to your template to pass the input text."
        )

    if actor_labels is not None and "actor_labels" not in template_vars:
        raise ValueError(
            f"Template '{template_name}' is missing the variable '{{{{ actor_labels }}}}', "
            "but actor_labels were provided. Either add '{{ actor_labels }}' to your template "
            "or remove the actor_labels argument."
        )

    if object_labels is not None and "object_labels" not in template_vars:
        raise ValueError(
            f"Template '{template_name}' is missing the variable '{{{{ object_labels }}}}', "
            "but object_labels were provided. Either add '{{ object_labels }}' to your template "
            "or remove the object_labels argument."
        )

    _reserved = {"text", "actor_labels", "object_labels"}
    template_columns = template_columns or []

    for col in template_columns:
        if col in _reserved:
            raise ValueError(
                f"'{col}' is a reserved template variable and cannot be used in template_columns."
            )
        if col not in data.columns:
            raise ValueError(
                f"Column '{col}' not found in data. Available columns: {list(data.columns)}."
            )
        if not pd.api.types.is_string_dtype(data[col]):
            raise ValueError(
                f"Column '{col}' must be string dtype (found {data[col].dtype}). "
                f"Cast it first with: data['{col}'] = data['{col}'].astype(str)"
            )
        if col not in template_vars:
            raise ValueError(
                f"Column '{col}' was passed via template_columns but is not used in template '{template_name}'. "
                "Remove it from template_columns or add the variable to your template."
            )

    unknown_vars = template_vars - _reserved - set(template_columns)
    if unknown_vars:
        raise ValueError(
            f"Template '{template_name}' references variables {unknown_vars} that are not provided. "
            "Add the missing columns to template_columns or remove the variables from your template."
        )

    _ensure_directory(RUN_DIR)

    if not resuming:
        with open(RUN_DIR / "run_config.json", "w") as f:
            json.dump(
                {
                    "model": backend.model_name,
                    "templates_dir": str(templates_dir),
                    "template": template_name.removesuffix(".txt"),
                    "timestamp": timestamp,
                    "quantisation": getattr(backend, "quantisation", False),
                    "template_columns": template_columns,
                },
                f,
                indent=2,
            )

    if not resuming:
        logging.info(
            f"Template: \t{Path(templates_dir, backend.model_name, template_name)}"
        )
        _preview_extra = {col: f"<{col.upper()}>" for col in template_columns}
        logging.info(
            f"Prompt preview:\n{template.render(text='<TEXT>', actor_labels=actor_labels, object_labels=object_labels, **_preview_extra)}"
        )

    # start loop
    logging.info("Start inference")
    for _, row in tqdm(enumerate(data.itertuples()), total=len(data)):

        if Path(RUN_DIR, f"{row.id}.txt").exists():
            logging.info(f"Skipping ID {row.id} (already processed)")
            continue

        logging.info(f"---------- ID {row.id} ----------\n")

        extra_vars = {col: getattr(row, col) for col in template_columns}
        var_summary = "\n".join(
            [f"  {col}: {val}" for col, val in extra_vars.items()]
            + [f"  text: {row.text}"]
        )
        logging.info(f"Variables:\n{var_summary}")

        output = extract_actants(
            input_text=row.text,
            backend=backend,
            prompt_template=template,
            actor_labels=actor_labels,
            object_labels=object_labels,
            **extra_vars,
        )

        # Parse result
        actant_dict = _parse_json(output)

        logging.info(f"Output:\n\t{actant_dict}")

        # save results
        file_path = Path(RUN_DIR, f"{row.id}.txt")
        file_path_full_response = Path(RUN_DIR, f"full_response_{row.id}.txt")

        logging.info(f"Writing output to: {file_path}")

        with open(file_path, "w") as f:
            json.dump(actant_dict, f)

        with open(file_path_full_response, "w") as f:
            json.dump(output, f)

        logging.info("##############################################################\n")
