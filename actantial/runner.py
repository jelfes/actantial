import os
import json
import pandas as pd
import logging

import yaml


from actantial.extract import extract_actants
from actantial.backends.base import LLMBackend
from tqdm import tqdm
from jinja2 import Environment, FileSystemLoader, Template
from pathlib import Path
from datetime import datetime
from actantial.io import parse_json, ensure_directory, configure_logging


os.environ["TOKENIZERS_PARALLELISM"] = (
    "false"  # to suppress warnings from transformers (TODO: check if still needed)
)


def run_extract(
    data: pd.DataFrame,
    backend: LLMBackend,
    output_dir: Path,
    template: str,
    actor_labels_path: str = None,
    object_labels_path: str = None
):
    """
    Hanldes logging, data loop, saving results, and calls extraction function.

    Args:
        data: DataFrame with at least 'id' and 'text' columns
        backend: Initialized LLMBackend instance
        output_dir: Base path for saving results and logs
        template: Name of the prompt template to use. Must be located in templates/{backend.model_name}/.
        actor_labels_path: Optional path to actor labels for annotation with predefined labels. Requires a template that supports labels.
        object_labels_path: Optional path to object labels for annotation with predefined labels. Requires a template that supports labels.
    """

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    RUN_ID = f"{backend.model_name}_{template}_{timestamp}"
    RUN_DIR = Path(
        output_dir, "actantial_models", backend.model_name, template, timestamp
    )
    print(RUN_DIR)
    LOG_DIR = Path(output_dir, "logs")

    print(f"Logging to: {RUN_ID}.log")
    configure_logging(LOG_DIR, RUN_ID)

    # get template
    environment = Environment(
        loader=FileSystemLoader(Path(__file__).parent / "templates")
    )

    if template.split(".")[-1] != "txt":
        template += ".txt"

    try:
        template = environment.get_template(str(Path(backend.model_name, template)))
    except Exception as e:
        error_message = f"Error loading template {e}. Please ensure that the template exists in the templates/{backend.model_name} directory and is named correctly."
        logging.error(error_message)
        raise FileNotFoundError(error_message)

    # handle labels (if provided)
    if actor_labels_path is not None:
        with open(actor_labels_path) as f:
            actor_labels = yaml.safe_load(f)        
    else:
        actor_labels = None

    if object_labels_path is not None:
        with open(object_labels_path) as f:
            object_labels = yaml.safe_load(f)
    else:
        object_labels = None

    # start logging
    logging.info(f"Prompt: {template.render(text='COMMENT HERE', actor_labels=actor_labels, object_labels=object_labels)}")
    ensure_directory(RUN_DIR)


    # start loop
    logging.info("Start inference")
    for _, row in tqdm(enumerate(data.itertuples()), total=len(data)):

        logging.info(f"---------- ID {row.id} ----------\n")
        logging.info(f"Comment: {row.text}")



        output = extract_actants(
            input_text=row.text,
            backend=backend,
            prompt_template=template,
            actor_labels=actor_labels,
            object_labels=object_labels
        )

        # Parse result
        actant_dict = parse_json(output)

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
