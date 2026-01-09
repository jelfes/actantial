import os
import json
import pandas as pd
import logging


from actantial.utils import parse_json
from actantial.extract import extract_actants
from actantial.backends.base import LLMBackend
from tqdm import tqdm
from jinja2 import Environment, FileSystemLoader, Template
from pathlib import Path
from datetime import datetime
from actantial.utils import ensure_directory


os.environ["TOKENIZERS_PARALLELISM"] = (
    "false"  # to suppress warnings from transformers (TODO: check if still needed)
)


def run_extract(
    data: pd.DataFrame, backend: LLMBackend, data_path: Path, template: str
):
    """
    Hanldes logging, data loop, saving results, and calls extraction function.

    Args:
        data: DataFrame with at least 'id' and 'text' columns
        backend: Initialized LLMBackend instance
        data_path: Base path for saving results and logs
        template: Name of the prompt template to use (without .txt extension)
    """

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    RUN_ID = f"{backend.model_name}_{template}_{timestamp}"
    RUN_DIR = Path(
        data_path, "actantial_models", backend.model_name, template, timestamp
    )
    print(RUN_DIR)
    LOG_DIR = Path(data_path, "logs")

    print(f"Logging to: {RUN_ID}.log")
    ensure_directory(LOG_DIR)
    logging.basicConfig(
        filename=f"{LOG_DIR}/{RUN_ID}.log",
        encoding="utf-8",
        format="%(asctime)s %(message)s",
        level=logging.INFO,
    )

    # get template
    environment = Environment(
        loader=FileSystemLoader(Path(__file__).parent / "templates")
    )
    template = environment.get_template(template + ".txt")

    # start logging
    logging.info(f"Prompt: {template.render(text='COMMENT HERE')}")
    ensure_directory(RUN_DIR)

    # start loop
    logging.info("Start inference")
    for _, row in tqdm(enumerate(data.itertuples()), total=len(data)):

        logging.info(f"---------- ID {row.id} ----------\n")
        logging.info(f"Comment: {row.text}")

        output = extract_actants(row.text, backend, template)

        # Parse result
        actant_dict = parse_json(output)

        logging.info(f"Output:\n{actant_dict}")

        # save results
        file_path = Path(RUN_DIR, f"{row.id}.txt")
        file_path_full_response = Path(RUN_DIR, f"full_response_{row.id}.txt")

        logging.info(f"Writing output to: {file_path}")

        with open(file_path, "w") as f:
            json.dump(actant_dict, f)

        with open(file_path_full_response, "w") as f:
            json.dump(output, f)

        logging.info("##############################################################\n")
