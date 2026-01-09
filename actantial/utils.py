import os
import re
import json
import logging

from typing import Dict


def ensure_directory(dir_path: str):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def parse_json(input_text: str) -> Dict:
    """
    Extract the first flat JSON object from a string.

    Args:
        input_text (str): The input string containing a JSON object.

    Returns:
        dict: The extracted JSON object, or empty dict if parsing fails.
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
