"""
Command-line interface for the actantial package.

Exposes two entry points: ``actantial`` for running the extraction
pipeline, and ``actantial-init-templates`` for copying the bundled
prompt templates to a local directory for customisation.
"""

import argparse
import shutil
from pathlib import Path

import pandas as pd
from .runner import run_extract

BUNDLED_TEMPLATES_DIR = Path(__file__).parent / "templates"


def main():
    """
    Entry point for the ``actantial`` CLI command.

    Parses command-line arguments, initialises the appropriate backend,
    and delegates to [`run_extract`][actantial.runner.run_extract].

    Args:
        --data_file: CSV file with ``id`` and ``text`` columns.
        --output_dir: Directory where results and logs will be saved.
        --backend: LLM backend to use for inference. One of ``anthropic``,
            ``openai``, ``huggingface``.
        --model: Model name or identifier understood by the chosen backend.
        --repository: HuggingFace repository name. Required when using the
            ``huggingface`` backend.
        --quantise: Run the model in 4-bit quantised mode. Only valid for
            the ``huggingface`` backend on a CUDA GPU.
        --template: Prompt template name. Must exist in
            ``templates_dir/{model}/``.
        --actor_labels_path: Path to a YAML file with predefined actor labels
            for closed-set annotation.
        --object_labels_path: Path to a YAML file with predefined object labels
            for closed-set annotation.
        --resume_timestamp: Timestamp of a previous run to resume, in
            ``YYYYMMDD_HHMMSS`` format. The model and template must match
            the original run.
        --templates_dir: Directory containing prompt templates. Defaults to
            the bundled templates.
    """
    parser = argparse.ArgumentParser(
        description="Actantial: LLM-based narrative role extraction",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--data_file",
        type=Path,
        required=True,
        help="CSV file with ``id`` and ``text`` columns.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Directory where results and logs will be saved.",
    )
    parser.add_argument(
        "--backend",
        choices=["anthropic", "huggingface", "openai"],
        required=True,
        help="LLM backend to use for inference.",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model name or identifier understood by the chosen backend",
    )
    parser.add_argument(
        "--repository",
        type=str,
        help="Hugging Face repository name (required for huggingface backend)",
    )
    parser.add_argument(
        "--quantise",
        dest="quantise",
        action="store_true",
        help="Run model in 4‑bit quantised mode (only valid for huggingface + CUDA)",
    )
    parser.add_argument(
        "--template",
        type=str,
        required=True,
        help="Prompt template file name located under `templates/<model>/`",
    )
    parser.add_argument(
        "--actor_labels_path",
        type=str,
        help="Path to a YAML file with predefined actor labels for closed-set annotation.",
    )
    parser.add_argument(
        "--object_labels_path",
        type=str,
        help="Path to a YAML file with predefined object labels for closed-set annotation.",
    )
    parser.add_argument(
        "--resume_timestamp",
        type=str,
        default=None,
        metavar="TIMESTAMP",
        help="Timestamp of a previous run to resume (format: YYYYMMDD_HHMMSS). Model and template must match the original run.",
    )
    parser.add_argument(
        "--templates_dir",
        type=Path,
        default=None,
        help="Directory containing prompt templates. Defaults to the bundled templates.",
    )

    # Parse arguments
    args = parser.parse_args()

    # read the dataset
    data = pd.read_csv(args.data_file)

    # create backend instance
    if args.backend == "anthropic":
        from .backends.anthropic import AnthropicBackend

        backend = AnthropicBackend(model_name=args.model)
    elif args.backend == "huggingface":
        if not args.repository:
            raise ValueError(
                "`--repository` is required when using the huggingface backend"
            )
        from .backends.huggingface import HuggingFaceBackend

        backend = HuggingFaceBackend(
            repository=args.repository,
            model_name=args.model,
            quantisation=args.quantise,
        )
    elif args.backend == "openai":
        from .backends.openai import OpenAIBackend

        backend = OpenAIBackend(model_name=args.model)
    else:
        # argparse should guard against this, but safety first
        raise ValueError(f"Unknown backend: {args.backend}")

    # compile kwargs for runner
    kwargs = {
        "data": data,
        "backend": backend,
        "output_dir": args.output_dir,
        "template": args.template,
        "actor_labels_path": args.actor_labels_path,
        "object_labels_path": args.object_labels_path,
        "resume_timestamp": args.resume_timestamp,
    }

    # only include templates_dir if provided, so not to override the default in runner.py
    if args.templates_dir is not None:
        kwargs["templates_dir"] = args.templates_dir

    run_extract(**kwargs)


def init_templates():
    """
    Entry point for the ``actantial-init-templates`` CLI command.

    Copies the bundled prompt templates to a local directory so they can
    be inspected and customised. The destination must not already exist.

    Args:
        --dest: Parent directory in which a ``templates/`` folder will be
            created. Defaults to the current directory.

    """
    parser = argparse.ArgumentParser(
        description="Copy bundled actantial templates to a local directory for customisation.",
    )
    parser.add_argument(
        "dest",
        type=Path,
        nargs="?",
        default=Path("."),
        help="Parent directory where a ``templates/`` folder will be created (default: current directory).",
    )
    args = parser.parse_args()

    dest = args.dest / "templates"

    if dest.exists():
        print(
            f"Error: '{dest}' already exists. Choose a different path or remove it first."
        )
        raise SystemExit(1)

    shutil.copytree(BUNDLED_TEMPLATES_DIR, dest)
    print(
        f"Templates copied to '{dest}'. Pass '--templates_dir {dest}' to actantial to use them."
    )


if __name__ == "__main__":
    main()
