# actantial/cli.py

import argparse
from pathlib import Path

import pandas as pd
from .runner import run_extract


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Actantial: LLM-based narrative role extraction",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--data_file",
        type=Path,
        required=True,
        help="CSV file with `id` and `text` columns. Will be read with pandas.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Directory where results and logs will be emitted.",
    )
    parser.add_argument(
        "--backend",
        choices=["anthropic", "huggingface", "openai"],
        required=True,
        help="LLM backend to use",
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
        help="Optional YAML file with predefined actor labels",
    )
    parser.add_argument(
        "--object_labels_path",
        type=str,
        help="Optional YAML file with predefined object labels",
    )
    parser.add_argument(
        "--resume_timestamp",
        type=str,
        default=None,
        metavar="TIMESTAMP",
        help="Timestamp of a previous run to resume (format: YYYYMMDD_HHMMSS). Model and template must match the original run.",
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

    run_extract(
        data=data,
        backend=backend,
        output_dir=args.output_dir,
        template=args.template,
        actor_labels_path=args.actor_labels_path,
        object_labels_path=args.object_labels_path,
        resume_timestamp=args.resume_timestamp,
    )


if __name__ == "__main__":
    main()
