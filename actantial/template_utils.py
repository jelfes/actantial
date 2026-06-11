from pathlib import Path
from typing import Union

DEFAULT_TEMPLATES_DIRNAME = "default"


def resolve_template_path(
    templates_dir: Union[str, Path], model_name: str, template_name: str
) -> tuple[Path, str]:
    """
    Resolve a template file to a path, checking model-specific templates
    before falling back to the shared ``default/`` templates.

    Args:
        templates_dir: Root directory containing per-model template
            subdirectories and the shared ``default/`` directory.
        model_name: Model identifier whose template directory is checked first.
        template_name: Template file name, including the ``.txt`` extension.

    Returns:
        A tuple of the resolved template path and its source, where source
            is either ``model_name`` or ``"default"``.

    Raises:
        ValueError: If a template with the same name exists in both the
            model-specific directory and ``default/``.
        FileNotFoundError: If the template is not found in either location.
    """
    templates_dir = Path(templates_dir)
    model_path = templates_dir / model_name / template_name
    default_path = templates_dir / DEFAULT_TEMPLATES_DIRNAME / template_name

    model_exists = model_path.exists()
    default_exists = default_path.exists()

    if model_exists and default_exists:
        raise ValueError(
            f"Template '{template_name}' exists in both '{model_name}/' and "
            f"'{DEFAULT_TEMPLATES_DIRNAME}/'. Names in '{DEFAULT_TEMPLATES_DIRNAME}/' "
            "are reserved for shared base templates — please rename your "
            "model-specific template."
        )

    if model_exists:
        return model_path, model_name

    if default_exists:
        return default_path, DEFAULT_TEMPLATES_DIRNAME

    raise FileNotFoundError(
        f"Template '{template_name}' not found in '{templates_dir / model_name}' "
        f"or '{templates_dir / DEFAULT_TEMPLATES_DIRNAME}'."
    )


def list_templates_for_model(
    templates_dir: Union[str, Path], model_name: str
) -> dict[str, list[str]]:
    """
    List the prompt templates available for a model, split by source.

    Args:
        templates_dir: Root directory containing per-model template
            subdirectories and the shared ``default/`` directory.
        model_name: Model identifier whose template directory is listed.

    Returns:
        A dict with keys ``"model_specific"`` and ``"default"``, each mapping
            to a sorted list of template names (without the ``.txt`` extension).

    Raises:
        ValueError: If a template name exists in both the model-specific
            directory and ``default/``.
        FileNotFoundError: If neither the model-specific directory nor
            ``default/`` exists.
    """
    templates_dir = Path(templates_dir)
    model_dir = templates_dir / model_name
    default_dir = templates_dir / DEFAULT_TEMPLATES_DIRNAME

    if not model_dir.exists() and not default_dir.exists():
        raise FileNotFoundError(f"Neither '{model_dir}' nor '{default_dir}' exists.")

    model_templates = (
        sorted(f.stem for f in model_dir.glob("*.txt")) if model_dir.exists() else []
    )
    default_templates = (
        sorted(f.stem for f in default_dir.glob("*.txt"))
        if default_dir.exists()
        else []
    )

    collisions = set(model_templates) & set(default_templates)
    if collisions:
        raise ValueError(
            f"Template name(s) {sorted(collisions)} exist in both '{model_dir}' "
            f"and '{default_dir}'. Names in '{DEFAULT_TEMPLATES_DIRNAME}/' are "
            "reserved for shared base templates — please rename your "
            "model-specific template(s)."
        )

    return {"model_specific": model_templates, "default": default_templates}
