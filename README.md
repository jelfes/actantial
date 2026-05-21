# actantial

`actantial` is a research tool for analysing narratives using Greimas' Actantial Model. It uses LLMs to annotate texts with six character roles — called actants — including Subject, Object, Sender, Receiver, Helper, and Opponent. These structured role representations can be used to analyse the underlying character constellations of a text and compare them across texts. The model can be applied to various sources, such as news articles, social media posts, or political speeches.

For further details on the theory and application, refer to the following resources:
- [Introduction to the Actantial Model]()
- [From narrative theory to automated annotation]() (Elfes, 2026)
- [Actantial model on news articles](https://dl.acm.org/doi/full/10.1145/3717867.3717868) (Elfes, 2025)
- [Actantial model on social media](https://arxiv.org/abs/2601.07398) (Elfes et al., 2026)


## Installation

For API-based backends (Anthropic or OpenAI):

```bash
pip install actantial
```

To run models locally on a GPU via HuggingFace:

```bash
pip install actantial[huggingface]
```

This additionally installs `torch`, `transformers`, `accelerate`, and `bitsandbytes`. It requires a CUDA-capable GPU.

## Key concepts

**Actants** are the six character roles defined by Greimas' Actantial Model: Subject, Object, Sender, Receiver, Helper, and Opponent. Given a text, `actantial` uses an LLM to assign actors from the text to each of these roles.

**Backends** are the LLM providers used for inference. `actantial` supports the Anthropic and OpenAI APIs, and local models via HuggingFace.

**Templates** are the prompts sent to the LLM. They control how the extraction task is framed and must contain at least a `{{ text }}` variable. `actantial` ships with built-in example templates, but custom templates are recommended for new use cases.

**Open vs. closed annotation** — in open mode, the LLM assigns actors freely from the text. In closed mode, you provide predefined lists of actor and object labels, constraining the LLM to choose from those options. Closed annotation is recommended when you want consistent, comparable labels across texts. However, it requires devising a concise label set (for details, see Elfes, 2026).

## Quick start

Your input data must have an `id` column and a `text` column.

```python
import pandas as pd
from actantial import OpenAIBackend, run_extract, load_annotations

data = pd.read_csv("data.csv")
backend = OpenAIBackend(model_name="gpt-4o-mini")

run_extract(
    data=data,
    backend=backend,
    output_dir="output",
    template="prompt_open",
)
```

Results are saved to `output/actantial_models/{model}/{template}/{timestamp}/`. To load them back into your DataFrame:

```python
annotations = load_annotations(
    data=data,
    label_folder="output/actantial_models/gpt-4o-mini/prompt_open/TIMESTAMP/",
)
```

This adds one column per actant role — Subject, Object, Sender, Receiver, Helper, Opponent — to your DataFrame.

## Backends

`actantial` supports three backends: Anthropic, OpenAI, and HuggingFace.

The **API backends** ([Anthropic](https://platform.claude.com/docs/en/about-claude/pricing) and [OpenAI](https://developers.openai.com/api/docs/pricing)) send requests to the respective company's servers. They require a subscription and an API key, but can be run from any machine. Check the pricing of your chosen model before running a large extraction.

The **HuggingFace backend** runs open-weight models locally on your machine, giving access to models such as Llama, DeepSeek, and Gemma on the [HuggingFace platform](https://huggingface.co/models?pipeline_tag=text-generation&sort=trending). It requires a GPU; running on a server is recommended for large-scale extractions.

### API

Set your API key in a `.env` file or as an environment variable:

```bash
ANTHROPIC_API_KEY=your-key-here
OPENAI_API_KEY=your-key-here
```

```python
from actantial import AnthropicBackend, OpenAIBackend

backend = AnthropicBackend(model_name="claude-haiku-4-5")
backend = OpenAIBackend(model_name="gpt-4o-mini")
```

Both backends validate the model name at initialisation — an invalid model raises a `ValueError`. An optional `system_prompt` argument is available for both.

### GPU

First install the HuggingFace dependencies (see [Installation](#installation)), then:

```python
from actantial.backends.huggingface import HuggingFaceBackend

backend = HuggingFaceBackend(
    repository="deepseek-ai",
    model_name="DeepSeek-R1-Distill-Qwen-32B",
    quantisation=True,  # optional: 4-bit quantisation, reduces VRAM requirements
)
```

## Templates

Templates are [Jinja2](https://jinja.palletsprojects.com/) files that define the prompt sent to the LLM. They are organised by model in a `templates/{model_name}/` directory structure. `actantial` ships with built-in example templates; to inspect and customise them, copy them to a local directory:

```bash
actantial-init-templates
```

This creates a `templates/` folder in your current directory.

To see which templates are available for your model, and to preview a template before running:

```python
backend.list_templates()       # returns a list of template names
backend.show_template("prompt_open")  # prints the rendered template
```

### Custom templates

To write your own template, create a `.txt` file with a `{{ text }}` variable:

```
Extract actants from the following text: {{ text }}
```

Place it at `templates/{model_name}/{template_name}.txt` and pass `templates_dir` and `template` to the runner:

```python
run_extract(
    data=data,
    backend=backend,
    output_dir="output",
    template="my_template",
    templates_dir="templates",
)
```

### Closed-set templates

For closed-set annotation you need to create a predefined label set for the model to apply. You might want to create a separate set of labels for actors and objects but that is optional. The labels are saved in YAML files:

```
- actor 1
- actor 2
...
```

Then add `{{ actor_labels }}` and `{{ object_labels }}` to your template and pass the corresponding YAML files to the runner:

```python
run_extract(
    data=data,
    backend=backend,
    output_dir="output",
    template="my_template",
    actor_labels_path="labels/actors.yaml",
    object_labels_path="labels/objects.yaml",
)
```

Note, not all models stick to the labels consistently!

### System prompt

An optional system prompt can be passed to API backends at initialisation:

```python
backend = OpenAIBackend(model_name="gpt-4o-mini", system_prompt="Always respond in JSON.")
```

## Validation

Validation of the labels is difficult. Especially open-label annotations can vary significantly between prompts and models. This is both due to the complexity of the model and the variation in label formulation without fixed label set. Thus, the validation workflow mostly makes sense for closed-set annotations. Either to compare different models, or to validate models against human annotations (for details, see [Elfes, 2026]()).

When loading annotations, if the LLM returned multiple actors for a role you can control how they are handled with `select_actor="first"` (default) or `select_actor="combine"` (joins them into a comma-separated string):

```python
llm_annotations = load_annotations(data, label_folder="output/actantial_models/model/prompt/timestamp", select_actor="first")
```

To validate `llm_annotations` against `reference_annotations`:

```python
from actantial import compare_labels

results = compare_labels(
    llm_annotations,        # DataFrame with 'id' + actant columns
    reference_annotations,  # DataFrame with 'id' + actant columns
    metric="accuracy",
)
```

`results` is a dict with `"per_actant"` scores and an `"avg"` mean across actants. Supported metrics: `accuracy`, `f1_micro`, `f1_macro`, `f1_weighted`, `krippendorff_alpha`.


## CLI

For large datasets, `actantial` can be run directly from the command line:

```bash
# OpenAI API
actantial \
    --data_file "data.csv" \
    --output_dir "output" \
    --backend openai \
    --model "gpt-4o-mini" \
    --templates_dir "templates" \
    --template "my_template"

# HuggingFace (GPU, 4-bit quantised)
actantial \
    --data_file "data.csv" \
    --output_dir "output" \
    --backend huggingface \
    --repository "deepseek-ai" \
    --model "DeepSeek-R1-Distill-Qwen-32B" \
    --templates_dir "templates" \
    --template "my_template" \
    --quantise

# With predefined label sets
actantial \
    --data_file "data.csv" \
    --output_dir "output" \
    --backend anthropic \
    --model "claude-haiku-4-5" \
    --templates_dir "templates" \
    --template "my_template_closed" \
    --actor_labels_path "labels/actors.yaml" \
    --object_labels_path "labels/objects.yaml"
```

**Resuming an interrupted run** — pass the timestamp printed at the start of the original run:

```bash
actantial \
    --data_file "data.csv" \
    --output_dir "output" \
    --backend openai \
    --model "gpt-4o-mini" \
    --templates_dir "templates" \
    --template "my_template" \
    --resume_timestamp "20260101_121500"
```

The model and template must match the original run. Already-processed IDs are skipped automatically.



---


## Acknowledgements
Part of this codebase was developed with assistance from [Claude Code](https://claude.ai/claude-code).
