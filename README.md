# actantial

`actantial` is a research tool for analysing narratives using Greimas' Actantial Model. It uses LLMs to annotate texts with six character roles — called actants — including the Subject, Object, Sender, Receiver, Helper, and Opponent. These structured role representations can be used to analyse the underlying character constellations of a text and compare them across texts. The model can be applied to various sources, such as news articles, social media posts, or political speeches.

Getting started:

- [Getting started notebook](https://github.com/jelfes/actantial/blob/main/examples/getting_started.ipynb) 🏄
- [Documentation](https://jelfes.github.io/actantial) 📖

Further details on the theory and application:

- [Theoretical background of the Actantial Model](https://jelfes.github.io/actantial/background/)
- From narrative theory to automated annotation (WIP)
- [Actantial model on news articles](https://dl.acm.org/doi/full/10.1145/3717867.3717868) (Elfes, 2025)
- [Actantial model on social media](https://arxiv.org/abs/2601.07398) (Elfes et al., 2026)


## Installation

```bash
pip install actantial
```

## Key concepts

**Actants** are the six character roles defined by Greimas' Actantial Model: Subject, Object, Sender, Receiver, Helper, and Opponent. Given a text, `actantial` uses an LLM to assign actors from the text to each of these roles.

**Backends** are the LLM providers used for inference. `actantial` supports the Anthropic and OpenAI APIs, and local models via HuggingFace.

**Templates** are the prompts sent to the LLM. They control the extraction process and significantly alter the way it is handled. The package provides a general-purpose base template that serves as a starting point for any model. However, performance varies significantly across models and use cases, so for systematic use it's important to adapt the template. This matters especially for closed-set annotations, where label definitions need careful, model-specific tuning. See [Templates](#templates) for details.

**Open vs. closed annotation** — in open mode, the LLM assigns actors freely from the text. In closed mode, you provide predefined lists of actor and object labels, constraining the LLM to choose from those options. Closed annotation is recommended when you want consistent, comparable labels across texts. However, it requires devising a concise label set (for details, see WIP).

## Key functions

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
    template="base_prompt",
)
```

Results are saved to `output/actantial_models/{model}/{template}/{timestamp}/`. To load them back into your DataFrame:

```python
annotations = load_annotations(
    data=data,
    label_folder="output/actantial_models/gpt-4o-mini/base_prompt/TIMESTAMP/",
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

```python
from actantial import HuggingFaceBackend

backend = HuggingFaceBackend(
    repository="deepseek-ai",
    model_name="DeepSeek-R1-Distill-Qwen-32B",
    quantisation=True,  # optional: 4-bit quantisation, requires a CUDA GPU!
)
```

## Templates

Templates are [Jinja2](https://jinja.palletsprojects.com/) files that define the prompt sent to the LLM. They are organised by model in a `templates/{model_name}/` directory structure, plus a shared `templates/default/` directory.

`templates/default/base_prompt.txt` is a general-purpose open-annotation template that serves as a starting point for any model. However, performance varies significantly across models and use cases, so for systematic use it's important to adapt templates for specific models. This can be a small change, e.g., adding `<think>` to the end of the template to trigger _thinking_ mode in a DeepSeek-R1 model. However, this can also entail adding additional details on the theory or providing more examples.

Besides the `base_prompt`, `actantial` also provides additional example templates that demonstrate some of these variations and are showcased in the example notebooks. You can copy these example templates to a local directory to start creating your own customisations:

```bash
actantial-init-templates path/to/directory/
```

This creates a `templates/` folder with the sample templates in the specified directory.

To see which templates are available for your model, and to preview a template before running:

```python
backend.list_templates()
# {'model_specific': ['prompt_closed', 'prompt_open_variables'], 'default': ['base_prompt']}

backend.show_template("base_prompt")  # prints the rendered template
```

### Custom templates

To write your own template, create a `.txt` file with a `{{ text }}` variable:

```
Extract actants from the following text: {{ text }}
```

Place it at `path/to/directory/templates/{model_name}/{template_name}.txt` and pass `templates_dir` and `template` to the runner:

```python
run_extract(
    data=data,
    backend=backend,
    output_dir="output",
    template="my_template",
    templates_dir="path/to/directory/templates",
)
```

### Closed-set templates

For closed-set annotation you need to create a predefined label set for the model to apply. You might want to create a separate set of labels for actors and objects but that is optional. The labels are saved in YAML files:

```yaml
# actors.yaml
- actor 1
- actor 2
...
```

Then add the variables `{{ actor_labels }}` and `{{ object_labels }}` to your template, load the YAML files, and pass the lists to the runner:

```python
import yaml

with open("path/to/directory/labels/actors.yaml") as f:
    actor_labels = yaml.safe_load(f)

with open("path/to/directory/labels/objects.yaml") as f:
    object_labels = yaml.safe_load(f)

run_extract(
    data=data,
    backend=backend,
    output_dir="output",
    template="my_template",
    actor_labels=actor_labels,
    object_labels=object_labels,
)
```

Note, not all models stick to the labels consistently! You can find a detailed example in the provided [case study](https://jelfes.github.io/actantial/examples/case_study/).

For additional guidance see [Elfes et al. (2026)](https://arxiv.org/abs/2601.07398). 

### Additional Variables

You can pass additional columns from your DataFrame as template variables using `template_columns`. This is useful when each data point has individual context beyond the text itself. For example, you can pass the `video_title` when annotating YouTube comments.

Two things are required:
1. A column of string dtype in your DataFrame (e.g. `video_title`).
2. A matching `{{ video_title }}` variable in your Jinja2 template.

```python
run_extract(
    data=data,
    backend=backend,
    output_dir="output",
    template="my_template",
    template_columns=["video_title"],
)
```

Multiple columns can be passed at once: `template_columns=["video_title", "video_creator"]`.

### System prompt

An optional system prompt can be passed to API backends at initialisation:

```python
backend = OpenAIBackend(model_name="gpt-4o-mini", system_prompt="Always respond in JSON.")
```

## Validation

Validation of the labels is difficult. Especially open-label annotations can vary significantly between prompts and models. This is both due to the complexity of the model and the variation in label formulation without fixed label set. Thus, the validation workflow mostly makes sense for closed-set annotations. Either to compare different models, or to validate models against human annotations (for details, see the [case study](https://jelfes.github.io/actantial/examples/case_study/)).

When loading annotations, if the LLM returned multiple actors for a role you can control how they are handled with `select_actor="first"` (default) or `select_actor="combine"` (joins them into a comma-separated string):

```python
llm_annotations = load_annotations(data, label_folder="output/actantial_models/model/prompt/timestamp", select_actor="first")
```

To validate `llm_annotations` against `reference_annotations`:

```python
from actantial import compare_annotations

results = compare_annotations(
    dfs=[llm_annotations, reference_annotations],
    names=["llm", "reference"],
    metric="accuracy",
)
```

`results` is a DataFrame with one row per actant and an `avg` summary row. Supported metrics: `accuracy`, `f1_micro`, `f1_macro`, `f1_weighted`, `krippendorff_alpha`.


## CLI

For large datasets, `actantial` can be run directly from the command line:

```bash
# OpenAI API
actantial \
    --data_file "data.csv" \
    --output_dir "output" \
    --backend openai \
    --model "gpt-4o-mini" \
    --templates_dir "path/to/directory/templates" \
    --template "my_template"

# HuggingFace (GPU, 4-bit quantised)
actantial \
    --data_file "data.csv" \
    --output_dir "output" \
    --backend huggingface \
    --repository "deepseek-ai" \
    --model "DeepSeek-R1-Distill-Qwen-32B" \
    --templates_dir "path/to/directory/templates" \
    --template "my_template" \
    --quantise

# With predefined label sets
actantial \
    --data_file "data.csv" \
    --output_dir "output" \
    --backend anthropic \
    --model "claude-haiku-4-5" \
    --templates_dir "path/to/directory/templates" \
    --template "my_template_closed" \
    --actor_labels_path "path/to/directory/labels/actors.yaml" \
    --object_labels_path "path/to/directory/labels/objects.yaml"
```

**With additional template variables** — pass `--template_column` once per column (repeatable):

```bash
actantial \
    --data_file "data.csv" \
    --output_dir "output" \
    --backend openai \
    --model "gpt-4o-mini" \
    --templates_dir "path/to/directory/templates" \
    --template "my_template" \
    --template_column video_title \
    --template_column video_creator
```

**Passing backend parameters** — create a YAML file with any constructor keyword arguments for the chosen backend:

```yaml
# params.yaml
temperature: 0.7
top_p: 0.9
do_sample: true
```

```bash
actantial \
    --data_file "data.csv" \
    --output_dir "output" \
    --backend huggingface \
    --repository "google" \
    --model "gemma-3-1b-it" \
    --templates_dir "path/to/directory/templates" \
    --template "my_template" \
    --backend_params_path "path/to/directory/params.yaml"
```

**Resuming an interrupted run** — pass the timestamp printed at the start of the original run:

```bash
actantial \
    --data_file "data.csv" \
    --output_dir "output" \
    --backend openai \
    --model "gpt-4o-mini" \
    --templates_dir "path/to/directory/templates" \
    --template "my_template" \
    --resume_timestamp "YYYYMMDD_HHMMSS"
```

The model and template must match the original run. Already-processed IDs are skipped automatically.



---


## Acknowledgements
[Claude Code](https://claude.ai/claude-code) was used for packaging, documentation, and code organisation.
