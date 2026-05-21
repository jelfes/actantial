# CLI

::: actantial.cli.main

::: actantial.cli.init_templates

## Examples

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

**Resuming an interrupted run:**

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
