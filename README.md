# actantial
`actantial` is a research tool for analyzing narratives using Greimas' Actantial Model. It provides functions to extract and analyse different character roles—actants—in text. Applications include news articles, social media posts, and more.

## Installation

### Minimal install (just core)
pip install actantial

### With HuggingFace backend
pip install actantial[huggingface]


## Components
- `actantial.extract`: extract actants from text either free-form or with a predifined label set
- `actantial.runner`: handles loop over multiple texts and merging results
- `actantial.backends`: different model backends (HuggingFace, Anthropic API, etc.)
- `actantial.templates`: prompt templates for different extraction tasks
- `actantial.cli`: command-line interface for easy usage

## Usage
### CLI
This is how you use the package in your terminal ...


### Notebook
This is how you use the package in a notebook ...

### Templates
This is how you add new templates ...

<!-- # TODO add some Some notes on constraints, difficulties -->

### Backends & Models
#### Huggingface
- do_sample=False
- GPU
- Quantisation
- LLMs that are available
- Resource for GPU constraints

#### Anthropic / OpenAI
- .env with token
- not really tested
- system prompt?

---

Parts of this codebase were developed with assistance from [Claude Code](https://claude.ai/claude-code).