# Roadmap to Publication

## Phase 1 — Solidify the Core

**Backends**
- [x] Fix Anthropic backend — rewrite using the `messages` API
- [x] Wire Anthropic backend into the CLI
- [x] Review and update the OpenAI system prompt
- [x] Add `model.eval()` to HuggingFace backend — called once after `from_pretrained`; needed because HF loads in training mode by default and `torch.no_grad()` alone does not disable dropout
- [x] Implement `cleanup()` on the base class so the context manager works correctly

**Correctness**
- [x] Write a basic test suite (`tests/`)
- [ ] Run end-to-end tests on GPU server with real HuggingFace models
- [ ] Main train 

---

## Phase 2 — Usability & Robustness

**Resumability**
- [x] Implement checkpoint/resume logic in the runner (skip already-processed IDs)

**Performance**
- [x] Investigate and fix slow module load time
    - Caused when loading torch/transformers

**Exploration**
- [ ] Recreate and revise `eda_actor_labels.ipynb`
- [ ] Add summarisation/visualisation utilities for extracted actants

---

## Phase 3 — Documentation & Examples

- [ ] Fill in README.md placeholders (CLI usage, notebook usage, template authoring, model recommendations)
- [ ] Add API reference documentation (e.g. via `mkdocs` or `sphinx`)
- [ ] Create `examples/` folder with sample dataset, labelset YAML, and walkthrough notebook
- [ ] Add more prompt templates for different extraction tasks and models
- [ ] Fix eof_token for open generation warning in huggingface

---

## Phase 4 — Packaging & Publication

- [ ] Review and finalise `pyproject.toml` (extras, minimal core deps)
- [ ] Add `LICENSE` file if not present
- [ ] Fine-tune dependency version bounds (lower and upper) before publication
- [ ] Audit all relative paths — ensure nothing assumes repo root as working directory
- [ ] Tag `v0.1.0` release on GitHub
- [ ] Publish to PyPI
- [ ] Consider a JOSS paper or short methods note

---

## Post-launch Features

- [ ] Add 8-bit quantisation support for HuggingFace backend (alongside existing 4-bit)
- [ ] System prompt support for HuggingFace backend via `apply_chat_template`

---

## Done

- [x] Basic CLI with argparse
- [x] OpenAI backend working
- [x] HuggingFace backend with 4-bit quantisation
- [x] Runner handles looping, logging, and saving
- [x] Support for predefined actor/object label sets via YAML
- [x] Validation function
- [x] Split dependencies into core and `[huggingface]` extras
