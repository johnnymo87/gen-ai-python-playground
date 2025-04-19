# Generative AI Python Playground

This repository is a collection of scripts and utilities that I use to interact with generative AI models from Anthropic (Claude), Google (Gemini), and OpenAI (GPT).

## Overview

This project leverages Python and [Poetry](https://python-poetry.org/) to manage dependencies, as well as [Direnv](https://direnv.net/) for managing environment variables. Each module (`claude/`, `gemini/`, `openai/`) in this repository has its own README, which provides task-specific or site-specific details. # Modified this line

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/johnnymo87/gen-ai-python-playground.git
   cd gen-ai-python-playground
   ```

2. **Environment variables:**
   - Rename `.envrc.example` to `.envrc` and fill in your API keys for Anthropic, Google, and OpenAI.
   - Allow direnv:
     ```bash
     direnv allow
     ```

3. **Python Setup:**
   - Use pyenv to install Python (see [pyenv installation](https://github.com/pyenv/pyenv#installation)).
   - The required Python version is specified in `.python-version`.

4. **Install Dependencies:**
   - Install Poetry if you haven't already:
     ```bash
     curl -sSL https://install.python-poetry.org | python3 -
     ```
   - Then install project dependencies:
     ```bash
     poetry install --with dev
     ```

5. **Pre-commit Hooks:**
   - Install pre-commit hooks:
     ```bash
     pre-commit install
     ```

6. **Running the linter:**
   - Run the linter with:
     ```bash
     pre-commit run --all-files
     ```

7. **Running Tests:**
   - Execute tests (if present) using pytest:
     ```bash
     poetry run pytest
     ```

## CI Pipeline

- A GitHub Actions workflow defined in `.github/workflows/ci.yaml` runs lint checks and tests on every push and pull request.

## Contributing

1. Fork the repository in GitHub.
2. Update your local clone to call your fork "origin" and my repository "upstream":
   ```bash
   git remote rename origin upstream
   git remote add origin YOUR_FORK_URL
   ```
3. Make a new branch for your changes.
4. Submit a Pull Request.
5. Confirm that it passes the CI pipeline.

---

Happy coding!
