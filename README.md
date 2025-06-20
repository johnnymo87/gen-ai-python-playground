# Generative AI Python Playground

This repository is a collection of scripts and utilities that I use to interact with generative AI models from Anthropic (Claude), Google (Gemini), and OpenAI (GPT).

These CLIs require either call the service providers' APIs directly, or use Google "Vertex AI" API. For the former approach, you will need an API key for each service provider. For the latter approach, you will need to set up the Google Cloud SDK and authenticate your account. The Google Cloud SDK allows you to interact with Google Cloud services, including Vertex AI, which can be used to access the same models.

## Overview

This project leverages Python and [uv](https://docs.astral.sh/uv/) to manage dependencies, as well as [Direnv](https://direnv.net/) for managing environment variables. Each module (`claude_cli/`, `gemini_cli/`, `openai_cli/`, `vertex_cli/`) in this repository has its own README, which provides task-specific or site-specific details.

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/johnnymo87/gen-ai-python-playground.git
   cd gen-ai-python-playground
   ```

2. **Install Direnv:**
   - Follow the [Direnv installation guide](https://direnv.net/docs/installation.html) for your operating system.
   - Ensure that your shell is configured to use Direnv (e.g., add `eval "$(direnv hook bash)"` to your `.bashrc` or `.bash_profile`).

3. **Install the Google Cloud SDK:** (optional, for using Vertex AI)
   - Follow the [Google Cloud SDK installation guide](https://cloud.google.com/sdk/docs/install-sdk) for your operating system.
   - After installation, run:
     ```bash
     gcloud auth application-default login
     ```
   - This command will set up your Google credentials for the project.

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
   - Install uv if you haven't already:
     ```bash
     curl -LsSf https://astral.sh/uv/install.sh | sh
     ```
   - Then install project dependencies:
     ```bash
     uv sync
     ```

5. **Git Hooks (Optional):**
   - Set up a git pre-commit hook that runs ruff automatically:
     ```bash
     ./setup-git-hooks.sh
     ```
   - This will run `ruff check` and `ruff format --check` before each commit
   - To skip the hook for a specific commit: `git commit --no-verify`

6. **Code Quality:**
   - Run ruff for linting and formatting:
     ```bash
     # Check for linting issues
     uv run ruff check .

     # Auto-fix linting issues
     uv run ruff check --fix .

     # Format code
     uv run ruff format .

     # Run type checking
     uv run mypy .
     ```

7. **Running Tests:**
   - Execute tests (if present) using pytest:
     ```bash
     uv run pytest
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
