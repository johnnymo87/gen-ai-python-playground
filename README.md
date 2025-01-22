# gen-ai-python-playground

This codebase is a collection of scripts, CLIs, etc. that use python to accomplish various ends by interacting with generative AI service providers' APIs.

## Install

### Environment variables

1. Copy `.envrc.example` to `.envrc`.
   ```
   cp .envrc.example .envrc
   ```

1. Fill out `.envrc`.

1. Source the environment variables defined in `.envrc`.
   ```
   direnv allow
   ```

### Python

1. Install or update `pyenv` (and `python-build` to get access to recent releases of python).
   ```
   brew update && brew install python-build pyenv
   brew update && brew upgrade python-build pyenv
   ```

1. Configure your shell to enable shims.
   ```
   # in e.g. ~/.bash_profile

   if which pyenv > /dev/null; then
     eval "$(pyenv init -)"
   fi
   ```

1. Install python.
   ```
   pyenv install $(cat .python-version)
   ```

### Poetry

1. Install Poetry [via its official installer](https://python-poetry.org/docs/#installing-with-the-official-installer).

### Python dependencies

1. Install dependencies.
   ```
   poetry install
   ```

1. Initialize the poetry virtual environment.
   ```
   poetry env activate
   ```

1. Install the pre-commit hooks.
   ```
   pre-commit install
   ```

## Development

* Initialize the poetry virtual environment.
  ```
  poetry env activate
  ```

* Run the auto formatter manually (although it will run automatically as a pre-commit hook).
  ```
  pre-commit run --all-files
  ```
