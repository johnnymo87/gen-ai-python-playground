# Claude Module

This module provides a CLI tool and library function for interacting with Anthropic's Claude via LangChain. It reads a text prompt from a file and returns Claude’s response.

## Configuration

This module uses the following environment variables:

- **ANTHROPIC_API_KEY**: Your API key for Anthropic's API.

## Using the CLI

You can run the CLI using Poetry:

### Required flags

The CLI requires the following flag:
* `input-file`: The path to a file containing the prompt.

```bash
poetry run python -m claude --input-file path/to/prompt.txt
```

## Optional flags

There are a few flags you can use to customize the behavior of the CLI:
* `--max-tokens`: The maximum number of tokens to generate in the response.
* `--thinking-budget-tokens`: The number of tokens to use for "thinking" before generating the response.

```
poetry run python -m claude \
  --max-tokens 15000 \
  --thinking-budget-tokens 10000 \
  --input-file path/to/prompt.txt
```

Happy prompting!
