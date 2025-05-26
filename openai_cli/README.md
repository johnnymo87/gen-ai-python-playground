# OpenAI Module

This module provides a CLI tool and library function for interacting with OpenAI's models (like GPT-4o, GPT-4o-mini) via the `openai` Python SDK. It expects the `OPENAI_API_KEY` environment variable to be set for authentication.

## Configuration

- **OPENAI_API_KEY**: Your API key for the OpenAI API.

## Using the CLI

You can run the CLI using Poetry:

### Required flag

*   `--prompt-file`: The path to a file containing the prompt.

```bash
poetry run python -m openai_cli --prompt-file path/to/prompt.txt
```

### Optional flags

*   `--system-prompt-file`: The file containing the system prompt (default: `system_prompts/coding_000000`).
*   `--model`: The OpenAI model name to use (default: `gpt-4o-mini`).
*   `--temperature`: The temperature for generation (default: 1.0).
*   `--max-tokens`: The maximum number of tokens to generate in the response (default: 16000).

### Example command

```bash
# Example usage with custom system prompt, model, temperature, and max tokens.
poetry run python -m openai_cli \
  --model gpt-4o \
  --temperature 0.5 \
  --max-tokens 2000 \
  --system-prompt-file path/to/system_prompt.txt \
  --prompt-file path/to/prompt.txt
```

## Logging

This module writes two logs to the `log/` directory:

1.  A conversation log (`<basename>_conversation`) that appends both the prompt and response.
2.  A response-only text file named `<basename>_openai_response_<timestamp>`.

Happy prompting!
