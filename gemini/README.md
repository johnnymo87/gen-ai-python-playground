# Gemini Module

This module provides a CLI tool and library function for interacting with Google's Gemini via the `google-genai` Python SDK. By default, it expects the `GOOGLE_API_KEY` environment variable to be set for authentication.

## Configuration

- **GOOGLE_API_KEY**: Your API key for the Gemini Developer API.

## Using the CLI

You can run the CLI using Poetry:

### Required flag

* `--prompt-file`: The path to a file containing the prompt.

```bash
poetry run python -m gemini --prompt-file path/to/prompt.txt
```

### Optional flags

* `--system-prompt-file`: The file containing the system prompt (default: `system_prompts/coding_000000`).
* `--model`: The Gemini model name to use (default: `gemini-2.5-pro-exp-03-25`).
* `--temperature`: The temperature for generation (default: 0.7).
* `--max-tokens`: The maximum number of tokens in the response (default: 2048).

### Example command

```bash
# Example usage with custom system prompt, model, temperature, and max tokens.
poetry run python -m gemini \
  --model gemini-2.5-pro-exp-03-25 \
  --temperature 0.6 \
  --max-tokens 1024 \
  --system-prompt-file path/to/system_prompt.txt
  --prompt-file path/to/prompt.txt
```

## Logging

This module writes two logs to the `log/` directory:

1. A conversation log (`<basename>_conversation`) that appends both the prompt and response.
2. A response-only text file named `<basename>_gemini_response_<timestamp>`.

Happy prompting!
