# Vertex Module

This module provides a CLI tool for interacting with models hosted on Google Cloud's Vertex AI platform (including Gemini, Imagen, Claude, and other models). It uses Google Cloud IAM for authentication rather than API keys.

## Configuration

This module uses Google Cloud authentication:

- **Authentication**: Uses Application Default Credentials (ADC) via `gcloud auth application-default login`
- **Project**: The Google Cloud project ID can be specified via the `--project` flag

## Using the CLI

You can run the CLI using uv:

### Required flag

* `--prompt-file`: The path to a file containing the prompt.

```bash
uv run python -m vertex_cli --prompt-file path/to/prompt.txt
```

### Optional flags

* `--system-prompt-file`: The file containing the system prompt (default: `system_prompts/coding_000000`).
* `--model`: The Vertex AI model name to use (default model varies by implementation).
* `--temperature`: The temperature for generation (default: 0.3).
* `--max-tokens`: The maximum number of tokens to generate in the response (default: 1000000, but limited by model).
* `--thinking-budget`: Thinking budget in tokens for Gemini and Claude models (default: 8192 tokens, 0 to disable thinking).
* `--project`: The Google Cloud project ID to use.

### Example command

```bash
# First authenticate with Google Cloud
gcloud auth application-default login

# Example usage with custom system prompt, model, temperature, and max tokens.
uv run python -m vertex_cli \
  --model gemini-1.5-pro \
  --temperature 0.7 \
  --max-tokens 8000 \
  --thinking-budget 4096 \
  --project my-gcp-project \
  --system-prompt-file path/to/system_prompt.txt \
  --prompt-file path/to/prompt.txt
```

## Logging

This module writes two logs to the `log/` directory:

1. A conversation log (`<basename>_conversation`) that appends both the prompt and response.
2. A response-only text file named `<basename>_vertex_response_<timestamp>`.

Happy prompting!
