# Claude Module

This module provides a CLI tool and library function for interacting with Anthropic's Claude via LangChain. It reads a text prompt from a file and returns Claude's response.

## Configuration

This module uses the following environment variables:

- **ANTHROPIC_API_KEY**: Your API key for Anthropic's API.

## Using the CLI

You can run the CLI using Poetry:

### Required flag

* `--prompt-file`: The path to a file containing the prompt.

```bash
poetry run python -m claude --prompt-file path/to/prompt.txt
```

### Optional flags

* `--system-prompt-file`: The file containing the system prompt (default: `system_prompts/coding_000000`).
* `--model`: The Claude model name to use (default: `claude-3-7-sonnet-latest`).
* `--temperature`: The temperature for generation (default: 1.0).
* `--max-tokens`: The maximum number of tokens to generate in the response (default: 20000).
* `--thinking-budget-tokens`: The number of tokens to use for "thinking" before generating the response (default: 16000).

### Example command

```bash
# Example usage with custom system prompt, model, temperature, and max tokens.
poetry run python -m claude \
  --model claude-3.5-sonnet-latest \
  --temperature 0.7 \
  --max-tokens 15000 \
  --thinking-budget-tokens 10000 \
  --system-prompt-file path/to/system_prompt.txt \
  --prompt-file path/to/prompt.txt
```

## Logging

This module writes two logs to the `log/` directory:

1. A conversation log (`<basename>_conversation`) that appends both the prompt and response.
2. A response-only text file named `<basename>_claude_response_<timestamp>`.

Happy prompting!
