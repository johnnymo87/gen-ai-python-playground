import os
import sys
from datetime import datetime
from typing import List, Union

import click
from anthropic import Anthropic
from anthropic.types import TextBlock, ThinkingBlock


@click.command()
@click.option(
    "--prompt-file",
    type=click.Path(exists=True),
    required=True,
    help="Path to a file containing the prompt text.",
)
@click.option(
    "--system-prompt-file",
    type=click.Path(exists=True),
    required=False,
    default="system_prompts/coding_000000",
    help="Path to a file containing the system prompt text.",
)
@click.option(
    "--model",
    default="claude-3-7-sonnet-latest",
    help="Anthropic model name to use. Defaults to claude-3-7-sonnet-latest.",
)
@click.option(
    "--temperature",
    type=float,
    default=0.3,
    help="Temperature for generation. Defaults to 0.3.",
)
@click.option(
    "--max-tokens",
    type=int,
    default=524288,
    help="Maximum tokens in the response. Defaults to about half a million.",
)
@click.option(
    "--thinking-budget-tokens",
    type=int,
    default=16000,
    help="Budget for the model's 'thinking' tokens.",
)
def main(
    prompt_file: str,
    system_prompt_file: str,
    model: str,
    temperature: float,
    max_tokens: int,
    thinking_budget_tokens: int,
) -> None:
    """
    Use the prompt text from --prompt-file (and a system prompt from
    --system-prompt-file) to query the Claude API via the Anthropic SDK,
    then store both the prompt and response in a conversation log file,
    and also write out a separate file containing just the response.
    """
    # Read the main prompt
    try:
        with open(prompt_file, "r", encoding="utf-8") as f:
            prompt = f.read()
    except OSError as exc:
        click.echo(f"Error reading prompt file: {exc}", err=True)
        sys.exit(1)

    # Read the system prompt
    try:
        with open(system_prompt_file, "r", encoding="utf-8") as f:
            system_prompt = f.read()
    except OSError as exc:
        click.echo(f"Error reading system prompt file: {exc}", err=True)
        sys.exit(1)

    # Call the Claude API
    try:
        response_text = get_claude_response(
            prompt,
            system_prompt,
            model,
            temperature,
            max_tokens,
            thinking_budget_tokens,
        )
    except Exception as exc:
        # Consider catching specific anthropic.APIError subclasses later
        click.echo(f"Error calling Anthropic API: {exc}", err=True)
        sys.exit(1)

    # Write the response to stdout
    click.echo(response_text)

    # Prepare for logging
    os.makedirs("log", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    base_name = os.path.splitext(os.path.basename(prompt_file))[0]

    # Append the prompt and response to a "conversation" log
    conversation_path = os.path.join("log", f"{base_name}_conversation")
    try:
        with open(conversation_path, "a", encoding="utf-8") as conv_f:
            # Using a slightly more structured log format
            conv_f.write(f"--- Prompt: {timestamp} ---\n")
            conv_f.write(f"{prompt}\n")
            conv_f.write(f"--- Response: {timestamp} ---\n")
            conv_f.write(f"{response_text}\n\n")  # Added newline for separation
    except OSError as exc:
        click.echo(f"Error writing conversation log: {exc}", err=True)

    # Write the response separately, with a timestamped filename
    response_path = os.path.join("log", f"{base_name}_claude_response_{timestamp}")
    try:
        with open(response_path, "w", encoding="utf-8") as resp_f:
            resp_f.write(response_text)
    except OSError as exc:
        click.echo(f"Error writing response file: {exc}", err=True)

    click.echo(f"Response written to {response_path}")


def get_claude_response(
    prompt: str,
    system_prompt: str,
    model: str,
    temperature: float,
    max_tokens: int,
    thinking_budget_tokens: int,
) -> str:
    """
    Uses the Claude API via the Anthropic Python SDK to process the
    given prompt. Make sure the ANTHROPIC_API_KEY environment variable is set.

    Returns the response text produced by the model.
    """
    # Client automatically reads ANTHROPIC_API_KEY from environment variables
    client = Anthropic()

    # Construct the thinking parameter dictionary
    thinking_param = {"type": "enabled", "budget_tokens": thinking_budget_tokens}

    try:
        message = client.messages.create(
            model=model,
            system=system_prompt,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            thinking=thinking_param,  # type: ignore
            # timeout=0, # Default timeout is 10 minutes, configure if needed
        )

        # Extract text content, ignoring thinking blocks for the final output
        response_parts: List[str] = []
        content_block: Union[TextBlock, ThinkingBlock]
        for content_block in message.content:
            if content_block.type == "text":
                response_parts.append(content_block.text)
            elif content_block.type == "thinking":
                # Optionally log thinking steps here if desired
                # click.echo(f"Thinking: {content_block.thinking}", err=True)
                pass  # Ignore thinking blocks in final concatenated output

        return "".join(response_parts)

    except Exception as e:
        # Re-raise for the main function to catch and report
        raise RuntimeError(f"Anthropic API call failed: {e}") from e


if __name__ == "__main__":
    main()
