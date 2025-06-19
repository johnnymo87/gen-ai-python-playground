import os
import sys
from datetime import datetime

import click

from gemini_common.api import get_gemini_response_via_genai


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
    help="Path to a file containing the prompt text.",
)
@click.option(
    "--model",
    default="gemini-2.5-pro",
    help="Gemini model name to use.",
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
    default=1000000,
    help="Maximum tokens in the response. Defaults to a million.",
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
    Use the prompt text from --prompt to query the Gemini API via
    google-genai, then store both the prompt and response in a conversation
    log file, and write out a separate file containing just the response.
    """
    # Read the prompt from the specified file
    try:
        with open(prompt_file, "r", encoding="utf-8") as f:
            prompt = f.read()
    except OSError as exc:
        click.echo(f"Error reading prompt file: {exc}", err=True)
        sys.exit(1)

    # Read the system prompt from the specified file
    try:
        with open(system_prompt_file, "r", encoding="utf-8") as f:
            system_prompt = f.read()
    except OSError as exc:
        click.echo(f"Error reading system prompt file: {exc}", err=True)
        sys.exit(1)

    # Make the request to Gemini
    try:
        response_text = get_gemini_response_via_genai(
            prompt=prompt,
            system_prompt=system_prompt,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            thinking_budget_tokens=thinking_budget_tokens,
        )
    except Exception as exc:
        click.echo(f"Error calling Gemini API: {exc}", err=True)
        sys.exit(1)

    # Prepare for logging
    os.makedirs("log", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    base_name = os.path.splitext(os.path.basename(prompt_file))[0]

    # Append the prompt and response to a conversation log
    conversation_path = os.path.join("log", f"{base_name}_conversation")
    try:
        with open(conversation_path, "a", encoding="utf-8") as conv_f:
            conv_f.write(f"{timestamp}\n")
            conv_f.write(f"{prompt}\n")
            conv_f.write(f"{timestamp}\n")
            conv_f.write(f"{response_text}\n")
    except OSError as exc:
        click.echo(f"Error writing conversation log: {exc}", err=True)

    # Write the response separately, with a timestamped filename
    response_path = os.path.join("log", f"{base_name}_gemini_response_{timestamp}")
    try:
        with open(response_path, "w", encoding="utf-8") as resp_f:
            resp_f.write(response_text)
    except OSError as exc:
        click.echo(f"Error writing response file: {exc}", err=True)

    click.echo(response_text)


if __name__ == "__main__":
    main()
