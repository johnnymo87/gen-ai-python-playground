import os
import sys
from datetime import datetime

import click
from anthropic import Anthropic

from anthropic_common.streaming import print_token_usage, stream_anthropic_response


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
    default="claude-sonnet-4-20250514",
    help="Anthropic model name to use. Defaults to claude-sonnet-4-20250514.",
)
@click.option(
    "--temperature",
    type=float,
    default=1.0,
    help=(
        "Temperature for generation. Defaults to 1.0, since that's "
        "required for reasoning models."
    ),
)
@click.option(
    "--max-tokens",
    type=int,
    default=32000,
    help="Maximum tokens in the response. Defaults to 32000.",
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
    streaming the response to log files.
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

    # Prepare for logging
    os.makedirs("log", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    base_name = os.path.splitext(os.path.basename(prompt_file))[0]
    conversation_path = os.path.join("log", f"{base_name}_conversation")
    response_path = os.path.join("log", f"{base_name}_claude_response_{timestamp}")

    try:
        # Open files for writing/appending *before* the API call
        with (
            open(conversation_path, "a", encoding="utf-8") as conv_f,
            open(response_path, "w", encoding="utf-8") as resp_f,
        ):

            # Log the prompt part to the conversation file
            conv_f.write(f"--- Prompt: {timestamp} ---\n")
            conv_f.write(f"{prompt}\n")
            conv_f.write(f"--- Response: {timestamp} ---\n")
            conv_f.flush()  # Ensure prompt is written before response starts

            # Create Anthropic client
            client = Anthropic()

            # Call the streaming API and process the stream
            response_text, token_usage = stream_anthropic_response(
                client=client,
                prompt=prompt,
                system_prompt=system_prompt,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                thinking_budget_tokens=thinking_budget_tokens,
                conv_log_writer=conv_f,
                resp_log_writer=resp_f,
                echo_to_terminal=False,  # We'll handle terminal output ourselves
            )

        click.echo("\nResponse stream finished.")  # Add a newline after streaming

        # Print token usage information
        print_token_usage(token_usage)

        click.echo(f"Conversation appended to {conversation_path}")
        click.echo(f"Response written to {response_path}")

    except OSError as exc:
        click.echo(f"Error opening or writing log files: {exc}", err=True)
        sys.exit(1)
    except Exception as exc:
        # Catch potential API errors raised from stream_anthropic_response
        click.echo(f"\nError during API call or streaming: {exc}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
