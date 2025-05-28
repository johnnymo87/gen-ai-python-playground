import os
import sys
from datetime import datetime
from typing import IO, Any, Dict

import click
from anthropic import Anthropic


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

            # Call the streaming API and process the stream
            token_usage = stream_claude_response(
                prompt=prompt,
                system_prompt=system_prompt,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                thinking_budget_tokens=thinking_budget_tokens,
                conv_log_writer=conv_f,
                resp_log_writer=resp_f,
            )

        click.echo("\nResponse stream finished.")  # Add a newline after streaming

        # Print token usage information
        click.echo("\n--- Token Usage ---")
        click.echo(f"Input tokens: {token_usage.get('input_tokens', 0)}")
        click.echo(f"Output tokens: {token_usage.get('output_tokens', 0)}")
        click.echo(
            f"Cache creation tokens: "
            f"{token_usage.get('cache_creation_input_tokens', 0)}"
        )
        click.echo(
            f"Cache read tokens: {token_usage.get('cache_read_input_tokens', 0)}"
        )
        total_input = (
            token_usage.get("input_tokens", 0)
            + token_usage.get("cache_creation_input_tokens", 0)
            + token_usage.get("cache_read_input_tokens", 0)
        )
        click.echo(f"Total input tokens: {total_input}")
        click.echo(f"Total tokens: {total_input + token_usage.get('output_tokens', 0)}")
        click.echo("-------------------\n")

        click.echo(f"Conversation appended to {conversation_path}")
        click.echo(f"Response written to {response_path}")

    except OSError as exc:
        click.echo(f"Error opening or writing log files: {exc}", err=True)
        sys.exit(1)
    except Exception as exc:
        # Catch potential API errors raised from stream_claude_response
        click.echo(f"\nError during API call or streaming: {exc}", err=True)
        sys.exit(1)


def stream_claude_response(
    prompt: str,
    system_prompt: str,
    model: str,
    temperature: float,
    max_tokens: int,
    thinking_budget_tokens: int,
    conv_log_writer: IO[str],
    resp_log_writer: IO[str],
) -> Dict[str, Any]:
    """
    Streams the Claude API response using the Anthropic Python SDK.
    Writes text chunks to the provided file writers.
    Make sure the ANTHROPIC_API_KEY environment variable is set.
    Returns a dictionary with token usage information.
    """
    # Client automatically reads ANTHROPIC_API_KEY from environment variables
    client = Anthropic()

    # Construct the thinking parameter dictionary
    thinking_param = {"type": "enabled", "budget_tokens": thinking_budget_tokens}

    # Initialize token usage tracking
    token_usage: Dict[str, Any] = {}

    try:
        # Use the stream context manager
        with client.messages.stream(
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
        ) as stream:
            # Process events to capture token usage
            for event in stream:
                # Capture token usage from message_start event
                if event.type == "message_start" and hasattr(event, "message"):
                    if hasattr(event.message, "usage"):
                        usage = event.message.usage
                        token_usage.update(
                            {
                                "input_tokens": getattr(usage, "input_tokens", 0),
                                "output_tokens": getattr(usage, "output_tokens", 0),
                                "cache_creation_input_tokens": getattr(
                                    usage, "cache_creation_input_tokens", 0
                                ),
                                "cache_read_input_tokens": getattr(
                                    usage, "cache_read_input_tokens", 0
                                ),
                            }
                        )

                # Update token usage from message_delta event (cumulative)
                elif event.type == "message_delta" and hasattr(event, "usage"):
                    delta_usage = event.usage
                    token_usage.update(
                        {
                            "output_tokens": getattr(
                                delta_usage,
                                "output_tokens",
                                token_usage.get("output_tokens", 0),
                            ),
                        }
                    )

                # Process text content
                elif event.type == "content_block_delta":
                    delta = event.delta
                    if delta.type == "text_delta":
                        text = delta.text
                        # Write to response log file
                        resp_log_writer.write(text)
                        resp_log_writer.flush()

                        # Write to conversation log file
                        conv_log_writer.write(text)
                        conv_log_writer.flush()

            # --- Optional: Handle other event types if needed ---
            # You can iterate through raw events instead of text_stream
            # for event in stream:
            #     if event.type == "text":
            #         # ... write event.text ...
            #     elif event.type == "thinking":
            #         click.echo(f"\n[Thinking: {event.thinking}]", err=True)
            #     elif event.type == "message_start":
            #         click.echo(
            #               f"\n[Message Start - ID: {event.message.id}]",
            #               err=True
            #         )
            #     elif event.type == "content_block_start":
            #         click.echo(
            #               f"\n[Content Block Start - Type: ",
            #               "{event.content_block.type}]",
            #               err=True
            #         )
            #     elif event.type == "content_block_delta":
            #         if event.delta.type == "text_delta":
            #             # ... write event.delta.text ...
            #     elif event.type == "content_block_stop":
            #         click.echo(f"\n[Content Block Stop]", err=True)
            #     elif event.type == "message_delta":
            #         click.echo(
            #               "\n[Message Delta - Stop Reason: ",
            #               f"{event.delta.stop_reason}]",
            #               err=True
            #         )
            #     elif event.type == "message_stop":
            #         click.echo(f"\n[Message Stop]", err=True)
            # ----------------------------------------------------

    except Exception as e:
        # Re-raise for the main function to catch and report
        # Add more specific error handling (e.g., APIStatusError) if needed
        raise RuntimeError(f"Anthropic API stream failed: {e}") from e

    return token_usage


if __name__ == "__main__":
    main()
