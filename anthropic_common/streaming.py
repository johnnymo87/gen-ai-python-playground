"""Common streaming utilities for Anthropic API."""

from typing import IO, TYPE_CHECKING, Any, Dict, Optional, Tuple, Union

# Runtime imports with type checking separation
import anthropic  # noqa: E402
import click
from anthropic.types import MessageParam  # noqa: E402, F401

# Define type aliases for better readability
if TYPE_CHECKING:
    from anthropic import Anthropic, AnthropicVertex  # noqa: F401

AnthropicClient = Union["anthropic.Anthropic", "anthropic.AnthropicVertex"]


def stream_anthropic_response(
    client: AnthropicClient,
    prompt: str,
    system_prompt: str,
    model: str,
    temperature: float,
    max_tokens: int,
    thinking_budget_tokens: int,
    conv_log_writer: Optional[IO[str]] = None,
    resp_log_writer: Optional[IO[str]] = None,
    echo_to_terminal: bool = True,
) -> Tuple[str, Dict[str, Any]]:
    """
    Streams the Anthropic API response using the provided client.

    Args:
        client: Anthropic or AnthropicVertex client instance
        prompt: User prompt text
        system_prompt: System prompt text
        model: Model name to use
        temperature: Temperature for generation
        max_tokens: Maximum tokens in the response
        thinking_budget_tokens: Budget for thinking tokens in extended thinking mode
        conv_log_writer: File writer for conversation log (optional)
        resp_log_writer: File writer for response log (optional)
        echo_to_terminal: Whether to echo response to terminal

    Returns:
        Tuple of (response_text, token_usage_dict)
    """
    # Build the messages list
    messages: list[MessageParam] = [{"role": "user", "content": prompt}]

    # Build the parameters for the API call
    api_params: Dict[str, Any] = {
        "model": model,
        "system": system_prompt,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    # Add thinking parameter if value is positive (for both direct and Vertex clients)
    if thinking_budget_tokens > 0:
        # Both direct Anthropic API and Vertex AI support the thinking parameter
        api_params["thinking"] = {
            "type": "enabled",
            "budget_tokens": thinking_budget_tokens,
        }

    # Initialize token usage tracking and text collection
    token_usage: Dict[str, Any] = {}
    text_chunks: list[str] = []

    try:
        # Use the stream context manager
        with client.messages.stream(**api_params) as stream:
            # Process events
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

                        # Collect text chunks
                        text_chunks.append(text)

                        # Echo to terminal if requested
                        if echo_to_terminal:
                            click.echo(text, nl=False)

                        # Write to response log file
                        if resp_log_writer:
                            resp_log_writer.write(text)
                            resp_log_writer.flush()

                        # Write to conversation log file
                        if conv_log_writer:
                            conv_log_writer.write(text)
                            conv_log_writer.flush()

    except Exception as e:
        # Re-raise with more context
        raise RuntimeError(f"Anthropic API stream failed: {e}") from e

    response_text = "".join(text_chunks)
    return response_text, token_usage


def print_token_usage(token_usage: Dict[str, Any]) -> None:
    """Print token usage information in a consistent format."""
    click.echo("\n\n--- Token Usage ---")
    click.echo(f"Input tokens: {token_usage.get('input_tokens', 0)}")
    click.echo(f"Output tokens: {token_usage.get('output_tokens', 0)}")
    click.echo(
        f"Cache creation tokens: "
        f"{token_usage.get('cache_creation_input_tokens', 0)}"
    )
    click.echo(f"Cache read tokens: {token_usage.get('cache_read_input_tokens', 0)}")
    total_input = (
        token_usage.get("input_tokens", 0)
        + token_usage.get("cache_creation_input_tokens", 0)
        + token_usage.get("cache_read_input_tokens", 0)
    )
    click.echo(f"Total input tokens: {total_input}")
    click.echo(f"Total tokens: {total_input + token_usage.get('output_tokens', 0)}")
    click.echo("-------------------\n")
