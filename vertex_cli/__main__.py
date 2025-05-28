"""
CLI for calling Gemini (or any other hosted) models on Vertex AI.

Auth: uses Google Cloud ADC.
Either run `gcloud auth application-default login` locally
or point GOOGLE_APPLICATION_CREDENTIALS at a service-account JSON file.
"""

import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import click
import vertexai  # type: ignore[import-untyped]

# --- Anthropic models ---
from anthropic import AnthropicVertex

# --- Google models ---
from vertexai.generative_models import (  # type: ignore[import-untyped]
    GenerationConfig,
    GenerativeModel,
)


# ---------- CLI ----------
@click.command()
@click.option("--prompt-file", type=click.Path(exists=True), required=True)
@click.option(
    "--system-prompt-file",
    type=click.Path(exists=True),
    default="system_prompts/coding_000000",
)
@click.option(
    "--model",
    default="gemini-2.5-pro-preview-05-06",
    help="Any model you have permission to call in Vertex.",
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
    help=(
        "Maximum tokens in the response. Defaults to a million, "
        "but overriden to provider-specific limits."
    ),
)
@click.option(
    "--project",
    envvar="GOOGLE_CLOUD_PROJECT",
    required=True,
    help="GCP project ID (env var fallback).",
)
def main(
    prompt_file: str,
    system_prompt_file: str,
    model: str,
    temperature: float,
    max_tokens: int,
    project: str,
) -> None:
    # ---------- read prompts ----------
    prompt = Path(prompt_file).read_text(encoding="utf-8")
    system_prompt = Path(system_prompt_file).read_text(encoding="utf-8")

    # ---------- Dispatch by publisher ----------
    if model.startswith(("claude", "publishers/anthropic")):
        client = AnthropicVertex(project_id=project, region="us-east5")
        max_tokens = min(max_tokens, 32000)  # Anthropic has a hard limit.

        # --- stream the response ---
        text_chunks: list[str] = []
        token_usage: Dict[str, Any] = {}

        with client.messages.stream(
            model=model,  # e.g. "claude-opus-4"
            system=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        ) as stream:
            for event in stream:  # yields SSE events
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

                elif event.type == "content_block_delta":
                    delta = event.delta
                    if delta.type == "text_delta":
                        piece = delta.text
                        click.echo(piece, nl=False)  # live to terminal
                        text_chunks.append(piece)

        response_text = "".join(text_chunks)

        # Print token usage information
        click.echo("\n\n--- Token Usage ---")
        click.echo(f"Input tokens: {token_usage.get('input_tokens', 0)}")
        click.echo(f"Output tokens: {token_usage.get('output_tokens', 0)}")
        click.echo(
            "Cache creation tokens: "
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

    elif model.startswith("gemini"):
        vertexai.init(project=project, location="us-central1")
        max_tokens = min(max_tokens, 65535)  # Gemini has a hard limit.

        gen_model = GenerativeModel(
            model_name=model,  # e.g. "gemini-2.5-pro-preview-05-06"
            system_instruction=system_prompt,
        )
        generation_config = GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
        )
        response = gen_model.generate_content(
            prompt,
            generation_config=generation_config,
            stream=False,
        )

        # Print the model dump JSON for debugging
        click.echo(response.model_dump_json())

        response_text = response.text
    else:
        raise ValueError(
            f"Unsupported model: {model}. "
            "Only Gemini and Claude models are supported at this time."
        )

    # ---------- log exactly like other modules ----------
    ts = datetime.now().strftime("%Y%m%d%H%M%S")
    base = Path(prompt_file).stem
    os.makedirs("log", exist_ok=True)
    conv_path = Path("log") / f"{base}_conversation"
    resp_path = Path("log") / f"{base}_vertex_response_{ts}"

    with conv_path.open("a", encoding="utf-8") as conv:
        conv.write(f"{ts}\n{prompt}\n{ts}\n{response_text}\n")
    resp_path.write_text(response_text, encoding="utf-8")

    click.echo(response_text)


if __name__ == "__main__":
    main()
