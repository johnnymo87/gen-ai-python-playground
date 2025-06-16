"""
CLI for calling Gemini (or any other hosted) models on Vertex AI.

Auth: uses Google Cloud ADC.
Either run `gcloud auth application-default login` locally
or point GOOGLE_APPLICATION_CREDENTIALS at a service-account JSON file.
"""

import os
from datetime import datetime
from pathlib import Path

import click
import vertexai  # type: ignore[import-untyped]

# --- Anthropic models ---
from anthropic import AnthropicVertex

from anthropic_common.streaming import print_token_usage, stream_anthropic_response
from gemini_common.api import get_gemini_response_via_vertex


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
    "--thinking-budget",
    type=int,
    default=8192,
    help=(
        "Thinking budget in tokens for Gemini and Claude models. "
        "Default: 8192 tokens. Set to 0 to disable thinking."
    ),
)
@click.option(
    "--project",
    envvar="GOOGLE_CLOUD_PROJECT",
    required=True,
    help="GCP project ID (env var fallback).",
)
@click.option(
    "--location",
    envvar="GOOGLE_CLOUD_LOCATION",
    required=True,
    help="GCP location (env var fallback).",
)
def main(
    prompt_file: str,
    system_prompt_file: str,
    model: str,
    temperature: float,
    max_tokens: int,
    thinking_budget: int,
    project: str,
    location: str,
) -> None:
    # ---------- read prompts ----------
    prompt = Path(prompt_file).read_text(encoding="utf-8")
    system_prompt = Path(system_prompt_file).read_text(encoding="utf-8")

    # ---------- Dispatch by publisher ----------
    if model.startswith(("claude", "publishers/anthropic")):
        client = AnthropicVertex(project_id=project, region="us-east5")
        max_tokens = min(max_tokens, 32000)  # Anthropic has a hard limit.

        # --- stream the response ---
        response_text, token_usage = stream_anthropic_response(
            client=client,
            prompt=prompt,
            system_prompt=system_prompt,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            # Extended thinking for Claude models
            thinking_budget_tokens=thinking_budget,
            conv_log_writer=None,  # We'll handle logging after
            resp_log_writer=None,  # We'll handle logging after
            echo_to_terminal=True,
        )

        # Print token usage information
        print_token_usage(token_usage)

    elif model.startswith("gemini"):
        vertexai.init(project=project, location=location)
        max_tokens = min(max_tokens, 65535)  # Gemini has a hard limit.

        response_text = get_gemini_response_via_vertex(
            prompt=prompt,
            system_prompt=system_prompt,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            thinking_budget_tokens=thinking_budget,
        )

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
