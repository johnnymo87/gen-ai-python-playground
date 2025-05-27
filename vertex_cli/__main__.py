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
        with client.messages.stream(
            model=model,  # e.g. "claude-opus-4"
            system=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        ) as stream:
            for event in stream:  # yields SSE events
                if event.type == "content_block_delta":
                    delta = event.delta
                    if delta.type == "text_delta":
                        piece = delta.text
                        click.echo(piece, nl=False)  # live to terminal
                        text_chunks.append(piece)

        response_text = "".join(text_chunks)
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
