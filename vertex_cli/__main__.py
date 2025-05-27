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
    default=65535,
    help="Maximum tokens in the response. Defaults to 65535.",
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
    default="us-central1",
    help="Vertex region (env var fallback).",
)
def main(
    prompt_file: str,
    system_prompt_file: str,
    model: str,
    temperature: float,
    max_tokens: int,
    project: str,
    location: str,
) -> None:
    # ---------- read prompts ----------
    prompt = Path(prompt_file).read_text(encoding="utf-8")
    system_prompt = Path(system_prompt_file).read_text(encoding="utf-8")

    # ---------- Vertex call ----------
    vertexai.init(project=project, location=location)

    gen_model = GenerativeModel(
        model_name=model,
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
