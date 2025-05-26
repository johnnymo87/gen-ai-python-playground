import os
import sys
from datetime import datetime

import click
from google import genai  # type: ignore[import-untyped]
from google.genai import types  # type: ignore[import-untyped]


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
    default="gemini-2.5-pro-preview-05-06",
    help="Gemini model name to use. Defaults to gemini-2.5-pro-preview-05-06.",
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
        response_text = get_gemini_response(
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

    click.echo(f"Response written to {response_path}")


def get_gemini_response(
    prompt: str,
    system_prompt: str,
    model: str,
    temperature: float,
    max_tokens: int,
    thinking_budget_tokens: int,
) -> str:
    """
    Uses the Gemini Developer API via the google-genai library to produce a response
    for the provided prompt. Make sure you have the environment variable:
      GOOGLE_API_KEY
    set to your Gemini Developer API key.
    """
    # Ensure the environment variable is set
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY environment variable not set.")

    # Create the client for Gemini
    client = genai.Client(api_key=api_key)

    # Generate text via Gemini
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=temperature,
            max_output_tokens=max_tokens,
            thinking_config=types.ThinkingConfig(
                thinking_budget=thinking_budget_tokens
            ),
        ),
    )

    # Print the model dump JSON for debugging
    click.echo(response.model_dump_json())

    # Return the text from the first (and typically only) candidate
    return response.text or ""


if __name__ == "__main__":
    main()
