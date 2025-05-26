import os
import sys
from datetime import datetime

import click
from openai import OpenAI


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
    default="gpt-4o-mini",
    help="OpenAI model name to use. Defaults to gpt-4o-mini.",
)
@click.option(
    "--temperature",
    type=float,
    default=1.0,
    help=(
        "Temperature for generation. Defaults to 1.0, since that's required ",
        "for reasoning models.",
    ),
)
@click.option(
    "--max-tokens",
    type=int,
    default=16000,
    help="Maximum tokens in the response. Defaults to 16000.",
)
def main(
    prompt_file: str,
    system_prompt_file: str,
    model: str,
    temperature: float,
    max_tokens: int,
) -> None:
    """
    Use the prompt text from --prompt-file (and a system prompt from
    --system-prompt-file) to query the OpenAI API, then store both the prompt
    and response in a conversation log file, and also write out a separate
    file containing just the response.
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

    # Call the OpenAI API
    try:
        response_text = get_openai_response(
            prompt, system_prompt, model, temperature, max_tokens
        )
    except Exception as exc:
        click.echo(f"Error calling OpenAI API: {exc}", err=True)
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
            conv_f.write(f"--- Prompt: {timestamp} ---\n")
            conv_f.write(f"{prompt}\n")
            conv_f.write(f"--- Response: {timestamp} ---\n")
            conv_f.write(f"{response_text}\n\n")
    except OSError as exc:
        click.echo(f"Error writing conversation log: {exc}", err=True)

    # Write the response separately, with a timestamped filename
    response_path = os.path.join("log", f"{base_name}_openai_response_{timestamp}")
    try:
        with open(response_path, "w", encoding="utf-8") as resp_f:
            resp_f.write(response_text)
    except OSError as exc:
        click.echo(f"Error writing response file: {exc}", err=True)


def get_openai_response(
    prompt: str,
    system_prompt: str,
    model: str,
    temperature: float,
    max_tokens: int,
) -> str:
    """
    Uses the OpenAI API via the openai-python library to produce a response.
    Requires the OPENAI_API_KEY environment variable to be set.

    Returns the response text produced by the model.
    """
    # OpenAI client automatically reads the API key from the OPENAI_API_KEY env var
    client = OpenAI()

    try:
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            max_completion_tokens=max_tokens,
        )

        if completion.choices and completion.choices[0].message:
            content = completion.choices[0].message.content
            return content if content else ""
        else:
            click.echo(
                f"Warning: Received unexpected response structure: {completion}",
                err=True,
            )
            return ""
    except Exception as e:
        # Re-raise a more specific error or handle as needed
        raise RuntimeError(f"OpenAI API call failed: {e}") from e


if __name__ == "__main__":
    main()
