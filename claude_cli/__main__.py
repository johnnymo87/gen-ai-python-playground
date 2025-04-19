import os
import sys
from datetime import datetime

import click
from langchain_anthropic import ChatAnthropic


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
    default="claude-3-7-sonnet-latest",
    help="Anthropic model name to use. Defaults to claude-3-7-sonnet-latest.",
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
    Use the prompt text from --prompt-file (and a system prompt from
    --system-prompt-file) to query the Claude API via LangChain, then store
    both the prompt and response in a conversation log file, and also write
    out a separate file containing just the response.
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

    # Call the Claude API
    try:
        response_text = get_claude_response(
            prompt,
            system_prompt,
            model,
            temperature,
            max_tokens,
            thinking_budget_tokens,
        )
    except Exception as exc:
        click.echo(f"Error: {exc}", err=True)
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
            conv_f.write(f"{timestamp}\n")
            conv_f.write(f"{prompt}\n")
            conv_f.write(f"{timestamp}\n")
            conv_f.write(f"{response_text}\n")
    except OSError as exc:
        click.echo(f"Error writing conversation log: {exc}", err=True)

    # Write the response separately, with a timestamped filename
    response_path = os.path.join("log", f"{base_name}_claude_response_{timestamp}")
    try:
        with open(response_path, "w", encoding="utf-8") as resp_f:
            resp_f.write(response_text)
    except OSError as exc:
        click.echo(f"Error writing response file: {exc}", err=True)


def get_claude_response(
    prompt: str,
    system_prompt: str,
    model: str,
    temperature: float,
    max_tokens: int,
    thinking_budget_tokens: int,
) -> str:
    """
    Uses the Claude API via LangChain's Anthropic integration to process the
    given prompt. Make sure the ANTHROPIC_API_KEY environment variable is set.

    Returns the response text produced by the model.
    """
    llm = ChatAnthropic(
        max_tokens_to_sample=max_tokens,
        model_name=model,
        stop=None,
        temperature=temperature,
        thinking={"type": "enabled", "budget_tokens": thinking_budget_tokens},
        timeout=0,
    )

    messages = [
        ("system", system_prompt),
        ("human", prompt),
    ]
    response = llm.invoke(messages)
    content = response.content

    if isinstance(content, str):
        return content
    else:
        joined = []
        for block in content:
            if isinstance(block, str):
                joined.append(block)
            elif isinstance(block, dict) and "text" in block:
                joined.append(block["text"])
            else:
                joined.append(str(block))
        return "".join(joined)


if __name__ == "__main__":
    main()
