import os
import sys
from datetime import datetime

import click
from langchain_anthropic import ChatAnthropic


@click.command()
@click.option(
    "--input-file",
    type=click.Path(exists=True),
    required=True,
    help="Path to a file containing the prompt text.",
)
@click.option(
    "--max-tokens",
    type=int,
    default=20000,
    help="Maximum tokens to sample for the model's response.",
)
@click.option(
    "--thinking-budget-tokens",
    type=int,
    default=16000,
    help="Budget for the model's 'thinking' tokens.",
)
def main(input_file: str, max_tokens: int, thinking_budget_tokens: int) -> None:
    """
    Use the prompt text from --input-file to query the Claude API via LangChain,
    then store both the prompt and response in a conversation log file, and also
    write out a separate file containing just the response.
    """
    # Read the prompt from the specified file
    try:
        with open(input_file, "r") as f:
            prompt = f.read()
    except OSError as exc:
        click.echo(f"Error reading prompt file: {exc}", err=True)
        sys.exit(1)

    # Call the Claude API
    try:
        response_text = get_claude_response(prompt, max_tokens, thinking_budget_tokens)
    except Exception as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)

    # Echo the response to stdout
    click.echo(response_text)

    # Prepare for logging
    os.makedirs("log", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    base_name = os.path.splitext(os.path.basename(input_file))[0]

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
    response_path = os.path.join("log", f"{base_name}_openai_response_{timestamp}")
    try:
        with open(response_path, "w", encoding="utf-8") as resp_f:
            resp_f.write(response_text)
    except OSError as exc:
        click.echo(f"Error writing response file: {exc}", err=True)


def get_claude_response(
    prompt: str, max_tokens: int, thinking_budget_tokens: int
) -> str:
    """
    Uses the Claude API via LangChain's Anthropic integration to process the
    given prompt. Make sure the ANTHROPIC_API_KEY environment variable is set.

    Returns a single concatenated string if multiple content blocks are returned.
    """
    llm = ChatAnthropic(
        max_tokens_to_sample=max_tokens,
        model_name="claude-3-7-sonnet-latest",
        stop=None,
        temperature=1,
        thinking={"type": "enabled", "budget_tokens": thinking_budget_tokens},
        timeout=0,
    )

    messages = [
        (
            "system",
            """
            As a general note when replying to me with code, for every file that
            needs to change, just write out the entire file for me, or at least large
            relevant chunks of it, so I can copy-paste it to my local file system.

            Never ever send me a diff or a patch file, even if I provide you with
            one. I will not be able to apply it. Instead, just send me the entire
            file(s) that need to change.

            However, in order to facilitate rapid code reviews, let's not change
            unrelated code for e.g. style reasons.
            """,
        ),
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
