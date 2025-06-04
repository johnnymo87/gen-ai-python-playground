"""Common API utilities for Gemini."""

import os

import click
from google import genai
from google.genai import types


def _get_gemini_response(
    client: genai.Client,
    prompt: str,
    system_prompt: str,
    model: str,
    temperature: float,
    max_tokens: int,
    thinking_budget_tokens: int,
) -> str:
    """
    Common function to get a response from Gemini using a provided client.

    Args:
        client: The Gemini client instance
        prompt: The user prompt text
        system_prompt: The system instructions
        model: The model name
        temperature: Generation temperature
        max_tokens: Maximum tokens to generate
        thinking_budget_tokens: Thinking budget in tokens (set to 0 to disable thinking)
    """
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


def get_gemini_response_via_genai(
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

    return _get_gemini_response(
        client=client,
        prompt=prompt,
        system_prompt=system_prompt,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        thinking_budget_tokens=thinking_budget_tokens,
    )


def get_gemini_response_via_vertex(
    prompt: str,
    system_prompt: str,
    model: str,
    temperature: float,
    max_tokens: int,
    thinking_budget_tokens: int,
) -> str:
    """
    Uses the Gemini models via Vertex AI to produce a response for the provided prompt.
    Requires proper Google Cloud authentication via ADC.

    Args:
        prompt: The user prompt text
        system_prompt: The system instructions
        model: The Vertex AI model name
        temperature: Generation temperature
        max_tokens: Maximum tokens to generate
        thinking_budget_tokens: Thinking budget in tokens
            (set to 0 to disable thinking)
    """
    # Create the client for Gemini
    client = genai.Client()

    return _get_gemini_response(
        client=client,
        prompt=prompt,
        system_prompt=system_prompt,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        thinking_budget_tokens=thinking_budget_tokens,
    )
