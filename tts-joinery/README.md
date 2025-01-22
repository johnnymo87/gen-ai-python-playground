# Using `tts-joinery` for Text-to-Speech

[`tts-joinery`](https://github.com/drien/tts-joinery) is a Python library and CLI tool that overcomes text length limitations in text-to-speech APIs by chunking the input text and stitching the audio outputs together.

### Prerequisites

- **FFmpeg**: Required for audio file processing.

  - **macOS:**

    ```bash
    brew install ffmpeg
    ```

  - **Ubuntu/Linux:**

    ```bash
    sudo apt-get install ffmpeg
    ```

  - **Windows:**

    Download and install from the [FFmpeg website](https://ffmpeg.org/download.html).

### Ensure API Key is Set

Your `OPENAI_API_KEY` should be set in your `.envrc` file and loaded using `direnv`.

### Basic Usage

Generate speech from a text file:

```bash
ttsjoin --input-file ~/Documents/hope-for-cynics/hope-for-cynics-chapter-01-signs-and-symptoms.txt \
        --output-file speech.mp3 \
        --model tts-1-hd \
        --voice ash
```

### Options and Flags

- `--input-file`: Path to the text file to convert.
- `--output-file`: Name of the output audio file.
- `--model`: TTS model to use (`tts-1`, `tts-1-hd`, etc.).
- `--voice`: Voice to use (`alloy`, `coral`, `onyx`, etc.).
- `--no-cache`: Disable caching of audio chunks.

### Advanced Usage

**Using Standard Input and Output:**

```bash
cat input.txt | ttsjoin --model tts-1-hd --voice coral > output.mp3
```

**Clearing the Cache:**

```bash
ttsjoin cache clear
```
