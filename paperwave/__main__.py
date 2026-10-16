"""Run the Paperwave command-line interface."""

import argparse
from pathlib import Path

from openai import OpenAIError
from pypdf.errors import PdfReadError

from .pipeline import LEVELS, generate


def main() -> None:
    """Parse generation settings and run the PDF-to-audio pipeline."""
    parser = argparse.ArgumentParser(prog="paperwave", description="Turn a PDF into an audio lecture.")
    commands = parser.add_subparsers(dest="command", required=True)
    command = commands.add_parser("generate", help="Generate a transcript and AI-narrated WAV.")
    command.add_argument("pdf", type=Path)
    command.add_argument("--duration", type=int, default=45, help="Target minutes (approximate).")
    command.add_argument("--level", choices=LEVELS, default="advanced")
    command.add_argument("--output", type=Path, help="Output directory (default: output/PDF_NAME).")
    command.add_argument("--voice", default="marin", help="OpenAI speech voice (default: marin).")
    command.add_argument("--model", default="gpt-5-mini", help="OpenAI transcript model.")
    args = parser.parse_args()
    try:
        transcript, audio = generate(
            args.pdf,
            duration=args.duration,
            level=args.level,
            output=args.output,
            voice=args.voice,
            model=args.model,
        )
    except (OSError, ValueError, RuntimeError, PdfReadError, OpenAIError) as error:
        parser.exit(1, f"Error: {error}\n")
    print(f"Transcript: {transcript}\nAI-generated audio: {audio}")


if __name__ == "__main__":
    main()
