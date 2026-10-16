# Paperwave

Turn research into understanding.

A minimal Python backend that turns research PDFs into English audio lectures: extract text, generate a transcript with OpenAI, then narrate it.

Requires Python 3.11+ and `OPENAI_API_KEY` set in your environment.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
python -m paperwave generate paper.pdf --duration 45 --level advanced
```

On macOS/Linux, activate with `source .venv/bin/activate` instead.

Saves `transcript.txt` and `audio.wav` in `output/<PDF_NAME>/`. No FFmpeg required.

- `--duration`: approximate target minutes (default: `45`).
- `--level`: `beginner`, `intermediate`, `advanced` (default), or `expert`.
- `--output`: output directory; choose a new one to regenerate existing output.
- `--voice`: narration voice (default: `marin`).
- `--model`: transcript model (default: `gpt-5-mini`).

Narration uses `gpt-4o-mini-tts`. Scanned PDFs need OCR first; encrypted PDFs must be unlocked.
