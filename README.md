# Audio Performance Rubric Evaluator

This repository contains a single-purpose script, `audio_analysis_smoke.py`, that scores an audio performance against the Voices Performance Rubric by calling an OpenAI audio-capable chat model. The script handles audio encoding, multi-part prompting, JSON-only enforcement, continuation handling when responses are truncated, and timestamped result storage.

## Repository Contents
- `audio_analysis_smoke.py` - main runner that sends an audio file plus rubric instructions to the model and persists the response.
- `sample.wav` - example input that you can use to test the workflow.
- `analysis_result_*.json` / `analysis_result_raw*.txt` - previously saved outputs illustrating what the script produces.

## Requirements
1. **Python 3.10+** (earlier versions may work but are untested).
2. **Dependencies**: `openai` Python SDK.
   ```bash
   python -m pip install --upgrade openai
   ```
3. **OpenAI API key** with access to an audio-capable chat model (defaults to `gpt-4o-audio-preview`). Set the key in your shell before running:
   - macOS/Linux: `export OPENAI_API_KEY=sk-...`
   - Windows (PowerShell): `$env:OPENAI_API_KEY='sk-...'`

## Step-by-Step: How the Script Works
1. **Configuration** (`MODEL`, `AUDIO_PATH`, `MAX_TOKENS_*`, `TEMPERATURE`): set near the top of `audio_analysis_smoke.py`. Adjust `AUDIO_PATH` if you want to score your own `.wav` or `.mp3`.
2. **Audio encoding** (`encode_audio_for_api`): validates the file exists, enforces `.wav/.mp3`, and returns the base64 payload required by the API.
3. **Prompt construction**: the script builds a system message that forces JSON-only replies and a user message that includes the full rubric plus the `input_audio` block.
4. **Primary model call** (`call_model`): submits the request with `modalities=["text", "audio"]` so the API knows to expect audio input/output.
5. **Response parsing** (`extract_text_from_response` + `parse_json_or_raise`): normalizes SDK response shapes, strips any non-JSON noise, and attempts to decode the payload into a Python dict.
6. **Continuation safety net** (`continue_if_truncated`): if the first reply is truncated or unparsable, a second request asks the model to finish the JSON, seeding it with the partial text.
7. **Result persistence** (`save_json`, `save_raw_text`): outputs now land in `Results/`, using the audio filename stem (e.g., `sample.json`, `sample_raw.txt`). That way every run stays grouped beside its artifacts.
8. **Console preview**: after saving, the script prints the path of the JSON file, the `finish_reason` returned by the API, and the first ~1200 characters of the model output for quick inspection.

## Running the Script
```bash
python audio_analysis_smoke.py
```
Expected output:
- A new `analysis_result_YYYY-MM-DD_HH-MM-SS.json` file with the full 26-metric rubric evaluation.
- Optional `analysis_result_raw*.txt` files if the first attempt needed manual inspection.

## Customizing & Tips
- **Different audio**: set `AUDIO_PATH` to your clip or pass it via environment variable logic if you extend the script.
- **Model / token tweaks**: raise `MAX_TOKENS_PRIMARY` or `MAX_TOKENS_CONTINUE` if you see repeated truncations; lower them if cost is a concern.
- **Automation**: wrap the script in a scheduler or loop if you need to score multiple takes - just be mindful of API rate limits and output file growth.
- **Error handling**: the script `sys.exit`s with helpful messages when prerequisites (API key, audio file) are missing, so start there if it stops early.

## Troubleshooting Checklist
- `OPENAI_API_KEY is not set`: run the export command in the same shell session before invoking the script.
- `Audio file not found`: confirm the relative path (default `sample.wav`) or provide an absolute path.
- `Could not parse JSON even after continuation`: inspect the saved raw `.txt` files, then consider increasing token limits or simplifying the rubric prompt.

With the steps above you can drop in any performance clip, run `audio_analysis_smoke.py`, and receive a rubric-aligned, machine-readable evaluation in a single file.
