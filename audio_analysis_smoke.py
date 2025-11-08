import os, base64, pathlib, sys, json, re, datetime
from typing import Optional, Tuple
from openai import OpenAI

# ========= CONFIG =========
MODEL = "gpt-4o-audio-preview"   # audio-capable chat model
AUDIO_PATH = "sample.wav"        # or "sample.mp3"
MAX_TOKENS_PRIMARY = 8000        # raise to help finish
MAX_TOKENS_CONTINUE = 9000       # one-time continuation budget
TEMPERATURE = 0
# ==========================


# ---------- File & audio helpers ----------
def encode_audio_for_api(path: str) -> Tuple[str, str]:
    """Return (format, base64_data) for .wav or .mp3 file."""
    p = pathlib.Path(path)
    if not p.exists():
        sys.exit(f"[!] Audio file not found: {p.resolve()}")
    fmt = p.suffix.lower().lstrip(".")
    if fmt not in {"wav", "mp3"}:
        sys.exit("[!] Use a .wav or .mp3 file for input_audio.")
    with open(p, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return fmt, b64


# ---------- Response text extraction ----------
def extract_text_from_response(resp) -> str:
    """
    Be defensive: depending on SDK/model, the text may be:
      - choices[0].message.content (string)
      - choices[0].message.content (list with parts that have .text or ['text'])
      - fallback transcript inside message.audio.transcript (rare)
    """
    msg = resp.choices[0].message

    # Case A: plain string
    if isinstance(getattr(msg, "content", None), str) and msg.content.strip():
        return msg.content.strip()

    # Case B: content parts
    parts = getattr(msg, "content", None)
    if isinstance(parts, list):
        texts = []
        for part in parts:
            if isinstance(part, dict) and isinstance(part.get("text"), str):
                texts.append(part["text"])
            elif hasattr(part, "text") and isinstance(part.text, str):
                texts.append(part.text)
        if texts:
            return "\n".join(t.strip() for t in texts if t and t.strip())

    # Case C: rare fallback
    audio_field = getattr(msg, "audio", None)
    if audio_field and hasattr(audio_field, "transcript"):
        return audio_field.transcript

    return ""


# ---------- JSON parsing & saving ----------
def _balanced_json_block(s: str) -> Optional[str]:
    """Try to extract the largest top-level {...} block (handles extra prose)."""
    start = s.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(s)):
        c = s[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return s[start : i + 1]
    return None

def parse_json_or_raise(text: str):
    """Attempt to parse JSON robustly; raise if impossible."""
    # direct
    try:
        return json.loads(text)
    except Exception:
        pass
    # balanced block
    block = _balanced_json_block(text)
    if block:
        return json.loads(block)
    # fenced code (rare)
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL | re.IGNORECASE)
    if m:
        return json.loads(m.group(1))
    raise ValueError("Could not parse JSON from model output.")

def timestamped_name(base: str, ext: str) -> str:
    stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return f"{base}_{stamp}.{ext}"

def save_json(payload, base_name="analysis_result") -> str:
    path = timestamped_name(base_name, "json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return path

def save_raw_text(text: str, base_name="analysis_result_raw") -> str:
    path = timestamped_name(base_name, "txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(text or "")
    return path


# ---------- Prompts ----------
SYSTEM_JSON_MODE = (
    "You are an acting-performance evaluator. Return ONLY valid JSON. "
    "No prose, no code fences. The JSON must conform to the format described by the user."
)

RUBRIC_PROMPT = r"""
Rate the attached audio performance using the Voices Performance Rubric (1–7 scale). Use the EXACT anchor wordings below when selecting scores. For each metric, provide:
- score (1–7)
- anchor_descriptor (quote the matching anchor text verbatim/near-verbatim)
- rationale (≤ 25 words, tied to what you hear)
- coaching_tip (≤ 20 words, one practical step)
If evidence is obvious, include brief time notes like "00:06 breath noise" when possible.

Return ONLY this top-level JSON:
{
  "summary": { "overall_comment": "...", "strengths": ["..."], "priorities": ["..."] },
  "scores": { "<Metric Name>": { "score": n, "anchor_descriptor": "...", "rationale": "...", "coaching_tip": "..." }, ... }
}

[Performance Expression]
Emotional Clarity: 1 Emotion not conveyed/confusing ... 7 Perfectly clear, immediately identifiable
Emotional Authenticity / Believability: 1 Very Poor ... 7 Excellent (compelling, professional-grade)
Emotional Intensity / Energy: 1 No emotional energy ... 7 Powerful, precisely balanced
Subtext / Inner Life: 1 Flat, no inner life ... 7 Deeply layered, fully present
Expressive Range: 1 Monotone/flat ... 7 Exceptional, nuanced range
Character Commitment: 1 Uncommitted ... 7 Completely immersed/transformative
Spontaneity / Naturalness: 1 Mechanical ... 7 Fully natural, reactive, present
Emotional Control: 1 Uncontrolled/erratic ... 7 Perfectly controlled/precise

[Vocal & Technical]
Vocal Clarity / Diction: 1 Mumbled ... 7 Crisp, precise throughout
Projection / Presence: 1 Weak/barely audible ... 7 Commanding presence with ease
Pacing / Rhythm: 1 Rushed/dragging ... 7 Masterful rhythm; phrasing elevates text
Pitch Variation / Prosody: 1 Monotone ... 7 Highly nuanced; emphasis perfect
Breath Control / Support: 1 Poor control ... 7 Flawless, effortless control
Technical Quality (Recording): 1 Noisy/clipping/unusable ... 7 Clean, quiet, studio-grade
Vocal Tone & Resonance Quality: 1 Harsh/unnatural ... 7 Excellent; rich, warm, well-resonant
Objective Vocal Stability (Jitter/Shimmer): 1 Severely unstable ... 7 Highly stable; clean/steady tone

[Interpretive]
Text Interpretation / Understanding: 1 Surface reading ... 7 Sophisticated; emphasis elevates text
Intent / Objective Clarity: 1 Objective unclear ... 7 Exceptionally focused/compelling
Listening / Reactivity: 1 Unresponsive ... 7 Fully engaged; truthful moment-to-moment
Storytelling Arc / Emotional Journey: 1 Static ... 7 Powerful, moving arc; memorable

[Overall Impact]
Presence / Charisma: 1 Disengaging ... 7 Utterly captivating; unforgettable
Uniqueness / Creativity of Choices: 1 Generic ... 7 Inventive/memorable; elevates material
Suitability / Casting Fit: 1 Miscast ... 7 Perfect fit; ideal for brief
Overall Impression / Professional Readiness: 1 Unprepared ... 7 Exceptional; industry-leading
Performance Consistency / Stamina: 1 Highly inconsistent ... 7 Rock-solid; last take as strong as first
Listener Engagement / Empathic Resonance: 1 Disengaging ... 7 Compelling/affecting; sustained immersion
"""


# ---------- Core call ----------
def call_model(client: OpenAI, fmt: str, b64: str, max_tokens: int):
    return client.chat.completions.create(
        model=MODEL,
        modalities=["text", "audio"],                  # sending audio, so include "audio"
        audio={"voice": "alloy", "format": "wav"},     # required by audio-preview models
        messages=[
            {"role": "system", "content": SYSTEM_JSON_MODE},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": RUBRIC_PROMPT},
                    {"type": "input_audio", "input_audio": {"data": b64, "format": fmt}},
                ],
            },
        ],
        # NOTE: response_format not supported by this model
        temperature=TEMPERATURE,
        max_tokens=max_tokens,
    )


def continue_if_truncated(client: OpenAI, fmt: str, b64: str, partial_text: str, max_tokens: int):
    """
    One-time continuation request if the first reply hit the length limit or was unparsable.
    We feed back the partial JSON and ask for a COMPLETE JSON object.
    """
    continuation_instruction = (
        "You previously returned a PARTIAL JSON object. "
        "Return a SINGLE, COMPLETE JSON object that merges and completes the result. "
        "Do not repeat duplicate keys. Ensure the final JSON includes all required fields and 26 rubric metrics. "
        "Return ONLY valid JSON. No prose."
    )
    return client.chat.completions.create(
        model=MODEL,
        modalities=["text", "audio"],
        audio={"voice": "alloy", "format": "wav"},
        messages=[
            {"role": "system", "content": SYSTEM_JSON_MODE},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": RUBRIC_PROMPT},
                    {"type": "input_audio", "input_audio": {"data": b64, "format": fmt}},
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": continuation_instruction},
                    {"type": "text", "text": f"PARTIAL_JSON_START\n{partial_text}\nPARTIAL_JSON_END"},
                ],
            },
        ],
        temperature=TEMPERATURE,
        max_tokens=max_tokens,
    )


# ---------- Main ----------
if __name__ == "__main__":
    # API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        sys.exit("[!] OPENAI_API_KEY is not set for this shell/session.")
    client = OpenAI(api_key=api_key)

    # Audio
    fmt, b64 = encode_audio_for_api(AUDIO_PATH)

    # Primary call
    resp = call_model(client, fmt, b64, MAX_TOKENS_PRIMARY)
    finish = getattr(resp.choices[0], "finish_reason", "")
    text = extract_text_from_response(resp)

    # Try to parse
    parsed = None
    try:
        parsed = parse_json_or_raise(text)
    except Exception:
        # Save partial raw for debugging
        raw_path = save_raw_text(text)
        print(f"[i] Saved partial raw output to: {raw_path}")

    # If truncated or unparsable, do a one-time continuation attempt
    if parsed is None or finish == "length":
        print("[info] Attempting continuation to complete JSON...")
        resp2 = continue_if_truncated(client, fmt, b64, text or "", MAX_TOKENS_CONTINUE)
        text2 = extract_text_from_response(resp2)
        try:
            parsed = parse_json_or_raise(text2)
            text = text2  # use the completed text for preview
            finish2 = getattr(resp2.choices[0], "finish_reason", "")
            if finish2 == "length":
                print("[warning] Continuation was also cut off (length). Consider increasing MAX_TOKENS_CONTINUE or tightening output.")
        except Exception:
            # Final failover: save raw continuation
            raw2_path = save_raw_text(text2, base_name="analysis_result_raw_continuation")
            print("[!] Could not parse JSON even after continuation.")
            print(f"[i] Raw continuation saved to: {raw2_path}")
            sys.exit(1)

    # Save JSON (CSV removed as requested)
    json_path = save_json(parsed)
    print(f"\nSaved JSON to: {json_path}")

    # Quick preview for console
    print("\n[debug] finish_reason (first call):", finish)
    preview = (text or "").strip()
    print("\nModel analysis (truncated preview):")
    print(preview[:1200], "..." if len(preview) > 1200 else "")
