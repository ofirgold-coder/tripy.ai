"""TripPilot (single-file deploy)

This file is a fully self-contained version of the original project (app.py + services/* + utils/*)
so you can deploy with only:
- app.py
- requirements.txt

Secrets required (Streamlit Cloud -> Advanced settings -> Secrets):
GOOGLE_API_KEY = "..."
Optional:
GEMINI_MODEL = "gemini-3-flash-preview"
GEMINI_TEMPERATURE = 0.5
GEMINI_MAX_OUTPUT_TOKENS = 8192
GEMINI_TIMEOUT_SEC = 60
LOG_LEVEL = "INFO"
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import re
import time
from datetime import datetime, timedelta
from json import JSONDecodeError
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional, Tuple

import httpx
import streamlit as st
from fpdf import FPDF
from google import genai
from google.genai import types
from ics import Calendar, Event


# =====================
# utils/cache.py
# =====================

def cached_data(ttl_seconds: int = 3600):
    """Wrapper for st.cache_data with sensible defaults and no mutation warning."""

    def decorator(func):
        return st.cache_data(ttl=ttl_seconds, show_spinner=False)(func)

    return decorator


def cached_resource(ttl_seconds: int = 3600):
    """Wrapper for st.cache_resource with sensible defaults."""

    def decorator(func):
        return st.cache_resource(ttl=ttl_seconds, show_spinner=False)(func)

    return decorator


# =====================
# utils/validators.py
# =====================

def validate_profile(profile: Dict) -> Tuple[bool, str]:
    required = ["destination", "days", "traveler_type", "budget", "pace", "mobility"]
    for field in required:
        if not profile.get(field):
            return False, f"Missing required field: {field}"
    try:
        days = int(profile.get("days", 0))
        if days <= 0:
            return False, "Days must be a positive number."
    except ValueError:
        return False, "Days must be numeric."
    return True, ""


def validate_itinerary_schema(itinerary: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    if not isinstance(itinerary, dict):
        return False, "INVALID_ROOT_TYPE"
    if "trip_summary" not in itinerary:
        return False, "MISSING_TRIP_SUMMARY"
    trip_summary = itinerary.get("trip_summary")
    if not isinstance(trip_summary, dict):
        return False, "TRIP_SUMMARY_NOT_OBJECT"
    if "days" not in itinerary:
        return False, "MISSING_DAYS_LIST"
    if not isinstance(itinerary.get("days"), list):
        return False, "DAYS_NOT_LIST"
    return True, None


# =====================
# utils/formatting.py
# =====================

RAW_SNIPPET_LIMIT = 4000


def mask_key(key: Optional[str], shown: int = 4) -> str:
    if not key:
        return ""
    if len(key) <= shown:
        return "*" * len(key)
    return "*" * (len(key) - shown) + key[-shown:]


def render_sources(sources: Any) -> None:
    if not sources:
        return
    if not isinstance(sources, list):
        return
    for src in sources:
        if not isinstance(src, dict):
            continue
        title = src.get("title", "Source")
        url = src.get("url")
        domain = src.get("domain", "")
        label = f"{title} ({domain})" if domain else title
        if url:
            st.markdown(f"- [{label}]({url})")
        else:
            st.markdown(f"- {label}")


def render_itinerary(itinerary: dict, raw_response: Optional[str] = None) -> None:
    if not itinerary:
        return

    summary = itinerary.get("trip_summary", {})
    if not isinstance(summary, dict):
        summary = {}
    st.subheader("Trip summary")
    cols = st.columns(3)
    cols[0].markdown(f"**Destination:** {summary.get('destination', 'N/A')}")
    cols[1].markdown(f"**Days:** {summary.get('days', 'N/A')}")
    cols[2].markdown(f"**Pace:** {summary.get('pace', 'N/A')}")
    st.markdown("**Why this fits you:**")
    for focus in summary.get("high_level_focus", []):
        st.markdown(f"- {focus}")

    lodging_list = itinerary.get("lodging") if isinstance(itinerary.get("lodging"), list) else []
    if lodging_list:
        st.subheader("Lodging recommendations")
        for lodging in lodging_list:
            if not isinstance(lodging, dict):
                st.markdown(f"- {lodging}")
                continue
            with st.expander(lodging.get("name", "Option")):
                st.markdown(f"**Type:** {lodging.get('type')}")
                st.markdown(f"**Area:** {lodging.get('area')}")
                st.markdown(f"**Why:** {lodging.get('why')}")
                st.markdown(f"**Price note:** {lodging.get('price_note')}")
                st.markdown("**Pros:**")
                for p in lodging.get("pros", []):
                    st.markdown(f"- {p}")
                st.markdown("**Cons:**")
                for c in lodging.get("cons", []):
                    st.markdown(f"- {c}")
                st.markdown("**Sources:**")
                render_sources(lodging.get("sources"))

    days_list = itinerary.get("days") if isinstance(itinerary.get("days"), list) else []
    if days_list:
        st.subheader("Day-by-day itinerary")
        for day in days_list:
            if not isinstance(day, dict):
                continue
            title = day.get("title") or day.get("theme", "")
            with st.expander(f"Day {day.get('day')}: {title}", expanded=False):
                st.markdown("**Activities:**")
                blocks = day.get("blocks") if isinstance(day.get("blocks"), list) else []
                if not blocks:
                    for slot in ("morning", "afternoon", "evening"):
                        slot_items = day.get(slot) if isinstance(day.get(slot), list) else []
                        for item in slot_items:
                            if isinstance(item, dict):
                                activity = item.get("activity")
                                notes = item.get("notes", "")
                                area = item.get("area", "")
                                duration = item.get("duration_est", "")
                            else:
                                activity = str(item)
                                notes = ""
                                area = ""
                                duration = ""
                            st.markdown(f"- **{slot.title()}**: {activity} ({area}) — {duration}. {notes}")
                            if isinstance(item, dict):
                                render_sources(item.get("sources"))
                else:
                    for block in blocks:
                        if not isinstance(block, dict):
                            continue
                        st.markdown(
                            f"- **{(block.get('time', '') or '').title()}**: {block.get('activity')} "
                            f"({block.get('area')}) — {block.get('duration_est')}. "
                            f"Transit: {block.get('transit_note')}. {block.get('notes', '')}"
                        )
                        render_sources(block.get("sources"))

                if day.get("food"):
                    st.markdown("**Food:**")
                    for food in day.get("food", []):
                        if not isinstance(food, dict):
                            st.markdown(f"- {food}")
                            continue
                        st.markdown(
                            f"- **{(food.get('type', '') or '').title()}**: {food.get('name')} "
                            f"({food.get('budget', '')} budget, {food.get('area', '')}) "
                            f"Diet: {', '.join(food.get('diet_fit', []))}"
                        )
                        render_sources(food.get("sources"))

                if day.get("plan_b"):
                    st.markdown("**Plan B:**")
                    for alt in day.get("plan_b", []):
                        if not isinstance(alt, dict):
                            st.markdown(f"- {alt}")
                            continue
                        st.markdown(f"- For **{alt.get('reason')}**: {alt.get('alternative')}")
                        render_sources(alt.get("sources"))

    assumptions = itinerary.get("assumptions") if isinstance(itinerary.get("assumptions"), list) else []
    if assumptions:
        st.subheader("Assumptions")
        for item in assumptions:
            st.markdown(f"- {item}")

    disclaimer = itinerary.get("disclaimer") if isinstance(itinerary.get("disclaimer"), str) else ""
    if disclaimer:
        st.info(disclaimer)

    with st.expander("Debug: raw JSON"):
        st.code(json.dumps(itinerary, indent=2))

    if raw_response:
        snippet = raw_response if len(raw_response) <= RAW_SNIPPET_LIMIT else raw_response[:RAW_SNIPPET_LIMIT] + "... [truncated]"
        with st.expander("Debug: raw model output"):
            st.code(snippet)


# =====================
# services/export.py
# =====================

def _set_font(pdf: FPDF) -> None:
    font_candidates = [
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        Path("/Library/Fonts/Arial Unicode.ttf"),
        Path("/System/Library/Fonts/Supplemental/Arial.ttf"),
        Path("C:/Windows/Fonts/arial.ttf"),
    ]
    for font_path in font_candidates:
        if font_path.exists():
            try:
                font_hash = hashlib.md5(str(font_path).encode()).hexdigest()[:8]
                font_name = f"{font_path.stem}_{font_hash}"
                pdf.add_font(font_name, "", str(font_path), uni=True)
                pdf.set_font(font_name, size=12)
                return
            except (OSError, RuntimeError):
                continue
    pdf.set_font("Helvetica", size=12)


def itinerary_to_pdf(itinerary: Dict) -> bytes:
    if not isinstance(itinerary, dict):
        raise ValueError("itinerary must be a dict")
    ok, reason = validate_itinerary_schema(itinerary)
    if not ok:
        raise ValueError(f"Invalid itinerary: {reason}")

    pdf = FPDF()
    pdf.add_page()
    _set_font(pdf)

    def write_line(text: str) -> None:
        try:
            pdf.multi_cell(0, 10, txt=text, new_x="LMARGIN", new_y="NEXT")
        except TypeError as err:
            if "new_x" not in str(err) and "new_y" not in str(err):
                raise
            pdf.multi_cell(0, 10, txt=text)

    summary = itinerary.get("trip_summary") if isinstance(itinerary.get("trip_summary"), dict) else {}
    write_line(f"Trip to {summary.get('destination', 'Destination')}")
    write_line(f"Days: {summary.get('days', 'N/A')} | Pace: {summary.get('pace', 'N/A')}")
    write_line("Highlights:")
    for focus in summary.get("high_level_focus", []):
        write_line(f"- {focus}")

    pdf.ln(4)
    write_line("Lodging:")
    lodging_list = itinerary.get("lodging") if isinstance(itinerary.get("lodging"), list) else []
    for lodging in lodging_list:
        if not isinstance(lodging, dict):
            write_line(f"- {lodging}")
            continue
        write_line(
            f"- {lodging.get('name')} ({lodging.get('type')}, {lodging.get('area')}): {lodging.get('why')}"
        )

    pdf.ln(4)
    day_entries = itinerary.get("days") if isinstance(itinerary.get("days"), list) else []
    for day in day_entries:
        if not isinstance(day, dict):
            continue
        write_line(f"Day {day.get('day')}: {day.get('theme', '')}")
        blocks = day.get("blocks") if isinstance(day.get("blocks"), list) else []
        for block in blocks:
            if not isinstance(block, dict):
                continue
            write_line(
                f"  {(block.get('time','') or '').title()}: {block.get('activity')} ({block.get('area')}) - "
                f"{block.get('duration_est')} | Transit: {block.get('transit_note')}"
            )
        foods = day.get("food") if isinstance(day.get("food"), list) else []
        for food in foods:
            if not isinstance(food, dict):
                write_line(f"  Food: {food}")
                continue
            write_line(
                f"  Food: {food.get('name')} ({food.get('type')}, {food.get('budget')} budget, {food.get('area')})"
            )
        plan_b_items = day.get("plan_b") if isinstance(day.get("plan_b"), list) else []
        for plan_b in plan_b_items:
            if not isinstance(plan_b, dict):
                write_line(f"  Plan B: {plan_b}")
                continue
            write_line(f"  Plan B ({plan_b.get('reason')}): {plan_b.get('alternative')}")
        pdf.ln(2)

    assumptions = itinerary.get("assumptions") if isinstance(itinerary.get("assumptions"), list) else []
    if assumptions:
        pdf.ln(4)
        write_line("Assumptions:")
        for a in assumptions:
            write_line(f"- {a}")

    disclaimer = itinerary.get("disclaimer") if isinstance(itinerary.get("disclaimer"), str) else ""
    if disclaimer:
        pdf.ln(4)
        write_line(f"Disclaimer: {disclaimer}")

    output = pdf.output(dest="S")
    if isinstance(output, (bytes, bytearray)):
        return bytes(output)
    return str(output).encode("latin-1", errors="replace")


def itinerary_to_ics(itinerary: Dict) -> bytes:
    if not isinstance(itinerary, dict):
        raise ValueError("itinerary must be a dict")

    cal = Calendar()
    start_date = datetime.utcnow().date()
    day_entries = itinerary.get("days") if isinstance(itinerary.get("days"), list) else []
    for day in day_entries:
        if not isinstance(day, dict):
            continue
        day_offset = int(day.get("day", 1)) - 1
        base_date = start_date + timedelta(days=day_offset)
        times = {"morning": 9, "afternoon": 13, "evening": 19}
        blocks = day.get("blocks") if isinstance(day.get("blocks"), list) else []
        for block in blocks:
            if not isinstance(block, dict):
                continue
            hour = times.get(block.get("time", "morning"), 9)
            event = Event()
            event.name = block.get("activity")
            event.begin = datetime.combine(base_date, datetime.min.time()) + timedelta(hours=hour)
            event.duration = timedelta(hours=2)
            event.location = block.get("area", "")
            event.description = block.get("notes", "")
            cal.events.add(event)
    return str(cal).encode("utf-8")


# =====================
# services/llm.py
# =====================

MIN_TIMEOUT = 30
LOW_TIMEOUT_ATTEMPTS = 2
DEFAULT_MAX_ATTEMPTS = 3
MAX_RETRY_SLEEP_SEC = 5


def _library_version() -> str:
    try:
        return importlib.metadata.version("google-genai")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _get_config() -> Dict[str, Any]:
    secrets = st.secrets
    api_key = secrets.get("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError(
            "Missing GOOGLE_API_KEY in Streamlit secrets. "
            "Add it via Streamlit Cloud -> Advanced settings -> Secrets."
        )

    def _safe_float(val: Any, default: float) -> float:
        try:
            return float(val)
        except (TypeError, ValueError):
            return default

    def _safe_int(val: Any, default: int) -> int:
        try:
            int_val = int(val)
            return int_val if int_val > 0 else default
        except (TypeError, ValueError):
            return default

    configured_timeout_sec = _safe_float(secrets.get("GEMINI_TIMEOUT_SEC", 60), 60.0)
    effective_timeout_sec = max(configured_timeout_sec, float(MIN_TIMEOUT))
    timeout_ms = int(effective_timeout_sec * 1000)
    temperature = _safe_float(secrets.get("GEMINI_TEMPERATURE", 0.5), 0.5)
    
    # FIX: Increased default max output tokens to 8192 to prevent TRUNCATED_JSON errors
    max_output_tokens = _safe_int(secrets.get("GEMINI_MAX_OUTPUT_TOKENS", 8192), 8192)

    return {
        "api_key": api_key,
        "model": secrets.get("GEMINI_MODEL", "gemini-3-flash-preview"),
        "temperature": temperature,
        "max_output_tokens": max_output_tokens,
        "configured_timeout_sec": configured_timeout_sec,
        "timeout_sec": effective_timeout_sec,
        "timeout_ms": timeout_ms,
        "log_level": secrets.get("LOG_LEVEL", "INFO"),
        "library_version": _library_version(),
    }


@cached_resource(ttl_seconds=3600)
def _client(api_key: str, timeout_ms: int):
    return genai.Client(api_key=api_key, http_options=types.HttpOptions(timeout=timeout_ms))


def _response_text(resp: Any) -> str:
    if not resp:
        return ""
    if isinstance(resp, dict):
        candidates = resp.get("candidates") or []
    else:
        candidates = getattr(resp, "candidates", None) or []

    text = getattr(resp, "text", "") or ""
    if text:
        return text

    if candidates:
        first = candidates[0]
        content = getattr(first, "content", None) or (first.get("content") if isinstance(first, dict) else {})
        parts = getattr(content, "parts", None) or (content.get("parts") if isinstance(content, dict) else []) or []
        for part in parts:
            part_text = getattr(part, "text", None) or (part.get("text") if isinstance(part, dict) else None)
            if part_text:
                return part_text
    return ""


def _build_contents(system_prompt: str, user_prompt: str, include_system_instruction: bool = True):
    if include_system_instruction:
        return [{"role": "user", "parts": [{"text": user_prompt}]}]
    return [
        {"role": "system", "parts": [{"text": system_prompt}]},
        {"role": "user", "parts": [{"text": user_prompt}]},
    ]


def _hash_key(payload: Dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _backoff_sleep(attempt_number: int, timeout_sec: float) -> float:
    """Exponential backoff with sane caps."""
    return min(2 ** max(attempt_number - 1, 0), timeout_sec / 2, MAX_RETRY_SLEEP_SEC)


def _log_attempt(
    attempt: int,
    elapsed: float,
    config: Dict[str, Any],
    tools_enabled: bool,
    status: str,
    error: Optional[str] = None,
) -> None:
    details = (
        f"Attempt {attempt} ({status}) in {elapsed:.1f}s — model={config['model']}, "
        f"timeout={config['timeout_sec']}s/{config['timeout_ms']}ms, tools={'on' if tools_enabled else 'off'}"
    )
    if error:
        details += f" | {error}"
    # Keep exactly as original: log in sidebar.
    st.sidebar.caption(details)


@cached_data(ttl_seconds=3600)
def _cached_generate(cache_key: str, system_prompt: str, user_prompt: str, config: Dict[str, Any]) -> str:
    client = _client(config["api_key"], config["timeout_ms"])
    tools = [types.Tool(google_search=types.GoogleSearch())]

    system_instruction_supported = True
    contents = _build_contents(system_prompt, user_prompt, include_system_instruction=True)

    try:
        gen_config = types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=config["temperature"],
            max_output_tokens=config["max_output_tokens"],
            tools=tools,
        )
    except TypeError:
        system_instruction_supported = False
        contents = _build_contents(system_prompt, user_prompt, include_system_instruction=False)
        gen_config = types.GenerateContentConfig(
            temperature=config["temperature"],
            max_output_tokens=config["max_output_tokens"],
            tools=tools,
        )

    last_error: Optional[Exception] = None
    max_attempts = LOW_TIMEOUT_ATTEMPTS if config["configured_timeout_sec"] < MIN_TIMEOUT else DEFAULT_MAX_ATTEMPTS
    attempt = 0

    while attempt < max_attempts:
        start = time.time()
        try:
            response = client.models.generate_content(
                model=config["model"],
                contents=contents,
                config=gen_config,
            )
            text = _response_text(response)
            elapsed = time.time() - start
            _log_attempt(attempt + 1, elapsed, config, tools_enabled=True, status="success")
            if text:
                return text
        except TypeError as err:
            elapsed = time.time() - start
            if system_instruction_supported:
                system_instruction_supported = False
                contents = _build_contents(system_prompt, user_prompt, include_system_instruction=False)
                gen_config = types.GenerateContentConfig(
                    temperature=config["temperature"],
                    max_output_tokens=config["max_output_tokens"],
                    tools=tools,
                )
                _log_attempt(
                    attempt + 1,
                    elapsed,
                    config,
                    tools_enabled=True,
                    status="retry",
                    error="system_instruction unsupported, retrying with in-content prompt",
                )
                continue
            _log_attempt(attempt + 1, elapsed, config, tools_enabled=True, status="error", error=str(err))
            last_error = err
        except (httpx.ReadTimeout, httpx.TimeoutException, TimeoutError) as err:
            elapsed = time.time() - start
            timeout_error_msg = (
                "Request timed out. Increase GEMINI_TIMEOUT_SEC and/or reduce output size. "
                f"(configured={config['configured_timeout_sec']}s, effective={config['timeout_sec']}s, "
                f"timeout_ms={config['timeout_ms']}), elapsed={elapsed:.1f}s, "
                f"exception={type(err).__name__}: {repr(err)}"
            )
            _log_attempt(attempt + 1, elapsed, config, tools_enabled=True, status="timeout", error=timeout_error_msg)
            last_error = TimeoutError(timeout_error_msg)
            break
        except Exception as err:
            elapsed = time.time() - start
            error_detail = (
                f"{type(err).__name__}: {repr(err)} (elapsed={elapsed:.1f}s, timeout_ms={config['timeout_ms']})"
            )
            _log_attempt(attempt + 1, elapsed, config, tools_enabled=True, status="error", error=error_detail)
            last_error = err

        attempt += 1
        if attempt >= max_attempts:
            break
        time.sleep(_backoff_sleep(attempt, float(config["timeout_sec"])))

    if last_error:
        raise last_error
    return ""


def generate_json(system_prompt: str, user_prompt: str, cache_key_payload: Dict[str, Any]) -> str:
    config = _get_config()
    cache_key = _hash_key({"payload": cache_key_payload, "model": config["model"]})
    return _cached_generate(cache_key, system_prompt, user_prompt, config)


def secrets_status() -> Dict[str, Any]:
    try:
        cfg = _get_config()
        warning = None
        if cfg["configured_timeout_sec"] < MIN_TIMEOUT:
            warning = (
                f"Configured timeout {cfg['configured_timeout_sec']}s is too low; clamped to {cfg['timeout_sec']}s."
            )
        return {"ok": True, "config": cfg, "warning": warning}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


# =====================
# services/planner.py
# =====================

SNIPPET_MAX_LENGTH = 500
MISSING_DAY_BY_DAY_PLAN = "MISSING_DAY_BY_DAY_PLAN"


class ItineraryResult(NamedTuple):
    data: Dict[str, Any]
    raw: str
    error: Optional[str]
    warnings: List[str]


SCHEMA_TEXT = r"""
Return ONLY JSON (no prose, no markdown) with ALL of these top-level keys:
{
  "trip_summary": {
    "destination": "...",
    "days": 5,  // NUMBER of days
    "traveler_type": "...",
    "budget": "...",
    "pace": "...",
    "high_level_focus": ["..."]
  },
  "days": [  // REQUIRED array with one object per day
    {
      "day": 1,
      "title": "Day 1 title",
      "morning": [ {"activity": "...", "area": "...", "duration_est": "...", "notes": "...", "sources": [{"title":"...","domain":"...","url":"..."}]} ],
      "afternoon": [ {"activity": "...", "area": "...", "duration_est": "...", "notes": "...", "sources": [{"title":"...","domain":"...","url":"..."}]} ],
      "evening": [ {"activity": "...", "area": "...", "duration_est": "...", "notes": "...", "sources": [{"title":"...","domain":"...","url":"..."}]} ],
      "notes": "",
      "plan_b": [ {"reason": "rain|low_energy|crowded", "alternative": "...", "sources": [{"title":"...","domain":"...","url":"..."}]} ],
      "food": [ {"name": "...", "type": "breakfast|lunch|dinner|snack", "budget": "low|mid|high", "area": "...", "diet_fit": ["kosher|vegan|halal|gluten-free|none"], "sources": [{"title":"...","domain":"...","url":"..."}]} ],
      "sources": [{"title":"...","domain":"...","url":"..."}]
    }
  ],
  "alternatives": [ {"destination": "...", "why": "...", "sources": [{"title":"...","domain":"...","url":"..."}]} ],
  "lodging": [ {"name": "...", "type": "hotel|hostel|apartment", "area": "...", "why": "...", "price_note": "...", "pros": ["..."], "cons": ["..."], "sources": [{"title":"...","domain":"...","url":"..."}]} ],
  "food": [],
  "transport": {}
}
Rules: trip_summary.days is a NUMBER, but top-level "days" MUST be an ARRAY of day objects. Always include all required top-level keys even if empty arrays/objects. JSON only, no markdown. Include citations from grounded search. If data is uncertain, note it in assumptions. Cap activities to 2-3 per day. Keep neighborhoods consistent.
"""


def _system_prompt() -> str:
    return (
        "You are TripPilot, a travel planner. Use Google Search tool for grounding; include "
        "source titles and domains. Never hallucinate URLs. If a tool fails, note that in assumptions. "
        "Never provide illegal, unsafe, or guaranteed claims. Keep output concise: max 2 items per time block, "
        "sources at the day level only (1-2 per day), and short notes. "
        "Always follow the schema exactly. "
        + SCHEMA_TEXT
    )


def _build_user_prompt(profile: Dict[str, Any], destination: str, refinement: Optional[str]) -> str:
    base = [
        f"Primary destination: {destination}",
        f"Days: {profile.get('days')}",
        f"Traveler type: {profile.get('traveler_type')}",
        f"Budget: {profile.get('budget')}",
        f"Pace: {profile.get('pace')}",
        f"Mobility: {profile.get('mobility')}",
        f"Interests: {', '.join(profile.get('interests', [])) or 'unspecified'}",
        f"Lodging preference: {profile.get('lodging')}",
        f"Dietary/accessibility constraints: {profile.get('constraints') or 'none'}",
        f"Open to alternatives: {profile.get('open_to_alternatives')}",
    ]
    if profile.get("dates"):
        base.append(f"Dates: {profile['dates']}")

    instructions = (
        "Generate a realistic day-by-day itinerary with morning/afternoon/evening blocks, "
        "restaurants matched to dietary and budget, lodging options with pros/cons and neighborhoods, "
        "and plan B options per day. Provide 2-3 alternatives destinations if open_to_alternatives. "
        "Ensure transit notes are brief and plausible. Top-level keys must exactly match the schema and "
        "include trip_summary, days (array of day objects), alternatives, lodging, food, and transport. "
        "Respond with JSON only. Keep outputs compact to avoid truncation; keep notes short and no more than 2 "
        "activities per time block with day-level sources only."
    )

    if refinement:
        instructions += f" Apply this refinement request: {refinement}. Only adjust necessary parts."

    return "\n".join(base + [instructions])


def _strip_code_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^
http://googleusercontent.com/immersive_entry_chip/0
