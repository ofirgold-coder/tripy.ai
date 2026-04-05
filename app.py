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
GEMINI_MAX_OUTPUT_TOKENS = 2048
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
    max_output_tokens = _safe_int(secrets.get("GEMINI_MAX_OUTPUT_TOKENS", 2048), 2048)

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
        text = re.sub(r"^```(?:json)?", "", text, flags=re.IGNORECASE).strip()
        if text.endswith("```"):
            text = text[:-3].strip()
    return text


def _structure_only_snippet(text: str) -> str:
    snippet = text[:SNIPPET_MAX_LENGTH]
    return re.sub(r"[^{}\[\]:,\"'\s]", "·", snippet)


def _is_balanced_braces(text: str) -> bool:
    depth = 0
    in_string = False
    string_delim = ""
    triple_delim = False
    escape = False
    i = 0
    text_len = len(text)

    while i < text_len:
        ch = text[i]
        if in_string:
            if escape:
                escape = False
                i += 1
                continue
            if ch == "\\":
                escape = True
                i += 1
                continue
            if triple_delim and i + 2 < text_len and text[i : i + 3] == string_delim * 3:
                in_string = False
                triple_delim = False
                string_delim = ""
                i += 3
                continue
            if not triple_delim and ch == string_delim:
                in_string = False
                string_delim = ""
            i += 1
            continue

        if ch in ('"', "'"):
            if i + 2 < text_len and text[i + 1] == ch and text[i + 2] == ch:
                triple_delim = True
                in_string = True
                string_delim = ch
                i += 3
                continue
            in_string = True
            string_delim = ch
            i += 1
            continue

        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
        if depth < 0:
            return False
        i += 1

    return depth == 0


def _ensure_dict(obj: Any) -> Tuple[Dict[str, Any], Optional[str]]:
    if isinstance(obj, dict):
        return obj, None
    if isinstance(obj, str):
        try:
            inner = json.loads(obj.strip())
            if isinstance(inner, dict):
                return inner, None
        except (JSONDecodeError, ValueError):
            pass
    return {}, f"Invalid JSON type: expected object, got {type(obj).__name__}"


def _extract_json(text: str) -> Tuple[Dict[str, Any], Optional[str]]:
    cleaned = _strip_code_fences(text)
    last_error: Optional[str] = None

    parsed: Any = None
    try:
        parsed = json.loads(cleaned)
    except (JSONDecodeError, ValueError) as err:
        last_error = str(err)
        if cleaned and not _is_balanced_braces(cleaned):
            snippet = _structure_only_snippet(cleaned)
            return {}, (
                "TRUNCATED_JSON: output cut mid-JSON. Increase max tokens or reduce detail. Snippet: " + snippet
            )

    if isinstance(parsed, str):
        try:
            parsed = json.loads(parsed.strip())
        except (JSONDecodeError, ValueError) as err:
            last_error = str(err)

    if isinstance(parsed, dict):
        return parsed, None

    if parsed is not None:
        ensured, err = _ensure_dict(parsed)
        if err is None:
            return ensured, None
        last_error = err

    snippet = _structure_only_snippet(cleaned)
    last_error = last_error or "No JSON object found"
    return {}, f"Invalid JSON returned from model. Last error: {last_error}. Snippet: {snippet}"


def _ensure_sources(container: Dict[str, Any]) -> None:
    sources = container.get("sources")
    if isinstance(sources, list):
        container["sources"] = [s for s in sources if isinstance(s, dict)]
    else:
        container["sources"] = []


def _clean_list_of_dicts(items: Any) -> List[Dict[str, Any]]:
    cleaned: List[Dict[str, Any]] = []
    if not isinstance(items, list):
        return cleaned
    for entry in items:
        if isinstance(entry, dict):
            _ensure_sources(entry)
            cleaned.append(entry)
    return cleaned


def normalize_itinerary(itin: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str], Optional[str]]:
    warnings: List[str] = []
    if not isinstance(itin, dict):
        return {}, warnings, "INVALID_ROOT_TYPE"

    trip_summary = itin.get("trip_summary")
    if not isinstance(trip_summary, dict):
        return {}, warnings, "MISSING_TRIP_SUMMARY" if trip_summary is None else "TRIP_SUMMARY_NOT_OBJECT"

    days = itin.get("days")
    normalized_days: List[Dict[str, Any]] = []

    if not isinstance(days, list):
        ts_days = trip_summary.get("days")
        if isinstance(ts_days, int) and ts_days > 0:
            for i in range(1, ts_days + 1):
                normalized_days.append(
                    {
                        "day": i,
                        "title": f"Day {i}",
                        "morning": [],
                        "afternoon": [],
                        "evening": [],
                        "notes": "",
                        "plan_b": [],
                        "food": [],
                        "sources": [],
                    }
                )
            warnings.append(MISSING_DAY_BY_DAY_PLAN)
        else:
            return {}, warnings, "MISSING_DAYS_LIST"
    else:
        for idx, day in enumerate(days):
            if not isinstance(day, dict):
                warnings.append(f"DAY_NOT_OBJECT_{idx + 1}")
                continue
            normalized_day = {
                "day": day.get("day", idx + 1),
                "title": day.get("title") or day.get("theme") or f"Day {idx + 1}",
                "morning": _clean_list_of_dicts(day.get("morning")),
                "afternoon": _clean_list_of_dicts(day.get("afternoon")),
                "evening": _clean_list_of_dicts(day.get("evening")),
                "notes": day.get("notes", ""),
                "plan_b": _clean_list_of_dicts(day.get("plan_b")),
                "food": _clean_list_of_dicts(day.get("food")),
                "sources": [],
                "blocks": _clean_list_of_dicts(day.get("blocks")),
            }
            _ensure_sources(normalized_day)
            normalized_days.append(normalized_day)

        if not normalized_days:
            if MISSING_DAY_BY_DAY_PLAN not in warnings:
                warnings.append(MISSING_DAY_BY_DAY_PLAN)

    itin["days"] = normalized_days

    for key in ["alternatives", "lodging", "food"]:
        val = itin.get(key)
        if val is None:
            itin[key] = []
            continue
        if not isinstance(val, list):
            warnings.append(f"{key.upper()}_NOT_LIST")
            itin[key] = []
        else:
            cleaned_items: List[Dict[str, Any]] = []
            for item in val:
                if not isinstance(item, dict):
                    warnings.append(f"{key.upper()}_ITEM_INVALID")
                    continue
                _ensure_sources(item)
                cleaned_items.append(item)
            itin[key] = cleaned_items

    transport = itin.get("transport")
    if transport is None:
        itin["transport"] = {}
    elif not isinstance(transport, dict):
        warnings.append("TRANSPORT_NOT_OBJECT")
        itin["transport"] = {}

    assumptions = itin.get("assumptions")
    if assumptions is not None and not isinstance(assumptions, list):
        warnings.append("ASSUMPTIONS_NOT_LIST")
        itin["assumptions"] = []

    disclaimer = itin.get("disclaimer")
    if disclaimer is not None and not isinstance(disclaimer, str):
        warnings.append("DISCLAIMER_NOT_STRING")
        itin["disclaimer"] = ""

    if "days" in itin and not isinstance(trip_summary.get("days"), int):
        trip_summary["days"] = len(itin.get("days", []))

    return itin, warnings, None


def generate_itinerary(
    profile: Dict[str, Any],
    destination: str,
    refinement: Optional[str] = None,
    current_itinerary: Optional[Dict[str, Any]] = None,
) -> ItineraryResult:
    system_prompt = _system_prompt()
    user_prompt = _build_user_prompt(profile, destination, refinement)

    if current_itinerary:
        user_prompt += (
            "\n\nUpdate the following existing itinerary to satisfy the request. Keep schema intact and return the "
            "full updated itinerary JSON only:\n" + json.dumps(current_itinerary)
        )

    payload = {
        "destination": destination,
        "profile": profile,
        "refinement": refinement or "",
        "has_current_itinerary": bool(current_itinerary),
        "current_itinerary": current_itinerary or {},
    }

    text = generate_json(system_prompt, user_prompt, payload)
    parsed, parse_error = _extract_json(text)

    warnings: List[str] = []
    if not parse_error:
        schema_ok, reason = validate_itinerary_schema(parsed)
        if not schema_ok:
            if reason == "MISSING_DAYS_LIST":
                trip_summary = parsed.get("trip_summary") if isinstance(parsed, dict) else {}
                ts_days = trip_summary.get("days") if isinstance(trip_summary, dict) else None
                if isinstance(ts_days, int) and ts_days > 0:
                    parsed["days"] = []
                    if MISSING_DAY_BY_DAY_PLAN not in warnings:
                        warnings.append(MISSING_DAY_BY_DAY_PLAN)
                else:
                    parse_error = reason
            else:
                parse_error = reason

    if not parse_error:
        parsed, norm_warnings, parse_error = normalize_itinerary(parsed)
        warnings.extend(norm_warnings)

    if parse_error:
        parsed = parsed if isinstance(parsed, dict) else {}

    return ItineraryResult(parsed, text, parse_error, warnings)


# =====================
# Original app.py (UI)
# =====================

st.set_page_config(page_title="TripPilot", page_icon="🛫", layout="wide")

FILTER_HE = {
    "form_title": "תפריט סינון",
    "destination": "יעד",
    "days": "מספר ימים",
    "traveler_type": "סוג מטיילים",
    "budget": "תקציב",
    "pace": "קצב",
    "mobility": "ניידות",
    "interests": "תחומי עניין",
    "constraints": "הגבלות",
    "lodging": "העדפת לינה",
    "dates": "תאריכים",
    "open_to_alternatives": "פתוח ליעדים חלופיים?",
    "save_button": "שמור פרטי טיול",
    "save_success": "פרטי הטיול נשמרו.",
}

TRAVELER_OPTIONS = {
    "יחיד": "solo",
    "זוג": "couple",
    "חברים": "friends",
    "משפחה": "family",
}
PACE_OPTIONS = {"רגוע": "relaxed", "מאוזן": "balanced", "אינטנסיבי": "packed"}
BUDGET_OPTIONS = {"נמוך": "low", "בינוני": "medium", "גבוה": "high"}
MOBILITY_OPTIONS = {"עם רכב": "with car", "בלי רכב": "no car"}
LODGING_OPTIONS = {"מלון": "hotel", "הוסטל": "hostel", "דירה": "apartment", "גמיש": "flexible"}
INTEREST_OPTIONS = {
    "חופים": "beaches",
    "חיי לילה": "nightlife",
    "אוכל": "food",
    "תרבות": "culture",
    "קניות": "shopping",
    "טבע": "nature",
    "ספורט": "sports",
    "מוזיאונים": "museums",
    "פארקי שעשועים": "theme parks",
}


def _init_state() -> None:
    defaults = {
        "started": False,
        "trip_profile": None,
        "chosen_destination": None,
        "itinerary_json": None,
        "itinerary_warnings": [],
        "revision_history": [],
        "generation_error": None,
        "last_raw_response": None,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val
    if not isinstance(st.session_state.get("revision_history"), list):
        st.session_state["revision_history"] = []
    if not isinstance(st.session_state.get("itinerary_warnings"), list):
        st.session_state["itinerary_warnings"] = []


def _reset_session_state() -> None:
    st.session_state["trip_profile"] = None
    st.session_state["chosen_destination"] = None
    st.session_state["itinerary_json"] = None
    st.session_state["itinerary_warnings"] = []
    st.session_state["revision_history"] = []
    st.session_state["generation_error"] = None
    st.session_state["last_raw_response"] = None
    st.session_state["started"] = False


def _get_valid_itinerary_from_state() -> Optional[Dict]:
    itinerary = st.session_state.get("itinerary_json")
    if itinerary is not None and not isinstance(itinerary, dict):
        err = f"Internal error: itinerary_json must be an object, got {type(itinerary).__name__}"
        st.session_state["generation_error"] = err
        st.session_state["itinerary_json"] = None
        st.session_state["itinerary_warnings"] = []
        st.error(err)
        return None
    return itinerary


def _handle_generate(refinement: Optional[str] = None, force_destination: Optional[str] = None) -> None:
    st.session_state["generation_error"] = None
    st.session_state["last_raw_response"] = None
    st.session_state["itinerary_warnings"] = []

    profile: Dict[str, Any] = st.session_state.get("trip_profile") or {}
    ok, msg = validate_profile(profile)
    if not ok:
        st.error(msg)
        return

    destination = force_destination or profile.get("destination")
    current_itinerary = _get_valid_itinerary_from_state()

    with st.spinner("Generating itinerary with Gemini..."):
        start = time.time()
        try:
            itinerary, raw_text, parse_error, warnings = generate_itinerary(
                profile, destination, refinement, current_itinerary=current_itinerary
            )
            st.session_state["last_raw_response"] = raw_text

            if parse_error:
                error_labels = {
                    "MISSING_DAYS_LIST": "Invalid itinerary shape: missing top-level days list.",
                    "DAYS_NOT_LIST": "Invalid itinerary shape: days must be an array.",
                    "MISSING_TRIP_SUMMARY": "Invalid itinerary shape: trip_summary missing.",
                    "TRIP_SUMMARY_NOT_OBJECT": "Invalid itinerary shape: trip_summary must be an object.",
                    "INVALID_ROOT_TYPE": "Invalid itinerary shape: root must be a JSON object.",
                }
                if isinstance(parse_error, str) and parse_error.startswith("TRUNCATED_JSON"):
                    friendly_error = parse_error
                else:
                    friendly_error = error_labels.get(parse_error, parse_error)
                st.session_state["generation_error"] = friendly_error
                st.error(f"Generation failed: {friendly_error}. Keeping previous itinerary.")
                return

            if not isinstance(itinerary, dict):
                err_msg = f"Invalid itinerary type: {type(itinerary).__name__}"
                st.session_state["generation_error"] = err_msg
                st.error(f"Generation failed: {err_msg}. Keeping previous itinerary.")
                return

            if not itinerary:
                st.session_state["generation_error"] = "No itinerary returned."
                st.error("No itinerary returned. Please retry. Keeping previous itinerary.")
                return

            if "trip_summary" not in itinerary:
                err_msg = "Invalid itinerary shape returned: missing trip_summary"
                st.session_state["generation_error"] = err_msg
                st.error(f"Generation failed: {err_msg}. Keeping previous itinerary.")
                return

            st.session_state["itinerary_json"] = itinerary
            st.session_state["itinerary_warnings"] = warnings
            st.session_state["chosen_destination"] = itinerary.get("trip_summary", {}).get("destination", destination)

            if not isinstance(st.session_state.get("revision_history"), list):
                st.session_state["revision_history"] = []
            st.session_state["revision_history"].append(
                {"timestamp": time.time(), "refinement": refinement, "destination": destination}
            )

        except Exception as exc:
            st.session_state["generation_error"] = str(exc)
            st.error(f"Generation failed: {exc}. Keeping previous itinerary.")
        finally:
            st.info(f"Completed in {time.time() - start:.1f}s")


def intake_form() -> None:
    st.subheader(FILTER_HE["form_title"])
    with st.form("trip_form"):
        col1, col2, col3 = st.columns(3)

        destination = col1.text_input(FILTER_HE["destination"], value="תל אביב", placeholder="הקלד יעד")
        days = col2.number_input(FILTER_HE["days"], min_value=1, max_value=21, value=5)
        traveler_label = col3.selectbox(FILTER_HE["traveler_type"], list(TRAVELER_OPTIONS.keys()))
        traveler_type = TRAVELER_OPTIONS[traveler_label]

        budget_label = col1.selectbox(FILTER_HE["budget"], list(BUDGET_OPTIONS.keys()))
        budget = BUDGET_OPTIONS[budget_label]
        pace_label = col2.selectbox(FILTER_HE["pace"], list(PACE_OPTIONS.keys()))
        pace = PACE_OPTIONS[pace_label]
        mobility_label = col3.selectbox(FILTER_HE["mobility"], list(MOBILITY_OPTIONS.keys()))
        mobility = MOBILITY_OPTIONS[mobility_label]

        interests = st.multiselect(
            FILTER_HE["interests"],
            list(INTEREST_OPTIONS.keys()),
            default=["אוכל", "חופים", "תרבות"],
        )

        constraints = st.text_input(FILTER_HE["constraints"], value="", placeholder="כשרות, אלרגיות, נגישות ועוד")
        lodging_label = st.selectbox(FILTER_HE["lodging"], list(LODGING_OPTIONS.keys()))
        lodging = LODGING_OPTIONS[lodging_label]
        dates = st.text_input(FILTER_HE["dates"], placeholder="לדוגמה: 12-18 ביוני")
        open_to_alternatives = st.checkbox(FILTER_HE["open_to_alternatives"], value=True)

        submitted = st.form_submit_button(FILTER_HE["save_button"])
        if submitted:
            profile = {
                "destination": destination,
                "days": int(days),
                "traveler_type": traveler_type,
                "budget": budget,
                "pace": pace,
                "mobility": mobility,
                "interests": [INTEREST_OPTIONS.get(label, label) for label in interests],
                "constraints": constraints,
                "lodging": lodging,
                "dates": dates,
                "open_to_alternatives": open_to_alternatives,
            }
            st.session_state["trip_profile"] = profile
            st.session_state["chosen_destination"] = destination
            st.success(FILTER_HE["save_success"])


def render_alternatives() -> Optional[str]:
    itinerary = _get_valid_itinerary_from_state()
    if not itinerary or not isinstance(itinerary, dict):
        return None

    alts = itinerary.get("alternatives") if isinstance(itinerary.get("alternatives"), list) else []
    if not alts:
        return None

    st.subheader("Destination alternatives")
    options = [alt.get("destination") for alt in alts if isinstance(alt, dict) and alt.get("destination")]
    if not options:
        return None

    chosen = st.radio("Pick an alternative or keep the original", ["Keep current"] + options, key="alternative_pick")
    if chosen != "Keep current":
        return chosen
    return None


def render_refinement() -> Optional[str]:
    st.subheader("Refinement")
    presets = {
        "None": "",
        "More relaxed": "Make the itinerary more relaxed with lighter pacing.",
        "More packed": "Increase activities while staying realistic.",
        "More nightlife": "Add nightlife and evening entertainment options.",
        "More nature": "Add nature and outdoors focused activities.",
    }
    choice = st.selectbox("Suggested refinements", list(presets.keys()), index=0, key="refinement_choice")
    custom = st.text_input("Custom edit request", key="custom_edit_request")
    refinement = custom.strip() if custom.strip() else presets.get(choice, "")
    return refinement or None


def render_export() -> None:
    itinerary = _get_valid_itinerary_from_state()
    if not itinerary:
        return

    st.subheader("Export")
    try:
        pdf_bytes = itinerary_to_pdf(itinerary)
        ics_bytes = itinerary_to_ics(itinerary)
    except ValueError as exc:
        st.error(str(exc))
        return

    st.download_button("Download PDF", data=pdf_bytes, file_name="itinerary.pdf", mime="application/pdf")
    st.download_button("Download ICS", data=ics_bytes, file_name="itinerary.ics", mime="text/calendar")


def sidebar() -> None:
    st.sidebar.title("Settings")
    status = secrets_status()
    if status.get("ok"):
        cfg = status.get("config", {})
        st.sidebar.success("API key detected.")
        st.sidebar.caption(f"Model: {cfg.get('model')}")
        st.sidebar.caption(f"Temperature: {cfg.get('temperature')}")
        st.sidebar.caption(f"Configured timeout: {cfg.get('configured_timeout_sec')}s")
        st.sidebar.caption(f"Effective timeout: {cfg.get('timeout_sec')}s")
        st.sidebar.caption(f"Client timeout: {cfg.get('timeout_ms')}ms")
        st.sidebar.caption(
            f"Max output tokens: {cfg.get('max_output_tokens')} (consider 4096 for detailed plans)"
        )
        st.sidebar.caption(f"google-genai: {cfg.get('library_version')}")
        st.sidebar.caption(f"API key: {mask_key(cfg.get('api_key'))}")
        if status.get("warning"):
            st.sidebar.warning(status["warning"])
    else:
        st.sidebar.error(status.get("error", "Missing configuration"))

    if st.sidebar.button("Reset session"):
        _reset_session_state()
        st.rerun()


def main() -> None:
    _init_state()
    sidebar()

    st.title("TripPilot: Plan your trip with Gemini 3 Flash Preview")
    st.write("Tell me your trip goals; I’ll build a grounded, time-aware itinerary.")

    if not st.session_state["started"]:
        if st.button("Start planning"):
            st.session_state["started"] = True
        else:
            st.info("Click **Start planning** to begin.")
            return

    intake_form()

    st.subheader("Itinerary generation")
    refinement_request = render_refinement()
    chosen_alternative = render_alternatives()

    if st.button("Send Request"):
        _handle_generate(refinement=refinement_request, force_destination=chosen_alternative)

    itinerary = _get_valid_itinerary_from_state()

    if st.session_state.get("generation_error"):
        st.error(st.session_state["generation_error"])

    warning_labels = st.session_state.get("itinerary_warnings") or []
    if warning_labels:
        if MISSING_DAY_BY_DAY_PLAN in warning_labels:
            st.warning("Partial itinerary: day-by-day plan missing; showing recommendations.")
        else:
            st.warning(f"Partial itinerary: {', '.join(warning_labels)}")

    if itinerary:
        render_itinerary(itinerary, raw_response=st.session_state.get("last_raw_response"))
        render_export()
    elif st.session_state.get("started") and not st.session_state.get("generation_error"):
        st.info("No itinerary yet. Provide details and click **Send Request** to generate one.")

    if st.session_state.get("last_raw_response") and not itinerary:
        with st.expander("Debug: last model output"):
            raw = st.session_state["last_raw_response"]
            limit = RAW_SNIPPET_LIMIT
            snippet = raw if len(raw) <= limit else raw[:limit] + "... [truncated]"
            st.code(snippet)


if __name__ == "__main__":
    main()
