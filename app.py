
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
JSON_REPAIR_MAX_ATTEMPTS = 2
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

@@ -477,144 +478,146 @@ def _hash_key(payload: Dict[str, Any]) -> str:

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
def _cached_generate(cache_key: str, system_prompt: str, user_prompt: str, config: Dict[str, Any], response_mime_type: Optional[str] = None) -> str:
    client = _client(config["api_key"], config["timeout_ms"])
    tools = [types.Tool(google_search=types.GoogleSearch())]

    system_instruction_supported = True
    contents = _build_contents(system_prompt, user_prompt, include_system_instruction=True)

    try:
        gen_config = types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=config["temperature"],
            max_output_tokens=config["max_output_tokens"],
            response_mime_type=response_mime_type,
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
                    response_mime_type=response_mime_type,
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
def generate_json(system_prompt: str, user_prompt: str, cache_key_payload: Dict[str, Any], response_mime_type: Optional[str] = "application/json") -> str:
    config = _get_config()
    cache_key = _hash_key({"payload": cache_key_payload, "model": config["model"]})
    return _cached_generate(cache_key, system_prompt, user_prompt, config)
    cache_key = _hash_key({"payload": cache_key_payload, "model": config["model"], "mime": response_mime_type})
    return _cached_generate(cache_key, system_prompt, user_prompt, config, response_mime_type=response_mime_type)


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
@@ -912,77 +915,96 @@ def normalize_itinerary(itin: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]
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




def _repair_json_with_model(system_prompt: str, broken_json: str) -> str:
    repair_prompt = (
        "You will receive malformed or truncated JSON for a travel itinerary. "
        "Return ONE valid JSON object only (no markdown), preserving existing values whenever possible. "
        "If data is missing due to truncation, complete minimally while keeping schema valid. "
        "Do not add commentary.\n\nMalformed JSON:\n" + broken_json
    )
    payload = {"repair": True, "broken_json": broken_json[:4000]}
    return generate_json(system_prompt, repair_prompt, payload, response_mime_type="application/json")

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
    repair_attempt = 0
    while parse_error and isinstance(parse_error, str) and parse_error.startswith("TRUNCATED_JSON") and repair_attempt < JSON_REPAIR_MAX_ATTEMPTS:
        warnings.append(f"JSON_REPAIR_ATTEMPT_{repair_attempt + 1}")
        repaired_text = _repair_json_with_model(system_prompt, text)
        parsed, parse_error = _extract_json(repaired_text)
        text = repaired_text
        repair_attempt += 1
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
cat /workspace/tripy.ai/app.py
