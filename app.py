import json
import time
from typing import Dict, Optional

import streamlit as st

from services import planner
from services import export as export_service
from services import llm
from utils import formatting
from utils import validators


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


def _init_state():
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


def _reset_session_state():
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


def _handle_generate(refinement: Optional[str] = None, force_destination: Optional[str] = None):
    st.session_state["generation_error"] = None
    st.session_state["last_raw_response"] = None
    st.session_state["itinerary_warnings"] = []
    profile: Dict = st.session_state.get("trip_profile") or {}
    ok, msg = validators.validate_profile(profile)
    if not ok:
        st.error(msg)
        return
    destination = force_destination or profile.get("destination")
    current_itinerary = _get_valid_itinerary_from_state()
    with st.spinner("Generating itinerary with Gemini..."):
        start = time.time()
        try:
            itinerary, raw_text, parse_error, warnings = planner.generate_itinerary(
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
            st.session_state["chosen_destination"] = itinerary.get("trip_summary", {}).get(
                "destination", destination
            )
            if not isinstance(st.session_state.get("revision_history"), list):
                st.session_state["revision_history"] = []
            st.session_state["revision_history"].append(
                {"timestamp": time.time(), "refinement": refinement, "destination": destination}
            )
        except Exception as exc:  # pragma: no cover - defensive
            st.session_state["generation_error"] = str(exc)
            st.error(f"Generation failed: {exc}. Keeping previous itinerary.")
        finally:
            st.info(f"Completed in {time.time() - start:.1f}s")


def intake_form():
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
    options = [alt.get("destination") for alt in alts if alt.get("destination")]
    if not options:
        return None
    chosen = st.radio("Pick an alternative or keep the original", ["Keep current"] + options, key="alternative_pick")
    if chosen != "Keep current":
        return chosen
    return None


def render_refinement():
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


def render_export():
    itinerary = _get_valid_itinerary_from_state()
    if not itinerary:
        return
    st.subheader("Export")
    try:
        pdf_bytes = export_service.itinerary_to_pdf(itinerary)
        ics_bytes = export_service.itinerary_to_ics(itinerary)
    except ValueError as exc:
        st.error(str(exc))
        return
    st.download_button("Download PDF", data=pdf_bytes, file_name="itinerary.pdf", mime="application/pdf")
    st.download_button("Download ICS", data=ics_bytes, file_name="itinerary.ics", mime="text/calendar")


def sidebar():
    st.sidebar.title("Settings")
    status = llm.secrets_status()
    if status["ok"]:
        cfg = status["config"]
        st.sidebar.success("API key detected.")
        st.sidebar.caption(f"Model: {cfg.get('model')}")
        st.sidebar.caption(f"Temperature: {cfg.get('temperature')}")
        st.sidebar.caption(f"Configured timeout: {cfg.get('configured_timeout_sec')}s")
        st.sidebar.caption(f"Effective timeout: {cfg.get('timeout_sec')}s")
        st.sidebar.caption(f"Client timeout: {cfg.get('timeout_ms')}ms")
        st.sidebar.caption(f"Max output tokens: {cfg.get('max_output_tokens')} (consider 4096 for detailed plans)")
        st.sidebar.caption(f"google-genai: {cfg.get('library_version')}")
        st.sidebar.caption(f"API key: {formatting.mask_key(cfg.get('api_key'))}")
        if status.get("warning"):
            st.sidebar.warning(status["warning"])
    else:
        st.sidebar.error(status["error"])
    if st.sidebar.button("Reset session"):
        _reset_session_state()
        st.rerun()


def main():
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
        if "MISSING_DAY_BY_DAY_PLAN" in warning_labels:
            st.warning("Partial itinerary: day-by-day plan missing; showing recommendations.")
        else:
            st.warning(f"Partial itinerary: {', '.join(warning_labels)}")
    if itinerary:
        formatting.render_itinerary(itinerary, raw_response=st.session_state.get("last_raw_response"))
        render_export()
    elif st.session_state.get("started") and not st.session_state.get("generation_error"):
        st.info("No itinerary yet. Provide details and click **Send Request** to generate one.")
    if st.session_state.get("last_raw_response") and not itinerary:
        with st.expander("Debug: last model output"):
            raw = st.session_state["last_raw_response"]
            limit = formatting.RAW_SNIPPET_LIMIT
            snippet = raw if len(raw) <= limit else raw[:limit] + "... [truncated]"
            st.code(snippet)


if __name__ == "__main__":
    main()
