import json
from datetime import datetime, timedelta

import streamlit as st
from openai import OpenAI
from fpdf import FPDF
from ics import Calendar, Event


# =======================
# CONFIG
# =======================

st.set_page_config(page_title="AI Trip Planner", layout="centered")

if "OPENAI_API_KEY" not in st.secrets:
    st.error("OPENAI_API_KEY missing in Streamlit secrets")
    st.stop()

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])


# =======================
# OPENAI GENERATION
# =======================

def generate_itinerary(data):

    system_prompt = """
You are a professional travel planner.
Return ONLY valid JSON in this structure:

{
  "trip_summary": {
    "destination": "",
    "days": 0,
    "pace": ""
  },
  "days": [
    {
      "day": 1,
      "title": "",
      "activities": [
        {
          "time": "",
          "name": "",
          "location": ""
        }
      ]
    }
  ]
}
"""

    user_prompt = f"""
Destination: {data['destination']}
Days: {data['days']}
Budget: {data['budget']}
Pace: {data['pace']}
Traveler type: {data['traveler_type']}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.7,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )

    content = response.choices[0].message.content

    try:
        return json.loads(content)
    except:
        st.error("Model returned invalid JSON")
        st.code(content)
        return None


# =======================
# PDF EXPORT
# =======================

def create_pdf(itinerary):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    summary = itinerary["trip_summary"]
    pdf.cell(0, 10, f"Trip to {summary['destination']}", ln=True)
    pdf.cell(0, 10, f"Days: {summary['days']} | Pace: {summary['pace']}", ln=True)
    pdf.ln(5)

    for day in itinerary["days"]:
        pdf.cell(0, 10, f"Day {day['day']}: {day['title']}", ln=True)
        for act in day["activities"]:
            pdf.cell(
                0,
                8,
                f"{act['time']} - {act['name']} ({act['location']})",
                ln=True,
            )
        pdf.ln(3)

    return pdf.output(dest="S").encode("latin-1")


# =======================
# ICS EXPORT
# =======================

def create_ics(itinerary):
    cal = Calendar()
    start_date = datetime.utcnow().date()

    for day in itinerary["days"]:
        base_date = start_date + timedelta(days=day["day"] - 1)

        for act in day["activities"]:
            event = Event()
            event.name = act["name"]
            event.begin = datetime.combine(base_date, datetime.min.time()) + timedelta(hours=9)
            event.duration = timedelta(hours=2)
            event.location = act["location"]
            cal.events.add(event)

    return str(cal).encode("utf-8")


# =======================
# UI
# =======================

st.title("✈️ AI Trip Planner")

with st.form("trip_form"):
    destination = st.text_input("Destination")
    days = st.number_input("Days", min_value=1, value=3)
    budget = st.selectbox("Budget", ["low", "medium", "high"])
    pace = st.selectbox("Pace", ["relaxed", "balanced", "intense"])
    traveler_type = st.text_input("Traveler type")
    submit = st.form_submit_button("Generate")

if submit:

    if not destination:
        st.error("Destination is required")
        st.stop()

    data = {
        "destination": destination,
        "days": days,
        "budget": budget,
        "pace": pace,
        "traveler_type": traveler_type,
    }

    with st.spinner("Generating itinerary..."):
        itinerary = generate_itinerary(data)

    if itinerary:
        st.success("Itinerary ready!")
        st.json(itinerary)

        pdf_bytes = create_pdf(itinerary)
        st.download_button(
            "Download PDF",
            pdf_bytes,
            file_name="itinerary.pdf",
            mime="application/pdf",
        )

        ics_bytes = create_ics(itinerary)
        st.download_button(
            "Download Calendar (.ics)",
            ics_bytes,
            file_name="itinerary.ics",
            mime="text/calendar",
        )
