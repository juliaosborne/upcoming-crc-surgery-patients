import streamlit as st
import os
import openai
import pandas as pd
import re
import json
import datetime
from dotenv import load_dotenv
from openai import OpenAI

# === Load OpenAI API key ===
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# === Page settings & title ===
st.set_page_config(page_title="Upcoming CRC surgery patients")
st.title("🧠 Upcoming CRC surgery patients")
st.write("Paste in the raw text — I'll extract and organize your upcoming colorectal surgery patients using AI.")

# === 1) SMART PATIENT CHUNKER ===
def split_patient_chunks(text):
    """
    Breaks messy text into separate patient 'chunks'
    whenever a key component (ID, Phone, Date, Surgeon) repeats.
    """
    tokens = re.findall(
        r"\b(?:Dr\.?|CAV|Pang|Garfinkle|Morin|Faria|Ghitulescu|Vasilevsky|U\d{5,7}|\d{5,7}|\d{3}[- ]?\d{3}[- ]?\d{4}|[A-Za-z]+|sx|OR|January|February|March|April|May|June|July|August|September|October|November|December|\d{1,2})\b",
        text
    )
    seen_types = set()
    current = []
    patients = []
    def classify(token):
        if re.match(r"\d{5,7}", token):
            return "ID"
        if re.search(r"\d{3}[- ]?\d{3}[- ]?\d{4}", token):
            return "Phone"
        if token.lower() in ["sx", "or"]:
            return "Context"
        if token.capitalize() in ["January", "February", "March", "April", "May", "June", "July",
                                   "August", "September", "October", "November", "December"]:
            return "Date"
        if token.lower() in ["pang", "ghitulescu", "garfinkle", "faria", "morin", "cav", "vasilevsky"]:
            return "Surgeon"
        return "Other"
    for tok in tokens:
        tok_type = classify(tok)
        if tok_type in seen_types and tok_type != "Other":
            patients.append(" ".join(current))
            current = []
            seen_types = set()
        current.append(tok)
        if tok_type != "Other":
            seen_types.add(tok_type)
    if current:
        patients.append(" ".join(current))
    return patients

# === 2) DATE NORMALIZATION & PARSING ===
def normalize_date_string(date_str):
    """
    Removes duplicate tokens (e.g., "Jan jan 21" becomes "Jan 21")
    """
    tokens = date_str.split()
    seen = []
    result = []
    for token in tokens:
        token_lower = token.lower()
        if token_lower not in [t.lower() for t in seen]:
            seen.append(token)
            result.append(token)
    return " ".join(result)

def parse_surgery_date(date_str):
    """
    Converts a date string like "Feb 2" or "March 25" (even with duplicates)
    into a datetime object (year 2025) for sorting.
    """
    date_str = date_str.strip()
    if not date_str:
        return None
    normalized = normalize_date_string(date_str)
    for fmt in ["%b %d %Y", "%B %d %Y"]:
        try:
            return datetime.datetime.strptime(normalized + " 2025", fmt)
        except ValueError:
            continue
    return None

# === 3) SURGEON NAME UNIFIER ===
def unify_surgeon_name(name):
    """
    Converts any variant of Dr V, CAV, or Carol Ann Vasilevsky to "Dr. Vasilevsky".
    Also standardizes other surgeon names.
    """
    if not name:
        return name
    normalized = name.lower().replace(".", "").strip()
    # Guarantee any indication of Dr V or similar becomes Dr. Vasilevsky.
    if normalized in ["dr v", "drv", "v", "cav"] or "dr v" in normalized or "vasilevsky" in normalized:
        return "Dr. Vasilevsky"
    synonyms = {
        "pang": "Dr. Pang",
        "ghitulescu": "Dr. Ghitulescu",
        "garfinkle": "Dr. Garfinkle",
        "faria": "Dr. Faria",
        "morin": "Dr. Morin"
    }
    for key, val in synonyms.items():
        if key in normalized:
            return val
    return name.title()

# === 4) GPT BATCH PARSER ===
def extract_with_gpt(raw_entries):
    """
    Sends all patient chunks to GPT in one batch and returns structured JSON.
    Instructs GPT to output surgeon names in canonical form and fix duplicate months.
    """
    try:
        formatted = "\n".join([f"{i+1}. {entry}" for i, entry in enumerate(raw_entries)])
        prompt = f"""
You will receive a list of patient scheduling blurbs. For each, extract:
- Surgeon
- Patient Name
- Patient ID
- Surgery Date
- Phone Number

Return a JSON array. Example:
[
  {{
    "Surgeon": "Dr. Morin",
    "Patient Name": "Raphael Benatar",
    "Patient ID": "176421",
    "Surgery Date": "Feb 2",
    "Phone": "514-867-4718"
  }},
  {{
    "Surgeon": "Dr. Vasilevsky",
    "Patient Name": "Maria Arvanitis",
    "Patient ID": "798709",
    "Surgery Date": "Jan 23",
    "Phone": "514-321-9177"
  }}
]
Only assign "Dr. Vasilevsky" if the original text clearly includes a name like "Dr V", "CAV", "Dr. V.", or "Vasilevsky". Do not change any other surgeon names.
If a month is duplicated (e.g., "Jan jan 21"), output it as "Jan 21".
If you sense a typo in a surgeons name (e.g., "Motin"), interpret it as "Morin" so you can output it as "Dr. Morin".
If the user inputs irrelevant text, such as email introductions like "Hi Julia, here are the upcoming surgeries", ignore this and only look for the relevant data. 

TEXT:
{formatted}
"""
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You extract and structure patient scheduling data from messy hospital text."},
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )
        content = response.choices[0].message.content.strip()
        if content.startswith("```json"):
            content = content[7:]
        elif content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        st.write("🧠 CLEANED GPT Response Text:", content)
        return json.loads(content)
    except Exception as e:
        return [{"Error": str(e)}]

# === STREAMLIT APP LOGIC ===
user_input = st.text_area("Paste your patient info here (no need to format it):", height=300)

if user_input:
    st.info("Parsing patient entries...")
    # 1) Chunk the raw text
    patient_chunks = split_patient_chunks(user_input)
    st.write("🔍 Patient Chunks Detected:", patient_chunks)
    # 2) Send them all to GPT at once
    gpt_data = extract_with_gpt(patient_chunks)
    st.write("🧠 GPT Raw Output:", gpt_data)
    # 3) Build a DataFrame from GPT data
    df = pd.DataFrame(gpt_data)
    expected_cols = ["Surgeon", "Patient Name", "Patient ID", "Surgery Date", "Phone"]
    for col in expected_cols:
        if col not in df.columns:
            df[col] = ""
    # 4) Normalize surgeon names (ensuring any "Dr V" becomes "Dr. Vasilevsky")
    df["Surgeon"] = df["Surgeon"].apply(unify_surgeon_name)
    # 5) Normalize patient IDs: remove any leading 'U' or 'u' and whitespace
    df["Patient ID"] = df["Patient ID"].str.replace(r"^[Uu]\s*", "", regex=True)
    # 6) Normalize and parse surgery dates; sort by date (earliest → latest)
    df["Surgery Date"] = df["Surgery Date"].apply(lambda x: normalize_date_string(x))
    df["ParsedDate"] = df["Surgery Date"].apply(parse_surgery_date)
    df.sort_values("ParsedDate", ascending=True, na_position="last", inplace=True)
    df.drop(columns=["ParsedDate"], inplace=True)
    df = df[expected_cols]
    st.success("✅ Done! Here's your organized schedule (earliest → latest):")
    st.dataframe(df.fillna(""))
    # 7) Download as CSV
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download as CSV", csv, "upcoming_crc_patients.csv", "text/csv")
