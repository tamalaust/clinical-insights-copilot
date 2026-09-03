import sys
from pathlib import Path

# streamlit run ui/app.py puts ui/ on sys.path, not the repo root — add
# the repo root explicitly so `ui.*` and `etl.*` imports resolve correctly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import plotly.express as px
import requests
import streamlit as st

from ui.charts import (
    admission_type_breakdown,
    avg_medications_by_age,
    readmission_rate_by_age,
    time_in_hospital_distribution,
)
from ui.chat_client import send_chat_message
from ui.data_access import load_cleaned_data

st.set_page_config(page_title="Clinical Insights Copilot", layout="wide")
st.title("Clinical Insights Copilot")

df = load_cleaned_data()

# --- Sidebar filters ---
st.sidebar.header("Filters")

gender_options = sorted(df["gender"].dropna().unique())
selected_genders = st.sidebar.multiselect("Gender", gender_options, default=gender_options)

race_options = sorted(df["race"].dropna().unique())
selected_races = st.sidebar.multiselect("Race", race_options, default=race_options)

filtered_df = df[df["gender"].isin(selected_genders) & df["race"].isin(selected_races)]
st.sidebar.caption(f"{len(filtered_df):,} of {len(df):,} encounters match the current filters")

dashboard_tab, chat_tab = st.tabs(["Dashboard", "Chat"])

with dashboard_tab:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Readmission rate by age group")
        readmit_data = readmission_rate_by_age(filtered_df)
        fig = px.bar(readmit_data, x="age", y="readmission_rate_pct",
                     labels={"age": "Age group", "readmission_rate_pct": "Readmission rate (%)"})
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Average medications by age group")
        meds_data = avg_medications_by_age(filtered_df)
        fig = px.bar(meds_data, x="age", y="avg_num_medications",
                     labels={"age": "Age group", "avg_num_medications": "Avg. medications"})
        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        st.subheader("Time in hospital distribution")
        los_data = time_in_hospital_distribution(filtered_df)
        fig = px.bar(los_data, x="time_in_hospital", y="encounter_count",
                     labels={"time_in_hospital": "Days in hospital", "encounter_count": "Encounters"})
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        st.subheader("Admission type breakdown")
        admission_data = admission_type_breakdown(filtered_df)
        fig = px.pie(admission_data, names="admission_type_id_description", values="encounter_count")
        st.plotly_chart(fig, use_container_width=True)

with chat_tab:
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    user_input = st.chat_input("Ask about the data, e.g. 'why is readmission higher for older patients?'")

    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.write(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    result = send_chat_message(user_input, st.session_state.chat_history[:-1])
                    answer = result["answer"]
                    from_cache = result["from_cache"]
                except requests.exceptions.RequestException:
                    answer = (
                        "Sorry, I couldn't reach the backend. "
                        "Make sure the FastAPI server is running (`uvicorn app.main:app --reload`)."
                    )
                    from_cache = False
            if from_cache:
                st.markdown(":red[⚡ Served from cache]")
            st.write(answer)

        st.session_state.chat_history.append({"role": "assistant", "content": answer})