import streamlit as st
import pandas as pd
import plotly.express as px
from model_utils import generate_full_report

st.set_page_config(page_title="Hospitalization Cost Estimator", layout="centered")
st.title("Hospitalization Cost Estimator")
st.write("Enter patient details to estimate hospitalization cost with an AI-generated explanation.")

with st.form("patient_form"):
    age = st.number_input("Age", min_value=0, max_value=120, value=40)
    bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0)
    smoker = st.selectbox("Smoker", ["yes", "no"])
    gender = st.selectbox("Gender", ["male", "female"])
    region = st.selectbox("Region", ["southeast", "southwest", "northeast", "northwest"])
    num_lab_procedures = st.number_input("Number of Lab Procedures", min_value=0, max_value=100, value=10)

    submitted = st.form_submit_button("Estimate Cost")

if submitted:
    new_patient = pd.DataFrame([{
        "age": age,
        "bmi": bmi,
        "smoker": smoker,
        "gender": gender,
        "region": region,
        "num_lab_procedures": num_lab_procedures
    }])

    with st.spinner("Analyzing patient data and generating report..."):
        predicted_cost, contribution_table, report_text = generate_full_report(new_patient)

    st.subheader("💰 Estimated Cost")
    st.metric(label="Predicted Hospitalization Cost", value=f"${predicted_cost:,.2f}")

    st.subheader("📊 Top Contributing Factors")

    # Clean feature names for display (remove ColumnTransformer prefixes)
    chart_df = contribution_table.copy()
    chart_df["feature"] = chart_df["feature"].str.split("__").str[-1]
    chart_df["direction"] = chart_df["shap_contribution"].apply(lambda x: "Increases cost" if x > 0 else "Decreases cost")
    chart_df = chart_df.sort_values("shap_contribution")

    fig = px.bar(
        chart_df,
        x="shap_contribution",
        y="feature",
        color="direction",
        orientation="h",
        color_discrete_map={"Increases cost": "#e74c3c", "Decreases cost": "#2ecc71"},
        labels={"shap_contribution": "Impact on predicted cost ($)", "feature": "Feature"},
    )
    fig.update_layout(showlegend=True, height=400)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("📝 Explanation Report")
    st.write(report_text)