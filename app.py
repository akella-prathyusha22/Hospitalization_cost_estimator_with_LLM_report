import streamlit as st
import pandas as pd
import datetime
import plotly.express as px
from model_utils import generate_full_report

st.set_page_config(page_title="Hospitalization Cost Estimator", layout="centered")
st.title("Hospitalization Cost Estimator")
st.write("Enter patient details to estimate hospitalization cost with an AI-generated explanation.")

with st.form("patient_form"):
    # Personal details
    gender = st.selectbox("Gender", ["male", "female"])
    age = st.number_input("Age", min_value=0, max_value=150)
    bmi = st.number_input("BMI", min_value=20)
    num_of_children = st.number_input("Number of Children", min_value=0)
    
    # Hospital details
    hospital_tier = st.selectbox("Hospital Tier", ["tier-1", "tier-2", "tier-3"])
    city_tier = st.selectbox("City Tier", ["tier-1", "tier-2", "tier-3"])
    state_id = st.selectbox("State ID", ["R1011", "R1012", "R1013"])
    
    # Medical history
    hba1c = st.number_input("HBA1C value for Diatebetes risk assessment", min_value=0.0)
    smoker = st.selectbox("Smoker", ["yes", "no"])
    cancer_history = st.selectbox("Any Cancer History?", ["yes", "no"])
    any_transplants = st.selectbox("Any Transplants?", ["yes", "no"])
    heart_issues = st.selectbox("Any Heart issues?", ["yes", "no"])
    num_of_major_surgeries = st.number_input("Number of Major Surgeries", min_value=0, max_value=50)
    
    submitted = st.form_submit_button("Estimate Cost")

if submitted:
    new_patient = pd.DataFrame([{
        "Age": age,
        "bmi": bmi,
        "children": num_of_children,
        "hospital_tier": hospital_tier,
        "city_tier": city_tier,
        "state_id_R1011" : ,
        "state_id_R1012": ,
        "state_id_R1013": ,
        "state_id": state_id,
        "hba1c": hba1c,
        "cancer_history": cancer_history,
        "any_transplants": any_transplants,
        "heart_issues": heart_issues,
        "smoker": smoker,
        "Gender": gender,
        "numberofmajorsurgeries": num_of_major_surgeries
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
