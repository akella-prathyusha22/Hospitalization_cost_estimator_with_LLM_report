import pickle
import numpy as np
import shap
import os
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from huggingface_hub import notebook_login
from huggingface_hub import hf_hub_download
import streamlit as st

#load_dotenv()
hf_token = st.secrets["HF_TOKEN"]

# ---------------------------------------------------------------
# 1. Load models (one-time, at import)
# ---------------------------------------------------------------

# Loading the Random forest model from the Hugging Face Cloud due to its large size        

@st.cache_resource # Keeps the model loaded in memory so it doesn't redownload on every click
def load_my_model():
    # 1. Download the file securely from Hugging Face
    model_path = hf_hub_download(
        repo_id="Prathyusha22/hospitalization-cost-estimator-random-forest-model", 
        filename="radom_forest.pkl",
        token=hf_token 
    )
    
    # 2. Load the downloaded pickle file
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model

# Call the cached function
model = load_my_model()
st.write("Model loaded successfully from Hugging Face Hub!")
        
#Gradient Boost model

with open("models/gradient_boosting.pkl", "rb") as f:
    gb_model = pickle.load(f)

#Ridge model

with open("models/ridge.pkl", "rb") as f:
    ridge_best = pickle.load(f)

ridge_best = model_ridge.best_estimator_
ridge_model_only = ridge_best.named_steps["regressor"]
preprocessor = ridge_best.named_steps["scaler"]
feature_names = preprocessor.get_feature_names_out()

# ---------------------------------------------------------------
# 2. Build explainers (one-time, at import)
# ---------------------------------------------------------------
def _load_background_data(sample_size=100):
    # X_train_raw should be saved separately during training, e.g. as a CSV/pickle
    X_train_raw = pd.read_pickle("models/X_train.pkl")
    X_train_transformed = preprocessor.transform(X_train_raw)
    if hasattr(X_train_transformed, "toarray"):
        X_train_transformed = X_train_transformed.toarray()
    return np.array(shap.sample(X_train_transformed, sample_size))

background_data = _load_background_data()

rf_explainer = shap.TreeExplainer(rf_model)
gb_explainer = shap.TreeExplainer(gb_model)
ridge_explainer = shap.LinearExplainer(ridge_model_only, background_data)

def _get_expected_value(explainer):
    ev = explainer.expected_value
    return float(ev.flatten()[0]) if isinstance(ev, np.ndarray) else float(ev)

base_value = (
    _get_expected_value(rf_explainer) +
    _get_expected_value(gb_explainer) +
    _get_expected_value(ridge_explainer)
) / 3

# ---------------------------------------------------------------
# 3. Hugging Face client
# ---------------------------------------------------------------
hf_client = InferenceClient(
    model="meta-llama/Llama-3.1-8B-Instruct",
    token= hf_token
        #os.getenv("HF_TOKEN")
)

def to_scalar(x):
    if isinstance(x, np.ndarray):
        return float(x.flatten()[0]) if x.size == 1 else float(x.mean())
    return float(x)

# ---------------------------------------------------------------
# 4. Core prediction + SHAP function
# ---------------------------------------------------------------
def explain_patient(new_patient_df, top_n=10):
    patient_transformed = preprocessor.transform(new_patient_df)
    if hasattr(patient_transformed, "toarray"):
        patient_transformed = patient_transformed.toarray()

    rf_pred = float(rf_model.predict(patient_transformed)[0])
    gb_pred = float(gb_model.predict(patient_transformed)[0])
    ridge_pred = float(ridge_model_only.predict(patient_transformed)[0])
    final_prediction = (rf_pred + gb_pred + ridge_pred) / 3

    rf_shap = np.array(rf_explainer.shap_values(patient_transformed)).squeeze()
    gb_shap = np.array(gb_explainer.shap_values(patient_transformed)).squeeze()
    ridge_shap = np.array(ridge_explainer.shap_values(patient_transformed)).squeeze()
    combined_shap = (rf_shap + gb_shap + ridge_shap) / 3

    df = pd.DataFrame({
        "feature": feature_names,
        "shap_contribution": combined_shap.astype(float)
    })
    df["abs_contribution"] = df["shap_contribution"].abs()
    df = df.sort_values("abs_contribution", ascending=False).drop(columns="abs_contribution").head(top_n)

    return final_prediction, df.reset_index(drop=True), new_patient_df.iloc[0].to_dict()

# ---------------------------------------------------------------
# 5. Prompt + LLM report generation
# ---------------------------------------------------------------
def build_shap_summary_text(contribution_table, raw_patient_dict):
    lines = []
    for _, row in contribution_table.iterrows():
        clean_name = row["feature"].split("__")[-1]
        direction = "increased" if row["shap_contribution"] > 0 else "decreased"
        lines.append(f"- {clean_name}: {direction} the estimated cost by ${abs(row['shap_contribution']):.2f}")

    patient_info = ", ".join([f"{k}: {v}" for k, v in raw_patient_dict.items()])
    shap_summary = "\n".join(lines)
    return patient_info, shap_summary

def build_llm_prompt(predicted_cost, base_value, patient_info, shap_summary):
    return f"""You are a medical cost analyst explaining a hospitalization cost prediction to a patient in plain, non-technical language.

Patient details: {patient_info}

Baseline average predicted cost: ${base_value:.2f}
Final predicted cost for this patient: ${predicted_cost:.2f}

The following factors influenced this patient's predicted cost, based on a machine learning model:
{shap_summary}

Write a short, clear explanation (3-5 sentences) for the patient describing:
1. What their estimated hospitalization cost is
2. The top 2-3 factors that most influenced this estimate, in plain English
3. Whether each factor increased or decreased their cost, and briefly why that makes sense medically

Avoid technical jargon like "SHAP values" or "feature contribution" — write as if explaining to someone with no data science background. Do not give medical advice."""

def generate_report(prompt, max_tokens=500):
    response = hf_client.chat_completion(
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens
    )
    return response.choices[0].message.content

# ---------------------------------------------------------------
# 6. Full pipeline — the one function app.py calls
# ---------------------------------------------------------------
def generate_full_report(new_patient_df, top_n=10):
    predicted_cost, contribution_table, raw_patient_dict = explain_patient(new_patient_df, top_n=top_n)
    patient_info, shap_summary = build_shap_summary_text(contribution_table, raw_patient_dict)
    prompt = build_llm_prompt(to_scalar(predicted_cost), to_scalar(base_value), patient_info, shap_summary)
    report_text = generate_report(prompt)
    return predicted_cost, contribution_table, report_text
