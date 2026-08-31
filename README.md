# Hospitalization Cost Estimator with AI-Generated Explanations

An end-to-end machine learning project that predicts hospitalization costs using an ensemble of models, explains *why* each prediction was made using SHAP (SHapley Additive exPlanations), and translates those explanations into plain-English reports using a Large Language Model (LLM) - all wrapped in an interactive Streamlit web app.

---

## Project Overview

Traditional ML models are often "black boxes" - they give a number, but not a reason. This project solves that by combining:

1. **Predictive modeling** - an ensemble of Random Forest, Gradient Boosting, and Ridge Regression to estimate hospitalization costs.
2. **Explainability** - SHAP values to quantify exactly which patient factors drove the prediction, and by how much.
3. **Generative AI** - an LLM that converts raw SHAP numbers into a short, human-readable explanation a patient (not a data scientist) can actually understand.

The result: given a patient's details, the app returns a predicted cost **and** a plain-language report explaining what influenced that number.

---

## Key Features

- **Ensemble regression model** combining Random Forest, Gradient Boosting, and Ridge Regression (tuned via `GridSearchCV`)
- **SHAP-based explainability** using `TreeExplainer` (RF, GB) and `LinearExplainer` (Ridge), combined to match the ensemble's averaging logic
- **LLM-generated natural language reports** via the Hugging Face Inference API (with fallback to Claude API)
- **Interactive Streamlit UI** with a patient input form, cost prediction, SHAP contribution bar chart, and AI-written explanation
- **Modular codebase** - model/ML logic (`model_utils.py`) cleanly separated from UI code (`app.py`)

---

## How It Works

```
Patient Input (Streamlit form)
        ↓
Preprocessing (fitted ColumnTransformer)
        ↓
Ensemble Prediction  → (RF + GB + Ridge) / 3
        ↓
SHAP Explanation      → (RF SHAP + GB SHAP + Ridge SHAP) / 3
        ↓
Top Contributing Features Table
        ↓
LLM Prompt Construction
        ↓
Hugging Face / Claude API → Plain-English Report
        ↓
Streamlit UI: Cost + Bar Chart + Report
```

---

## Project Structure

```
├── app.py                  # Streamlit UI
├── model_utils.py          # Model loading, SHAP explainers, LLM prompt + API logic
├── models/
│   ├── random_forest.pkl
│   ├── gradient_boosting.pkl
│   ├── ridge.pkl            # Ridge model wrapped in a preprocessing Pipeline
│   └── X_train.pkl          # Background reference data for SHAP's LinearExplainer
├── notebooks/
│   └── data_cleaning_and_model_training.ipynb  # Full training pipeline
├── .env                     # Stores HF_TOKEN or ANTHROPIC_API_KEY (not committed)
├── requirements.txt
└── README.md
```

---

## Tech Stack

| Component | Tool |
|---|---|
| Modeling | scikit-learn (RandomForestRegressor, GradientBoostingRegressor, Ridge, GridSearchCV) |
| Explainability | SHAP (TreeExplainer, LinearExplainer) |
| Generative AI | Hugging Face Inference API (Llama 3.1) or Claude API |
| Web App | Streamlit |
| Visualization | Plotly |
| Data Handling | pandas, NumPy |

---

## Getting Started

### 1. Clone the repo and install dependencies

```bash
git clone <your-repo-url>
cd hospitalization-cost-estimator
pip install -r requirements.txt
```

### 2. Set up your LLM API token

**Hugging Face (free tier)**
- Create a free account at [huggingface.co](https://huggingface.co)
- Generate a **fine-grained** access token with **"Make calls to Inference Providers"** permission
- Create a `.env` file:
  ```
  HF_TOKEN=your_token_here
  ```

### 3. Run the app locally

```bash
streamlit run app.py
```

Or deploy directly to Streamlit Community Cloud - it will build the environment from `requirements.txt` automatically.

---

## Example Output

Given a patient's details (age, BMI, smoker status, region, etc.), the app returns:

- **Predicted Cost** - e.g., `$14,320.50`
- **Top Contributing Factors** - a horizontal bar chart showing which features pushed the estimate up (red) or down (green)
- **AI-Generated Report** - a short paragraph explaining the estimate in plain language

---

## Model Validation & Known Limitations

### Important Finding: Spurious Patterns in the Ensemble

During model validation, **hypothesis testing revealed that smoking status is statistically independent of hospitalization cost** in the training data (p-value > 0.05). However, **the ensemble models learned a strong negative relationship** between smoking and cost (SHAP: -$4248).

This indicates the models are **overfitting to spurious patterns** - learning noise in the training set rather than true causal relationships. A similar pattern was observed for BMI and Age.

### Implications

- **Explanations for smoking status, BMI, and Age should be interpreted with caution** - they may not reflect true medical relationships, but rather artifacts of the training data.
- The model predictions may still be useful as a *statistical* baseline, but the *feature explanations* should not be treated as ground truth.
- For production use, consider:
  - Feature selection or regularization to reduce overfitting
  - Removing features that show statistical independence from the target despite high model importance
  - Cross-validation against domain expertise

### How This is Handled

The LLM prompt includes guidance to be skeptical of counterintuitive factors (e.g., "smoking reducing cost") and to focus explanations on medically plausible relationships instead. This doesn't hide the issue - it acknowledges it transparently.

---

## Portfolio Learning Value

This project demonstrates:

1. **End-to-end ML pipeline** - from data cleaning to model training to deployment
2. **Model explainability** - using industry-standard SHAP libraries
3. **LLM integration** - calling external APIs to generate human-readable outputs
4. **Critical validation** - testing model assumptions against domain knowledge and identifying limitations
5. **Graceful degradation** - documenting and flagging when models behave unexpectedly rather than blindly trusting them

The validation finding (spurious patterns) is actually a *strength* of this portfolio, not a weakness, it shows you validate models rigorously and communicate limitations transparently, which is exactly what good data scientists do.

---

## Future Improvements

- Add model performance metrics (RMSE, MAE, R²) to the UI for transparency
- Implement feature importance filtering to exclude statistically independent features
- Create a comparison view showing how RF, GB, and Ridge differ in their explanations
  
---

## License

This project is for educational and portfolio purposes.

---

## Questions & Feedback

If reviewing this project:

- **Why include the limitation?** - Transparency and critical thinking are more valuable than hiding flaws. The validation finding actually demonstrates stronger data science practice than ignoring it.
- **Should I retrain?** - For a portfolio piece, documenting the limitation and showing you detected it is more impressive than silently retraining. In production, yes, address the overfitting.
- **Is the model still useful?** - Yes, as a statistical baseline. The ensemble still captures broader cost patterns. Just don't over-interpret individual feature explanations.
