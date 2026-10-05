"""
Customer Churn Predictor - Streamlit app
Run:  streamlit run app.py
"""
import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).parent
st.set_page_config(page_title="Customer Churn Predictor", page_icon="📉", layout="wide")


@st.cache_resource
def load_artifacts():
    model = joblib.load(ROOT / "models" / "churn_model.joblib")
    metrics = json.loads((ROOT / "models" / "metrics.json").read_text())
    return model, metrics


def explain(model, row: pd.DataFrame, raw_cols: list[str]) -> pd.Series:
    """Per-feature contribution to the churn log-odds (works for the logistic model)."""
    prep, clf = model.named_steps["prep"], model.named_steps["model"]
    if not hasattr(clf, "coef_"):
        return pd.Series(dtype=float)
    x = prep.transform(row)
    x = x.toarray() if hasattr(x, "toarray") else x
    contrib = pd.Series(clf.coef_[0] * x[0], index=prep.get_feature_names_out())
    grouped = {}
    for name, val in contrib.items():
        base = name.split("__", 1)[1]
        raw = next((c for c in raw_cols if base == c or base.startswith(c + "_")), base)
        grouped[raw] = grouped.get(raw, 0.0) + val
    return pd.Series(grouped).sort_values(key=abs, ascending=False)


model, metrics = load_artifacts()
RAW_COLS = ["gender", "SeniorCitizen", "Partner", "Dependents", "tenure", "PhoneService",
            "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
            "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
            "PaperlessBilling", "PaymentMethod", "MonthlyCharges", "TotalCharges"]

st.title("📉 Customer Churn Predictor")
st.caption("Predict which telecom customers are likely to leave - and why.")

# ----------------------------------------------------------------- sidebar inputs
with st.sidebar:
    st.header("Customer profile")
    tenure = st.slider("Tenure (months)", 0, 72, 12)
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    internet = st.selectbox("Internet service", ["Fiber optic", "DSL", "No"])
    monthly = st.slider("Monthly charges ($)", 18.0, 120.0, 70.0, 0.5)
    payment = st.selectbox("Payment method", [
        "Electronic check", "Mailed check", "Bank transfer (automatic)",
        "Credit card (automatic)"])
    paperless = st.selectbox("Paperless billing", ["Yes", "No"])
    senior = st.selectbox("Senior citizen", ["No", "Yes"])
    partner = st.selectbox("Has partner", ["No", "Yes"])
    dependents = st.selectbox("Has dependents", ["No", "Yes"])
    with st.expander("More services"):
        gender = st.selectbox("Gender", ["Female", "Male"])
        phone = st.selectbox("Phone service", ["Yes", "No"])
        lines = st.selectbox("Multiple lines", ["No", "Yes", "No phone service"])
        sec = st.selectbox("Online security", ["No", "Yes", "No internet service"])
        backup = st.selectbox("Online backup", ["No", "Yes", "No internet service"])
        protect = st.selectbox("Device protection", ["No", "Yes", "No internet service"])
        support = st.selectbox("Tech support", ["No", "Yes", "No internet service"])
        tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
        movies = st.selectbox("Streaming movies", ["No", "Yes", "No internet service"])

row = pd.DataFrame([{
    "gender": gender, "SeniorCitizen": senior, "Partner": partner, "Dependents": dependents,
    "tenure": tenure, "PhoneService": phone, "MultipleLines": lines,
    "InternetService": internet, "OnlineSecurity": sec, "OnlineBackup": backup,
    "DeviceProtection": protect, "TechSupport": support, "StreamingTV": tv,
    "StreamingMovies": movies, "Contract": contract, "PaperlessBilling": paperless,
    "PaymentMethod": payment, "MonthlyCharges": monthly,
    "TotalCharges": round(monthly * tenure, 2),
}])[RAW_COLS]

prob = float(model.predict_proba(row)[0, 1])
if prob >= 0.6:
    level, color = "High risk", "🔴"
elif prob >= 0.35:
    level, color = "Medium risk", "🟠"
else:
    level, color = "Low risk", "🟢"

tab1, tab2 = st.tabs(["🔮 Prediction", "📊 Model performance"])

with tab1:
    c1, c2 = st.columns([1, 2])
    with c1:
        st.metric("Churn probability", f"{prob:.0%}")
        st.progress(prob)
        st.subheader(f"{color} {level}")
        if prob >= 0.6:
            st.info("Suggested action: offer a discounted 1-2 year contract or a loyalty perk.")
        elif prob >= 0.35:
            st.info("Suggested action: proactive check-in and a tech-support / security bundle.")
        else:
            st.info("Suggested action: no urgent action - keep the customer happy.")
    with c2:
        st.subheader("Why this prediction?")
        contrib = explain(model, row, RAW_COLS)
        if contrib.empty:
            st.write("Explanations are available for the logistic regression model.")
        else:
            top = contrib.head(8).sort_values()
            st.bar_chart(top.rename("Impact on churn risk"), horizontal=True)
            st.caption("Positive = pushes towards churn, negative = pushes towards staying.")

with tab2:
    best = metrics["best_model"]
    res = metrics["test_results"][best]
    st.write(f"**Selected model:** {best} (trained on {metrics['dataset']['rows']:,} customers, "
             f"{metrics['dataset']['churn_rate']}% churn rate)")
    m = st.columns(5)
    for col, key, label in zip(m, ["roc_auc", "accuracy", "precision", "recall", "f1"],
                               ["ROC-AUC", "Accuracy", "Precision", "Recall", "F1"]):
        col.metric(label, f"{res[key]:.2f}")
    st.dataframe(pd.DataFrame(metrics["test_results"]).T, width="stretch")
    i1, i2 = st.columns(2)
    i1.image(str(ROOT / "images" / "02_roc_curves.png"))
    i2.image(str(ROOT / "images" / "04_feature_importance.png"))
