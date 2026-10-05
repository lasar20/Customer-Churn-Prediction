"""
Customer Churn Prediction - training pipeline
---------------------------------------------
1. Load & clean the Telco customer dataset
2. Exploratory data analysis (saved as images)
3. Build preprocessing + model pipelines
4. Compare 3 models with cross-validation
5. Evaluate the best model on a held-out test set
6. Save the model, metrics and charts

Run:  python train.py
"""
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, RocCurveDisplay, accuracy_score,
                             classification_report, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_STATE = 42
ROOT = Path(__file__).parent
IMG = ROOT / "images"
sns.set_theme(style="whitegrid", context="talk")
PALETTE = {"No": "#4C78A8", "Yes": "#E45756"}


# ------------------------------------------------------------------ 1. DATA
def load_data() -> pd.DataFrame:
    df = pd.read_csv(ROOT / "data" / "telco_churn.csv")
    # TotalCharges has blank strings for brand-new customers (tenure = 0)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)
    df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})
    return df.drop(columns=["customerID"])


# ------------------------------------------------------------------ 2. EDA
def run_eda(df: pd.DataFrame) -> dict:
    stats = {"rows": len(df), "churn_rate": round((df["Churn"] == "Yes").mean() * 100, 1)}

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    df["Churn"].value_counts().reindex(["No", "Yes"]).plot.bar(
        ax=axes[0, 0], color=[PALETTE["No"], PALETTE["Yes"]], rot=0)
    axes[0, 0].set_title("Churn distribution (imbalanced)")
    axes[0, 0].set_xlabel("")

    rate = (df.assign(c=df["Churn"].eq("Yes")).groupby("Contract")["c"].mean() * 100)
    rate.reindex(["Month-to-month", "One year", "Two year"]).plot.bar(
        ax=axes[0, 1], color="#E45756", rot=0)
    axes[0, 1].set_title("Churn rate by contract type (%)")
    axes[0, 1].set_xlabel("")
    stats["churn_by_contract"] = rate.round(1).to_dict()

    sns.kdeplot(data=df, x="tenure", hue="Churn", fill=True, common_norm=False,
                palette=PALETTE, ax=axes[1, 0])
    axes[1, 0].set_title("Tenure (months) vs churn")

    sns.kdeplot(data=df, x="MonthlyCharges", hue="Churn", fill=True, common_norm=False,
                palette=PALETTE, ax=axes[1, 1])
    axes[1, 1].set_title("Monthly charges vs churn")

    plt.tight_layout()
    plt.savefig(IMG / "01_eda.png", dpi=130)
    plt.close()
    return stats


# ------------------------------------------------------------------ 3. PIPELINES
def build_models(X: pd.DataFrame) -> dict:
    num_cols = X.select_dtypes(include="number").columns.tolist()
    cat_cols = X.select_dtypes(exclude="number").columns.tolist()

    def prep(scale: bool) -> ColumnTransformer:
        return ColumnTransformer([
            ("num", StandardScaler() if scale else "passthrough", num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ])

    return {
        "Logistic Regression": Pipeline([
            ("prep", prep(True)),
            ("model", LogisticRegression(max_iter=2000, class_weight="balanced"))]),
        "Random Forest": Pipeline([
            ("prep", prep(False)),
            ("model", RandomForestClassifier(n_estimators=400, min_samples_leaf=5,
                                             class_weight="balanced",
                                             random_state=RANDOM_STATE, n_jobs=-1))]),
        "Gradient Boosting": Pipeline([
            ("prep", prep(False)),
            ("model", HistGradientBoostingClassifier(learning_rate=0.05, max_iter=200,
                                                     max_depth=4, class_weight="balanced",
                                                     random_state=RANDOM_STATE))]),
    }


# ------------------------------------------------------------------ 4-6. MAIN
def main():
    df = load_data()
    stats = run_eda(df)
    print(f"Rows: {stats['rows']}  |  Churn rate: {stats['churn_rate']}%")

    X = df.drop(columns=["Churn"])
    y = (df["Churn"] == "Yes").astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)

    # ---- model comparison with 5-fold stratified CV (ROC-AUC)
    models = build_models(X)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = {}
    for name, pipe in models.items():
        s = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
        cv_scores[name] = (s.mean(), s.std())
        print(f"{name:22s} CV ROC-AUC = {s.mean():.4f} (+/- {s.std():.4f})")

    best_name = max(cv_scores, key=lambda k: cv_scores[k][0])
    print(f"\nBest model: {best_name}")

    # ---- fit every model on train, evaluate on the untouched test set
    results = {}
    fig, ax = plt.subplots(figsize=(9, 8))
    for name, pipe in models.items():
        pipe.fit(X_train, y_train)
        proba = pipe.predict_proba(X_test)[:, 1]
        pred = (proba >= 0.5).astype(int)
        results[name] = {
            "roc_auc": round(roc_auc_score(y_test, proba), 4),
            "accuracy": round(accuracy_score(y_test, pred), 4),
            "precision": round(precision_score(y_test, pred), 4),
            "recall": round(recall_score(y_test, pred), 4),
            "f1": round(f1_score(y_test, pred), 4),
            "cv_roc_auc": round(cv_scores[name][0], 4),
        }
        RocCurveDisplay.from_predictions(y_test, proba, name=name, ax=ax)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_title("ROC curves on the test set")
    plt.tight_layout()
    plt.savefig(IMG / "02_roc_curves.png", dpi=130)
    plt.close()

    best = models[best_name]
    proba = best.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    print("\n" + classification_report(y_test, pred, target_names=["Stayed", "Churned"]))

    # ---- confusion matrix
    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay.from_predictions(
        y_test, pred, display_labels=["Stayed", "Churned"], cmap="Blues", ax=ax)
    ax.set_title(f"Confusion matrix - {best_name}")
    ax.grid(False)
    plt.tight_layout()
    plt.savefig(IMG / "03_confusion_matrix.png", dpi=130)
    plt.close()

    # ---- permutation importance on the raw (pre-encoding) features
    imp = permutation_importance(best, X_test, y_test, scoring="roc_auc",
                                 n_repeats=10, random_state=RANDOM_STATE, n_jobs=-1)
    imp_df = (pd.DataFrame({"feature": X.columns, "importance": imp.importances_mean})
              .sort_values("importance", ascending=True).tail(10))
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(imp_df["feature"], imp_df["importance"], color="#4C78A8")
    ax.set_title("Top 10 churn drivers (permutation importance)")
    ax.set_xlabel("Drop in ROC-AUC when feature is shuffled")
    plt.tight_layout()
    plt.savefig(IMG / "04_feature_importance.png", dpi=130)
    plt.close()
    top_drivers = imp_df.sort_values("importance", ascending=False)["feature"].tolist()

    # ---- save artefacts
    (ROOT / "models").mkdir(exist_ok=True)
    joblib.dump(best, ROOT / "models" / "churn_model.joblib")
    report = {"best_model": best_name, "dataset": stats, "test_results": results,
              "top_drivers": top_drivers}
    (ROOT / "models" / "metrics.json").write_text(json.dumps(report, indent=2))
    print("\nSaved model, metrics and charts.")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
