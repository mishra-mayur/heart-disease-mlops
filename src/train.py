"""Train, compare and register reproducible scikit-learn pipelines with MLflow."""
import json
import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, RocCurveDisplay
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from src.config import PROCESSED_DATA, MODEL_PATH, ARTIFACTS_DIR, PLOTS_DIR
from src.data import FEATURES, split_features_target

def make_pipeline(model):
    preprocessor = ColumnTransformer([("numeric", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), FEATURES)])
    return Pipeline([("preprocess", preprocessor), ("model", model)])

def evaluate(model, x_train, y_train, x_test, y_test):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_auc = cross_val_score(model, x_train, y_train, cv=cv, scoring="roc_auc")
    model.fit(x_train, y_train)
    prob = model.predict_proba(x_test)[:, 1]
    pred = (prob >= 0.5).astype(int)
    return {"cv_roc_auc_mean": float(np.mean(cv_auc)), "accuracy": accuracy_score(y_test, pred), "precision": precision_score(y_test, pred, zero_division=0), "recall": recall_score(y_test, pred, zero_division=0), "roc_auc": roc_auc_score(y_test, prob)}, prob

def main():
    if not PROCESSED_DATA.exists():
        raise FileNotFoundError("Run: python src/eda.py")
    ARTIFACTS_DIR.mkdir(exist_ok=True); PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    mlflow.set_experiment("heart-disease-risk")
    df = pd.read_csv(PROCESSED_DATA); x, y = split_features_target(df)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=42)
    candidates = {"logistic_regression": (LogisticRegression(max_iter=2000, C=1.0, random_state=42), {"C": 1.0}), "random_forest": (RandomForestClassifier(n_estimators=350, max_depth=7, min_samples_leaf=3, random_state=42, class_weight="balanced"), {"n_estimators": 350, "max_depth": 7, "min_samples_leaf": 3})}
    results, fitted = {}, {}
    for name, (estimator, params) in candidates.items():
        with mlflow.start_run(run_name=name):
            pipeline = make_pipeline(estimator)
            metrics, prob = evaluate(pipeline, x_train, y_train, x_test, y_test)
            mlflow.log_params({"model": name, **params, "test_size": 0.2, "random_state": 42})
            mlflow.log_metrics(metrics)
            fig, ax = plt.subplots(); RocCurveDisplay.from_predictions(y_test, prob, name=name, ax=ax); ax.set_title(f"ROC curve - {name}")
            plot_path = PLOTS_DIR / f"roc_{name}.png"; fig.savefig(plot_path, dpi=180, bbox_inches="tight"); plt.close(fig)
            mlflow.log_artifact(str(plot_path), artifact_path="plots")
            mlflow.sklearn.log_model(pipeline, artifact_path="model")
            results[name] = metrics; fitted[name] = pipeline
    best_name = max(results, key=lambda name: results[name]["roc_auc"])
    joblib.dump(fitted[best_name], MODEL_PATH)
    (ARTIFACTS_DIR / "metrics.json").write_text(json.dumps({"selected_model": best_name, "results": results}, indent=2))
    print(json.dumps({"selected_model": best_name, "metrics": results[best_name]}, indent=2))

if __name__ == "__main__":
    main()

