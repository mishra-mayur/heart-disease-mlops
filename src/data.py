import pandas as pd

FEATURES = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
]
TARGET = "target"


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the UCI CSV while retaining only reproducible model columns."""
    frame = df.copy()
    raw_target = next(
        (name for name in ("num", TARGET, "class") if name in frame.columns), None
    )
    if raw_target is None:
        raise ValueError("Expected a 'num', 'target', or 'class' outcome column")
    numeric_target = pd.to_numeric(frame[raw_target], errors="coerce")
    if numeric_target.notna().any():
        frame[TARGET] = (numeric_target > 0).astype("Int64")
    else:
        labels = frame[raw_target].astype(str).str.lower().str.strip()
        frame[TARGET] = labels.isin({"1", "yes", "true", "present", "disease"}).astype(
            "Int64"
        )
    for col in FEATURES:
        if col not in frame.columns:
            raise ValueError(f"Missing required feature: {col}")
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame[TARGET] = pd.to_numeric(frame[TARGET], errors="coerce")
    return (
        frame[FEATURES + [TARGET]].dropna().astype({TARGET: int}).reset_index(drop=True)
    )


def split_features_target(df: pd.DataFrame):
    return df[FEATURES], df[TARGET]
