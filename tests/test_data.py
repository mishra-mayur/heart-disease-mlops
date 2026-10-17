import pandas as pd
from src.data import FEATURES, clean_data


def test_clean_data_creates_binary_target_and_drops_missing():
    row = {c: 1 for c in FEATURES}
    row.update(age=50, num=2)
    missing = row.copy()
    missing["chol"] = None
    result = clean_data(pd.DataFrame([row, missing]))
    assert list(result.columns) == FEATURES + ["target"]
    assert len(result) == 1 and result.iloc[0]["target"] == 1
