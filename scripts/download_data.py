"""Download the processed UCI heart disease dataset from OpenML."""
from pathlib import Path
import pandas as pd
from sklearn.datasets import fetch_openml

OUT = Path("data/raw/heart_disease_uci.csv")
OUT.parent.mkdir(parents=True, exist_ok=True)

def main():
    dataset = fetch_openml(name="heart-disease", version=1, as_frame=True, parser="auto")
    df = dataset.frame
    # Normalize common UCI/OpenML column spellings to the project schema.
    df.columns = [str(c).lower().replace(" ", "_") for c in df.columns]
    aliases = {"trestbps": "trestbps", "thalach": "thalach", "oldpeak": "oldpeak", "target": "num"}
    df = df.rename(columns=aliases)
    df.to_csv(OUT, index=False)
    print(f"Saved {len(df)} rows to {OUT}")

if __name__ == "__main__":
    main()

