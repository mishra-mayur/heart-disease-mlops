from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DATA = DATA_DIR / "raw" / "heart_disease_uci.csv"
PROCESSED_DATA = DATA_DIR / "processed" / "heart_disease_clean.csv"
ARTIFACTS_DIR = ROOT / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "best_model.joblib"
PLOTS_DIR = ARTIFACTS_DIR / "plots"

