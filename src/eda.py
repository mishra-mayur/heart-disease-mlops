"""Generate EDA assets required for the report."""
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from src.config import RAW_DATA, PROCESSED_DATA, PLOTS_DIR
from src.data import clean_data

def main():
    if not RAW_DATA.exists():
        raise FileNotFoundError("Run: python scripts/download_data.py")
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    df = clean_data(__import__("pandas").read_csv(RAW_DATA))
    PROCESSED_DATA.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_DATA, index=False)
    sns.set_theme(style="whitegrid")
    ax = sns.countplot(data=df, x="target", hue="target", legend=False, palette="Set2")
    ax.set(title="Class balance after cleaning", xlabel="Heart disease", ylabel="Patients")
    plt.tight_layout(); plt.savefig(PLOTS_DIR / "class_balance.png", dpi=180); plt.close()
    df.hist(figsize=(14, 10), bins=20)
    plt.suptitle("Feature distributions", y=1.02); plt.tight_layout(); plt.savefig(PLOTS_DIR / "histograms.png", dpi=180); plt.close()
    plt.figure(figsize=(12, 9))
    sns.heatmap(df.corr(numeric_only=True), cmap="vlag", center=0, square=True)
    plt.title("Feature correlation heatmap"); plt.tight_layout(); plt.savefig(PLOTS_DIR / "correlation_heatmap.png", dpi=180); plt.close()
    print(f"Clean dataset and plots written to {PROCESSED_DATA.parent} and {PLOTS_DIR}")

if __name__ == "__main__":
    main()

