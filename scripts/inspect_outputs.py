from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "gpt_outputs"

csv_path = OUTPUT_DIR / "experiments.csv"
text_path = OUTPUT_DIR / "generated_text.txt"
model_path = OUTPUT_DIR / "mini_gpt.pt"

print("=" * 70)
print("MY_GPT PORTFOLIO OUTPUT INSPECTOR")
print("=" * 70)

if csv_path.exists():
    print("\n[experiments.csv]")
    try:
        df = pd.read_csv(csv_path)
        print("Rows:", len(df))
        print("Columns:", list(df.columns))
        print("\nPreview:")
        print(df.head().to_string(index=False))
    except Exception as exc:
        print("Could not read experiments.csv:", exc)
else:
    print("\nexperiments.csv not found.")

if text_path.exists():
    print("\n[generated_text.txt]")
    text = text_path.read_text(encoding="utf-8", errors="replace")
    print(text[:2000])
else:
    print("\ngenerated_text.txt not found.")

if model_path.exists():
    size_mb = model_path.stat().st_size / (1024 * 1024)
    print(f"\n[mini_gpt.pt]\nCheckpoint found — {size_mb:.2f} MB")
else:
    print("\nmini_gpt.pt not found.")
