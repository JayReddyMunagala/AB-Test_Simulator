# A/B Test Simulator (Python)

This is a **Python rewrite** of the original TypeScript/React A/B Test Simulator.

## Features
- Generate synthetic A/B experiment data (optional user segments)
- Two-proportion z-test
  - conversion rates, z-score, p-value
  - confidence interval on the difference (B − A)
- Lift
  - absolute lift (difference)
  - relative lift (%)
- Simple recommendation text (ship / don’t ship / consider context)
- Power analysis
  - estimated power for the current sample size
  - recommended sample size per group to reach ~80% power
- Experiment history (saved to SQLite)
- CSV export

## Quickstart

```bash
# from this folder
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate

pip install -r requirements.txt
streamlit run app.py
```

## Storage
A local SQLite database file `abtest.db` is created in this project folder the first time you run the app.

## How this relates to the original TS project
A React/Vite frontend can’t be "converted" into Python line-by-line (Python is typically backend / data / scripting).
So this rewrite keeps the same *logic* in Python (`abtest/`) and rebuilds the UI in Python using **Streamlit**.
