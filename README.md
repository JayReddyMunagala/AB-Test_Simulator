# A/B Test Simulator

An interactive A/B testing simulator that generates synthetic experiment data, runs statistical significance testing, estimates power, and provides decision-ready insights. It helps understand how sample size, baseline conversion, and expected uplift affect experiment outcomes.

## Features
- Generates user level A/B experiment data (Control vs Treatment)
- Computes conversion rates, absolute lift, and relative lift
- Runs a two-proportion z-test (z-score + p-value)
- Computes a 95% confidence interval for lift
- Estimates statistical power via repeated simulation
- Suggests a recommended sample size to reach a target power level
- Optional segmentation (e.g., new vs returning) with segment-level results
- Exports generated data to CSV
- Saves experiment runs and supports viewing history

## Run Locally

### Requirements
- Python 3.10+

### Setup
```bash
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows (PowerShell)
# .venv\Scripts\Activate.ps1

pip install -r requirements.txt
````
### Start App
```bash
streamlit run app.py
````

### Usage

Click Run Experiment to generate data and view results.

(Optional) Enable Segments for segment-level analysis.

Click Run Power Analysis to estimate detection probability.

Use Download CSV to export the generated dataset.

Check History to view prior experiment runs.

### Project Structure

app.py — Streamlit UI and visualizations

abtest/data_generator.py — synthetic experiment data generation

abtest/statistics.py — hypothesis testing, confidence intervals, power simulation

abtest/storage.py — local persistence for experiment history

requirements.txt — dependencies
