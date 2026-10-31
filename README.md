# Pankaj Singh Rawat — Independent Data Science Portfolio Demos

These are educational portfolio demonstrations built specifically for this portfolio using generated synthetic data. They contain no employer or client data, code, results, or confidential materials. Results are illustrative and must not be presented as production performance or client outcomes.

## Projects

### 1. Transaction risk scoring

`fraud_risk_demo.py` generates synthetic transactions with a small fraud class, engineers simple behavioral signals, trains a baseline logistic model and a tree model, compares precision/recall, and illustrates threshold selection. It emphasizes evaluation under class imbalance rather than accuracy alone.

For the included fixed-seed sample run, average precision was 0.367 for logistic regression and 0.346 for random forest. These figures describe only this generated toy dataset.

### 2. Marketing mix modeling

`marketing_mix_demo.py` creates synthetic weekly media spend and sales with carryover, diminishing returns, seasonality, and noise. It applies adstock and saturation transformations, fits a regularized regression, and visualizes the synthetic series and modeled channel effects. This is a teaching example, not causal proof or a calibrated budget recommendation.

For the included sample run, holdout MAE was 19.23 sales-index points and holdout R-squared was 0.724. These are illustrative outputs on generated data, not client performance.

### 3. Reusable analytics pipeline

`analytics_pipeline_demo.py` builds a reproducible synthetic customer and transaction dataset, validates schema and key constraints, creates customer-level features, and writes clean outputs plus a data-quality summary. It demonstrates a compact, auditable Python workflow that can be adapted to Spark for larger datasets.

The included sample run creates 2,000 synthetic customers and 17,732 synthetic transactions, removes no orphan records, and produces a customer-level view with no null values.

## Requirements

Python 3.10 or newer, plus `numpy`, `pandas`, `scikit-learn`, and `matplotlib`.

```bash
python -m pip install numpy pandas scikit-learn matplotlib
python fraud_risk_demo.py --output-dir output/fraud
python marketing_mix_demo.py --output-dir output/marketing_mix
python analytics_pipeline_demo.py --output-dir output/pipeline
```

Each script uses a fixed random seed and writes its generated data and figures only to the selected output directory. Do not add client data to these samples. Review the code and outputs before uploading them to a marketplace.

The repository contains the scripts and instructions; generated CSVs, summaries, and figures are not committed. Run each script to recreate the outputs locally. The `--output-dir` option can be used to choose a different output location.
