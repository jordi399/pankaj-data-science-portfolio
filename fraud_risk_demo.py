"""Synthetic transaction-risk scoring demo; no real or client data is used."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


SEED = 42
FEATURES = [
    "log_amount",
    "hour_sin",
    "hour_cos",
    "account_age_days",
    "transactions_last_hour",
    "device_risk_score",
    "geo_mismatch",
    "merchant_risk_score",
]


def make_synthetic_data(rows: int = 20_000, seed: int = SEED) -> pd.DataFrame:
    """Generate a toy, imbalanced transaction dataset with plausible signals."""
    rng = np.random.default_rng(seed)
    hour = rng.integers(0, 24, size=rows)
    amount = rng.lognormal(mean=3.5, sigma=1.0, size=rows)
    account_age = rng.integers(1, 3_000, size=rows)
    velocity = rng.poisson(lam=1.7, size=rows)
    device_risk = rng.beta(2, 5, size=rows)
    geo_mismatch = rng.binomial(1, 0.035, size=rows)
    merchant_risk = rng.beta(2, 6, size=rows)

    logit = (
        -4.8
        + 0.55 * np.log1p(amount)
        - 0.00035 * account_age
        + 0.20 * velocity
        + 2.1 * device_risk
        + 1.35 * geo_mismatch
        + 1.65 * merchant_risk
        + 0.55 * ((hour <= 4) | (hour >= 23))
    )
    probability = 1 / (1 + np.exp(-logit))
    fraud = rng.binomial(1, probability)

    return pd.DataFrame(
        {
            "transaction_id": [f"SYN-{i:06d}" for i in range(rows)],
            "amount": amount.round(2),
            "hour": hour,
            "account_age_days": account_age,
            "transactions_last_hour": velocity,
            "device_risk_score": device_risk.round(4),
            "geo_mismatch": geo_mismatch,
            "merchant_risk_score": merchant_risk.round(4),
            "is_fraud": fraud,
        }
    )


def engineer_features(data: pd.DataFrame) -> pd.DataFrame:
    result = data.copy()
    result["log_amount"] = np.log1p(result["amount"])
    result["hour_sin"] = np.sin(2 * np.pi * result["hour"] / 24)
    result["hour_cos"] = np.cos(2 * np.pi * result["hour"] / 24)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("output/fraud"))
    parser.add_argument("--rows", type=int, default=20_000)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    data = engineer_features(make_synthetic_data(rows=args.rows))
    train, test = train_test_split(
        data, test_size=0.25, random_state=SEED, stratify=data["is_fraud"]
    )
    X_train, y_train = train[FEATURES], train["is_fraud"]
    X_test, y_test = test[FEATURES], test["is_fraud"]

    logistic = Pipeline(
        [
            ("scale", StandardScaler()),
            ("model", LogisticRegression(class_weight="balanced", max_iter=1_000)),
        ]
    )
    forest = RandomForestClassifier(
        n_estimators=100,
        max_depth=12,
        min_samples_leaf=8,
        class_weight="balanced_subsample",
        n_jobs=1,
        random_state=SEED,
    )

    summary = []
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, model in [("Logistic regression", logistic), ("Random forest", forest)]:
        model.fit(X_train, y_train)
        scores = model.predict_proba(X_test)[:, 1]
        precision, recall, thresholds = precision_recall_curve(y_test, scores)
        ap = average_precision_score(y_test, scores)
        auc = roc_auc_score(y_test, scores)
        f1_scores = 2 * precision[:-1] * recall[:-1] / (
            precision[:-1] + recall[:-1] + 1e-12
        )
        best_index = int(np.argmax(f1_scores))
        threshold = float(thresholds[best_index]) if len(thresholds) else 0.5
        summary.append(
            {
                "model": name,
                "average_precision": round(ap, 4),
                "roc_auc": round(auc, 4),
                "illustrative_f1_threshold": round(threshold, 4),
                "note": "synthetic test split; not a production result",
            }
        )
        ax.plot(recall, precision, label=f"{name} (AP={ap:.3f})")

    ax.set(
        title="Precision–recall trade-off (synthetic data)",
        xlabel="Recall",
        ylabel="Precision",
    )
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.output_dir / "precision_recall.png", dpi=160)
    plt.close(fig)

    data.drop(columns=["log_amount", "hour_sin", "hour_cos"]).to_csv(
        args.output_dir / "synthetic_transactions.csv", index=False
    )
    pd.DataFrame(summary).to_csv(args.output_dir / "model_summary.csv", index=False)
    print(pd.DataFrame(summary).to_string(index=False))
    print(f"\nSynthetic demo outputs written to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
