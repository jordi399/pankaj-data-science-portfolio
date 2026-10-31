"""Illustrative marketing-mix demo on generated weekly data only."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler


SEED = 24
CHANNELS = ["search", "social", "tv"]


def geometric_adstock(values: np.ndarray, decay: float) -> np.ndarray:
    result = np.zeros_like(values, dtype=float)
    for index, value in enumerate(values):
        result[index] = value + (decay * result[index - 1] if index else 0.0)
    return result


def make_synthetic_data(weeks: int = 156, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    time = np.arange(weeks)
    spend = {
        "search": rng.gamma(7, 110, weeks),
        "social": rng.gamma(6, 95, weeks),
        "tv": rng.gamma(5, 150, weeks),
    }
    carry = {
        "search": geometric_adstock(spend["search"], 0.25),
        "social": geometric_adstock(spend["social"], 0.45),
        "tv": geometric_adstock(spend["tv"], 0.65),
    }
    response = {
        channel: np.log1p(carry[channel] / 350) for channel in CHANNELS
    }
    seasonal = 35 * np.sin(2 * np.pi * time / 52) + 16 * np.cos(2 * np.pi * time / 26)
    sales = (
        900
        + 130 * response["search"]
        + 110 * response["social"]
        + 155 * response["tv"]
        + seasonal
        + 0.35 * time
        + rng.normal(0, 20, weeks)
    )
    data = pd.DataFrame({"week": time + 1, "sales": sales, **{f"{c}_spend": spend[c] for c in CHANNELS}})
    for channel in CHANNELS:
        adstock = geometric_adstock(data[f"{channel}_spend"].to_numpy(), {"search": 0.25, "social": 0.45, "tv": 0.65}[channel])
        data[f"{channel}_response"] = np.log1p(adstock / 350)
    data["annual_sin"] = np.sin(2 * np.pi * time / 52)
    data["annual_cos"] = np.cos(2 * np.pi * time / 52)
    data["time_trend"] = time / weeks
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("output/marketing_mix"))
    parser.add_argument("--weeks", type=int, default=156)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    data = make_synthetic_data(weeks=args.weeks)
    features = [f"{c}_response" for c in CHANNELS] + ["annual_sin", "annual_cos", "time_trend"]
    split = int(len(data) * 0.8)
    X_train, X_test = data[features].iloc[:split], data[features].iloc[split:]
    y_train, y_test = data["sales"].iloc[:split], data["sales"].iloc[split:]

    # Standardize using training data only. The resulting coefficients are useful
    # for a reproducible demo, not direct estimates of causal contribution.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    model = Ridge(alpha=4.0)
    model.fit(X_train_scaled, y_train)
    prediction = model.predict(X_test_scaled)

    coefficients = pd.DataFrame({"feature": features, "standardized_coefficient": model.coef_})
    coefficients.to_csv(args.output_dir / "illustrative_model_coefficients.csv", index=False)
    data.to_csv(args.output_dir / "synthetic_weekly_data.csv", index=False)

    fig, axes = plt.subplots(2, 1, figsize=(9, 7), constrained_layout=True)
    axes[0].plot(data["week"], data["sales"], color="#223b67", linewidth=1.6)
    axes[0].set(title="Synthetic weekly sales", xlabel="Week", ylabel="Sales index")
    axes[0].grid(alpha=0.2)
    axes[1].bar(coefficients["feature"], coefficients["standardized_coefficient"], color="#36a28b")
    axes[1].set(title="Fitted standardized coefficients (illustrative)", ylabel="Coefficient")
    axes[1].tick_params(axis="x", rotation=25)
    axes[1].grid(axis="y", alpha=0.2)
    fig.savefig(args.output_dir / "marketing_mix_demo.png", dpi=160)
    plt.close(fig)

    print(f"Holdout MAE: {mean_absolute_error(y_test, prediction):.2f}")
    print(f"Holdout R-squared: {r2_score(y_test, prediction):.3f}")
    print("All results are illustrative, generated from synthetic data, and not causal findings.")
    print(f"Demo outputs written to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
