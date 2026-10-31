"""Reproducible synthetic customer analytics pipeline with quality checks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


SEED = 17


def generate_inputs(customers: int = 2_000, seed: int = SEED) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    customer_ids = [f"C-{i:05d}" for i in range(customers)]
    customer_table = pd.DataFrame(
        {
            "customer_id": customer_ids,
            "signup_days_ago": rng.integers(1, 2_000, customers),
            "region": rng.choice(["north", "south", "east", "west"], customers),
            "segment": rng.choice(["standard", "plus", "premium"], customers, p=[0.62, 0.28, 0.10]),
        }
    )
    txn_count = rng.poisson(9, customers)
    transaction_customer = np.repeat(customer_ids, txn_count)
    transactions = pd.DataFrame(
        {
            "transaction_id": [f"T-{i:07d}" for i in range(len(transaction_customer))],
            "customer_id": transaction_customer,
            "amount": rng.lognormal(3.3, 0.8, len(transaction_customer)).round(2),
            "status": rng.choice(["approved", "declined"], len(transaction_customer), p=[0.96, 0.04]),
        }
    )
    return customer_table, transactions


def run_pipeline(customer_table: pd.DataFrame, transactions: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    required_customer = {"customer_id", "signup_days_ago", "region", "segment"}
    required_txn = {"transaction_id", "customer_id", "amount", "status"}
    if missing := required_customer - set(customer_table.columns):
        raise ValueError(f"Customer input is missing columns: {sorted(missing)}")
    if missing := required_txn - set(transactions.columns):
        raise ValueError(f"Transaction input is missing columns: {sorted(missing)}")
    if customer_table["customer_id"].duplicated().any():
        raise ValueError("Customer IDs must be unique")
    if transactions["transaction_id"].duplicated().any():
        raise ValueError("Transaction IDs must be unique")
    if (transactions["amount"] < 0).any():
        raise ValueError("Transaction amounts must be non-negative")

    valid_customers = set(customer_table["customer_id"])
    orphan_rows = int((~transactions["customer_id"].isin(valid_customers)).sum())
    clean_txn = transactions[transactions["customer_id"].isin(valid_customers)].copy()
    features = (
        clean_txn.assign(is_approved=clean_txn["status"].eq("approved").astype(int))
        .groupby("customer_id", as_index=False)
        .agg(
            transaction_count=("transaction_id", "count"),
            total_amount=("amount", "sum"),
            average_amount=("amount", "mean"),
            approval_rate=("is_approved", "mean"),
        )
    )
    output = customer_table.merge(features, on="customer_id", how="left", validate="one_to_one")
    for column in ["transaction_count", "total_amount", "average_amount", "approval_rate"]:
        output[column] = output[column].fillna(0)
    quality = {
        "input_customers": int(len(customer_table)),
        "input_transactions": int(len(transactions)),
        "orphan_transactions_removed": orphan_rows,
        "output_customers": int(len(output)),
        "output_duplicate_customer_ids": int(output["customer_id"].duplicated().sum()),
        "output_nulls": int(output.isna().sum().sum()),
    }
    return output, quality


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("output/pipeline"))
    parser.add_argument("--customers", type=int, default=2_000)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    customer_table, transactions = generate_inputs(customers=args.customers)
    customer_table.to_csv(args.output_dir / "synthetic_customers.csv", index=False)
    transactions.to_csv(args.output_dir / "synthetic_transactions.csv", index=False)
    output, quality = run_pipeline(customer_table, transactions)
    output.to_csv(args.output_dir / "customer_analytics_view.csv", index=False)
    (args.output_dir / "data_quality_summary.json").write_text(json.dumps(quality, indent=2) + "\n")
    print(json.dumps(quality, indent=2))
    print(f"Pipeline outputs written to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
