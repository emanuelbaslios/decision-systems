
import pandas as pd
from decision_engine import evaluate_application

def evaluate_portfolio(applications):
    """Evaluate application-level decisions across a portfolio."""
    required = ["credit_score", "pd", "requested_amount"]
    missing = set(required) - set(applications.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = applications.copy()
    if df[required].isna().any().any():
        raise ValueError("Policy inputs must not contain missing values.")
    if not df["pd"].between(0, 1).all():
        raise ValueError("PD must be between 0 and 1.")
    if (df["requested_amount"] < 0).any():
        raise ValueError("Requested amount must be nonnegative.")

    df["decision"] = df.apply(
        lambda row: evaluate_application(
            credit_score=row["credit_score"],
            pd=row["pd"],
            requested_amount=row["requested_amount"]
        ), axis=1
    )
    return df

def summarize_policy(results):
    """Summarize decision distribution and requested exposure."""
    decisions = ["Approve", "Manual Review", "Decline"]
    summary = results.groupby("decision").agg(
        Applications=("decision", "size"),
        Requested_Exposure=("requested_amount", "sum"),
        Average_PD=("pd", "mean")
    ).reindex(decisions, fill_value=0)
    summary["Application_Rate"] = (
        summary["Applications"] / len(results)
        if len(results) else 0
    )
    return summary.reset_index()

if __name__ == "__main__":
    applications = pd.DataFrame({
        "credit_score": [720, 650, 490, 610, 580, 700],
        "pd": [0.02, 0.09, 0.04, 0.12, 0.06, 0.03],
        "requested_amount": [40000, 60000, 30000, 80000, 120000, 50000]
    })
    results = evaluate_portfolio(applications)
    print(results.to_string(index=False))
    print(summarize_policy(results).round(4).to_string(index=False))
