
import numpy as np
import pandas as pd
from credit_policy import evaluate_portfolio
from policy_comparison import evaluate_challenger

def evaluate_economics(applications, lgd=0.65, funding_cost=0.04):
    """Evaluate expected one-period economics under two credit policies."""
    required = ["credit_score", "pd", "requested_amount", "interest_rate"]
    missing = set(required) - set(applications.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    if applications.empty:
        raise ValueError("Applications cannot be empty.")
    if not 0 <= lgd <= 1 or not np.isfinite(funding_cost) or funding_cost < 0:
        raise ValueError("Invalid LGD or funding cost.")
    if applications[required].isna().any().any():
        raise ValueError("Inputs cannot contain missing values.")
    if not np.isfinite(applications[required].to_numpy(dtype=float)).all():
        raise ValueError("Inputs must be finite.")
    if not applications["pd"].between(0, 1).all():
        raise ValueError("PD must be between 0 and 1.")
    if (applications[["requested_amount", "interest_rate"]] < 0).any().any():
        raise ValueError("Amounts and rates must be nonnegative.")

    policies = {
        "Champion": evaluate_portfolio(applications),
        "Challenger": evaluate_challenger(applications)
    }
    summaries = []
    for name, df in policies.items():
        approved = df.loc[df["decision"] == "Approve"].copy()
        exposure = approved["requested_amount"]
        pd_values = approved["pd"]

        expected_loss = (pd_values * lgd * exposure).sum()
        expected_income = (
            (1 - pd_values) * approved["interest_rate"] * exposure
        ).sum()
        funding_expense = (funding_cost * exposure).sum()

        summaries.append({
            "Policy": name,
            "Applications": len(df),
            "Approvals": len(approved),
            "Approval_Rate": len(approved) / len(df),
            "Approved_Exposure": exposure.sum(),
            "Expected_Loss": expected_loss,
            "Expected_Income": expected_income,
            "Funding_Expense": funding_expense,
            "Expected_Contribution": (
                expected_income - expected_loss - funding_expense
            )
        })
    return pd.DataFrame(summaries)

def incremental_impact(results):
    """Calculate challenger minus champion economic outcomes."""
    metrics = [
        "Approvals", "Approval_Rate", "Approved_Exposure",
        "Expected_Loss", "Expected_Income",
        "Funding_Expense", "Expected_Contribution"
    ]
    indexed = results.set_index("Policy")
    delta = indexed.loc["Challenger", metrics] - indexed.loc["Champion", metrics]
    return pd.DataFrame({
        "Metric": metrics,
        "Champion": indexed.loc["Champion", metrics].to_numpy(),
        "Challenger": indexed.loc["Challenger", metrics].to_numpy(),
        "Incremental_Impact": delta.to_numpy()
    })

if __name__ == "__main__":
    applications = pd.DataFrame({
        "credit_score": [720, 650, 490, 610, 580, 700, 510, 690],
        "pd": [0.02, 0.09, 0.04, 0.12, 0.06, 0.03, 0.04, 0.07],
        "requested_amount": [
            40000, 60000, 30000, 80000,
            120000, 50000, 45000, 70000
        ],
        "interest_rate": [0.15, 0.19, 0.16, 0.21, 0.18, 0.14, 0.17, 0.20]
    })
    results = evaluate_economics(applications)
    print("Policy Economics")
    print(results.round(4).to_string(index=False))
    print("\nIncremental Impact")
    print(incremental_impact(results).round(4).to_string(index=False))
