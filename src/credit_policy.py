
import numpy as np
import pandas as pd

def evaluate_policy(data, pd_cutoff, lgd=0.65, funding_cost=0.04):
    """Evaluate a credit approval policy using expected unit economics."""
    df = data.copy()
    required = ["pd", "exposure", "interest_rate"]
    if not set(required).issubset(df.columns):
        raise ValueError(f"Required columns: {required}")
    if not 0 <= pd_cutoff <= 1 or not 0 <= lgd <= 1:
        raise ValueError("PD cutoff and LGD must be between 0 and 1.")
    if not np.isfinite(df[required].to_numpy(dtype=float)).all():
        raise ValueError("Inputs must contain finite numeric values.")
    if ((df["pd"] < 0) | (df["pd"] > 1)).any():
        raise ValueError("PD values must be between 0 and 1.")
    if (df["exposure"].lt(0) | df["interest_rate"].lt(0)).any():
        raise ValueError("Exposure and interest rates must be nonnegative.")

    df["approved"] = df["pd"] <= pd_cutoff
    df["expected_loss"] = df["pd"] * lgd * df["exposure"]
    df["expected_income"] = (
        (1 - df["pd"]) * df["interest_rate"] * df["exposure"]
    )
    df["funding_expense"] = funding_cost * df["exposure"]
    df["expected_contribution"] = (
        df["expected_income"] - df["expected_loss"]
        - df["funding_expense"]
    )

    approved = df.loc[df["approved"]]
    return {
        "PD Cutoff": pd_cutoff,
        "Approval Rate": df["approved"].mean(),
        "Approved Exposure": approved["exposure"].sum(),
        "Expected Loss": approved["expected_loss"].sum(),
        "Expected Contribution": approved["expected_contribution"].sum()
    }

def compare_policies(data, cutoffs, lgd=0.65, funding_cost=0.04):
    """Compare alternative credit approval thresholds."""
    results = [
        evaluate_policy(data, cutoff, lgd, funding_cost)
        for cutoff in cutoffs
    ]
    return pd.DataFrame(results)

if __name__ == "__main__":
    rng = np.random.default_rng(42)
    applications = pd.DataFrame({
        "pd": rng.beta(2, 18, 1000),
        "exposure": rng.uniform(5000, 100000, 1000),
        "interest_rate": rng.uniform(0.12, 0.25, 1000)
    })
    results = compare_policies(
        applications, cutoffs=[0.03, 0.05, 0.08, 0.10, 0.15]
    )
    print(results.round(4).to_string(index=False))
