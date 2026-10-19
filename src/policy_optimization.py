
import numpy as np
import pandas as pd

def optimize_credit_policy(applications, pd_cutoffs=None, max_loss_rate=0.04,
                           lgd=0.65, funding_cost=0.04, min_score=500,
                           max_amount=100000):
    """Optimize a PD cutoff subject to an expected loss-rate constraint."""
    required = ["credit_score", "pd", "requested_amount", "interest_rate"]
    missing = set(required) - set(applications.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    if applications.empty or applications[required].isna().any().any():
        raise ValueError("Applications must be nonempty with complete inputs.")
    values = applications[required].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Inputs must be finite.")
    if not applications["pd"].between(0, 1).all():
        raise ValueError("PD must be between 0 and 1.")
    if (applications[["requested_amount", "interest_rate"]] < 0).any().any():
        raise ValueError("Amounts and rates must be nonnegative.")
    if not 0 <= max_loss_rate <= 1 or not 0 <= lgd <= 1 or funding_cost < 0:
        raise ValueError("Invalid policy assumptions.")

    if pd_cutoffs is None:
        pd_cutoffs = np.arange(0.02, 0.151, 0.01)
    pd_cutoffs = np.asarray(pd_cutoffs, dtype=float)
    if pd_cutoffs.ndim != 1 or len(pd_cutoffs) == 0:
        raise ValueError("Provide at least one PD cutoff.")
    if not np.isfinite(pd_cutoffs).all() or ((pd_cutoffs < 0) | (pd_cutoffs > 1)).any():
        raise ValueError("PD cutoffs must be between 0 and 1.")

    results = []
    for cutoff in pd_cutoffs:
        approved = applications.loc[
            (applications["credit_score"] >= min_score)
            & (applications["pd"] <= cutoff)
            & (applications["requested_amount"] <= max_amount)
        ]
        exposure = approved["requested_amount"]
        expected_loss = (approved["pd"] * lgd * exposure).sum()
        expected_income = (
            (1 - approved["pd"]) * approved["interest_rate"] * exposure
        ).sum()
        funding_expense = (funding_cost * exposure).sum()
        total_exposure = exposure.sum()
        loss_rate = expected_loss / total_exposure if total_exposure > 0 else np.nan

        results.append({
            "PD_Cutoff": round(float(cutoff), 4),
            "Approvals": len(approved),
            "Approval_Rate": len(approved) / len(applications),
            "Approved_Exposure": total_exposure,
            "Expected_Loss": expected_loss,
            "Expected_Loss_Rate": loss_rate,
            "Expected_Contribution": expected_income - expected_loss - funding_expense,
            "Feasible": total_exposure > 0 and loss_rate <= max_loss_rate
        })

    results = pd.DataFrame(results)
    feasible = results.loc[results["Feasible"]]
    optimal = (
        feasible.sort_values(
            ["Expected_Contribution", "PD_Cutoff"],
            ascending=[False, True]
        ).iloc[0]
        if not feasible.empty else None
    )
    return results, optimal

if __name__ == "__main__":
    rng = np.random.default_rng(42)
    applications = pd.DataFrame({
        "credit_score": rng.integers(450, 800, 1000),
        "pd": rng.beta(2, 18, 1000),
        "requested_amount": rng.uniform(5000, 100000, 1000),
        "interest_rate": rng.uniform(0.12, 0.25, 1000)
    })
    results, optimal = optimize_credit_policy(applications)
    print(results.round(4).to_string(index=False))
    print("\nOptimal Feasible Policy:")
    print(optimal.to_string() if optimal is not None else "No feasible policy")
