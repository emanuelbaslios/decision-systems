
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from policy_optimization import optimize_credit_policy

def validate_credit_policy(applications, test_size=0.30,
                           random_state=42, max_loss_rate=0.04,
                           **policy_kwargs):
    """Select a policy on development data and evaluate it out of sample."""
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")
    if len(applications) < 2:
        raise ValueError("At least two applications are required.")

    development, validation = train_test_split(
        applications, test_size=test_size, random_state=random_state
    )
    _, optimal = optimize_credit_policy(
        development, max_loss_rate=max_loss_rate, **policy_kwargs
    )
    if optimal is None:
        raise ValueError("No feasible development policy identified.")

    cutoff = float(optimal["PD_Cutoff"])
    results = []
    for sample, df in [("Development", development),
                       ("Validation", validation)]:
        evaluated, _ = optimize_credit_policy(
            df, pd_cutoffs=[cutoff],
            max_loss_rate=max_loss_rate, **policy_kwargs
        )
        row = evaluated.iloc[0].to_dict()
        row["Sample"] = sample
        results.append(row)

    return pd.DataFrame(results), cutoff

if __name__ == "__main__":
    rng = np.random.default_rng(42)
    applications = pd.DataFrame({
        "credit_score": rng.integers(450, 800, 2000),
        "pd": rng.beta(2, 18, 2000),
        "requested_amount": rng.uniform(5000, 100000, 2000),
        "interest_rate": rng.uniform(0.12, 0.25, 2000)
    })
    results, cutoff = validate_credit_policy(applications)
    print(f"Selected PD Cutoff: {cutoff:.2%}")
    print(results.round(4).to_string(index=False))
