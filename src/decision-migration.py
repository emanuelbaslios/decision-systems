
import pandas as pd
from credit_policy import evaluate_portfolio
from policy_comparison import evaluate_challenger

DECISIONS = ["Approve", "Manual Review", "Decline"]

def decision_migration(applications):
    """Compare application decisions under champion and challenger policies."""
    champion = evaluate_portfolio(applications)
    challenger = evaluate_challenger(applications)
    migration = pd.DataFrame({
        "Champion": champion["decision"].to_numpy(),
        "Challenger": challenger["decision"].to_numpy(),
        "Requested_Amount": applications["requested_amount"].to_numpy()
    })
    migration["Changed"] = migration["Champion"] != migration["Challenger"]
    return migration

def migration_matrix(migration, normalize=False):
    """Summarize application movement between policy decisions."""
    matrix = pd.crosstab(
        migration["Champion"], migration["Challenger"]
    ).reindex(index=DECISIONS, columns=DECISIONS, fill_value=0)
    if normalize:
        matrix = matrix.div(matrix.sum(axis=1).replace(0, float("nan")), axis=0)
    return matrix

def migration_summary(migration):
    """Summarize decision changes and affected requested exposure."""
    changed = migration["Changed"]
    return pd.DataFrame([{
        "Applications": len(migration),
        "Changed_Decisions": int(changed.sum()),
        "Migration_Rate": changed.mean() if len(migration) else 0,
        "Affected_Requested_Exposure": migration.loc[
            changed, "Requested_Amount"
        ].sum()
    }])

if __name__ == "__main__":
    applications = pd.DataFrame({
        "credit_score": [720, 650, 490, 610, 580, 700, 510, 690],
        "pd": [0.02, 0.09, 0.04, 0.12, 0.06, 0.03, 0.04, 0.07],
        "requested_amount": [
            40000, 60000, 30000, 80000, 120000, 50000, 45000, 70000
        ]
    })
    migration = decision_migration(applications)
    print("Decision Migration Matrix")
    print(migration_matrix(migration))
    print("\nMigration Summary")
    print(migration_summary(migration).round(4).to_string(index=False))
