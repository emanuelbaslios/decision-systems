
def evaluate_application(credit_score, pd, requested_amount):
    """ Evaluate a credit application using illustrative policy rules. """
    if credit_score < 500 or pd > 0.15:
        return "Decline"
    if requested_amount > 100000 or pd > 0.08:
        return "Manual Review"
    return "Approve"


if __name__ == "__main__":
    decision = evaluate_application(
        credit_score = 650,
        pd = 0.04,
        requested_amount = 50000
    )
    print(f"Decision: {decision}")
