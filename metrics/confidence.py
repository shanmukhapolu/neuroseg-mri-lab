def confidence(metrics):

    score = 100

    if metrics["Circularity"] < 0.55:
        score -= 15

    if metrics["Solidity"] < 0.80:
        score -= 15

    if metrics["Irregularity"] > 1.6:
        score -= 20

    score = max(score,40)

    return score