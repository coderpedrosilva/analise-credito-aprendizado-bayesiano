import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score


def evaluate_naive_bayes(model, X_test, y_test):
    """
    Avalia o modelo Naive Bayes.
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    return {
        "model": "Naive Bayes",
        "accuracy": accuracy_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba)
    }


def evaluate_bayesian_logistic(trace, X_test, y_test):
    """
    Avalia a Regressão Logística Bayesiana usando a média posterior.
    """
    intercept_samples = trace.posterior["intercept"].values
    coef_samples = trace.posterior["coefficients"].values

    intercept_mean = intercept_samples.mean()
    coef_mean = coef_samples.mean(axis=(0, 1))

    logits = intercept_mean + np.dot(X_test, coef_mean)
    probs = 1 / (1 + np.exp(-logits))

    y_pred = (probs >= 0.5).astype(int)

    return {
        "model": "Bayesian Logistic Regression",
        "accuracy": accuracy_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, probs)
    }
