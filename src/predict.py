import numpy as np
import pymc as pm

def predict_probabilities(model, trace, X):
    """
    Retorna a probabilidade média de aprovação por cliente.
    """
    posterior = trace.posterior
    intercept = posterior["intercept"].mean(dim=("chain", "draw")).values
    coefs = posterior["coefficients"].mean(dim=("chain", "draw")).values

    logits = intercept + np.dot(X, coefs)
    probs = 1 / (1 + np.exp(-logits))

    return probs
