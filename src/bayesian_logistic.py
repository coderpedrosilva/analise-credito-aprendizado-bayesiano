import pymc as pm
import numpy as np
import arviz as az

def train_bayesian_logistic(X_train, y_train, draws=1000, tune=1000):
    """
    Treina uma regressão logística bayesiana usando PyMC.
    """

    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train)

    n_features = X_train.shape[1]

    with pm.Model() as model:
        # Priors
        intercept = pm.Normal("intercept", mu=0, sigma=5)
        coefficients = pm.Normal("coefficients", mu=0, sigma=5, shape=n_features)

        # Linear combination
        logits = intercept + pm.math.dot(X_train, coefficients)

        # Likelihood
        y_obs = pm.Bernoulli(
            "y_obs",
            logit_p=logits,
            observed=y_train
        )

        # Sampling
        trace = pm.sample(
            draws=draws,
            tune=tune,
            target_accept=0.9,
            return_inferencedata=True,
            progressbar=True
        )

    return model, trace


def summarize_model(trace):
    """
    Retorna um resumo estatístico do modelo bayesiano.
    """
    return az.summary(trace, hdi_prob=0.95)
