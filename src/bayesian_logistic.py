import pymc as pm
import numpy as np
import arviz as az

def train_bayesian_logistic(X_train, y_train, draws=200, tune=200):
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

        # Sampling (mais rápido)
        trace = pm.sample(
            draws=draws,
            tune=tune,
            chains=2,
            cores=2,
            init="adapt_diag",
            target_accept=0.9,
            return_inferencedata=True,
            progressbar=True
        )

    _print_convergence_diagnostics(trace)

    return model, trace


def _print_convergence_diagnostics(trace):
    rhat = az.rhat(trace)
    ess = az.ess(trace)

    rhat_max = float(rhat.to_array().max())
    ess_min = float(ess.to_array().min())

    rhat_ok = "✅" if rhat_max <= 1.01 else "⚠️"
    ess_ok = "✅" if ess_min >= 100 else "⚠️"

    print("\n🔍 Diagnósticos de convergência MCMC:")
    print(f"  R-hat máximo : {rhat_max:.4f}  {rhat_ok}  (ideal ≤ 1.01)")
    print(f"  ESS mínimo   : {ess_min:.0f}      {ess_ok}  (ideal ≥ 100)")

    n_divergences = int(trace.sample_stats["diverging"].values.sum())
    div_ok = "✅" if n_divergences == 0 else "⚠️"
    print(f"  Divergências : {n_divergences}         {div_ok}  (ideal = 0)")


def summarize_model(trace):
    """
    Retorna um resumo estatístico do modelo bayesiano.
    """
    return az.summary(trace, hdi_prob=0.95)
