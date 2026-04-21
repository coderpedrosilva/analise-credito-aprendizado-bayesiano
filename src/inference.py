import os
import numpy as np
import arviz as az

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRACE_PATH = os.path.join(BASE_DIR, "models", "bayesian_credit_trace.nc")

if not os.path.exists(TRACE_PATH):
    raise FileNotFoundError(
        f"Trace do modelo não encontrado em '{TRACE_PATH}'. "
        "Execute 'python main.py' para treinar o modelo primeiro."
    )

trace = az.from_netcdf(TRACE_PATH)

posterior = trace.posterior
intercept = posterior["intercept"].mean(dim=("chain", "draw")).values
coefs = posterior["coefficients"].mean(dim=("chain", "draw")).values


def predict_proba(X):
    logits = intercept + np.dot(X, coefs)
    return 1 / (1 + np.exp(-logits))
