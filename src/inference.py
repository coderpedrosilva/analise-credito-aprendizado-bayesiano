import numpy as np
import arviz as az
import joblib

MODEL_PATH = "models/bayesian_credit_model.pkl"
TRACE_PATH = "models/bayesian_credit_trace.nc"

model = joblib.load(MODEL_PATH)
trace = az.from_netcdf(TRACE_PATH)

posterior = trace.posterior
intercept = posterior["intercept"].mean(dim=("chain","draw")).values
coefs = posterior["coefficients"].mean(dim=("chain","draw")).values

def predict_proba(X):
    logits = intercept + np.dot(X, coefs)
    return 1 / (1 + np.exp(-logits))
