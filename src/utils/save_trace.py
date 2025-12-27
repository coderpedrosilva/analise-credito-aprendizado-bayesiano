import os

def save_bayesian_trace(trace, output_path="results/bayesian_trace.nc"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    trace.to_netcdf(output_path)
