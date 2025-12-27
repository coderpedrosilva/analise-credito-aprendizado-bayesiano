import arviz as az
import os

def summarize_and_save_coefficients(trace, output_path="results/coefficients_summary.csv"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    summary = az.summary(trace, hdi_prob=0.95)
    summary = summary[["mean", "sd", "hdi_2.5%", "hdi_97.5%"]]

    summary.to_csv(output_path)

    return summary
