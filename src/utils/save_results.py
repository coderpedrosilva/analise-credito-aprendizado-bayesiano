import json
import os

def save_metrics(metrics: list, output_path="results/metrics.json"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)
