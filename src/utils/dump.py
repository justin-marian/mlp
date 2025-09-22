import csv
import json
from pathlib import Path
from typing import Union


def save_metrics_json(metrics: dict, save_path: Union[str, Path]) -> None:
    if not isinstance(metrics, dict):
        raise TypeError("metrics must be a dictionary.")

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, 'w', encoding='utf-8-sig') as f:
        json.dump(metrics, f, indent=4)


def save_metrics_csv(metrics: dict, save_path: Union[str, Path]) -> None:
    if not isinstance(metrics, dict):
        raise TypeError("metrics must be a dictionary.")

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    with open(save_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["key", "value"])
        for k, v in metrics.items():
            writer.writerow([k, v])


def load_metrics_json(load_path: Union[str, Path]) -> dict:
    load_path = Path(load_path)
    if not load_path.is_file():
        raise FileNotFoundError(f"No such file: '{load_path}'")

    with open(load_path, encoding='utf-8-sig') as f:
        return json.load(f)


def load_metrics_csv(load_path: Union[str, Path]) -> dict:
    no_cols = 2  # key, value

    load_path = Path(load_path)
    if not load_path.is_file():
        raise FileNotFoundError(f"No such file: '{load_path}'")

    metrics = {}
    with open(load_path,  newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        for row in reader:
            if len(row) != no_cols:
                continue
            key, value = row
            try:
                value = float(value)
            except ValueError:
                pass
            metrics[key] = value
    return metrics
