#!/usr/bin/env python3

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
import argparse
from pathlib import Path
from typing import Iterable, List, Tuple


DEFAULT_RESULTS_ROOT = Path(__file__).resolve().parent.parent / "testing_results"


@dataclass
class Metrics:
    accuracy: float
    precision: float
    recall: float


def load_json_metrics(results_dir: Path) -> Metrics:
    evaluation_file = results_dir / "evaluation_results.json"
    with evaluation_file.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    metrics = data.get("metrics", {})
    return Metrics(
        accuracy=float(metrics.get("accuracy", 0.0)),
        precision=float(metrics.get("precision", 0.0)),
        recall=float(metrics.get("recall", 0.0)),
    )


def load_predictions(results_dir: Path) -> Tuple[List[float], List[int]]:
    predictions_file = results_dir / "predictions.csv"
    labels: List[int] = []
    probs: List[float] = []
    with predictions_file.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            labels.append(int(float(row["true_label"])))
            probs.append(float(row["prediction_prob"]))
    return probs, labels


def compute_metrics(threshold: float, probs: Iterable[float], labels: Iterable[int]) -> Metrics:
    tp = fp = tn = fn = 0
    for prob, label in zip(probs, labels):
        pred = 1 if prob >= threshold else 0
        if pred == 1 and label == 1:
            tp += 1
        elif pred == 1 and label == 0:
            fp += 1
        elif pred == 0 and label == 0:
            tn += 1
        else:
            fn += 1

    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return Metrics(accuracy=accuracy, precision=precision, recall=recall)


def find_best_threshold(probs: List[float], labels: List[int], step: float = 0.01) -> Tuple[float, Metrics]:
    best_threshold = 0.5
    best_metrics = compute_metrics(best_threshold, probs, labels)
    best_score = f1_score(best_metrics)

    threshold = 0.0
    while threshold <= 1.0:
        metrics = compute_metrics(threshold, probs, labels)
        score = f1_score(metrics)
        if score > best_score or (score == best_score and metrics.accuracy > best_metrics.accuracy):
            best_threshold = threshold
            best_metrics = metrics
            best_score = score
        threshold = round(threshold + step, 10)  # avoid floating point drift
    return best_threshold, best_metrics


def f1_score(metrics: Metrics) -> float:
    precision = metrics.precision
    recall = metrics.recall
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize evaluation metrics for multiple model result folders."
    )
    parser.add_argument(
        "results_root",
        nargs="?",
        default=str(DEFAULT_RESULTS_ROOT),
        help="Path to the directory containing per-model result subdirectories.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_root = Path(args.results_root).expanduser().resolve()

    if not results_root.exists():
        raise SystemExit(f"Results directory does not exist: {results_root}")
    if not results_root.is_dir():
        raise SystemExit(f"Results path is not a directory: {results_root}")

    for results_dir in sorted(results_root.iterdir()):
        evaluation_file = results_dir / "evaluation_results.json"
        predictions_file = results_dir / "predictions.csv"
        if not evaluation_file.exists() or not predictions_file.exists():
            continue

        json_metrics = load_json_metrics(results_dir)
        probs, labels = load_predictions(results_dir)
        total_examples = len(labels)
        positives = sum(labels)
        negatives = total_examples - positives

        best_threshold, best_metrics = find_best_threshold(probs, labels)

        print(f"{results_dir.name}:")
        print(
            "  samples           : "
            f"{total_examples} "
            f"(pos={positives} [{positives / total_examples:.1%}]"
            f", neg={negatives} [{negatives / total_examples:.1%}])"
            if total_examples
            else "  samples           : 0"
        )
        print(f"  reported accuracy : {json_metrics.accuracy:.4f}")
        print(f"  reported precision: {json_metrics.precision:.4f}")
        print(f"  reported recall   : {json_metrics.recall:.4f}")
        print(f"  best threshold    : {best_threshold:.2f}")
        print(
            f"    accuracy={best_metrics.accuracy:.4f}, "
            f"precision={best_metrics.precision:.4f}, "
            f"recall={best_metrics.recall:.4f}"
        )
        print()


if __name__ == "__main__":
    main()

