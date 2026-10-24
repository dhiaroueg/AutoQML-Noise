"""Mesure le VQC de référence sur plusieurs intensités de bruit simulé."""

import argparse
import json

from _common import ROOT, get_partitions, library_versions, prepare_features
from src.config import CircuitConfig, DATASET_NAME, N_QUBITS, NOISE_LEVELS, SEED, TRAINING_STEPS
from src.evaluation import classification_metrics, save_csv_rows
from src.quantum.vqc_model import VQCModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    parts = get_partitions(quick=args.quick)
    config = CircuitConfig(n_layers=2, encoding="angle", structure="linear")
    _, x_train, _, x_test = prepare_features(parts, config.encoding)
    model = VQCModel(N_QUBITS, config, seed=SEED, steps=2 if args.quick else TRAINING_STEPS)
    model.fit(x_train, parts.y_train)
    levels = (0.0, 0.02) if args.quick else NOISE_LEVELS
    rows = []
    for probability in levels:
        model.set_evaluation_noise(probability)
        metrics = classification_metrics(parts.y_test, model.predict(x_test))
        metrics["evaluation_seconds"] = model.predict_seconds
        rows.append({"dataset": DATASET_NAME, "model": "VQC", "seed": SEED,
                     "noise_probability_per_wire_layer": probability,
                     "fit_seconds": model.fit_seconds, "n_parameters": model.n_parameters,
                     "n_qubits": N_QUBITS, "circuit_depth": model.n_layers,
                     "estimated_gate_count": model.gate_count,
                     **library_versions(), **metrics})
    output = ROOT / "results" / "metrics" / "noise_experiments.csv"
    save_csv_rows(rows, output)
    print(json.dumps(rows, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
