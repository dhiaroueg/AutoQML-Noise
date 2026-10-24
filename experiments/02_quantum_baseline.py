"""Entraîne et évalue la baseline VQC sur le sous-ensemble commun."""

import argparse
import json

from _common import ROOT, get_partitions, library_versions, prepare_features
from src.config import CircuitConfig, DATASET_NAME, N_QUBITS, SEED, TRAINING_STEPS
from src.evaluation import classification_metrics, save_metrics
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
    metrics = classification_metrics(parts.y_test, model.predict(x_test))
    metrics.update({"dataset": DATASET_NAME, "model": "VQC", "seed": SEED,
                    "fit_seconds": model.fit_seconds, "n_parameters": model.n_parameters,
                    "n_qubits": N_QUBITS, "circuit_depth": model.n_layers,
                    "estimated_gate_count": model.gate_count,
                    "noise_probability": config.noise_probability,
                    "n_train": len(parts.y_train), "n_test": len(parts.y_test)})
    metrics.update(library_versions())
    save_metrics(metrics, ROOT / "results" / "metrics" / "quantum_baseline.json")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
