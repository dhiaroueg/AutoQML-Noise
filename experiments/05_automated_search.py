"""Recherche AutoQML-Noise à budget limité, avec sélection sur validation."""

import argparse
import json

from _common import ROOT, get_partitions, library_versions, prepare_features
from src.config import (
    CircuitConfig,
    DATASET_NAME,
    N_QUBITS,
    NOISE_LEVELS,
    SEED,
    TRAINING_STEPS,
)
from src.evaluation import classification_metrics, save_csv_rows, save_metrics
from src.quantum.circuit import estimated_gate_count
from src.quantum.vqc_model import VQCModel


def objective(mean_validation_f1: float, config: CircuitConfig, fit_seconds: float) -> float:
    """Score = F1 moyen - pénalités profondeur (0.02), portes (0.01), temps (0.005).

    Profondeur et nombre de portes sont normalisés par les maxima du catalogue
    de recherche; la durée est plafonnée à une minute avant normalisation.
    """
    max_depth = 2
    max_gates = max(
        estimated_gate_count(N_QUBITS, candidate_depth, structure, encoding)
        for candidate_depth in (1, 2)
        for structure in ("linear", "ring")
        for encoding in ("angle", "amplitude")
    )
    gates = estimated_gate_count(
        N_QUBITS, config.n_layers, config.structure, config.encoding
    )
    return (
        mean_validation_f1
        - 0.02 * config.n_layers / max_depth
        - 0.01 * gates / max_gates
        - 0.005 * min(fit_seconds / 60.0, 1.0)
    )


def evaluate_noise_sweep(model, features, labels, levels):
    metrics_by_noise = {}
    for probability in levels:
        model.set_evaluation_noise(probability)
        predictions = model.predict(features)
        metrics_by_noise[probability] = classification_metrics(labels, predictions)
        metrics_by_noise[probability]["evaluation_seconds"] = model.predict_seconds
    return metrics_by_noise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Réduit les exemples et le nombre d'étapes")
    args = parser.parse_args()
    parts = get_partitions(quick=args.quick)
    levels = (0.0, 0.02) if args.quick else NOISE_LEVELS
    steps = 2 if args.quick else TRAINING_STEPS

    # Catalogue prédéfini, budget de quatre configurations; le test n'intervient
    # qu'après le choix du gagnant par la validation.
    candidates = [
        CircuitConfig(1, "angle", "linear"),
        CircuitConfig(2, "angle", "ring"),
        CircuitConfig(1, "amplitude", "linear"),
        CircuitConfig(2, "amplitude", "ring"),
    ]
    search_rows = []
    fitted_candidates = []
    search_fit_seconds_total = 0.0
    search_evaluation_seconds_total = 0.0
    for index, config in enumerate(candidates):
        _, x_train, x_validation, _ = prepare_features(parts, config.encoding)
        model = VQCModel(N_QUBITS, config, seed=SEED, steps=steps)
        model.fit(x_train, parts.y_train)
        search_fit_seconds_total += model.fit_seconds
        validation_metrics = evaluate_noise_sweep(
            model, x_validation, parts.y_validation, levels
        )
        mean_f1 = sum(item["f1"] for item in validation_metrics.values()) / len(levels)
        validation_eval_seconds = sum(item["evaluation_seconds"] for item in validation_metrics.values())
        search_evaluation_seconds_total += validation_eval_seconds
        score = objective(mean_f1, config, model.fit_seconds)
        row = {
            "candidate_id": index,
            "dataset": DATASET_NAME,
            "seed": SEED,
            "n_layers": config.n_layers,
            "encoding": config.encoding,
            "structure": config.structure,
            "training_noise_probability": config.noise_probability,
            "fit_seconds": model.fit_seconds,
            "validation_evaluation_seconds": validation_eval_seconds,
            "n_parameters": model.n_parameters,
            "n_qubits": N_QUBITS,
            "estimated_gate_count": model.gate_count,
            "mean_validation_f1": mean_f1,
            "objective_score": score,
            **library_versions(),
        }
        for probability, metrics in validation_metrics.items():
            row[f"validation_f1_noise_{probability:g}"] = metrics["f1"]
            row[f"validation_accuracy_noise_{probability:g}"] = metrics["accuracy"]
        search_rows.append(row)
        fitted_candidates.append((score, config, model))

    search_csv = ROOT / "results" / "metrics" / "automated_search_candidates.csv"
    save_csv_rows(search_rows, search_csv)
    _, selected_config, selected_model = max(fitted_candidates, key=lambda item: item[0])

    # Référence fixe définie avant examen des métriques de validation/test.
    fixed_config = CircuitConfig(2, "angle", "linear")
    _, x_train_fixed, _, x_test_fixed = prepare_features(parts, fixed_config.encoding)
    fixed_model = VQCModel(N_QUBITS, fixed_config, seed=SEED, steps=steps)
    fixed_model.fit(x_train_fixed, parts.y_train)

    final_rows = []
    final_evaluation_seconds_total = 0.0
    for label, model, config in (
        ("automated_search", selected_model, selected_config),
        ("fixed_reference", fixed_model, fixed_config),
    ):
        _, _, _, x_test = prepare_features(parts, config.encoding)
        test_metrics = evaluate_noise_sweep(model, x_test, parts.y_test, levels)
        for probability, metrics in test_metrics.items():
            final_evaluation_seconds_total += metrics["evaluation_seconds"]
            final_rows.append({
                "selection": label,
                "dataset": DATASET_NAME,
                "seed": SEED,
                "n_layers": config.n_layers,
                "encoding": config.encoding,
                "structure": config.structure,
                "noise_probability_per_wire_layer": probability,
                "fit_seconds": model.fit_seconds,
                "evaluation_seconds": metrics["evaluation_seconds"],
                "n_parameters": model.n_parameters,
                "n_qubits": N_QUBITS,
                "estimated_gate_count": model.gate_count,
                **library_versions(),
                **metrics,
            })
    final_csv = ROOT / "results" / "metrics" / "automated_search_test_comparison.csv"
    save_csv_rows(final_rows, final_csv)
    summary = {
        "selected_config": selected_config.__dict__,
        "fixed_reference_config": fixed_config.__dict__,
        "candidate_budget": len(candidates),
        "search_fit_seconds_total": search_fit_seconds_total,
        "search_validation_evaluation_seconds_total": search_evaluation_seconds_total,
        "final_test_evaluation_seconds_total": final_evaluation_seconds_total,
        "selection_metric": "mean validation F1 over noise levels minus documented complexity/time penalties",
        "candidate_csv": str(search_csv),
        "test_comparison_csv": str(final_csv),
        "test_results": final_rows,
    }
    save_metrics(summary, ROOT / "results" / "metrics" / "automated_search_summary.json")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
