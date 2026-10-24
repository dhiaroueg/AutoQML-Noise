"""Point d'entrée et vérification rapide de QML-Robustness."""

import argparse
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def quick_test() -> None:
    """Vérifie les imports, le prétraitement et un très court entraînement VQC."""
    import numpy as np
    import pennylane as qml
    import sklearn

    from src.config import CircuitConfig
    from src.preprocessing import QuantumPreprocessor, load_binary_dataset
    from src.quantum.vqc_model import VQCModel

    features, labels, _ = load_binary_dataset()
    selected = np.concatenate(
        [np.flatnonzero(labels == label)[:4] for label in np.unique(labels)]
    )
    x = features[selected]
    y = labels[selected]
    preprocessor = QuantumPreprocessor(n_qubits=2, encoding="angle")
    encoded = preprocessor.fit_transform(x)
    model = VQCModel(2, CircuitConfig(n_layers=1), steps=1)
    model.fit(encoded, y)
    prediction_count = len(model.predict(encoded[:2]))
    assert prediction_count == 2
    print({"status": "ok", "python": __import__("sys").version.split()[0],
           "pennylane": qml.__version__, "scikit_learn": sklearn.__version__,
           "numpy": np.__version__, "vqc_fit_seconds": model.fit_seconds})


def main() -> None:
    parser = argparse.ArgumentParser(description="AutoQML-Noise project tools")
    parser.add_argument("--quick-test", action="store_true", help="Smoke test rapide imports + VQC")
    args = parser.parse_args()
    if args.quick_test:
        quick_test()
        return
    print("AutoQML-Noise — recherche de circuits VQC robustes au bruit")
    print(f"Répertoire : {ROOT}")
    print("Lancez python experiments/05_automated_search.py pour l'expérience de recherche.")


if __name__ == "__main__":
    main()
