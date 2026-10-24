"""Point de départ pour entraîner la baseline VQC sur un petit sous-ensemble."""

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from src.quantum.preprocessing import AnglePreprocessor


def main():
    data = load_breast_cancer()
    x_train, x_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.25, random_state=42, stratify=data.target
    )
    preprocessor = AnglePreprocessor(n_qubits=4)
    x_train = preprocessor.fit_transform(x_train)
    x_test = preprocessor.transform(x_test)
    print(f"Données préparées : train={x_train.shape}, test={x_test.shape}")
    print("L'entraînement VQC et son protocole d'optimisation restent à implémenter.")
    print(f"Classes : {sorted(set(y_train))}; test={len(y_test)} exemples")


if __name__ == "__main__":
    main()
