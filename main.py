"""Point d'entrée du projet QML-Robustness."""

from pathlib import Path


ROOT = Path(__file__).resolve().parent


def main() -> None:
    print("QML-Robustness — squelette du projet")
    print(f"Répertoire : {ROOT}")
    print("Lancez les scripts du dossier experiments/ pour exécuter les baselines.")


if __name__ == "__main__":
    main()
