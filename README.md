# QML-Robustness

**Évaluation de la robustesse et de l'efficacité des modèles hybrides de Quantum Machine Learning sous bruit quantique**

*Robustness and Efficiency of Hybrid Quantum Machine Learning under Quantum Noise: A Comparative Study*

## Question de recherche

Comment le bruit quantique, l'encodage des données et la profondeur des circuits influencent-ils les performances et le coût des modèles hybrides quantiques par rapport aux modèles classiques ?

## Objectifs

- Comparer un classifieur classique à un classifieur quantique variationnel (VQC) sur les mêmes partitions de données.
- Étudier l'effet de canaux de bruit simulés et du nombre de couches du circuit.
- Comparer plusieurs schémas d'encodage avec le même budget expérimental.
- Rapporter les scores prédictifs avec le temps d'entraînement et le nombre de paramètres.

## Structure

```text
data/{raw,processed}/       données brutes et préparées
src/classical/              références scikit-learn
src/quantum/                circuit, VQC, bruit, prétraitement
src/evaluation.py           métriques et export des résultats
experiments/                expériences reproductibles
results/{metrics,figures,logs}/ résultats générés
notebooks/                  exploration et visualisation
app/                        future interface de démonstration
main.py                     point d'entrée
```

## Démarrage

Dans PowerShell, depuis ce dossier :

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python main.py
```

Les expériences sont conçues pour être lancées avec `python experiments/01_classical_baseline.py` et les scripts suivants. Le simulateur par défaut est PennyLane `default.qubit`; les expériences bruitées utilisent des canaux de bruit simulés et ne prétendent pas représenter un matériel quantique réel.

## Principes de comparaison

Fixer la graine, les partitions train/test et le prétraitement entre modèles. Réduire les données à une dimension compatible avec le nombre de qubits, ajuster les transformations uniquement sur l'entraînement, puis conserver les mêmes exemples pour toutes les variantes. Comparer les performances (accuracy, précision, rappel, F1) et les coûts (durée, paramètres entraînables, qubits et profondeur). Les résultats dépendent du jeu de données et du simulateur.

## État

Squelette initial. Le jeu de données principal, le protocole expérimental et la cible matérielle sont à préciser avant les conclusions scientifiques.
