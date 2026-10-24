# AutoQML-Noise

**Automated Search for Noise-Robust Quantum Machine Learning Circuits**

Projet expérimental consacré à l'évaluation et à la sélection de petits circuits variationnels pour la classification binaire. L'originalité éventuelle de la contribution reste à vérifier par une revue bibliographique : ce dépôt ne revendique pas de nouveauté scientifique établie.

## Problème scientifique

Comment le bruit quantique simulé, l'encodage des données, la structure et la profondeur d'un circuit influencent-ils les performances, la robustesse et le coût des classifieurs hybrides, par rapport à des références classiques ?

Le programme AutoQML-Noise compare une petite recherche de configurations à une référence VQC fixe choisie avant l'évaluation finale. Il mesure l'accuracy, la précision, le rappel, le F1, la matrice de confusion, les durées d'entraînement et d'évaluation, le nombre de paramètres, les qubits, une estimation de portes et le comportement à plusieurs niveaux de bruit. Chaque résultat inclut les versions de Python, PennyLane, scikit-learn et NumPy.

## Objectifs et contribution envisagée

- Comparer régression logistique, SVM RBF et VQC sur les mêmes partitions source.
- Comparer angle embedding et amplitude embedding, ainsi que deux profondeurs et deux topologies.
- Évaluer un modèle entraîné sur un circuit idéal sous plusieurs probabilités dépolarisantes simulées.
- Sélectionner sur validation avec un objectif qui combine le F1 moyen sous bruit et des pénalités de complexité/coût.
- Publier les configurations candidates et les mesures dans des fichiers CSV/JSON pour permettre la reproduction.

Il s'agit d'une étude expérimentale, et non d'une affirmation d'avantage quantique. La valeur scientifique de la méthode et son éventuelle nouveauté doivent être vérifiées dans la littérature et sur des expériences répétées.

## Architecture

```text
src/config.py                 graines et paramètres centralisés
src/preprocessing.py          données, partitions, scaler/PCA ajustés sur train
src/classical/                régression logistique et SVM
src/quantum/circuit.py        ansatz, encodages et canaux simulés
src/quantum/vqc_model.py      entraînement VQC et prédiction
src/evaluation.py             métriques et exports JSON/CSV
experiments/01...05           baselines, bruit, encodage et recherche
results/metrics/              mesures générées par les commandes
results/figures/              réservé aux figures reproductibles
results/logs/                 réservé aux journaux
notebooks/                    exploration facultative
app/                          future interface, non implémentée
main.py                       point d'entrée et smoke test rapide
```

## Installation sous Windows / PowerShell

Depuis la racine du dépôt, sur Python 3.11 :

```powershell
py -3.11 -m venv .venv
```

```powershell
.\.venv\Scripts\Activate.ps1
```

```powershell
python -m pip install --upgrade pip
```

```powershell
python -m pip install -r requirements.txt
```

Si la stratégie PowerShell bloque l'activation, appelez directement `\.venv\Scripts\python.exe` devant les commandes Python.

## Commandes d'expérience

Toutes les expériences peuvent recevoir `--quick` pour réduire le sous-échantillon et le nombre d'étapes, afin de vérifier rapidement le flux. Les résultats rapides ne sont pas destinés à une conclusion scientifique.

```powershell
python main.py --quick-test
```

```powershell
python experiments/01_classical_baseline.py
```

```powershell
python experiments/02_quantum_baseline.py
```

```powershell
python experiments/03_noise_experiments.py
```

```powershell
python experiments/04_encoding_comparison.py
```

```powershell
python experiments/05_automated_search.py
```

Pour un contrôle rapide d'une expérience : ajoutez `--quick`, par exemple `python experiments/05_automated_search.py --quick`.

## Protocole de validation

- Jeu de données : Breast Cancer Wisconsin (Diagnostic), chargé depuis `sklearn.datasets.load_breast_cancer` (569 observations, 30 variables, deux classes).
- Graine par défaut : 42. Les partitions stratifiées train/validation/test sont créées une fois et partagées par les modèles.
- Les expériences quantiques utilisent un sous-échantillon stratifié commun pour borner les coûts de simulation ; les tailles sont centralisées dans `src/config.py`.
- Le `StandardScaler`, la PCA et le scaler d'angles sont ajustés uniquement sur train, puis appliqués à validation et test. En encodage amplitude, la PCA produit `2**n_qubits` composantes avant normalisation d'état dans PennyLane.
- La sélection automatique compare quatre configurations, avec `n_qubits=4` : profondeur 1/2, encodage angle/amplitude et topologie linéaire/en anneau. Le score est la moyenne du F1 de validation sur les bruits configurés, moins `0.02 × profondeur/max_profondeur`, `0.01 × portes_estimées/max_portes` et `0.005 × min(temps_fit/60, 1)`. Le catalogue est fixe ; le fichier CSV conserve toutes ses mesures.
- La référence fixe est `2 couches + angle + topologie linéaire`. Le test n'est consulté qu'après la sélection sur validation. Les résultats test de la configuration retenue et de la référence fixe sont exportés ensuite.
- Chaque modèle utilise les mêmes partitions et le même nombre d'étapes d'entraînement ; la recherche automatique évalue quatre candidats alors que la référence est préfixée. Le temps agrégé de recherche et d'évaluation validation est donc rapporté séparément dans le résumé, car son coût total est supérieur à celui d'une configuration fixe.
- Les expériences finales doivent être répétées avec plusieurs graines avant toute conclusion robuste. Les temps mesurés dépendent de la machine, de PennyLane et du simulateur.

## Simulation et limites

Les circuits sont exécutés dans `default.qubit` (idéal) et `default.mixed` (états mixtes avec canal dépolarisant). Le paramètre de bruit est appliqué après chaque couche à chaque qubit ; il ne modélise pas un appareil particulier. La différentiation idéale utilise la rétropropagation du simulateur et les circuits bruités l'estimateur parameter-shift. Les mesures sont analytiques (`shots=None`) sauf évolution ultérieure du protocole.

La durée rapportée est celle d'une simulation classique, pas une estimation du temps d'exécution sur un ordinateur quantique physique. Le nombre de portes est un compte des opérations explicites de l'ansatz et de l'encodage angle ; la décomposition en portes élémentaires d'`AmplitudeEmbedding` dépend du compilateur et n'est pas estimée. Le petit nombre de qubits et de données borne la portée des conclusions. Quantum Kernel SVM et application de démonstration ne sont pas implémentés.

## État réel du développement

Implémentés dans cette version : baselines logistique/SVM, préparation avec partitions sans fuite, VQC entraînable, encodages angle/amplitude, bruit dépolarisant simulé, métriques et recherche sur quatre configurations. Les scripts exportent leurs résultats uniquement lorsqu'ils sont exécutés. Aucun benchmark n'est publié par défaut dans ce dépôt ; les premières mesures devront être interprétées à la lumière des limites ci-dessus.

## Références

1. V. Havlíček et al., “Supervised learning with quantum-enhanced feature spaces,” *Nature* 567, 209–212 (2019). [doi:10.1038/s41586-019-0980-2](https://doi.org/10.1038/s41586-019-0980-2).
2. M. Schuld and N. Killoran, “Quantum machine learning in feature Hilbert spaces,” *Physical Review Letters* 122, 040504 (2019). [doi:10.1103/PhysRevLett.122.040504](https://doi.org/10.1103/PhysRevLett.122.040504).
3. M. Schuld, M. Sweke, and J. J. Meyer, “Effect of data encoding on the expressive power of variational quantum-machine-learning models,” *Physical Review A* 103, 032430 (2021). [doi:10.1103/PhysRevA.103.032430](https://doi.org/10.1103/PhysRevA.103.032430).
4. PennyLane, [Release notes 0.45.1](https://docs.pennylane.ai/en/stable/development/release_notes.html), [interfaces and differentiation](https://docs.pennylane.ai/en/stable/introduction/interfaces.html), and [`default.mixed` device documentation](https://docs.pennylane.ai/en/stable/code/api/pennylane.devices.default_mixed.html).
5. scikit-learn, [Breast Cancer Wisconsin dataset documentation](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html).
