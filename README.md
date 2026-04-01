# IDS Bayésien Naïf — Détection d'intrusions réseau

## Projet M1 Cybersécurité — Probabilités & Statistiques

---

## 1. Présentation

Ce projet implémente un **Système de Détection d'Intrusions (IDS)** basé sur un **classifieur bayésien naïf**.  
Il distingue automatiquement le trafic réseau **normal** du trafic **anormal (attaque)** en appliquant le **théorème de Bayes**.

**Dataset** : NSL-KDD (téléchargeable depuis [ici](https://www.unb.ca/cic/datasets/nsl.html))

---

## 2. Notions théoriques clés

### 2.1 Théorème de Bayes

```
P(C | X) = P(X | C) * P(C) / P(X)
```

- **P(C)** : probabilité a priori de la classe (Normal ou Attaque)
- **P(X | C)** : vraisemblance des features sachant la classe
- **P(C | X)** : probabilité a posteriori (ce qu'on cherche)

### 2.2 Hypothèse naïve (indépendance conditionnelle)

```
P(X | C) = P(x1 | C) * P(x2 | C) * ... * P(xn | C)
```

On suppose que chaque feature est **indépendante** des autres sachant la classe.

### 2.3 Règle de décision

```
Si P(Attaque | X) > P(Normal | X) → Déclencher une alerte
```

### 2.4 Estimation des probabilités

- **Attributs catégoriels** : fréquences relatives + lissage de Laplace
- **Attributs continus** : hypothèse de distribution gaussienne (moyenne + écart-type)

### 2.5 Lissage de Laplace

Pour éviter les probabilités nulles :
```
P(xi | C) = (count(xi, C) + alpha) / (count(C) + alpha * |V|)
```

---

## 3. Étapes méthodologiques

| Étape | Description |
|-------|-------------|
| 1 | **Compréhension des données** — Explorer le dataset NSL-KDD (41 features) |
| 2 | **Prétraitement** — Encoder les variables catégorielles, normaliser, binariser les labels |
| 3 | **Séparation** — Train/Test split (80/20) |
| 4 | **Implémentation manuelle** — Coder Bayes naïf depuis les formules |
| 5 | **Implémentation sklearn** — Utiliser GaussianNB pour comparaison |
| 6 | **Évaluation** — Accuracy, Precision, Recall, F1-Score, Matrice de confusion |
| 7 | **Analyse** — Comparer les deux approches, interpréter les résultats |

---

## 4. Points critiques

- **Déséquilibre des classes** : le dataset contient plus d'attaques que de trafic normal
- **Attributs continus** : utiliser la distribution gaussienne pour estimer P(xi | C)
- **Lissage de Laplace** : indispensable pour les attributs catégoriels rares
- **Corrélation entre features** : l'hypothèse naïve est violée en pratique, mais fonctionne bien

---

## 5. Structure du projet

```
proba stat/
├── main.py                  # Point d'entrée
├── gui.py                   # Interface Tkinter
├── data_loader.py           # Chargement et prétraitement des données
├── naive_bayes_manual.py    # Implémentation manuelle du Bayes naïf
├── naive_bayes_sklearn.py   # Implémentation avec scikit-learn
├── evaluation.py            # Métriques et visualisations
├── requirements.txt         # Dépendances Python
├── data/                    # Dossier pour les fichiers CSV du dataset
│   └── (placer KDDTrain+.txt et KDDTest+.txt ici)
└── README.md                # Ce fichier
```

---

## 6. Installation et lancement

```bash
# 1. Installer les dépendances
pip install -r requirements.txt

# 2. Lancer l'application
python main.py
```

> **Note** : Le dataset NSL-KDD est **téléchargé automatiquement** au premier lancement.  
> Si le téléchargement échoue (pas de connexion), l'application utilise des données synthétiques.

---

## 7. Feuille de route (4 semaines)

| Semaine | Objectifs |
|---------|-----------|
| S1 | Comprendre Bayes, explorer le dataset, prétraitement |
| S2 | Implémenter le classifieur manuellement |
| S3 | Comparer avec sklearn, évaluer, visualiser |
| S4 | Interface Tkinter, rédaction du rapport, soutenance |

---

## Auteur

Projet réalisé dans le cadre du cours de Probabilités & Statistiques — M1 Cybersécurité (INSI S7).
