# 4. Implémentation Manuelle — Ligne par Ligne

Ce document explique **chaque méthode** de la classe `NaiveBayesManual` dans
`naive_bayes_manual.py`, avec les formules mathématiques correspondantes.

---

## 4.1 Structure de la classe

```python
class NaiveBayesManual:
    def __init__(self, var_smoothing=1e-9):
        self.var_smoothing = var_smoothing   # ε pour le lissage
        self.classes_    = None              # [0, 1] → Normal, Attaque
        self.priors_     = None              # P(C) pour chaque classe
        self.means_      = None              # μ(xᵢ | C) pour chaque classe
        self.variances_  = None              # σ²(xᵢ | C) pour chaque classe
```

**var_smoothing** : petite valeur ajoutée à chaque variance pour éviter la division par zéro.

---

## 4.2 Méthode `fit(X, y)` — Entraînement

**But** : estimer les paramètres du modèle à partir des données.

### Étape 1 — Probabilités a priori P(C)

```
P(C) = nombre d'échantillons de la classe C / nombre total d'échantillons
```

```python
for idx, c in enumerate(self.classes_):
    X_c = X[y == c]                            # Filtrer les échantillons de la classe c
    self.priors_[idx] = X_c.shape[0] / X.shape[0]   # P(C) = |X_c| / |X|
```

**Résultat NSL-KDD** :
- P(Normal)  = 67343 / 125973 = **0.5346**
- P(Attaque) = 58630 / 125973 = **0.4654**

### Étape 2 — Moyennes μ(xᵢ | C)

```
μ(xᵢ | C) = (1/|X_c|) × Σ xᵢ    pour tous les échantillons de la classe C
```

```python
self.means_[idx, :] = X_c.mean(axis=0)
```

On obtient un tableau de moyennes : une valeur par caractéristique, par classe.

### Étape 3 — Variances σ²(xᵢ | C)

```
σ²(xᵢ | C) = (1/|X_c|) × Σ (xᵢ - μ)²  + ε
```

```python
self.variances_[idx, :] = X_c.var(axis=0) + self.var_smoothing
```

Le `+ self.var_smoothing` (ε = 10⁻⁹) est le lissage pour la stabilité numérique.

---

## 4.3 Méthode `_log_likelihood(X)` — Calcul de log P(X|C)

**But** : pour chaque échantillon et chaque classe, calculer la log-vraisemblance.

### Formule de la densité gaussienne en log :

```
log P(xᵢ | C) = -0.5 × log(2π σ²) - (xᵢ - μ)² / (2σ²)
```

### Code :

```python
for idx in range(n_classes):
    mean = self.means_[idx]       # μ pour cette classe
    var = self.variances_[idx]    # σ² pour cette classe

    # Terme 1 : -0.5 × log(2π σ²)
    log_prob = -0.5 * np.log(2 * np.pi * var)

    # Terme 2 : -(xᵢ - μ)² / (2σ²)
    log_prob = log_prob - (X - mean) ** 2 / (2 * var)

    # Hypothèse naïve : somme des log (= produit des probabilités)
    log_likelihoods[:, idx] = log_prob.sum(axis=1)
```

### Pourquoi `.sum(axis=1)` ?

C'est l'hypothèse naïve en action :
```
log P(X | C) = log P(x₁|C) + log P(x₂|C) + ... + log P(xₙ|C)
             = Σᵢ log P(xᵢ | C)
```

La somme remplace le produit grâce au logarithme.

---

## 4.4 Méthode `predict_log_proba(X)` — Log-probabilité a posteriori

**But** : combiner la vraisemblance avec la probabilité a priori.

```
log P(C | X) ∝ log P(C) + log P(X | C)
```

```python
log_prior = np.log(self.priors_)            # log P(C)
log_likelihood = self._log_likelihood(X)     # Σ log P(xᵢ | C)
log_posterior = log_prior + log_likelihood    # Théorème de Bayes en log
```

---

## 4.5 Méthode `predict_proba(X)` — Probabilités normalisées

**But** : convertir les log-probabilités en probabilités réelles (somme = 1).

### Astuce du log-sum-exp (stabilité numérique) :

```python
log_max = log_posterior.max(axis=1, keepdims=True)
log_posterior_shifted = log_posterior - log_max          # Évite les grands nombres
proba = np.exp(log_posterior_shifted)                     # Retour en espace normal
proba = proba / proba.sum(axis=1, keepdims=True)         # Normalisation
```

**Pourquoi soustraire le max ?**
Les log-probabilités peuvent être très négatives (ex: -1500). `exp(-1500)` donne 0 en
machine. En soustrayant le max, on ramène les valeurs dans une plage calculable.

---

## 4.6 Méthode `predict(X)` — Décision finale

**Règle de décision** :
```
ŷ = argmax_C  log P(C | X)
```

Si log P(Attaque | X) > log P(Normal | X) → classe = **Attaque** → **ALERTE**

```python
def predict(self, X):
    log_posterior = self.predict_log_proba(X)
    return self.classes_[np.argmax(log_posterior, axis=1)]
```

---

## 4.7 Résumé du flux complet

```
Données d'entraînement
        │
        ▼
    fit(X, y)
    ├── Calcul de P(C)      → priors_
    ├── Calcul de μ(xᵢ|C)   → means_
    └── Calcul de σ²(xᵢ|C)  → variances_

Nouvel échantillon X
        │
        ▼
    predict(X)
    ├── _log_likelihood(X)     → Σ log P(xᵢ|C)
    ├── predict_log_proba(X)   → log P(C) + Σ log P(xᵢ|C)
    └── argmax                 → Classe prédite (Normal ou Attaque)
```
