# 3. Estimation des Probabilités

## 3.1 Probabilité a priori P(C)

C'est la proportion de chaque classe dans les données d'entraînement.

```
P(Normal)  = nombre d'échantillons normaux / nombre total d'échantillons
P(Attaque) = nombre d'échantillons d'attaque / nombre total d'échantillons
```

### Dans notre dataset NSL-KDD :
```
P(Normal)  = 67 343 / 125 973 = 0.5346  (53.46%)
P(Attaque) = 58 630 / 125 973 = 0.4654  (46.54%)
```

### Correspondance dans le code :

```python
# Dans naive_bayes_manual.py — méthode fit()

# P(C) = |X_c| / |X|
self.priors_[idx] = X_c.shape[0] / X.shape[0]
```

---

## 3.2 Vraisemblance P(xᵢ | C) — Cas gaussien

Pour les caractéristiques **continues** (durée, octets, taux...), on suppose que chaque
caractéristique suit une **distribution normale (gaussienne)** dans chaque classe.

### Formule de la densité gaussienne :

```
P(xᵢ | C) = (1 / √(2π σ²)) × exp(-(xᵢ - μ)² / (2σ²))
```

Où :
- **μ** = moyenne de xᵢ pour la classe C (estimée sur les données d'entraînement)
- **σ²** = variance de xᵢ pour la classe C (estimée sur les données d'entraînement)

### Exemple :

Pour la caractéristique "durée de session" :
- Classe Normal  : μ = 30s, σ² = 100
- Classe Attaque : μ = 3s,  σ² = 4

Si on observe une durée de 2 secondes :
```
P(durée=2 | Normal)  = gaussienne(2, μ=30, σ²=100) = très faible
P(durée=2 | Attaque) = gaussienne(2, μ=3,  σ²=4)   = élevé
```
→ Cela penche vers "Attaque"

### Correspondance dans le code :

```python
# Dans naive_bayes_manual.py — méthode fit()

# μ(xᵢ | C) — moyenne par caractéristique et par classe
self.means_[idx, :] = X_c.mean(axis=0)

# σ²(xᵢ | C) — variance par caractéristique et par classe + lissage
self.variances_[idx, :] = X_c.var(axis=0) + self.var_smoothing
```

```python
# Dans naive_bayes_manual.py — méthode _log_likelihood()

# Log de la densité gaussienne :
# log P(xᵢ|C) = -0.5 × log(2π σ²) - (xᵢ - μ)² / (2σ²)
log_prob = -0.5 * np.log(2 * np.pi * var)
log_prob = log_prob - (X - mean) ** 2 / (2 * var)

# Somme sur toutes les caractéristiques (hypothèse d'indépendance)
log_likelihoods[:, idx] = log_prob.sum(axis=1)
```

---

## 3.3 Pourquoi utiliser le logarithme ?

Le produit de nombreuses probabilités (entre 0 et 1) donne un nombre **extrêmement petit**
qui cause des dépassements numériques par le bas (valeur arrondie à 0 par l'ordinateur).

**Solution** : travailler en espace logarithmique.

```
log(a × b × c) = log(a) + log(b) + log(c)
```

Les produits deviennent des **sommes**, ce qui est numériquement stable.

### Correspondance dans le code :

```python
# Au lieu de :  P(C) × ∏ P(xᵢ | C)        → risque de dépassement
# On calcule :  log P(C) + Σ log P(xᵢ | C)  → stable numériquement

log_posterior = log_prior + log_likelihood
```

---

## 3.4 Lissage (var_smoothing)

On ajoute une petite valeur ε = 10⁻⁹ à toutes les variances pour éviter la **division
par zéro** quand une caractéristique a une variance nulle (valeur constante).

```python
self.variances_[idx, :] = X_c.var(axis=0) + self.var_smoothing  # ε = 1e-9
```

C'est l'équivalent du **lissage de Laplace** pour les variables continues.
