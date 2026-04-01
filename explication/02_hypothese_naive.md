# 2. L'Hypothèse Naïve (Indépendance Conditionnelle)

## 2.1 Le problème de la dimension

Si on a n caractéristiques (features), estimer P(X | C) directement nécessite un nombre
**exponentiel** d'échantillons. Avec 17 caractéristiques, c'est impossible en pratique.

## 2.2 L'hypothèse naïve

On suppose que **chaque caractéristique est indépendante des autres, sachant la classe** :

```
P(X | C) = P(x₁ | C) × P(x₂ | C) × ... × P(xₙ | C)
         = ∏ᵢ P(xᵢ | C)
```

### Avant l'hypothèse naïve (impossible à calculer) :
```
P(durée, paquets, connexions, erreurs... | Attaque)
→ Il faudrait estimer la distribution jointe de TOUTES les variables ensemble
```

### Après l'hypothèse naïve (faisable) :
```
P(durée | Attaque) × P(paquets | Attaque) × P(connexions | Attaque) × ...
→ On estime chaque variable SÉPARÉMENT
```

## 2.3 Pourquoi "naïve" ?

L'hypothèse est dite **naïve** car en réalité, les caractéristiques réseau ne sont PAS
indépendantes. Par exemple :
- Le nombre de paquets/s est corrélé avec le nombre de connexions
- La durée de session est liée aux octets envoyés

**Malgré cela**, le classifieur naïf bayésien fonctionne très bien en pratique. C'est un
résultat empirique bien connu en apprentissage automatique.

## 2.4 Formule complète de classification

```
ŷ = argmax_C [ P(C) × ∏ᵢ P(xᵢ | C) ]
```

En français : on choisit la classe C qui maximise le produit de la probabilité a priori
par toutes les vraisemblances individuelles.

## 2.5 Correspondance dans le code

```python
# Dans naive_bayes_manual.py — méthode predict_log_proba()

log_prior = np.log(self.priors_)           # log P(C)
log_likelihood = self._log_likelihood(X)    # Σ log P(xᵢ | C)

# log P(C|X) ∝ log P(C) + Σ log P(xᵢ|C)
log_posterior = log_prior + log_likelihood
```
