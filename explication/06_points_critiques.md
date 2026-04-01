# 6. Points Critiques et Limites

## 6.1 Déséquilibre des classes

### Le problème
Dans le dataset NSL-KDD :
- Entraînement : 53% Normal, 47% Attaque (relativement équilibré)
- Test : 43% Normal, 57% Attaque (déséquilibré)

Un déséquilibre fort biaise le classifieur vers la classe majoritaire.

### Solutions possibles
1. **Sous-échantillonnage** de la classe majoritaire
2. **Sur-échantillonnage** de la classe minoritaire (SMOTE)
3. **Pondération des classes** dans le calcul de P(C)

---

## 6.2 Hypothèse d'indépendance (violée en pratique)

### Le problème
Les caractéristiques réseau sont corrélées :
- `src_bytes` et `dst_bytes` sont liés
- `serror_rate` et `srv_serror_rate` sont quasi-identiques
- `count` et `srv_count` sont dépendants

### Impact
L'hypothèse naïve surestime la confiance du classifieur (les probabilités sont
trop proches de 0 ou 1). Mais la **décision finale** reste souvent correcte.

### Solution possible
Utiliser un classifieur bayésien **non naïf** ou réduire les caractéristiques corrélées
par **Analyse en Composantes Principales (ACP)**.

---

## 6.3 Attributs continus — Hypothèse gaussienne

### Le problème
On suppose que chaque caractéristique suit une loi normale. Or :
- `duration` suit plutôt une loi exponentielle
- `src_bytes` a une distribution très asymétrique (beaucoup de 0)
- Certaines caractéristiques sont binaires (0 ou 1)

### Impact
Le modèle peut mal estimer P(xᵢ | C) pour les distributions non gaussiennes.

### Solutions possibles
1. **Transformation logarithmique** des caractéristiques asymétriques
2. **Discrétisation** des caractéristiques continues
3. Utiliser un **Naive Bayes multinomial** pour les données discrètes

---

## 6.4 Lissage de Laplace / Var Smoothing

### Le problème
Si une caractéristique a une variance nulle dans une classe (même valeur pour tous
les échantillons), la formule gaussienne donne une division par zéro.

### Notre solution
```python
self.variances_[idx, :] = X_c.var(axis=0) + self.var_smoothing  # ε = 10⁻⁹
```

Le paramètre `var_smoothing = 1e-9` est identique à celui utilisé par `sklearn.naive_bayes.GaussianNB`.

---

## 6.5 Sélection des caractéristiques

### Le problème
Le dataset NSL-KDD a **41 caractéristiques**. Certaines sont :
- **Redondantes** (plusieurs mesures de la même chose)
- **Non informatives** (variance très faible)
- **Bruitées** (ajoutent du bruit sans signal)

### Notre approche
On sélectionne **14 caractéristiques numériques + 3 catégorielles = 17 au total**.

Les caractéristiques sélectionnées :
```
Numériques : duration, src_bytes, dst_bytes, count, srv_count,
             serror_rate, rerror_rate, same_srv_rate, diff_srv_rate,
             dst_host_count, dst_host_srv_count, dst_host_same_srv_rate,
             dst_host_serror_rate, dst_host_rerror_rate

Catégorielles : protocol_type, service, flag
```

---

## 6.6 Résumé des points critiques

| Point | Gravité | Notre solution |
|-------|---------|----------------|
| Déséquilibre des classes | Moyenne | Données relativement équilibrées |
| Indépendance violée | Faible | Fonctionne bien en pratique |
| Hypothèse gaussienne | Moyenne | Normalisation MinMax |
| Division par zéro | Critique | var_smoothing = 10⁻⁹ |
| Trop de caractéristiques | Moyenne | Sélection de 17/41 |
