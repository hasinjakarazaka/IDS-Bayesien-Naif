# 5. Métriques d'Évaluation

## 5.1 Matrice de confusion

La matrice de confusion résume toutes les prédictions du modèle :

```
                    Prédit Normal    Prédit Attaque
Réel Normal      │     VN          │     FP          │
Réel Attaque     │     FN          │     VP          │
```

| Terme | Signification | Exemple IDS |
|-------|---------------|-------------|
| **VP** (Vrais Positifs) | Attaque correctement détectée | Le système a raison de déclencher l'alerte |
| **VN** (Vrais Négatifs) | Normal correctement identifié | Le système laisse passer le trafic légitime |
| **FP** (Faux Positifs) | Normal classé comme attaque | **Fausse alerte** — dérange l'administrateur |
| **FN** (Faux Négatifs) | Attaque non détectée | **Danger** — l'attaque passe inaperçue |

---

## 5.2 Les 4 métriques principales

### Exactitude (Accuracy)

```
Exactitude = (VP + VN) / (VP + VN + FP + FN)
```

**Interprétation** : proportion de prédictions correctes parmi toutes les prédictions.

⚠️ **Piège** : si 90% du trafic est normal, un modèle qui dit toujours "normal" a 90%
d'exactitude mais ne détecte aucune attaque !

---

### Précision (Precision)

```
Précision = VP / (VP + FP)
```

**Interprétation** : parmi toutes les alertes déclenchées, combien sont de vraies attaques ?

- Précision **élevée** → peu de fausses alertes
- Précision **faible** → beaucoup de fausses alertes (fatigue de l'administrateur)

---

### Rappel (Recall / Sensibilité)

```
Rappel = VP / (VP + FN)
```

**Interprétation** : parmi toutes les vraies attaques, combien ont été détectées ?

- Rappel **élevé** → on détecte la plupart des attaques
- Rappel **faible** → beaucoup d'attaques passent inaperçues

**Pour un IDS, le rappel est CRITIQUE** : une attaque non détectée peut être catastrophique.

---

### Score F1

```
F1 = 2 × (Précision × Rappel) / (Précision + Rappel)
```

**Interprétation** : moyenne harmonique entre précision et rappel. C'est le meilleur
indicateur unique de la qualité du classifieur.

- F1 = 1 → parfait
- F1 = 0 → aucune détection correcte

---

## 5.3 Courbe ROC et AUC

### Courbe ROC (Receiver Operating Characteristic)

Elle trace le **Taux de Vrais Positifs (TVP = Rappel)** en fonction du
**Taux de Faux Positifs (TFP)** pour différents seuils de décision.

```
TVP = VP / (VP + FN)    ← Rappel
TFP = FP / (FP + VN)    ← Taux de fausses alertes
```

### AUC (Aire Sous la Courbe)

- **AUC = 1.0** → classifieur parfait
- **AUC = 0.5** → classifieur aléatoire (inutile)
- **AUC > 0.8** → bon classifieur

---

## 5.4 Résultats de notre IDS

### Implémentation manuelle :
```
Exactitude : 75.42%
Précision  : 92.61%  → Peu de fausses alertes
Rappel     : 61.74%  → Détecte ~62% des attaques
Score F1   : 74.09%
```

### Interprétation :
- **Précision très élevée (92.61%)** : quand le système dit "attaque", il a raison 93 fois sur 100
- **Rappel moyen (61.74%)** : environ 38% des attaques passent inaperçues
- **Piste d'amélioration** : ajuster le seuil de décision pour augmenter le rappel au prix d'un peu de précision

### Comparaison Manuel vs Sklearn :
Les résultats sont quasiment identiques (75.42% vs 75.38%), ce qui **valide** notre
implémentation manuelle.
