# 1. Le Théorème de Bayes

## 1.1 Formule fondamentale

Le théorème de Bayes relie la probabilité **a posteriori** d'une classe C sachant une observation X :

```
P(C | X) = P(X | C) × P(C) / P(X)
```

### Signification de chaque terme

| Terme | Nom | Signification |
|-------|-----|---------------|
| P(C) | **Probabilité a priori** | Probabilité de la classe avant toute observation. Ex : 53% du trafic est normal |
| P(X \| C) | **Vraisemblance** | Probabilité d'observer X si on est dans la classe C |
| P(C \| X) | **Probabilité a posteriori** | Ce qu'on cherche : la probabilité de la classe sachant l'observation |
| P(X) | **Évidence** | Constante de normalisation (même pour toutes les classes) |

### Exemple concret (IDS)

On observe un paquet réseau X avec :
- 500 paquets/seconde
- 80 connexions simultanées
- durée de session : 2 secondes

On veut calculer :
- P(Attaque | X) = ?
- P(Normal | X) = ?

Si P(Attaque | X) > P(Normal | X) → **ALERTE**

## 1.2 Pourquoi Bayes est adapté à l'IDS ?

1. **Décision probabiliste** : on ne dit pas juste "attaque" ou "normal", on donne une probabilité
2. **Prise en compte du contexte** : P(C) reflète la fréquence réelle des attaques
3. **Mise à jour** : on peut recalculer les probabilités quand de nouvelles données arrivent
4. **Interprétable** : on peut expliquer pourquoi le système a déclenché une alerte

## 1.3 Simplification par comparaison

Comme P(X) est identique pour les deux classes, on compare directement :

```
P(Attaque | X) ∝ P(X | Attaque) × P(Attaque)
P(Normal  | X) ∝ P(X | Normal)  × P(Normal)
```

→ On choisit la classe avec le produit le plus grand.
