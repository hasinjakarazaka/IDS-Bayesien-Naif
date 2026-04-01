# Explication Complète — IDS Bayésien Naïf

## Sommaire

Ce dossier contient l'explication détaillée de l'implémentation manuelle du classifieur
bayésien naïf, avec les formules mathématiques, le code correspondant et les résultats.

### Documents

| # | Fichier | Contenu |
|---|---------|---------|
| 1 | [01_theoreme_de_bayes.md](01_theoreme_de_bayes.md) | Formule de Bayes, signification de chaque terme, application à l'IDS |
| 2 | [02_hypothese_naive.md](02_hypothese_naive.md) | Indépendance conditionnelle, pourquoi "naïve", formule complète |
| 3 | [03_estimation_des_probabilites.md](03_estimation_des_probabilites.md) | P(C), densité gaussienne, logarithme, lissage |
| 4 | [04_implementation_manuelle_detaillee.md](04_implementation_manuelle_detaillee.md) | Chaque méthode du code expliquée ligne par ligne |
| 5 | [05_metriques_evaluation.md](05_metriques_evaluation.md) | Matrice de confusion, exactitude, précision, rappel, F1, ROC/AUC |
| 6 | [06_points_critiques.md](06_points_critiques.md) | Déséquilibre, indépendance violée, hypothèse gaussienne, lissage |

### Ordre de lecture recommandé

Pour la soutenance, présenter dans cet ordre :
1. **Théorème de Bayes** → poser le cadre mathématique
2. **Hypothèse naïve** → justifier la simplification
3. **Estimation des probabilités** → montrer comment on calcule chaque terme
4. **Implémentation manuelle** → faire le lien formules ↔ code
5. **Métriques d'évaluation** → présenter et interpréter les résultats
6. **Points critiques** → montrer la maturité de l'analyse
