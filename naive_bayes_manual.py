"""
naive_bayes_manual.py — Implémentation manuelle du classifieur bayésien naïf gaussien

Formules utilisées :
    P(C|X) ∝ P(C) * ∏ P(xi|C)

    Pour les features continues (hypothèse gaussienne) :
    P(xi|C) = (1 / sqrt(2π σ²)) * exp(-(xi - μ)² / (2σ²))

    Règle de décision :
    ŷ = argmax_C [ log P(C) + Σ log P(xi|C) ]
"""

import numpy as np


class NaiveBayesManual:
    """
    Classifieur Bayésien Naïf Gaussien — Implémentation depuis zéro.

    Attributs après fit() :
        classes_     : les classes uniques (ex: [0, 1])
        priors_      : P(C) pour chaque classe
        means_       : μ(xi | C) — moyenne de chaque feature par classe
        variances_   : σ²(xi | C) — variance de chaque feature par classe
    """

    def __init__(self, var_smoothing=1e-9):
        """
        Paramètres:
            var_smoothing : petite valeur ajoutée à la variance pour
                            éviter la division par zéro (lissage)
        """
        self.var_smoothing = var_smoothing
        self.classes_ = None
        self.priors_ = None
        self.means_ = None
        self.variances_ = None

    def fit(self, X, y):
        """
        Entraîne le classifieur en estimant les paramètres :
        - P(C) : probabilités a priori
        - μ(xi|C) : moyennes par feature et par classe
        - σ²(xi|C) : variances par feature et par classe

        Paramètres:
            X : array de shape (n_samples, n_features)
            y : array de shape (n_samples,) avec les labels
        """
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        n_features = X.shape[1]

        self.priors_ = np.zeros(n_classes)
        self.means_ = np.zeros((n_classes, n_features))
        self.variances_ = np.zeros((n_classes, n_features))

        for idx, c in enumerate(self.classes_):
            X_c = X[y == c]

            # P(C) = |X_c| / |X|
            self.priors_[idx] = X_c.shape[0] / X.shape[0]

            # μ(xi | C)
            self.means_[idx, :] = X_c.mean(axis=0)

            # σ²(xi | C) + lissage
            self.variances_[idx, :] = X_c.var(axis=0) + self.var_smoothing

        return self

    def _log_likelihood(self, X):
        """
        Calcule log P(X|C) pour chaque classe en utilisant la densité gaussienne.

        log P(xi|C) = -0.5 * log(2π σ²) - (xi - μ)² / (2σ²)

        Retourne: array de shape (n_samples, n_classes)
        """
        n_classes = len(self.classes_)
        n_samples = X.shape[0]
        log_likelihoods = np.zeros((n_samples, n_classes))

        for idx in range(n_classes):
            mean = self.means_[idx]
            var = self.variances_[idx]

            # Log de la densité gaussienne pour chaque feature
            log_prob = -0.5 * np.log(2 * np.pi * var)
            log_prob = log_prob - (X - mean) ** 2 / (2 * var)

            # Somme sur toutes les features (hypothèse d'indépendance)
            log_likelihoods[:, idx] = log_prob.sum(axis=1)

        return log_likelihoods

    def predict_log_proba(self, X):
        """
        Calcule log P(C|X) ∝ log P(C) + log P(X|C)

        Retourne: array de shape (n_samples, n_classes)
        """
        log_prior = np.log(self.priors_)
        log_likelihood = self._log_likelihood(X)

        # log P(C|X) ∝ log P(C) + Σ log P(xi|C)
        log_posterior = log_prior + log_likelihood

        return log_posterior

    def predict_proba(self, X):
        """
        Calcule P(C|X) normalisé (somme = 1 pour chaque sample).

        Retourne: array de shape (n_samples, n_classes)
        """
        log_posterior = self.predict_log_proba(X)

        # Normalisation avec le log-sum-exp trick pour la stabilité numérique
        log_max = log_posterior.max(axis=1, keepdims=True)
        log_posterior_shifted = log_posterior - log_max
        proba = np.exp(log_posterior_shifted)
        proba = proba / proba.sum(axis=1, keepdims=True)

        return proba

    def predict(self, X):
        """
        Prédit la classe pour chaque sample.
        Règle : ŷ = argmax_C P(C|X)

        Retourne: array de shape (n_samples,)
        """
        log_posterior = self.predict_log_proba(X)
        return self.classes_[np.argmax(log_posterior, axis=1)]

    def score(self, X, y):
        """
        Retourne l'accuracy sur le jeu de données (X, y).
        """
        predictions = self.predict(X)
        return np.mean(predictions == y)

    def get_params(self):
        """
        Retourne les paramètres appris sous forme de dictionnaire.
        """
        return {
            'classes': self.classes_,
            'priors': self.priors_,
            'means': self.means_,
            'variances': self.variances_,
        }
