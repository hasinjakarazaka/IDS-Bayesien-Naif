"""
naive_bayes_sklearn.py — Classifieur bayésien naïf avec scikit-learn

Utilise GaussianNB pour comparer avec l'implémentation manuelle.
"""

from sklearn.naive_bayes import GaussianNB


def create_sklearn_model(var_smoothing=1e-9):
    """
    Crée et retourne un classifieur GaussianNB de scikit-learn.

    Paramètres:
        var_smoothing : lissage ajouté à la variance (équivalent au
                        var_smoothing de notre implémentation manuelle)
    """
    model = GaussianNB(var_smoothing=var_smoothing)
    return model


def train_and_predict(X_train, y_train, X_test, var_smoothing=1e-9):
    """
    Entraîne le modèle sklearn et retourne les prédictions + probabilités.

    Retourne: model, y_pred, y_proba
    """
    model = create_sklearn_model(var_smoothing=var_smoothing)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    return model, y_pred, y_proba
