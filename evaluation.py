"""
evaluation.py — Métriques d'évaluation et visualisations

Métriques calculées :
    - Accuracy  : (TP + TN) / (TP + TN + FP + FN)
    - Precision : TP / (TP + FP)
    - Recall    : TP / (TP + FN)
    - F1-Score  : 2 * (Precision * Recall) / (Precision + Recall)
    - Matrice de confusion
"""

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc
)


def compute_metrics(y_true, y_pred):
    """
    Calcule toutes les métriques d'évaluation.

    Retourne: dict avec accuracy, precision, recall, f1
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
    }
    return metrics


def get_classification_report(y_true, y_pred):
    """
    Retourne le rapport de classification complet sous forme de string.
    """
    target_names = ['Normal (0)', 'Attaque (1)']
    return classification_report(y_true, y_pred, target_names=target_names)


def format_metrics_text(metrics, model_name="Modèle"):
    """
    Formate les métriques en texte lisible.
    """
    text = f"=== {model_name} ===\n"
    text += f"  Exactitude : {metrics['accuracy']:.4f}  ({metrics['accuracy']*100:.2f}%)\n"
    text += f"  Précision  : {metrics['precision']:.4f}\n"
    text += f"  Rappel     : {metrics['recall']:.4f}\n"
    text += f"  Score F1   : {metrics['f1']:.4f}\n"
    return text


def plot_confusion_matrix(y_true, y_pred, title="Matrice de Confusion"):
    """
    Affiche la matrice de confusion sous forme de heatmap.
    """
    cm = confusion_matrix(y_true, y_pred)
    labels = ['Normal', 'Attaque']

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=labels, yticklabels=labels, ax=ax
    )
    ax.set_xlabel('Prédit')
    ax.set_ylabel('Réel')
    ax.set_title(title)
    plt.tight_layout()
    return fig


def plot_roc_curve(y_true, y_proba, title="Courbe ROC"):
    """
    Trace la courbe ROC et calcule l'AUC.

    Paramètres:
        y_proba : probabilités de la classe positive (Attaque)
    """
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color='darkorange', lw=2,
            label=f'ROC (AUC = {roc_auc:.4f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Taux de Faux Positifs (TFP)')
    ax.set_ylabel('Taux de Vrais Positifs (TVP)')
    ax.set_title(title)
    ax.legend(loc='lower right')
    plt.tight_layout()
    return fig


def plot_comparison(metrics_manual, metrics_sklearn):
    """
    Compare visuellement les deux modèles (barres côte à côte).
    """
    metric_names = ['Exactitude', 'Précision', 'Rappel', 'Score F1']
    keys = ['accuracy', 'precision', 'recall', 'f1']

    values_manual = [metrics_manual[k] for k in keys]
    values_sklearn = [metrics_sklearn[k] for k in keys]

    x = np.arange(len(metric_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar(x - width/2, values_manual, width, label='Manuel', color='#2196F3')
    bars2 = ax.bar(x + width/2, values_sklearn, width, label='Sklearn', color='#FF9800')

    ax.set_ylabel('Score')
    ax.set_title('Comparaison : Implémentation manuelle vs Sklearn')
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names)
    ax.set_ylim(0, 1.1)
    ax.legend()

    # Ajouter les valeurs au-dessus des barres
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
                     xytext=(0, 3), textcoords="offset points", ha='center', fontsize=9)
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
                     xytext=(0, 3), textcoords="offset points", ha='center', fontsize=9)

    plt.tight_layout()
    return fig


def plot_class_distribution(y_train, y_test):
    """
    Affiche la distribution des classes dans les données train/test.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    for ax, data, title in zip(axes, [y_train, y_test], ['Entraînement', 'Test']):
        unique, counts = np.unique(data, return_counts=True)
        labels = ['Normal' if u == 0 else 'Attaque' for u in unique]
        colors = ['#4CAF50' if u == 0 else '#F44336' for u in unique]

        ax.bar(labels, counts, color=colors)
        ax.set_title(f'Distribution des classes — {title}')
        ax.set_ylabel('Nombre')

        for i, (label, count) in enumerate(zip(labels, counts)):
            pct = count / sum(counts) * 100
            ax.annotate(f'{count}\n({pct:.1f}%)', xy=(i, count),
                        xytext=(0, 5), textcoords="offset points",
                        ha='center', fontsize=10)

    plt.tight_layout()
    return fig
