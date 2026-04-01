"""
gui.py — Interface graphique Tkinter pour l'IDS Bayésien Naïf
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, filedialog, messagebox
import threading
import os
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from data_loader import load_and_prepare
from naive_bayes_manual import NaiveBayesManual
from naive_bayes_sklearn import train_and_predict
from evaluation import (
    compute_metrics, format_metrics_text, get_classification_report,
    plot_confusion_matrix, plot_roc_curve, plot_comparison,
    plot_class_distribution
)


class IDSApp:
    """Application principale de l'IDS Bayésien Naïf."""

    def __init__(self, root):
        self.root = root
        self.root.title("IDS Bayésien Naïf — Détection d'Intrusions Réseau")
        self.root.geometry("1000x700")
        self.root.minsize(900, 600)

        # Variables d'état
        self.data_loaded = False
        self.models_trained = False
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_names = None
        self.data_source = None
        self.model_manual = None
        self.model_sklearn = None
        self.metrics_manual = None
        self.metrics_sklearn = None
        self.y_pred_manual = None
        self.y_pred_sklearn = None
        self.y_proba_manual = None
        self.y_proba_sklearn = None

        self._build_ui()

    def _build_ui(self):
        """Construit l'interface utilisateur."""

        # --- Barre de titre ---
        header = tk.Frame(self.root, bg="#1565C0", height=50)
        header.pack(fill=tk.X)
        header.pack_propagate(False)
        tk.Label(
            header, text="🛡️  IDS Bayésien Naïf — Détection d'Intrusions Réseau",
            bg="#1565C0", fg="white", font=("Segoe UI", 14, "bold")
        ).pack(side=tk.LEFT, padx=15, pady=10)

        # --- Frame principal ---
        main_frame = tk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # --- Panneau gauche : contrôles ---
        left_panel = tk.LabelFrame(main_frame, text="Contrôles", width=280,
                                   font=("Segoe UI", 10, "bold"))
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 5), pady=5)
        left_panel.pack_propagate(False)
        left_panel.configure(width=280)

        # Section données
        data_frame = tk.LabelFrame(left_panel, text="1. Données",
                                   font=("Segoe UI", 9, "bold"))
        data_frame.pack(fill=tk.X, padx=5, pady=5)

        # Chemin absolu vers le dossier data/ (relatif au script)
        default_data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
        self.data_dir_var = tk.StringVar(value=default_data_dir)
        tk.Label(data_frame, text="Dossier des données :",
                 font=("Segoe UI", 9)).pack(anchor=tk.W, padx=5, pady=(5, 0))

        dir_frame = tk.Frame(data_frame)
        dir_frame.pack(fill=tk.X, padx=5, pady=2)
        tk.Entry(dir_frame, textvariable=self.data_dir_var,
                 font=("Segoe UI", 9)).pack(side=tk.LEFT, fill=tk.X, expand=True)
        tk.Button(dir_frame, text="...", width=3,
                  command=self._browse_dir).pack(side=tk.RIGHT, padx=(3, 0))

        btn_load = tk.Button(
            data_frame, text="📂  Charger les données",
            command=self._load_data, font=("Segoe UI", 9, "bold"),
            bg="#4CAF50", fg="white", relief=tk.FLAT, cursor="hand2"
        )
        btn_load.pack(fill=tk.X, padx=5, pady=5)

        # Section entraînement
        train_frame = tk.LabelFrame(left_panel, text="2. Entraînement",
                                    font=("Segoe UI", 9, "bold"))
        train_frame.pack(fill=tk.X, padx=5, pady=5)

        btn_train = tk.Button(
            train_frame, text="🧠  Entraîner les modèles",
            command=self._train_models, font=("Segoe UI", 9, "bold"),
            bg="#2196F3", fg="white", relief=tk.FLAT, cursor="hand2"
        )
        btn_train.pack(fill=tk.X, padx=5, pady=5)

        # Section visualisations
        viz_frame = tk.LabelFrame(left_panel, text="3. Visualisations",
                                  font=("Segoe UI", 9, "bold"))
        viz_frame.pack(fill=tk.X, padx=5, pady=5)

        buttons_viz = [
            ("📊  Distribution des classes", self._show_distribution),
            ("🔢  Matrice de confusion", self._show_confusion),
            ("📈  Courbe ROC", self._show_roc),
            ("⚖️  Comparaison des modèles", self._show_comparison),
        ]
        for text, cmd in buttons_viz:
            tk.Button(
                viz_frame, text=text, command=cmd,
                font=("Segoe UI", 9), relief=tk.GROOVE, cursor="hand2"
            ).pack(fill=tk.X, padx=5, pady=2)

        # Section prédiction manuelle
        pred_frame = tk.LabelFrame(left_panel, text="4. Test manuel",
                                   font=("Segoe UI", 9, "bold"))
        pred_frame.pack(fill=tk.X, padx=5, pady=5)

        tk.Label(pred_frame, text="Index de l'échantillon (dans le jeu de test) :",
                 font=("Segoe UI", 8)).pack(anchor=tk.W, padx=5, pady=(5, 0))

        self.sample_idx_var = tk.StringVar(value="0")
        tk.Entry(pred_frame, textvariable=self.sample_idx_var,
                 font=("Segoe UI", 9), width=10).pack(anchor=tk.W, padx=5, pady=2)

        tk.Button(
            pred_frame, text="🔍  Prédire cet échantillon",
            command=self._predict_sample, font=("Segoe UI", 9),
            bg="#FF9800", fg="white", relief=tk.FLAT, cursor="hand2"
        ).pack(fill=tk.X, padx=5, pady=5)

        # --- Panneau droit : résultats ---
        right_panel = tk.Frame(main_frame)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, pady=5)

        # Onglets
        self.notebook = ttk.Notebook(right_panel)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # Onglet Console
        console_tab = tk.Frame(self.notebook)
        self.notebook.add(console_tab, text="  Console  ")

        self.console = scrolledtext.ScrolledText(
            console_tab, wrap=tk.WORD, font=("Consolas", 10),
            bg="#1E1E1E", fg="#D4D4D4", insertbackground="white"
        )
        self.console.pack(fill=tk.BOTH, expand=True)

        # Onglet Graphiques
        graph_tab = tk.Frame(self.notebook)
        self.notebook.add(graph_tab, text="  Graphiques  ")

        self.graph_frame = tk.Frame(graph_tab)
        self.graph_frame.pack(fill=tk.BOTH, expand=True)

        # Onglet Théorie
        theory_tab = tk.Frame(self.notebook)
        self.notebook.add(theory_tab, text="  Théorie  ")

        theory_text = scrolledtext.ScrolledText(
            theory_tab, wrap=tk.WORD, font=("Consolas", 10),
            bg="#FAFAFA", fg="#333333"
        )
        theory_text.pack(fill=tk.BOTH, expand=True)
        theory_text.insert(tk.END, THEORY_TEXT)
        theory_text.config(state=tk.DISABLED)

        # Barre de statut
        self.status_var = tk.StringVar(value="Prêt — Chargez les données pour commencer.")
        status_bar = tk.Label(
            self.root, textvariable=self.status_var, bd=1, relief=tk.SUNKEN,
            anchor=tk.W, font=("Segoe UI", 9), bg="#E0E0E0"
        )
        status_bar.pack(fill=tk.X, side=tk.BOTTOM)

        # Message initial
        self._log("=" * 60)
        self._log("  IDS Bayésien Naïf — Projet M1 Cybersécurité")
        self._log("  Probabilités & Statistiques")
        self._log("=" * 60)
        self._log("")
        self._log("Instructions :")
        self._log("  1. Cliquer sur 'Charger les données'")
        self._log("     (données synthétiques si NSL-KDD absent)")
        self._log("  2. Cliquer sur 'Entraîner les modèles'")
        self._log("  3. Explorer les visualisations et les prédictions")
        self._log("")

    def _log(self, message):
        """Écrit un message dans la console."""
        self.console.insert(tk.END, message + "\n")
        self.console.see(tk.END)

    def _set_status(self, message):
        """Met à jour la barre de statut."""
        self.status_var.set(message)
        self.root.update_idletasks()

    def _browse_dir(self):
        """Ouvre un dialogue pour choisir le dossier des données."""
        directory = filedialog.askdirectory(title="Sélectionner le dossier des données")
        if directory:
            self.data_dir_var.set(directory)

    def _load_data(self):
        """Charge les données (NSL-KDD ou synthétiques)."""
        self._set_status("Chargement des données en cours...")
        self._log("--- Chargement des données ---")

        try:
            data_dir = self.data_dir_var.get()
            (self.X_train, self.X_test, self.y_train, self.y_test,
             self.feature_names, self.data_source) = load_and_prepare(data_dir=data_dir)

            self.data_loaded = True
            self.models_trained = False

            self._log(f"  Source : {self.data_source}")
            self._log(f"  Caractéristiques : {self.feature_names}")
            self._log(f"  Entraînement : {self.X_train.shape}")
            self._log(f"  Test         : {self.X_test.shape}")

            n_normal_train = np.sum(self.y_train == 0)
            n_attack_train = np.sum(self.y_train == 1)
            n_normal_test = np.sum(self.y_test == 0)
            n_attack_test = np.sum(self.y_test == 1)

            self._log(f"  Entraînement — Normal: {n_normal_train}, Attaque: {n_attack_train}")
            self._log(f"  Test         — Normal: {n_normal_test}, Attaque: {n_attack_test}")
            self._log("  ✅ Données chargées avec succès !")
            self._log("")

            self._set_status(f"Données chargées ({self.data_source})")

        except Exception as e:
            self._log(f"  ❌ Erreur : {e}")
            self._set_status("Erreur lors du chargement")
            messagebox.showerror("Erreur", str(e))

    def _train_models(self):
        """Entraîne les deux modèles (manuel + sklearn)."""
        if not self.data_loaded:
            messagebox.showwarning("Attention", "Veuillez d'abord charger les données.")
            return

        self._set_status("Entraînement en cours...")
        self._log("--- Entraînement des modèles ---")

        try:
            # --- Modèle manuel ---
            self._log("  [1/2] Entraînement du modèle MANUEL...")
            self.model_manual = NaiveBayesManual(var_smoothing=1e-9)
            self.model_manual.fit(self.X_train, self.y_train)

            self.y_pred_manual = self.model_manual.predict(self.X_test)
            self.y_proba_manual = self.model_manual.predict_proba(self.X_test)
            self.metrics_manual = compute_metrics(self.y_test, self.y_pred_manual)

            self._log(format_metrics_text(self.metrics_manual, "Bayes Naïf Manuel"))

            # Afficher les paramètres appris
            params = self.model_manual.get_params()
            self._log("  Paramètres appris (Manuel) :")
            self._log(f"    P(Normal)  = {params['priors'][0]:.4f}")
            self._log(f"    P(Attaque) = {params['priors'][1]:.4f}")
            self._log("")

            # --- Modèle sklearn ---
            self._log("  [2/2] Entraînement du modèle SKLEARN...")
            self.model_sklearn, self.y_pred_sklearn, self.y_proba_sklearn = (
                train_and_predict(self.X_train, self.y_train, self.X_test)
            )
            self.metrics_sklearn = compute_metrics(self.y_test, self.y_pred_sklearn)

            self._log(format_metrics_text(self.metrics_sklearn, "Bayes Naïf Sklearn"))

            # Rapport complet
            self._log("--- Rapport de classification (Manuel) ---")
            self._log(get_classification_report(self.y_test, self.y_pred_manual))

            self._log("--- Rapport de classification (Sklearn) ---")
            self._log(get_classification_report(self.y_test, self.y_pred_sklearn))

            self.models_trained = True
            self._log("  ✅ Entraînement terminé !")
            self._log("")
            self._set_status("Modèles entraînés — Explorez les visualisations")

        except Exception as e:
            self._log(f"  ❌ Erreur : {e}")
            self._set_status("Erreur lors de l'entraînement")
            messagebox.showerror("Erreur", str(e))

    def _show_plot(self, fig):
        """Affiche un graphique matplotlib dans l'onglet Graphiques."""
        # Nettoyer l'ancien graphique
        for widget in self.graph_frame.winfo_children():
            widget.destroy()

        canvas = FigureCanvasTkAgg(fig, master=self.graph_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Basculer sur l'onglet Graphiques
        self.notebook.select(1)

    def _show_distribution(self):
        """Affiche la distribution des classes."""
        if not self.data_loaded:
            messagebox.showwarning("Attention", "Veuillez d'abord charger les données.")
            return
        fig = plot_class_distribution(self.y_train, self.y_test)
        self._show_plot(fig)

    def _show_confusion(self):
        """Affiche les matrices de confusion."""
        if not self.models_trained:
            messagebox.showwarning("Attention", "Veuillez d'abord entraîner les modèles.")
            return

        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        from sklearn.metrics import confusion_matrix
        import seaborn as sns

        for ax, y_pred, title in zip(
            axes,
            [self.y_pred_manual, self.y_pred_sklearn],
            ["Manuel", "Sklearn"]
        ):
            cm = confusion_matrix(self.y_test, y_pred)
            labels = ['Normal', 'Attaque']
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=labels, yticklabels=labels, ax=ax)
            ax.set_xlabel('Prédit')
            ax.set_ylabel('Réel')
            ax.set_title(f'Matrice de Confusion — {title}')

        plt.tight_layout()
        self._show_plot(fig)

    def _show_roc(self):
        """Affiche les courbes ROC."""
        if not self.models_trained:
            messagebox.showwarning("Attention", "Veuillez d'abord entraîner les modèles.")
            return

        import matplotlib.pyplot as plt
        from sklearn.metrics import roc_curve, auc

        fig, ax = plt.subplots(figsize=(7, 5))

        for y_proba, label, color in [
            (self.y_proba_manual[:, 1], "Manuel", "#2196F3"),
            (self.y_proba_sklearn[:, 1], "Sklearn", "#FF9800"),
        ]:
            fpr, tpr, _ = roc_curve(self.y_test, y_proba)
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, color=color, lw=2,
                    label=f'{label} (AUC = {roc_auc:.4f})')

        ax.plot([0, 1], [0, 1], 'k--', lw=1)
        ax.set_xlabel('Taux de Faux Positifs (TFP)')
        ax.set_ylabel('Taux de Vrais Positifs (TVP)')
        ax.set_title('Courbes ROC — Manuel vs Sklearn')
        ax.legend(loc='lower right')
        plt.tight_layout()
        self._show_plot(fig)

    def _show_comparison(self):
        """Affiche la comparaison des métriques."""
        if not self.models_trained:
            messagebox.showwarning("Attention", "Veuillez d'abord entraîner les modèles.")
            return
        fig = plot_comparison(self.metrics_manual, self.metrics_sklearn)
        self._show_plot(fig)

    def _predict_sample(self):
        """Prédit la classe d'un sample spécifique du test set."""
        if not self.models_trained:
            messagebox.showwarning("Attention", "Veuillez d'abord entraîner les modèles.")
            return

        try:
            idx = int(self.sample_idx_var.get())
            if idx < 0 or idx >= len(self.X_test):
                messagebox.showwarning(
                    "Attention",
                    f"Index invalide. Plage valide : 0 - {len(self.X_test) - 1}"
                )
                return

            sample = self.X_test[idx].reshape(1, -1)
            true_label = self.y_test[idx]

            # Prédiction manuelle
            pred_manual = self.model_manual.predict(sample)[0]
            proba_manual = self.model_manual.predict_proba(sample)[0]

            # Prédiction sklearn
            pred_sklearn = self.model_sklearn.predict(sample)[0]
            proba_sklearn = self.model_sklearn.predict_proba(sample)[0]

            label_map = {0: "Normal", 1: "Attaque"}

            self._log(f"--- Prédiction pour l'échantillon #{idx} ---")
            self._log(f"  Vraie classe       : {label_map[true_label]} ({true_label})")
            self._log(f"  Caractéristiques   : {np.round(sample[0], 4)}")
            self._log("")
            self._log(f"  [Manuel]  Prédit : {label_map[pred_manual]}")
            self._log(f"    P(Normal)  = {proba_manual[0]:.6f}")
            self._log(f"    P(Attaque) = {proba_manual[1]:.6f}")
            self._log(f"    → {'✅ Correct' if pred_manual == true_label else '❌ Incorrect'}")
            self._log("")
            self._log(f"  [Sklearn] Prédit : {label_map[pred_sklearn]}")
            self._log(f"    P(Normal)  = {proba_sklearn[0]:.6f}")
            self._log(f"    P(Attaque) = {proba_sklearn[1]:.6f}")
            self._log(f"    → {'✅ Correct' if pred_sklearn == true_label else '❌ Incorrect'}")
            self._log("")

            # Alerte si attaque détectée
            if pred_manual == 1 or pred_sklearn == 1:
                self._log("  🚨 ALERTE : Trafic anormal détecté !")
                self._log("")

            # Basculer sur la console
            self.notebook.select(0)

        except ValueError:
            messagebox.showwarning("Attention", "Veuillez entrer un index numérique valide.")


# --- Texte théorique pour l'onglet Théorie ---
THEORY_TEXT = """
══════════════════════════════════════════════════════════
     THÉORÈME DE BAYES & CLASSIFIEUR NAÏF BAYÉSIEN
══════════════════════════════════════════════════════════

1. THÉORÈME DE BAYES
─────────────────────
Le théorème de Bayes permet de calculer la probabilité
a posteriori d'une classe C sachant une observation X :

    P(C | X) = P(X | C) × P(C) / P(X)

Où :
  • P(C)     = probabilité a priori de la classe
  • P(X | C) = vraisemblance
  • P(C | X) = probabilité a posteriori
  • P(X)     = évidence (constante de normalisation)


2. HYPOTHÈSE NAÏVE
────────────────────
On suppose que les caractéristiques sont indépendantes
conditionnellement sachant la classe :

    P(X | C) = P(x₁|C) × P(x₂|C) × ... × P(xₙ|C)

Cette hypothèse simplifie considérablement le calcul.


3. CARACTÉRISTIQUES CONTINUES (GAUSSIENNE)
────────────────────────────────────────────
Pour chaque caractéristique continue xᵢ, on suppose une
distribution normale :

    P(xᵢ | C) = 1/√(2π σ²) × exp(-(xᵢ - μ)² / (2σ²))

Où μ et σ² sont estimés sur les données d'entraînement.


4. LISSAGE DE LAPLACE
──────────────────────
Pour les caractéristiques catégorielles, on ajoute un terme α
pour éviter les probabilités nulles :

    P(xᵢ | C) = (count(xᵢ, C) + α) / (count(C) + α × |V|)


5. RÈGLE DE DÉCISION
──────────────────────
Pour classifier une observation X :

    ŷ = argmax_C [ P(C) × ∏ P(xᵢ | C) ]

En pratique, on utilise le logarithme pour éviter les
dépassements numériques par le bas :

    ŷ = argmax_C [ log P(C) + Σ log P(xᵢ | C) ]


6. APPLICATION À L'IDS
────────────────────────
  • Classes : Normal (0) / Attaque (1)
  • Si P(Attaque | X) > P(Normal | X) → ALERTE
  • Caractéristiques : paquets/s, connexions, durée, erreurs...


7. MÉTRIQUES D'ÉVALUATION
───────────────────────────
  • Exactitude = (VP + VN) / Total
  • Précision  = VP / (VP + FP)
  • Rappel     = VP / (VP + FN)
  • Score F1   = 2 × Précision × Rappel / (Précision + Rappel)

Où :
  VP = Vrais Positifs  (attaque détectée correctement)
  VN = Vrais Négatifs  (normal identifié correctement)
  FP = Faux Positifs   (normal classé comme attaque)
  FN = Faux Négatifs   (attaque non détectée)


══════════════════════════════════════════════════════════
"""
