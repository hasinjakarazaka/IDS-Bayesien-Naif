"""
main.py — Point d'entrée de l'IDS Bayésien Naïf

Projet M1 Cybersécurité — Probabilités & Statistiques
Sujet : Détection d'intrusions réseau par classifieur bayésien naïf

Lancement :
    python main.py
"""

import sys
import os
import tkinter as tk

# Ajouter le dossier du projet au path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gui import IDSApp


def main():
    """Lance l'application Tkinter."""
    root = tk.Tk()

    # Icône et style
    try:
        root.iconbitmap(default='')
    except tk.TclError:
        pass

    # Style ttk
    style = tk.ttk.Style()
    available_themes = style.theme_names()
    if 'clam' in available_themes:
        style.theme_use('clam')
    elif 'vista' in available_themes:
        style.theme_use('vista')

    app = IDSApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
