"""
data_loader.py — Chargement et prétraitement du dataset NSL-KDD
"""

import os
import urllib.request
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler


# URLs de téléchargement du dataset NSL-KDD
NSL_KDD_URLS = {
    "KDDTrain+.txt": "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain%2B.txt",
    "KDDTest+.txt": "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTest%2B.txt",
}


def download_nsl_kdd(data_dir="data"):
    """
    Télécharge automatiquement le dataset NSL-KDD si absent.
    """
    os.makedirs(data_dir, exist_ok=True)
    for filename, url in NSL_KDD_URLS.items():
        filepath = os.path.join(data_dir, filename)
        if not os.path.exists(filepath):
            print(f"Téléchargement de {filename}...")
            urllib.request.urlretrieve(url, filepath)
            print(f"  → {filepath} ({os.path.getsize(filepath) // 1024} Ko)")
    print("Dataset NSL-KDD prêt.")


# Les 41 features du dataset NSL-KDD + label + difficulty
KDD_COLUMNS = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
    'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins',
    'logged_in', 'num_compromised', 'root_shell', 'su_attempted', 'num_root',
    'num_file_creations', 'num_shells', 'num_access_files',
    'num_outbound_cmds', 'is_host_login', 'is_guest_login', 'count',
    'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate',
    'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
    'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
    'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
    'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate',
    'dst_host_serror_rate', 'dst_host_srv_serror_rate',
    'dst_host_rerror_rate', 'dst_host_srv_rerror_rate',
    'label', 'difficulty'
]

# Features catégorielles
CATEGORICAL_FEATURES = ['protocol_type', 'service', 'flag']

# Features numériques sélectionnées (les plus pertinentes pour l'IDS)
SELECTED_NUMERIC = [
    'duration', 'src_bytes', 'dst_bytes', 'count', 'srv_count',
    'serror_rate', 'rerror_rate', 'same_srv_rate', 'diff_srv_rate',
    'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate',
    'dst_host_serror_rate', 'dst_host_rerror_rate'
]


def load_nsl_kdd(train_path, test_path=None):
    """
    Charge le dataset NSL-KDD depuis des fichiers texte.
    Retourne un DataFrame avec les colonnes nommées.
    """
    df_train = pd.read_csv(train_path, header=None, names=KDD_COLUMNS)

    if test_path and os.path.exists(test_path):
        df_test = pd.read_csv(test_path, header=None, names=KDD_COLUMNS)
    else:
        df_test = None

    return df_train, df_test


def binarize_labels(df):
    """
    Convertit les labels multi-classes en binaire : Normal (0) / Attaque (1).
    """
    df = df.copy()
    df['label_bin'] = df['label'].apply(lambda x: 0 if x.strip() == 'normal' else 1)
    return df


def preprocess(df, scaler=None, encoders=None, fit=True):
    """
    Prétraitement complet :
    1. Binarisation des labels
    2. Encodage des variables catégorielles (LabelEncoder)
    3. Normalisation des variables numériques (MinMaxScaler)

    Paramètres:
        df       : DataFrame brut
        scaler   : MinMaxScaler existant (pour le test set)
        encoders : dict de LabelEncoders existants (pour le test set)
        fit      : True si on doit fit (train), False si on applique (test)

    Retourne: X (array), y (array), scaler, encoders
    """
    df = binarize_labels(df)

    # Supprimer la colonne difficulty si présente
    if 'difficulty' in df.columns:
        df = df.drop(columns=['difficulty'])

    # --- Encodage des features catégorielles ---
    if encoders is None:
        encoders = {}

    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            if fit:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                encoders[col] = le
            else:
                le = encoders[col]
                # Gérer les valeurs inconnues
                known = set(le.classes_)
                df[col] = df[col].astype(str).apply(
                    lambda x: x if x in known else le.classes_[0]
                )
                df[col] = le.transform(df[col])

    # --- Sélection des features ---
    feature_cols = SELECTED_NUMERIC + CATEGORICAL_FEATURES
    feature_cols = [c for c in feature_cols if c in df.columns]
    X = df[feature_cols].values.astype(np.float64)
    y = df['label_bin'].values

    # --- Normalisation ---
    if fit:
        scaler = MinMaxScaler()
        X = scaler.fit_transform(X)
    else:
        X = scaler.transform(X)

    return X, y, scaler, encoders, feature_cols


def generate_synthetic_data(n_samples=2000, random_state=42):
    """
    Génère des données synthétiques simulant du trafic réseau
    pour tester le système sans le dataset NSL-KDD.

    Features simulées :
    - packets_per_sec    : nombre de paquets par seconde
    - connections        : nombre de connexions simultanées
    - duration           : durée de la session (secondes)
    - error_rate         : taux d'erreurs (0 à 1)
    - src_bytes          : octets envoyés
    - dst_bytes          : octets reçus
    """
    np.random.seed(random_state)
    n_normal = n_samples // 2
    n_attack = n_samples - n_normal

    # --- Trafic normal ---
    normal = np.column_stack([
        np.random.normal(50, 15, n_normal),      # packets_per_sec
        np.random.normal(10, 4, n_normal),        # connections
        np.random.exponential(30, n_normal),       # duration
        np.random.beta(2, 20, n_normal),           # error_rate
        np.random.normal(5000, 2000, n_normal),    # src_bytes
        np.random.normal(8000, 3000, n_normal),    # dst_bytes
    ])

    # --- Trafic attaque ---
    attack = np.column_stack([
        np.random.normal(500, 150, n_attack),     # packets_per_sec (beaucoup plus)
        np.random.normal(80, 30, n_attack),       # connections (beaucoup plus)
        np.random.exponential(3, n_attack),        # duration (très court)
        np.random.beta(10, 5, n_attack),           # error_rate (élevé)
        np.random.normal(200, 100, n_attack),      # src_bytes (petit payload)
        np.random.normal(500, 200, n_attack),      # dst_bytes (petit)
    ])

    X = np.vstack([normal, attack])
    y = np.array([0] * n_normal + [1] * n_attack)

    # Assurer que toutes les valeurs sont positives
    X = np.clip(X, 0, None)

    feature_names = [
        'packets_per_sec', 'connections', 'duration',
        'error_rate', 'src_bytes', 'dst_bytes'
    ]

    return X, y, feature_names


def load_and_prepare(data_dir="data", test_size=0.2, random_state=42):
    """
    Fonction principale : charge les données NSL-KDD si disponibles,
    sinon génère des données synthétiques.

    Retourne: X_train, X_test, y_train, y_test, feature_names, data_source
    """
    train_path = os.path.join(data_dir, "KDDTrain+.txt")
    test_path = os.path.join(data_dir, "KDDTest+.txt")

    # Télécharger automatiquement si les fichiers sont absents
    if not os.path.exists(train_path):
        try:
            download_nsl_kdd(data_dir)
        except Exception as e:
            print(f"Téléchargement échoué ({e}), utilisation de données synthétiques.")

    if os.path.exists(train_path):
        # --- Charger NSL-KDD ---
        df_train, df_test = load_nsl_kdd(train_path, test_path)

        X_train, y_train, scaler, encoders, feature_names = preprocess(
            df_train, fit=True
        )

        if df_test is not None:
            X_test, y_test, _, _, _ = preprocess(
                df_test, scaler=scaler, encoders=encoders, fit=False
            )
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X_train, y_train, test_size=test_size,
                random_state=random_state, stratify=y_train
            )

        data_source = "NSL-KDD"
    else:
        # --- Données synthétiques ---
        X, y, feature_names = generate_synthetic_data(
            n_samples=3000, random_state=random_state
        )

        # Normaliser
        scaler = MinMaxScaler()
        X = scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size,
            random_state=random_state, stratify=y
        )

        data_source = "Synthétique (NSL-KDD non trouvé dans data/)"

    return X_train, X_test, y_train, y_test, feature_names, data_source
