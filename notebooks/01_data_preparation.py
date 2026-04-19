import pandas as pd
import numpy as np
import zipfile
import warnings

warnings.filterwarnings("ignore")

# -----------------------------
# 📦 Extraction du fichier ZIP
# -----------------------------
with zipfile.ZipFile("data/archive (2).zip", "r") as z:
    print("Fichiers dans l'archive :", z.namelist())
    z.extractall("data/kaggle")

# -----------------------------
# 📊 Chargement du dataset Kaggle
# -----------------------------
df_kaggle = pd.read_csv("data/kaggle/USA Housing Dataset.csv")

print(f"\n✅ Dataset Kaggle chargé : {df_kaggle.shape[0]} lignes, {df_kaggle.shape[1]} colonnes")
print(df_kaggle.head())
print("\nColonnes :", df_kaggle.columns.tolist())

# -----------------------------
# 📊 Chargement des données BKAM (IPAI)
# -----------------------------
xl = pd.ExcelFile("data/BKAM Séries IPAI T4 2025.xlsx")
print("\n📊 Onglets BKAM :", xl.sheet_names)

df_ipai_raw = pd.read_excel(
    "data/BKAM Séries IPAI T4 2025.xlsx",
    sheet_name=xl.sheet_names[0],
    skiprows=3
)

print(f"\n✅ IPAI chargé : {df_ipai_raw.shape}")
print(df_ipai_raw.head(10))

# -----------------------------
# 🧹 Nettoyage du dataset Kaggle
# -----------------------------

# Suppression des colonnes avec trop de valeurs manquantes (>40%)
cols_a_supprimer = [c for c in df_kaggle.columns if df_kaggle[c].isnull().mean() > 0.4]
df_kaggle = df_kaggle.drop(columns=cols_a_supprimer)

# Remplissage des valeurs numériques
for col in df_kaggle.select_dtypes(include=np.number).columns:
    df_kaggle[col] = df_kaggle[col].fillna(df_kaggle[col].median())

# Remplissage des variables catégorielles
for col in df_kaggle.select_dtypes(include="object").columns:
    df_kaggle[col] = df_kaggle[col].fillna(df_kaggle[col].mode()[0])

# -----------------------------
# 🎯 Détection de la variable cible
# -----------------------------
if "SalePrice" in df_kaggle.columns:
    target_col = "SalePrice"
elif "Price" in df_kaggle.columns:
    target_col = "Price"
elif "price" in df_kaggle.columns:
    target_col = "price"
else:
    target_col = df_kaggle.select_dtypes(include=np.number).columns[-1]

print(f"\n✅ Colonne cible : {target_col}")

# -----------------------------
# 🚫 Suppression des outliers
# -----------------------------
Q1 = df_kaggle[target_col].quantile(0.01)
Q3 = df_kaggle[target_col].quantile(0.99)

avant = len(df_kaggle)
df_kaggle = df_kaggle[(df_kaggle[target_col] >= Q1) & (df_kaggle[target_col] <= Q3)]

print(f"✅ Outliers supprimés : {avant - len(df_kaggle)} | Reste : {len(df_kaggle)}")

# -----------------------------
# 🔢 Encodage des variables catégorielles
# -----------------------------
df_encoded = pd.get_dummies(df_kaggle, drop_first=False)

print(f"✅ Encodage : {df_encoded.shape[1]} colonnes")

# -----------------------------
# 💾 Sauvegarde des fichiers
# -----------------------------
df_kaggle.to_csv("data/kaggle_clean.csv", index=False)
df_encoded.to_csv("data/kaggle_encoded.csv", index=False)
df_ipai_raw.to_csv("data/ipai_clean.csv", index=False)

print("\n✅ Fichiers sauvegardés dans data/")