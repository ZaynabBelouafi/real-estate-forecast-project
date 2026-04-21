import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings("ignore")

os.chdir(r"C:\Users\hp\Desktop\Prévision économique")

df_kaggle = pd.read_csv(r"data\train.csv")
print(f"✅ Dataset chargé : {df_kaggle.shape[0]} lignes, {df_kaggle.shape[1]} colonnes")
print(df_kaggle.head())

xl = pd.ExcelFile(r"data\BKAM Séries IPAI T4 2025.xlsx")
df_ipai_raw = pd.read_excel(r"data\BKAM Séries IPAI T4 2025.xlsx",
                             sheet_name=xl.sheet_names[0], skiprows=3)
print(f"✅ IPAI chargé : {df_ipai_raw.shape}")

cols_a_supprimer = [c for c in df_kaggle.columns if df_kaggle[c].isnull().mean() > 0.4]
df_kaggle = df_kaggle.drop(columns=cols_a_supprimer)

for col in df_kaggle.select_dtypes(include=np.number).columns:
    df_kaggle[col] = df_kaggle[col].fillna(df_kaggle[col].median())

for col in df_kaggle.select_dtypes(include="object").columns:
    df_kaggle[col] = df_kaggle[col].fillna(df_kaggle[col].mode()[0])

target_col = "SalePrice" if "SalePrice" in df_kaggle.columns else df_kaggle.select_dtypes(include=np.number).columns[-1]
print(f"✅ Colonne cible : {target_col}")

Q1 = df_kaggle[target_col].quantile(0.01)
Q3 = df_kaggle[target_col].quantile(0.99)
avant = len(df_kaggle)
df_kaggle = df_kaggle[(df_kaggle[target_col] >= Q1) & (df_kaggle[target_col] <= Q3)]
print(f"✅ Outliers supprimés : {avant - len(df_kaggle)} | Reste : {len(df_kaggle)}")

df_encoded = pd.get_dummies(df_kaggle, drop_first=False)
print(f"✅ Encodage : {df_encoded.shape[1]} colonnes")

df_kaggle.to_csv(r"data\kaggle_clean.csv", index=False)
df_encoded.to_csv(r"data\kaggle_encoded.csv", index=False)
df_ipai_raw.to_csv(r"data\ipai_clean.csv", index=False)

print("\n✅ Fichiers sauvegardés !")
print("🎯 Lance : python notebooks\\02_eda.py")