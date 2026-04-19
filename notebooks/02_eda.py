import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings("ignore")

os.chdir(r"C:\Users\hp\Desktop\Prévision économique")

plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["axes.facecolor"]   = "#f9f9f9"
plt.rcParams["axes.grid"]        = True
plt.rcParams["grid.alpha"]       = 0.3

# Chargement
df      = pd.read_csv(r"data\kaggle_clean.csv")
df_ipai = pd.read_csv(r"data\ipai_clean.csv")
target_col = "price"

print(f"✅ Données chargées : {df.shape[0]} biens, {df.shape[1]} colonnes")

# ── 2.1 Distribution du prix ──────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Distribution des prix", fontsize=13, fontweight="bold")

axes[0].hist(df[target_col], bins=60, color="#534AB7", alpha=0.8, edgecolor="white")
axes[0].set_title("Distribution brute")
axes[0].set_xlabel("Prix ($)")
axes[0].set_ylabel("Fréquence")

axes[1].hist(np.log1p(df[target_col]), bins=60, color="#1D9E75", alpha=0.8, edgecolor="white")
axes[1].set_title("Distribution log-transformée")
axes[1].set_xlabel("log(Prix + 1)")
axes[1].set_ylabel("Fréquence")

plt.tight_layout()
plt.savefig(r"outputs\02a_distribution_prix.png", dpi=150, bbox_inches="tight")
plt.show()
print("✅ outputs\\02a_distribution_prix.png")

# ── 2.2 Corrélations avec le prix ─────────────────────────────────────────────
num_cols     = df.select_dtypes(include=np.number).columns.tolist()
correlations = df[num_cols].corr()[target_col].drop(target_col).sort_values()
top_all      = pd.concat([correlations.head(5), correlations.tail(10)])

fig, ax = plt.subplots(figsize=(10, 8))
colors = ["#A32D2D" if v < 0 else "#534AB7" for v in top_all.values]
bars   = ax.barh(top_all.index, top_all.values, color=colors, alpha=0.85)
ax.axvline(0, color="black", linewidth=0.8)
ax.set_title("Variables les plus corrélées avec le prix", fontsize=13, fontweight="bold")
ax.set_xlabel("Corrélation de Pearson")
for bar, val in zip(bars, top_all.values):
    ax.text(val + (0.01 if val >= 0 else -0.01),
            bar.get_y() + bar.get_height() / 2,
            f"{val:.2f}", va="center",
            ha="left" if val >= 0 else "right", fontsize=9)
plt.tight_layout()
plt.savefig(r"outputs\02b_correlations.png", dpi=150, bbox_inches="tight")
plt.show()
print("✅ outputs\\02b_correlations.png")

# ── 2.3 Heatmap ───────────────────────────────────────────────────────────────
top_vars = correlations.abs().sort_values(ascending=False).head(10).index.tolist()
top_vars.append(target_col)

fig, ax = plt.subplots(figsize=(12, 10))
corr_matrix = df[top_vars].corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, ax=ax, annot=True, fmt=".2f",
            cmap="RdYlGn", center=0, mask=mask,
            linewidths=0.5, annot_kws={"size": 9})
ax.set_title("Heatmap des corrélations", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(r"outputs\02c_heatmap.png", dpi=150, bbox_inches="tight")
plt.show()
print("✅ outputs\\02c_heatmap.png")

# ── 2.4 Surface vs Prix ───────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(df["sqft_living"], df[target_col], alpha=0.4, s=15, color="#534AB7")
ax.set_xlabel("Surface habitable (sqft)")
ax.set_ylabel("Prix ($)")
ax.set_title("Surface vs Prix", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(r"outputs\02d_surface_vs_prix.png", dpi=150, bbox_inches="tight")
plt.show()
print("✅ outputs\\02d_surface_vs_prix.png")

# ── 2.5 Prix par ville (top 10) ───────────────────────────────────────────────
top_cities = df.groupby("city")[target_col].median().sort_values(ascending=False).head(10)

fig, ax = plt.subplots(figsize=(12, 6))
bars = ax.bar(top_cities.index, top_cities.values / 1e6, color="#534AB7", alpha=0.85)
ax.set_title("Prix médian par ville — Top 10 (MDH)", fontsize=13, fontweight="bold")
ax.set_ylabel("Prix médian (M$)")
ax.tick_params(axis="x", rotation=30)
for bar, val in zip(bars, top_cities.values / 1e6):
    ax.text(bar.get_x() + bar.get_width() / 2, val + 0.01,
            f"{val:.1f}", ha="center", fontsize=9)
plt.tight_layout()
plt.savefig(r"outputs\02e_prix_par_ville.png", dpi=150, bbox_inches="tight")
plt.show()
print("✅ outputs\\02e_prix_par_ville.png")

# ── 2.6 Évolution IPAI ────────────────────────────────────────────────────────
try:
    col_date   = df_ipai.columns[0]
    col_indice = df_ipai.columns[1]

    df_ipai[col_date] = pd.to_datetime(df_ipai[col_date], errors="coerce")
    df_ipai = df_ipai.dropna(subset=[col_date]).sort_values(col_date)
    df_ipai[col_indice] = pd.to_numeric(df_ipai[col_indice], errors="coerce")
    df_ipai = df_ipai.dropna(subset=[col_indice])
    df_ipai["ma4"] = df_ipai[col_indice].rolling(4).mean()

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(df_ipai[col_date], df_ipai[col_indice],
            color="#534AB7", linewidth=2, label="Indice IPAI")
    ax.fill_between(df_ipai[col_date], df_ipai[col_indice].min(),
                    df_ipai[col_indice], alpha=0.1, color="#534AB7")
    ax.plot(df_ipai[col_date], df_ipai["ma4"],
            color="#D85A30", linewidth=2, linestyle="--", label="Moyenne mobile 4T")
    ax.set_title("Indice des Prix des Actifs Immobiliers (IPAI) — Bank Al-Maghrib",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Trimestre")
    ax.set_ylabel("Indice IPAI")
    ax.legend()
    plt.tight_layout()
    plt.savefig(r"outputs\02f_ipai_evolution.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("✅ outputs\\02f_ipai_evolution.png")

except Exception as e:
    print(f"⚠️  Erreur IPAI : {e}")

print("\n✅ EDA terminée — tous les graphiques sont dans outputs\\")
