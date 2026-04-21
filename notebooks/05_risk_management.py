import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import warnings
warnings.filterwarnings("ignore")

os.chdir(r"C:\Users\hp\Desktop\Prévision économique")

plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["axes.facecolor"]   = "#f9f9f9"
plt.rcParams["axes.grid"]        = True
plt.rcParams["grid.alpha"]       = 0.3

# ── Chargement IPAI ───────────────────────────────────────────────────────────
df_ipai = pd.read_csv(r"data\ipai_clean.csv")

def trimestre_to_date(t):
    try:
        t = str(t).strip()
        q     = int(t[1])
        year  = int(t[2:])
        month = {1: 1, 2: 4, 3: 7, 4: 10}[q]
        return pd.Timestamp(year=year, month=month, day=1)
    except:
        return pd.NaT

premiere_date   = trimestre_to_date(df_ipai.columns[0])
premiere_valeur = pd.to_numeric(df_ipai.columns[1], errors="coerce")
dates   = [trimestre_to_date(t) for t in df_ipai.iloc[:, 0]]
valeurs = pd.to_numeric(df_ipai.iloc[:, 1], errors="coerce").tolist()
dates.insert(0, premiere_date)
valeurs.insert(0, premiere_valeur)

ipai_df = pd.DataFrame({"date": dates, "ipai": valeurs})
ipai_df = ipai_df.dropna().sort_values("date").reset_index(drop=True)
ipai_ts = ipai_df.set_index("date")["ipai"]

print(f" Série IPAI chargée : {len(ipai_ts)} trimestres")

# ── Paramètres de base ────────────────────────────────────────────────────────
rendements  = ipai_ts.pct_change().dropna()
mu          = rendements.mean()
sigma       = rendements.std()
VALEUR_BIEN = 2_000_000

print(f"\n Statistiques des rendements trimestriels IPAI")
print(f"   Rendement moyen  μ : {mu*100:.3f}%")
print(f"   Volatilité       σ : {sigma*100:.3f}%")
print(f"   Skewness           : {rendements.skew():.3f}")
print(f"   Kurtosis           : {rendements.kurtosis():.3f}")

# ── 5.1 Simulation Monte Carlo ────────────────────────────────────────────────
N_SIM   = 10_000
N_STEPS = 8
np.random.seed(42)

chocs   = np.random.normal(mu, sigma, (N_STEPS, N_SIM))
chemins = np.ones((N_STEPS + 1, N_SIM))
for t in range(1, N_STEPS + 1):
    chemins[t] = chemins[t-1] * (1 + chocs[t-1])

valeurs_finales = chemins[-1] * VALEUR_BIEN
pertes          = VALEUR_BIEN - valeurs_finales

print(f"\n Monte Carlo : {N_SIM:,} scénarios × {N_STEPS} trimestres (2 ans)")

# ── 5.2 VaR & Expected Shortfall ─────────────────────────────────────────────
niveaux  = [0.90, 0.95, 0.99]
var_dict = {}

print("\n Value at Risk (VaR) et Expected Shortfall (ES)")
print(f"   Valeur du bien : {VALEUR_BIEN:,.0f} DH")
print("-" * 58)
print(f"{'Confiance':>12} | {'VaR (DH)':>15} | {'ES (DH)':>15} | {'VaR (%)':>8}")
print("-" * 58)
for conf in niveaux:
    var = np.percentile(pertes, conf * 100)
    es  = pertes[pertes >= var].mean()
    var_dict[conf] = var
    print(f"  {conf*100:.0f}%       | {var:>15,.0f} | {es:>15,.0f} | {var/VALEUR_BIEN*100:>7.1f}%")
print("-" * 58)
print(f"\n→ Interprétation VaR 95% : avec 95% de confiance,")
print(f"  la perte max sur 2 ans ne dépassera pas {var_dict[0.95]:,.0f} DH")
print(f"  soit {var_dict[0.95]/VALEUR_BIEN*100:.1f}% de la valeur du bien.")

# ── 5.3 Stress Testing ────────────────────────────────────────────────────────
scenarios = {
    "Crise économique (-25%)"  : -0.25,
    "Hausse taux BAM (+200bp)" : -0.12,
    "Chute demande (-15%)"     : -0.15,
}

print("\n--- Stress Testing ---")
print(f"{'Scénario':<30} | {'Choc':>6} | {'Perte (DH)':>14} | {'Perte (%)':>10}")
print("-" * 68)
for nom, choc in scenarios.items():
    perte = abs(choc) * VALEUR_BIEN
    print(f"  {nom:<28} | {choc*100:>5.0f}% | {perte:>14,.0f} | {abs(choc)*100:>9.1f}%")
print("-" * 68)

# ── 5.4 Score de risque par ville ─────────────────────────────────────────────
villes = ["Casablanca", "Rabat", "Marrakech", "Tanger", "Agadir"]
vol    = [0.068, 0.055, 0.082, 0.075, 0.063]
liq    = [0.9,   0.8,   0.7,   0.6,   0.5  ]
prix   = [14000, 12000, 11000, 9500,  8500  ]

scores = []
for v, vo, li, px in zip(villes, vol, liq, prix):
    px_norm = (px - 8000) / (14000 - 8000)
    score   = round((0.5 * vo/0.082 + 0.3 * (1-li) + 0.2 * px_norm) * 10, 2)
    scores.append(score)

scores_df = pd.DataFrame({"Ville": villes, "Score": scores})
scores_df = scores_df.sort_values("Score", ascending=False).reset_index(drop=True)
print("\n--- Score de risque par ville (/10) ---")
print(scores_df.to_string(index=False))

# ── 5.5 Visualisation complète ────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 12))
fig.suptitle("Analyse de Risque Immobilier — Maroc", fontsize=15, fontweight="bold")
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

# Distribution Monte Carlo
ax1 = fig.add_subplot(gs[0, :2])
ax1.hist(valeurs_finales / 1e6, bins=100, color="#534AB7", alpha=0.75, edgecolor="white")
ax1.axvline(VALEUR_BIEN / 1e6, color="black", linewidth=2,
            label=f"Valeur initiale : {VALEUR_BIEN/1e6:.1f} MDH")
for conf, color, ls in zip([0.95, 0.99], ["#D85A30", "#A32D2D"], ["--", "-."]):
    seuil = (VALEUR_BIEN - var_dict[conf]) / 1e6
    ax1.axvline(seuil, color=color, linewidth=2, linestyle=ls,
                label=f"VaR {conf*100:.0f}% = {var_dict[conf]/1e6:.2f} MDH")
ax1.set_xlabel("Valeur du bien dans 2 ans (MDH)")
ax1.set_ylabel("Nombre de scénarios")
ax1.set_title(f"Monte Carlo — {N_SIM:,} scénarios")
ax1.legend(fontsize=9)

# Score de risque
ax2 = fig.add_subplot(gs[0, 2])
couleurs = ["#A32D2D", "#D85A30", "#BA7517", "#1D9E75", "#0F6E56"]
bars = ax2.barh(scores_df["Ville"], scores_df["Score"], color=couleurs, alpha=0.85)
ax2.set_title("Score de risque par ville (/10)")
ax2.set_xlabel("Score (plus élevé = plus risqué)")
for bar, val in zip(bars, scores_df["Score"]):
    ax2.text(val + 0.05, bar.get_y() + bar.get_height()/2,
             f"{val}", va="center", fontsize=10, fontweight="bold")

# Chemins Monte Carlo
ax3 = fig.add_subplot(gs[1, :2])
for i in range(300):
    ax3.plot(range(N_STEPS+1), chemins[:, i] * VALEUR_BIEN / 1e6,
             alpha=0.04, color="#534AB7", linewidth=0.8)
ax3.plot(range(N_STEPS+1), np.percentile(chemins, 5,  axis=1) * VALEUR_BIEN / 1e6,
         color="#A32D2D", linewidth=2.5, label="Percentile 5% (pire cas)")
ax3.plot(range(N_STEPS+1), np.percentile(chemins, 50, axis=1) * VALEUR_BIEN / 1e6,
         color="black",   linewidth=2.5, linestyle="--", label="Médiane")
ax3.plot(range(N_STEPS+1), np.percentile(chemins, 95, axis=1) * VALEUR_BIEN / 1e6,
         color="#1D9E75", linewidth=2.5, label="Percentile 95% (meilleur cas)")
ax3.set_xlabel("Trimestres")
ax3.set_ylabel("Valeur du bien (MDH)")
ax3.set_title("Évolution simulée de la valeur sur 2 ans")
ax3.legend(fontsize=9)

# Stress testing
ax4 = fig.add_subplot(gs[1, 2])
noms_sc   = [s.split("(")[0].strip() for s in scenarios.keys()]
pertes_sc = [abs(v) * VALEUR_BIEN / 1e6 for v in scenarios.values()]
bars_sc   = ax4.bar(range(len(noms_sc)), pertes_sc,
                    color=["#A32D2D", "#D85A30", "#BA7517"], alpha=0.85)
ax4.set_xticks(range(len(noms_sc)))
ax4.set_xticklabels(noms_sc, rotation=15, ha="right", fontsize=9)
ax4.set_ylabel("Perte simulée (MDH)")
ax4.set_title("Stress Testing — 3 scénarios")
for bar, val in zip(bars_sc, pertes_sc):
    ax4.text(bar.get_x() + bar.get_width()/2, val + 0.01,
             f"{val:.2f} MDH", ha="center", fontsize=10, fontweight="bold")

plt.savefig(r"outputs\05_risk_management.png", dpi=150, bbox_inches="tight")
plt.show()
print(" outputs\\05_risk_management.png")

print("""
╔══════════════════════════════════════════════════════════╗
║              PROJET COMPLET                           ║
╠══════════════════════════════════════════════════════════╣
║  outputs\\02*.png  → EDA                                ║
║  outputs\\03*.png  → Modèles ML + SHAP                  ║
║  outputs\\04*.png  → Séries temporelles SARIMA+Prophet  ║
║  outputs\\05*.png  → Risk Management                    ║
╚══════════════════════════════════════════════════════════╝
""")
print(" Dernière étape : git add . && git commit -m 'Projet complet' && git push")