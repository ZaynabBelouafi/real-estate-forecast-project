import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings("ignore")

from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools         import adfuller
from statsmodels.tsa.seasonal          import seasonal_decompose
from statsmodels.graphics.tsaplots     import plot_acf, plot_pacf
from sklearn.metrics                   import mean_squared_error

os.chdir(r"C:\Users\hp\Desktop\Prévision économique")

plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["axes.facecolor"]   = "#f9f9f9"
plt.rcParams["axes.grid"]        = True
plt.rcParams["grid.alpha"]       = 0.3

# ── Chargement IPAI ───────────────────────────────────────────────────────────
df_ipai = pd.read_csv(r"data\ipai_clean.csv")

# La structure réelle : première colonne = trimestres, autres = indices par ville
col_trim = df_ipai.columns[0]
print("Aperçu :", df_ipai.head())
print("Colonnes :", df_ipai.columns.tolist())

# Conversion trimestres → dates
def trimestre_to_date(t):
    try:
        t = str(t).strip()
        q    = int(t[1])
        year = int(t[2:])
        month = {1: 1, 2: 4, 3: 7, 4: 10}[q]
        return pd.Timestamp(year=year, month=month, day=1)
    except:
        return pd.NaT

# Reconstruction complète de la série
# La première ligne d'en-tête contient T22006 + sa valeur → on la récupère
premiere_date   = trimestre_to_date(df_ipai.columns[0])
premiere_valeur = pd.to_numeric(df_ipai.columns[1], errors="coerce")

dates   = [trimestre_to_date(t) for t in df_ipai.iloc[:, 0]]
valeurs = pd.to_numeric(df_ipai.iloc[:, 1], errors="coerce").tolist()

dates.insert(0, premiere_date)
valeurs.insert(0, premiere_valeur)

ipai_df = pd.DataFrame({"date": dates, "ipai": valeurs})
ipai_df = ipai_df.dropna().sort_values("date").reset_index(drop=True)
ipai_ts = ipai_df.set_index("date")["ipai"]

print(f"\n✅ Série IPAI : {len(ipai_ts)} trimestres")
print(f"   De : {ipai_ts.index[0].date()} → {ipai_ts.index[-1].date()}")
print(f"   Min : {ipai_ts.min():.2f} | Max : {ipai_ts.max():.2f}")

# ── 4.1 Visualisation de la série brute ───────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(ipai_ts.index, ipai_ts.values, color="#534AB7", linewidth=2, label="IPAI")
ipai_ts_ma = ipai_ts.rolling(4).mean()
ax.plot(ipai_ts.index, ipai_ts_ma, color="#D85A30", linewidth=2,
        linestyle="--", label="Moyenne mobile 4T")
ax.set_title("Indice des Prix des Actifs Immobiliers — Bank Al-Maghrib",
             fontsize=13, fontweight="bold")
ax.set_ylabel("Indice IPAI")
ax.legend()
plt.tight_layout()
plt.savefig(r"outputs\04a_ipai_serie.png", dpi=150, bbox_inches="tight")
plt.show()
print("✅ outputs\\04a_ipai_serie.png")

# ── 4.2 Décomposition STL ─────────────────────────────────────────────────────
decomp = seasonal_decompose(ipai_ts, model="additive", period=4)

fig, axes = plt.subplots(4, 1, figsize=(14, 12))
fig.suptitle("Décomposition STL — Indice IPAI", fontsize=13, fontweight="bold")
for ax, data, titre, color in zip(axes,
    [ipai_ts, decomp.trend, decomp.seasonal, decomp.resid],
    ["Série originale", "Tendance", "Saisonnalité", "Résidus"],
    ["#534AB7", "#1D9E75", "#D85A30", "#888780"]):
    ax.plot(data, color=color, linewidth=1.5)
    ax.set_title(titre)
plt.tight_layout()
plt.savefig(r"outputs\04b_decomposition.png", dpi=150, bbox_inches="tight")
plt.show()
print("✅ outputs\\04b_decomposition.png")

# ── 4.3 Test ADF ──────────────────────────────────────────────────────────────
def test_adf(serie, nom):
    result = adfuller(serie.dropna())
    print(f"\n--- Test ADF : {nom} ---")
    print(f"   Statistique : {result[0]:.4f}")
    print(f"   P-value     : {result[1]:.4f}")
    print(f"   → {'✅ Stationnaire' if result[1] < 0.05 else '❌ Non stationnaire'}")
    return result[1] < 0.05

stat = test_adf(ipai_ts, "IPAI brut")
if not stat:
    test_adf(ipai_ts.diff().dropna(), "IPAI différencié (d=1)")

# ── 4.4 ACF / PACF ────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
plot_acf(ipai_ts.diff().dropna(),  lags=16, ax=axes[0], title="ACF")
plot_pacf(ipai_ts.diff().dropna(), lags=16, ax=axes[1], title="PACF")
plt.tight_layout()
plt.savefig(r"outputs\04c_acf_pacf.png", dpi=150, bbox_inches="tight")
plt.show()
print("✅ outputs\\04c_acf_pacf.png")

# ── 4.5 SARIMA ────────────────────────────────────────────────────────────────
n_test      = 8
train_ts    = ipai_ts.iloc[:-n_test]
test_ts     = ipai_ts.iloc[-n_test:]

print("\n--- Entraînement SARIMA(1,1,1)(1,1,0)[4] ---")
sarima      = SARIMAX(train_ts, order=(1,1,1), seasonal_order=(1,1,0,4),
                      enforce_stationarity=False, enforce_invertibility=False)
sarima_fit  = sarima.fit(disp=False)
pred_sarima = sarima_fit.forecast(steps=n_test)
rmse_sarima = np.sqrt(mean_squared_error(test_ts, pred_sarima))
print(f"✅ RMSE SARIMA : {rmse_sarima:.2f} points")

# ── 4.6 Prophet ───────────────────────────────────────────────────────────────
try:
    from prophet import Prophet

    df_p = ipai_ts.reset_index()
    df_p.columns = ["ds", "y"]
    train_p = df_p.iloc[:-n_test]

    model = Prophet(changepoint_prior_scale=0.05,
                    yearly_seasonality=False,
                    weekly_seasonality=False)
    model.add_seasonality(name="quarterly", period=365.25/4, fourier_order=5)
    model.fit(train_p)

    future   = model.make_future_dataframe(periods=n_test + 8, freq="QS")
    forecast = model.predict(future)

    pred_prophet = forecast.iloc[-n_test-8:-8]["yhat"].values
    rmse_prophet = np.sqrt(mean_squared_error(test_ts.values, pred_prophet))
    print(f"✅ RMSE Prophet : {rmse_prophet:.2f} points")

    prev_future = forecast[forecast["ds"] > ipai_ts.index[-1]][["ds","yhat","yhat_lower","yhat_upper"]]
    print("\n📈 Prévisions IPAI futures :")
    print(prev_future.to_string(index=False))
    prev_future.to_csv(r"outputs\04_previsions_ipai.csv", index=False)

    # Visualisation SARIMA vs Prophet
    fig, axes = plt.subplots(1, 2, figsize=(17, 6))
    fig.suptitle("Prévision IPAI — SARIMA vs Prophet", fontsize=13, fontweight="bold")
    for ax, nom, pred, rmse in zip(axes,
        ["SARIMA(1,1,1)(1,1,0)[4]", "Prophet"],
        [pred_sarima.values, pred_prophet],
        [rmse_sarima, rmse_prophet]):
        ax.plot(ipai_ts.index, ipai_ts.values, color="#534AB7", label="Historique")
        ax.plot(test_ts.index, pred, color="#D85A30", linestyle="--",
                linewidth=2, marker="o", markersize=5, label="Prévision")
        ax.plot(test_ts.index, test_ts.values, color="#1D9E75",
                linewidth=2, marker="s", markersize=5, label="Réel")
        ax.axvline(train_ts.index[-1], color="gray", linestyle=":", alpha=0.7)
        ax.set_title(f"{nom}\nRMSE = {rmse:.2f}")
        ax.legend(fontsize=9)
        ax.set_ylabel("Indice IPAI")
    plt.tight_layout()
    plt.savefig(r"outputs\04d_sarima_vs_prophet.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("✅ outputs\\04d_sarima_vs_prophet.png")

except ImportError:
    print("⚠️  Prophet non installé : pip install prophet")

print("\n✅ Séries temporelles terminées !")
print("🎯 Lance : python notebooks\\05_risk_management.py")