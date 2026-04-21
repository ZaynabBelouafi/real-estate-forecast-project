import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.linear_model    import Ridge
from sklearn.ensemble        import RandomForestRegressor
from sklearn.preprocessing   import StandardScaler
from sklearn.metrics         import mean_absolute_error, mean_squared_error, r2_score
from xgboost                 import XGBRegressor

os.chdir(r"C:\Users\hp\Desktop\Prévision économique")

plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["axes.facecolor"]   = "#f9f9f9"
plt.rcParams["axes.grid"]        = True
plt.rcParams["grid.alpha"]       = 0.3

df_encoded = pd.read_csv(r"data\kaggle_encoded.csv")
target_col = "SalePrice"
print(f" Dataset chargé : {df_encoded.shape}")

y = np.log1p(df_encoded[target_col])
X = df_encoded.drop(columns=[target_col]).select_dtypes(include=np.number).fillna(0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler     = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)
print(f"Train : {X_train.shape[0]} | Test : {X_test.shape[0]} | Features : {X_train.shape[1]}")

def evaluer(nom, modele, X_tr, X_te, y_tr, y_te):
    modele.fit(X_tr, y_tr)
    pred   = np.expm1(modele.predict(X_te))
    y_reel = np.expm1(y_te)
    mae    = mean_absolute_error(y_reel, pred)
    rmse   = np.sqrt(mean_squared_error(y_reel, pred))
    r2     = r2_score(y_reel, pred)
    print(f"\n {nom} | MAE : {mae:,.0f}$ | RMSE : {rmse:,.0f}$ | R² : {r2:.4f}")
    return {"Modèle": nom, "MAE": mae, "RMSE": rmse, "R²": r2}, pred, np.expm1(y_te)

resultats = []
res, pred_ridge, y_reel = evaluer("Ridge", Ridge(alpha=10), X_train_sc, X_test_sc, y_train, y_test)
resultats.append(res)
res, pred_rf, _ = evaluer("Random Forest",
    RandomForestRegressor(n_estimators=200, max_depth=15, min_samples_leaf=5, random_state=42, n_jobs=-1),
    X_train, X_test, y_train, y_test)
resultats.append(res)
res, pred_xgb, _ = evaluer("XGBoost",
    XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.05,
                 subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0),
    X_train, X_test, y_train, y_test)
resultats.append(res)

df_resultats = pd.DataFrame(resultats)
print("\n--- Tableau comparatif ---")
print(df_resultats.to_string(index=False))
df_resultats.to_csv(r"outputs\03_resultats_ml.csv", index=False)

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle("Comparaison modèles ML — Réel vs Prédit", fontsize=13, fontweight="bold")
for ax, nom, pred, color in zip(axes,
    ["Ridge", "Random Forest", "XGBoost"],
    [pred_ridge, pred_rf, pred_xgb],
    ["#534AB7", "#1D9E75", "#D85A30"]):
    ax.scatter(y_reel, pred, alpha=0.35, s=15, color=color)
    lim = max(y_reel.max(), pred.max())
    ax.plot([0, lim], [0, lim], "r--", linewidth=1.5, label="Parfait")
    ax.set_title(f"{nom}\nR² = {r2_score(y_reel, pred):.3f}")
    ax.set_xlabel("Réel ($)")
    ax.set_ylabel("Prédit ($)")
    ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(r"outputs\03a_reel_vs_predit.png", dpi=150, bbox_inches="tight")
plt.show()
print(" outputs\\03a_reel_vs_predit.png")

rf_final = RandomForestRegressor(n_estimators=200, max_depth=15, min_samples_leaf=5, random_state=42, n_jobs=-1)
rf_final.fit(X_train, y_train)
importance = pd.Series(rf_final.feature_importances_, index=X_train.columns)
top15 = importance.sort_values(ascending=True).tail(15)
fig, ax = plt.subplots(figsize=(10, 7))
ax.barh(top15.index, top15.values, color="#534AB7", alpha=0.85)
ax.set_title("Top 15 variables — Random Forest", fontsize=13, fontweight="bold")
ax.set_xlabel("Importance")
plt.tight_layout()
plt.savefig(r"outputs\03b_importance_variables.png", dpi=150, bbox_inches="tight")
plt.show()
print(" outputs\\03b_importance_variables.png")

try:
    import shap
    xgb_final = XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.05,
                               subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0)
    xgb_final.fit(X_train, y_train)
    explainer   = shap.TreeExplainer(xgb_final)
    shap_values = explainer.shap_values(X_test.iloc[:300])
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle("SHAP — Explainability XGBoost", fontsize=13, fontweight="bold")
    plt.sca(axes[0])
    shap.summary_plot(shap_values, X_test.iloc[:300], plot_type="bar", max_display=12, show=False)
    axes[0].set_title("Importance globale (SHAP)")
    plt.sca(axes[1])
    shap.summary_plot(shap_values, X_test.iloc[:300], max_display=12, show=False)
    axes[1].set_title("Impact sur le prix (SHAP)")
    plt.tight_layout()
    plt.savefig(r"outputs\03c_shap.png", dpi=150, bbox_inches="tight")
    plt.show()
    print(" outputs\\03c_shap.png")
except ImportError:
    print("  pip install shap")

print("\n ML terminé !")
print(" Lance : python notebooks\\04_series_temporelles.py")