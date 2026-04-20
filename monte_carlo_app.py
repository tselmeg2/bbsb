# ==================================================
# Monte Carlo Simulation - Streamlit Web App
# Data is pre-loaded. User only inputs DTI range.
# ==================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import os

try:
    from xgboost import XGBRegressor
except ImportError:
    st.error("XGBoost not installed. Run: pip install xgboost")
    st.stop()

st.set_page_config(page_title="Monte Carlo DTI Simulation", page_icon="📊", layout="wide")

st.title("📊 Monte Carlo Simulation: Neutral DTIbbsb Rate Finder")
st.markdown("Set the **DTI policy shock range** and click **Run Simulation** to find the optimal neutral DTIbbsb rate.")

TARGET_GDP      = "realgdp"
TARGET_CPI      = "cpi"
TARGET_NPCL     = "Nonperloanconsumer"
ENDOGENOUS_VARS = [TARGET_GDP, TARGET_CPI, TARGET_NPCL]
DATA_PATH       = os.path.join(os.path.dirname(__file__), "DSR.xlsx")

@st.cache_data(show_spinner=False)
def load_data():
    df = pd.read_excel(DATA_PATH)
    for col in df.columns:
        if col not in ["Fiscal year", "Quarter"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["date"] = pd.PeriodIndex.from_fields(
        year=df["Fiscal year"].astype(int), quarter=df["Quarter"].astype(int), freq="Q"
    )
    df = df.set_index("date").sort_index()
    q_dummies = pd.get_dummies(df.index.quarter, prefix="Q", drop_first=True).astype(int)
    q_dummies.index = df.index
    return pd.concat([df, q_dummies], axis=1)

def add_lags(df, cols, lags=(1,2,3,4)):
    df = df.copy()
    for col in cols:
        for lag in lags:
            df[f"{col}_lag{lag}"] = df[col].shift(lag)
    return df

def train_xgb(X_train, y_train):
    model = XGBRegressor(n_estimators=400, learning_rate=0.1, max_depth=5,
                         subsample=0.6, colsample_bytree=0.6,
                         objective="reg:squarederror", missing=np.nan, random_state=42)
    model.fit(X_train, y_train)
    return model

def run_monte_carlo(df, shock_min, shock_max, n_sim, n_forecast, g_star, pi_star, npcl_star):
    np.random.seed(42)
    base_cols = [c for c in df.columns if c not in ["Fiscal year","Quarter"] and not c.startswith("Q_")]
    df = add_lags(df, base_cols)
    X  = df.drop(columns=["Fiscal year","Quarter"], errors="ignore")

    X_dict, y_dict = {}, {}
    for var in ENDOGENOUS_VARS:
        mask = df[var].notna()
        X_dict[var] = X.drop(columns=[var], errors="ignore").loc[mask]
        y_dict[var] = df[var].loc[mask]

    var_models   = {var: train_xgb(X_dict[var], y_dict[var]) for var in ENDOGENOUS_VARS}
    policy_draws = np.linspace(shock_min, shock_max, n_sim)
    last_obs     = y_dict[TARGET_GDP].dropna().index.max()
    forecast_periods = pd.period_range(start=last_obs+1, periods=n_forecast, freq="Q")
    lag_cols_all = [c for c in X.columns if "_lag" in c]

    mc_results, mc_input_data = [], []
    progress = st.progress(0, text="Running simulations…")

    for i, shock in enumerate(policy_draws):
        if i % max(1, n_sim//100) == 0:
            progress.progress(i/n_sim, text=f"Simulation {i}/{n_sim}…")

        history = {var: y_dict[var].loc[:last_obs].copy() for var in ENDOGENOUS_VARS}
        X_prev  = X.loc[last_obs].copy()
        gdp_f, cpi_f, npcl_f = [], [], []

        for p in forecast_periods:
            X_curr = X_prev.copy()
            for var in ENDOGENOUS_VARS:
                X_curr[var] = np.nan
            if p in df.index:
                for col in X_curr.index:
                    if col in df.columns and not pd.isna(df.loc[p, col]):
                        X_curr[col] = df.loc[p, col]
            X_curr["DTIbbsb"] = shock

            for lag_col in lag_cols_all:
                base, lag_num = lag_col.rsplit("_lag",1)
                lag_num = int(lag_num)
                if base in ENDOGENOUS_VARS:
                    X_curr[lag_col] = history[base].iloc[-lag_num] if len(history[base])>=lag_num else np.nan
                elif base == "DTIbbsb":
                    X_curr[lag_col] = X_prev["DTIbbsb"] if lag_num==1 else X_prev.get(f"DTIbbsb_lag{lag_num-1}", np.nan)
                else:
                    X_curr[lag_col] = X_prev.get(base, np.nan) if lag_num==1 else X_prev.get(f"{base}_lag{lag_num-1}", np.nan)

            for col in X_curr.index:
                if col.startswith("Q_"):
                    X_curr[col] = 1 if col==f"Q_{p.quarter}" else 0

            preds = {}
            for var in ENDOGENOUS_VARS:
                if p in y_dict[var].index and not pd.isna(y_dict[var].loc[p]):
                    pred = y_dict[var].loc[p]
                else:
                    Xi   = pd.DataFrame([X_curr]).reindex(columns=X_dict[var].columns, fill_value=0)
                    pred = var_models[var].predict(Xi)[0]
                X_curr[var] = pred
                history[var] = pd.concat([history[var], pd.Series([pred], index=[p])])
                preds[var]   = pred

            gdp_f.append(preds[TARGET_GDP]); cpi_f.append(preds[TARGET_CPI]); npcl_f.append(preds[TARGET_NPCL])
            row = X_curr.copy()
            row.update({"sim_index":i,"forecast_period":p.start_time,"DTIbbsb_policy_shock":shock,
                        "predicted_gdp":preds[TARGET_GDP],"predicted_cpi":preds[TARGET_CPI],"predicted_npcl":preds[TARGET_NPCL]})
            mc_input_data.append(row)
            X_prev = X_curr.copy()
            for var in ENDOGENOUS_VARS: X_prev[var] = preds[var]

        loss = 0.33*(cpi_f[-1]-pi_star)**2 + 0.33*(gdp_f[-1]-g_star)**2 + 0.33*(npcl_f[-1]-npcl_star)**2
        mc_results.append({"DTIbbsb":shock,"Et_gdp":gdp_f[-1],"Et_cpi":cpi_f[-1],"Et_npcl":npcl_f[-1],"loss":loss})

    progress.progress(1.0, text="✅ Done!")
    mc_df       = pd.DataFrame(mc_results)
    neutral_row = mc_df.loc[mc_df["loss"].idxmin()]
    return mc_df, pd.DataFrame(mc_input_data), neutral_row["DTIbbsb"], neutral_row

def to_excel_bytes(df):
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w: df.to_excel(w, index=False)
    return buf.getvalue()

# --- Load data ---
with st.spinner("Loading data…"):
    df = load_data()

# --- Info strip ---
c1, c2, c3 = st.columns(3)
c1.info(f"📅 Data: **{df.index.min()} → {df.index.max()}**")
c2.info(f"📊 Quarters: **{len(df)}**")
c3.info("🔢 Variables: GDP · CPI · NPCL · DTIbbsb")

with st.expander("🔍 Preview source data"):
    st.dataframe(df.reset_index().rename(columns={"date":"Period"}), use_container_width=True)

# --- Sidebar: inputs only ---
st.sidebar.header("⚙️ Simulation Settings")
st.sidebar.subheader("📌 DTIbbsb Policy Shock Range")
shock_min = st.sidebar.number_input("Minimum DTIbbsb rate", value=0.30, step=0.01, format="%.2f")
shock_max = st.sidebar.number_input("Maximum DTIbbsb rate", value=0.90, step=0.01, format="%.2f")

st.sidebar.markdown("---")
st.sidebar.subheader("🔧 Options")
n_sim      = st.sidebar.slider("Number of Simulations", 100, 2000, 1000, step=100)
n_forecast = st.sidebar.slider("Forecast Quarters",       1,    8,    4)

st.sidebar.markdown("---")
st.sidebar.subheader("🎯 Taylor Loss Targets")
g_star    = st.sidebar.number_input("GDP target (g*)",     value=0.058, format="%.3f", step=0.001)
pi_star   = st.sidebar.number_input("CPI target (π*)",     value=0.060, format="%.3f", step=0.001)
npcl_star = st.sidebar.number_input("NPCL target (npcl*)", value=0.036, format="%.3f", step=0.001)

st.sidebar.markdown("---")
run_btn = st.sidebar.button("▶️ Run Simulation", type="primary", use_container_width=True)

if not run_btn:
    st.markdown("---")
    st.info("👈 Set the DTIbbsb range in the sidebar and click **▶️ Run Simulation**.")
    st.stop()

if shock_min >= shock_max:
    st.warning("⚠️ Minimum must be less than Maximum.")
    st.stop()

st.markdown("---")
try:
    mc_df, input_df, neutral_rate, neutral_row = run_monte_carlo(
        df, shock_min, shock_max, n_sim, n_forecast, g_star, pi_star, npcl_star)

    st.subheader("🎯 Result")
    m1,m2,m3,m4 = st.columns(4)
    m1.metric("🏆 Neutral DTIbbsb Rate", f"{neutral_rate:.4f}")
    m2.metric("📈 Forecasted GDP",        f"{neutral_row['Et_gdp']:.4f}")
    m3.metric("💹 Forecasted CPI",        f"{neutral_row['Et_cpi']:.4f}")
    m4.metric("📉 Forecasted NPCL",       f"{neutral_row['Et_npcl']:.4f}")

    st.markdown("---")
    st.subheader("📉 DTIbbsb Rate vs Loss Function")
    fig, ax = plt.subplots(figsize=(12,5))
    ax.scatter(mc_df["DTIbbsb"], mc_df["loss"], alpha=0.3, s=10, color="#4C8BF5")
    ax.axvline(neutral_rate, color="red", linestyle="--", linewidth=2, label=f"Neutral Rate = {neutral_rate:.4f}")
    ax.set_xlabel("ББСБ-ын хэрэглээний зээлийн өр, орлогын харьцаа (DTIbbsb)", fontsize=11)
    ax.set_ylabel("Өргөтгөсөн функцын утга (Loss)", fontsize=11)
    ax.set_title(f"Monte Carlo | Range [{shock_min:.2f} – {shock_max:.2f}] | N={n_sim}", fontsize=13)
    ax.legend(); ax.grid(True, alpha=0.3)
    st.pyplot(fig)

    st.markdown("---")
    st.subheader("📊 Forecast Distributions")
    fig2, axes = plt.subplots(1,3,figsize=(15,4))
    for ax2, col, label, target in zip(axes, ["Et_gdp","Et_cpi","Et_npcl"], ["GDP","CPI","NPCL"], [g_star,pi_star,npcl_star]):
        ax2.hist(mc_df[col], bins=40, color="#4C8BF5", alpha=0.7, edgecolor="white")
        ax2.axvline(mc_df[col].mean(), color="red",   linestyle="--", label=f"Mean: {mc_df[col].mean():.4f}")
        ax2.axvline(target,            color="green", linestyle="--", label=f"Target: {target:.4f}")
        ax2.set_title(f"Forecasted {label}"); ax2.set_xlabel(label); ax2.legend(fontsize=8); ax2.grid(True,alpha=0.3)
    plt.tight_layout(); st.pyplot(fig2)

    st.markdown("---")
    st.subheader("📋 Top 10 Lowest Loss Scenarios")
    top10 = mc_df.nsmallest(10,"loss").reset_index(drop=True); top10.index+=1
    st.dataframe(top10.style.format({"DTIbbsb":"{:.4f}","Et_gdp":"{:.4f}","Et_cpi":"{:.4f}","Et_npcl":"{:.4f}","loss":"{:.6f}"}), use_container_width=True)

    st.markdown("---")
    st.subheader("📥 Download Results")
    d1,d2 = st.columns(2)
    with d1:
        st.download_button("⬇️ MC Results (Excel)", data=to_excel_bytes(mc_df),
            file_name="mc_results.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    with d2:
        st.download_button("⬇️ Full Simulation Data (Excel)", data=to_excel_bytes(input_df),
            file_name="mc_input_data.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

except Exception as e:
    st.error(f"❌ Simulation failed: {e}")
    st.exception(e)
