from flask import Flask, render_template_string, request, send_file
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io, os, base64
from xgboost import XGBRegressor

app = Flask(__name__)

TARGET_GDP      = "realgdp"
TARGET_CPI      = "cpi"
TARGET_NPCL     = "Nonperloanconsumer"
ENDOGENOUS_VARS = [TARGET_GDP, TARGET_CPI, TARGET_NPCL]
DATA_PATH       = os.path.join(os.path.dirname(__file__), "DSR.xlsx")

HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Monte Carlo DTI Simulation</title>
<style>
  *{box-sizing:border-box;margin:0;padding:0}
  body{font-family:'Segoe UI',sans-serif;background:#f0f4f8;color:#222}
  header{background:#1a3c5e;color:white;padding:24px 40px}
  header h1{font-size:1.6rem}
  header p{font-size:0.9rem;opacity:0.8;margin-top:4px}
  .container{max-width:920px;margin:32px auto;padding:0 20px}
  .card{background:white;border-radius:12px;padding:28px;margin-bottom:24px;box-shadow:0 2px 8px rgba(0,0,0,0.08)}
  h2{font-size:1.05rem;color:#1a3c5e;margin-bottom:18px;border-bottom:2px solid #e2e8f0;padding-bottom:8px}
  .grid2{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:16px}
  .grid3{display:grid;grid-template-columns:1fr 1fr 1fr;gap:16px;margin-bottom:16px}
  .grid4{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}
  label{display:block;font-size:0.82rem;color:#555;margin-bottom:5px;font-weight:600}
  input[type=number]{width:100%;padding:10px 12px;border:1.5px solid #cbd5e0;border-radius:8px;font-size:0.95rem}
  input[type=number]:focus{outline:none;border-color:#1a3c5e}
  button[type=submit]{width:100%;padding:14px;background:#1a3c5e;color:white;border:none;border-radius:8px;
                      font-size:1rem;font-weight:700;cursor:pointer;margin-top:8px}
  button[type=submit]:hover{background:#2a5080}
  .metric{background:#f7fafc;border-radius:10px;padding:16px;text-align:center;border:1px solid #e2e8f0}
  .metric .val{font-size:1.45rem;font-weight:700;color:#1a3c5e}
  .metric .lbl{font-size:0.75rem;color:#718096;margin-top:4px}
  .highlight{background:#ebf8ff;border:2px solid #1a3c5e}
  img.chart{width:100%;border-radius:8px;margin-top:8px}
  table{width:100%;border-collapse:collapse;font-size:0.86rem}
  th{background:#1a3c5e;color:white;padding:10px 12px;text-align:left}
  td{padding:9px 12px;border-bottom:1px solid #e2e8f0}
  tr:nth-child(even) td{background:#f7fafc}
  .dl-btn{display:inline-block;margin:6px 6px 0 0;padding:10px 22px;background:#2a5080;
          color:white;border-radius:8px;text-decoration:none;font-size:0.9rem;font-weight:600}
  .dl-btn:hover{background:#1a3c5e}
  .info-strip{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin-bottom:24px}
  .info-box{background:#ebf8ff;border:1px solid #bee3f8;border-radius:8px;
            padding:10px 14px;font-size:0.84rem;color:#2c5282;text-align:center}
  .error{background:#fff5f5;border:1px solid #fc8181;border-radius:8px;padding:16px;color:#c53030;margin-bottom:16px}
  #spinner{display:none;text-align:center;padding:16px;font-size:1rem;color:#1a3c5e;font-weight:600}
</style>
</head>
<body>
<header>
  <h1>📊 Monte Carlo Simulation: Neutral DTIbbsb Rate Finder</h1>
  <p>Set the DTI policy shock range to find the optimal neutral rate minimizing the Taylor loss function</p>
</header>
<div class="container">
  <div class="info-strip">
    <div class="info-box">📅 Data: <strong>{{ info.start }} → {{ info.end }}</strong></div>
    <div class="info-box">📊 Quarters: <strong>{{ info.rows }}</strong></div>
    <div class="info-box">🔢 Variables: GDP · CPI · NPCL · DTIbbsb</div>
  </div>

  <div class="card">
    <h2>⚙️ Simulation Settings</h2>
    <form method="POST" onsubmit="document.getElementById('spinner').style.display='block';this.querySelector('button').disabled=true">
      <div class="grid2">
        <div><label>Minimum DTIbbsb Rate</label>
             <input type="number" name="shock_min" step="0.01" value="{{ f.shock_min }}" required></div>
        <div><label>Maximum DTIbbsb Rate</label>
             <input type="number" name="shock_max" step="0.01" value="{{ f.shock_max }}" required></div>
      </div>
      <div class="grid2">
        <div><label>Number of Simulations</label>
             <input type="number" name="n_sim" min="100" max="2000" step="100" value="{{ f.n_sim }}" required></div>
        <div><label>Forecast Quarters</label>
             <input type="number" name="n_forecast" min="1" max="8" value="{{ f.n_forecast }}" required></div>
      </div>
      <div class="grid3">
        <div><label>GDP Target (g*)</label>
             <input type="number" name="g_star" step="0.001" value="{{ f.g_star }}" required></div>
        <div><label>CPI Target (π*)</label>
             <input type="number" name="pi_star" step="0.001" value="{{ f.pi_star }}" required></div>
        <div><label>NPCL Target (npcl*)</label>
             <input type="number" name="npcl_star" step="0.001" value="{{ f.npcl_star }}" required></div>
      </div>
      <button type="submit">▶️ Run Simulation</button>
      <div id="spinner">⏳ Running simulation, please wait (this may take 1–2 minutes)…</div>
    </form>
  </div>

  {% if error %}<div class="error">❌ {{ error }}</div>{% endif %}

  {% if result %}
  <div class="card">
    <h2>🎯 Result</h2>
    <div class="grid4">
      <div class="metric highlight">
        <div class="val">{{ "%.4f"|format(result.neutral_rate) }}</div>
        <div class="lbl">🏆 Neutral DTIbbsb Rate</div>
      </div>
      <div class="metric">
        <div class="val">{{ "%.4f"|format(result.Et_gdp) }}</div>
        <div class="lbl">📈 Forecasted GDP</div>
      </div>
      <div class="metric">
        <div class="val">{{ "%.4f"|format(result.Et_cpi) }}</div>
        <div class="lbl">💹 Forecasted CPI</div>
      </div>
      <div class="metric">
        <div class="val">{{ "%.4f"|format(result.Et_npcl) }}</div>
        <div class="lbl">📉 Forecasted NPCL</div>
      </div>
    </div>
  </div>

  <div class="card">
    <h2>📉 DTIbbsb Rate vs Loss Function</h2>
    <img class="chart" src="data:image/png;base64,{{ result.chart1 }}">
  </div>

  <div class="card">
    <h2>📊 Forecast Distributions</h2>
    <img class="chart" src="data:image/png;base64,{{ result.chart2 }}">
  </div>

  <div class="card">
    <h2>📋 Top 10 Lowest Loss Scenarios</h2>
    {{ result.table | safe }}
  </div>

  <div class="card">
    <h2>📥 Download Results</h2>
    <a class="dl-btn" href="/download/mc_results">⬇️ MC Results (Excel)</a>
    <a class="dl-btn" href="/download/mc_input">⬇️ Full Simulation Data (Excel)</a>
  </div>
  {% endif %}
</div>
</body>
</html>
"""

def load_data():
    df = pd.read_excel(DATA_PATH)
    for col in df.columns:
        if col not in ["Fiscal year", "Quarter"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["date"] = pd.PeriodIndex.from_fields(
        year=df["Fiscal year"].astype(int), quarter=df["Quarter"].astype(int), freq="Q")
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

def run_simulation(df, shock_min, shock_max, n_sim, n_forecast, g_star, pi_star, npcl_star):
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
    fperiods     = pd.period_range(start=last_obs+1, periods=n_forecast, freq="Q")
    lag_cols_all = [c for c in X.columns if "_lag" in c]
    mc_results, mc_input = [], []
    for i, shock in enumerate(policy_draws):
        history = {var: y_dict[var].loc[:last_obs].copy() for var in ENDOGENOUS_VARS}
        X_prev  = X.loc[last_obs].copy()
        gdp_f, cpi_f, npcl_f = [], [], []
        for p in fperiods:
            X_curr = X_prev.copy()
            for var in ENDOGENOUS_VARS: X_curr[var] = np.nan
            if p in df.index:
                for col in X_curr.index:
                    if col in df.columns and not pd.isna(df.loc[p, col]):
                        X_curr[col] = df.loc[p, col]
            X_curr["DTIbbsb"] = shock
            for lag_col in lag_cols_all:
                base, ln = lag_col.rsplit("_lag",1); ln=int(ln)
                if base in ENDOGENOUS_VARS:
                    X_curr[lag_col] = history[base].iloc[-ln] if len(history[base])>=ln else np.nan
                elif base=="DTIbbsb":
                    X_curr[lag_col] = X_prev["DTIbbsb"] if ln==1 else X_prev.get(f"DTIbbsb_lag{ln-1}", np.nan)
                else:
                    X_curr[lag_col] = X_prev.get(base,np.nan) if ln==1 else X_prev.get(f"{base}_lag{ln-1}", np.nan)
            for col in X_curr.index:
                if col.startswith("Q_"): X_curr[col]=1 if col==f"Q_{p.quarter}" else 0
            preds = {}
            for var in ENDOGENOUS_VARS:
                if p in y_dict[var].index and not pd.isna(y_dict[var].loc[p]):
                    pred = y_dict[var].loc[p]
                else:
                    Xi = pd.DataFrame([X_curr]).reindex(columns=X_dict[var].columns, fill_value=0)
                    pred = var_models[var].predict(Xi)[0]
                X_curr[var]=pred
                history[var]=pd.concat([history[var], pd.Series([pred],index=[p])])
                preds[var]=pred
            gdp_f.append(preds[TARGET_GDP]); cpi_f.append(preds[TARGET_CPI]); npcl_f.append(preds[TARGET_NPCL])
            row=X_curr.copy()
            row.update({"sim_index":i,"forecast_period":p.start_time,"DTIbbsb_policy_shock":shock,
                        "predicted_gdp":preds[TARGET_GDP],"predicted_cpi":preds[TARGET_CPI],"predicted_npcl":preds[TARGET_NPCL]})
            mc_input.append(row)
            X_prev=X_curr.copy()
            for var in ENDOGENOUS_VARS: X_prev[var]=preds[var]
        loss=0.33*(cpi_f[-1]-pi_star)**2+0.33*(gdp_f[-1]-g_star)**2+0.33*(npcl_f[-1]-npcl_star)**2
        mc_results.append({"DTIbbsb":shock,"Et_gdp":gdp_f[-1],"Et_cpi":cpi_f[-1],"Et_npcl":npcl_f[-1],"loss":loss})
    mc_df = pd.DataFrame(mc_results)
    nr    = mc_df.loc[mc_df["loss"].idxmin()]
    return mc_df, pd.DataFrame(mc_input), nr["DTIbbsb"], nr

def fig_to_b64(fig):
    buf=io.BytesIO(); fig.savefig(buf,format="png",bbox_inches="tight",dpi=110); buf.seek(0)
    return base64.b64encode(buf.read()).decode()

def to_excel_bytes(df):
    buf=io.BytesIO()
    with pd.ExcelWriter(buf,engine="openpyxl") as w: df.to_excel(w,index=False)
    buf.seek(0); return buf.read()

_store = {}

@app.route("/", methods=["GET","POST"])
def index():
    df   = load_data()
    info = {"start":str(df.index.min()),"end":str(df.index.max()),"rows":len(df)}
    defaults = dict(shock_min=0.30,shock_max=0.90,n_sim=1000,n_forecast=4,
                    g_star=0.058,pi_star=0.060,npcl_star=0.036)
    error, result = None, None

    if request.method=="POST":
        f = {k: float(request.form[k]) if "." in request.form[k] or k in ("g_star","pi_star","npcl_star","shock_min","shock_max")
               else int(request.form[k]) for k in defaults}
        if f["shock_min"] >= f["shock_max"]:
            error = "Minimum must be less than Maximum."
        else:
            try:
                mc_df, inp_df, neutral_rate, nr = run_simulation(
                    df, f["shock_min"], f["shock_max"], int(f["n_sim"]), int(f["n_forecast"]),
                    f["g_star"], f["pi_star"], f["npcl_star"])
                _store["mc_df"]  = mc_df
                _store["inp_df"] = inp_df

                # Chart 1
                fig1,ax=plt.subplots(figsize=(11,4.5))
                ax.scatter(mc_df["DTIbbsb"],mc_df["loss"],alpha=0.3,s=8,color="#4C8BF5")
                ax.axvline(neutral_rate,color="red",linestyle="--",linewidth=2,label=f"Neutral Rate = {neutral_rate:.4f}")
                ax.set_xlabel("DTIbbsb Rate"); ax.set_ylabel("Loss"); ax.legend(); ax.grid(True,alpha=0.3)
                ax.set_title(f"Monte Carlo | Range [{f['shock_min']:.2f}–{f['shock_max']:.2f}] | N={int(f['n_sim'])}")
                c1=fig_to_b64(fig1); plt.close(fig1)

                # Chart 2
                fig2,axes=plt.subplots(1,3,figsize=(14,4))
                for ax2,col,lbl,tgt in zip(axes,["Et_gdp","Et_cpi","Et_npcl"],["GDP","CPI","NPCL"],[f["g_star"],f["pi_star"],f["npcl_star"]]):
                    ax2.hist(mc_df[col],bins=40,color="#4C8BF5",alpha=0.7,edgecolor="white")
                    ax2.axvline(mc_df[col].mean(),color="red",linestyle="--",label=f"Mean: {mc_df[col].mean():.4f}")
                    ax2.axvline(tgt,color="green",linestyle="--",label=f"Target: {tgt:.4f}")
                    ax2.set_title(f"Forecasted {lbl}"); ax2.legend(fontsize=8); ax2.grid(True,alpha=0.3)
                plt.tight_layout(); c2=fig_to_b64(fig2); plt.close(fig2)

                # Top 10 table
                top10=mc_df.nsmallest(10,"loss").reset_index(drop=True); top10.index+=1
                tbl=top10.to_html(classes="",border=0,float_format=lambda x:f"{x:.4f}",index=True)

                result=dict(neutral_rate=neutral_rate,Et_gdp=nr["Et_gdp"],Et_cpi=nr["Et_cpi"],
                            Et_npcl=nr["Et_npcl"],chart1=c1,chart2=c2,table=tbl)
                defaults=f
            except Exception as e:
                error=str(e)
        if error:
            pass

    return render_template_string(HTML, info=info, f=defaults, error=error, result=result)

@app.route("/download/<name>")
def download(name):
    if name=="mc_results" and "mc_df" in _store:
        return send_file(io.BytesIO(to_excel_bytes(_store["mc_df"])),
                         download_name="mc_results.xlsx",as_attachment=True,
                         mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    if name=="mc_input" and "inp_df" in _store:
        return send_file(io.BytesIO(to_excel_bytes(_store["inp_df"])),
                         download_name="mc_input_data.xlsx",as_attachment=True,
                         mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    return "No data yet. Run simulation first.", 404

if __name__=="__main__":
    port=int(os.environ.get("PORT",5000))
    app.run(host="0.0.0.0",port=port)
