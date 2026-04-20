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
DATA_PATH       = os.path.join(os.path.abspath(os.path.dirname(__file__) or "."), "DSR.xlsx")

# ==================================================
# BASE LAYOUT
# ==================================================
BASE = """
<!DOCTYPE html>
<html lang="mn">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>ББСБ Судалгаа</title>
<style>
  *{box-sizing:border-box;margin:0;padding:0}
  body{font-family:'Segoe UI',sans-serif;background:#f0f4f8;color:#222;display:flex;min-height:100vh}

  /* ---- Sidebar ---- */
  .sidebar{
    width:260px;min-width:260px;background:#1a3c5e;color:white;
    display:flex;flex-direction:column;min-height:100vh;position:fixed;top:0;left:0;z-index:100
  }
  .sidebar-logo{
    padding:28px 24px 20px;border-bottom:1px solid rgba(255,255,255,0.1)
  }
  .sidebar-logo h2{font-size:1.05rem;font-weight:700;line-height:1.4}
  .sidebar-logo p{font-size:0.75rem;opacity:0.6;margin-top:4px}
  .sidebar-nav{padding:16px 0;flex:1}
  .nav-section{padding:8px 24px 4px;font-size:0.7rem;text-transform:uppercase;
               letter-spacing:1px;opacity:0.45;font-weight:600}
  .nav-item{
    display:flex;align-items:center;gap:12px;padding:11px 24px;
    color:rgba(255,255,255,0.75);text-decoration:none;font-size:0.9rem;
    transition:all 0.15s;cursor:pointer;border:none;background:none;width:100%;text-align:left
  }
  .nav-item:hover{background:rgba(255,255,255,0.08);color:white}
  .nav-item.active{background:rgba(255,255,255,0.15);color:white;font-weight:600;
                   border-left:3px solid #63b3ed}
  .nav-item .icon{font-size:1.1rem;width:20px;text-align:center}
  .nav-item .badge{margin-left:auto;background:#63b3ed;color:#1a3c5e;
                   font-size:0.65rem;padding:2px 7px;border-radius:10px;font-weight:700}
  .sidebar-footer{padding:16px 24px;border-top:1px solid rgba(255,255,255,0.1);
                  font-size:0.75rem;opacity:0.45}

  /* ---- Main ---- */
  .main{margin-left:260px;flex:1;display:flex;flex-direction:column;min-height:100vh}
  .topbar{background:white;padding:16px 32px;border-bottom:1px solid #e2e8f0;
          display:flex;align-items:center;justify-content:space-between}
  .topbar h1{font-size:1.1rem;color:#1a3c5e;font-weight:700}
  .topbar .sub{font-size:0.82rem;color:#718096;margin-top:2px}
  .content{padding:28px 32px;flex:1}

  /* ---- Cards ---- */
  .card{background:white;border-radius:12px;padding:24px;margin-bottom:20px;
        box-shadow:0 1px 6px rgba(0,0,0,0.07)}
  h2{font-size:1rem;color:#1a3c5e;margin-bottom:16px;border-bottom:2px solid #e2e8f0;padding-bottom:8px}
  .grid2{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:16px}
  .grid3{display:grid;grid-template-columns:1fr 1fr 1fr;gap:16px;margin-bottom:16px}
  .grid4{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}
  label{display:block;font-size:0.8rem;color:#555;margin-bottom:5px;font-weight:600}
  input[type=number]{width:100%;padding:9px 12px;border:1.5px solid #cbd5e0;
                     border-radius:8px;font-size:0.92rem}
  input[type=number]:focus{outline:none;border-color:#1a3c5e}
  .btn-run{width:100%;padding:13px;background:#1a3c5e;color:white;border:none;
           border-radius:8px;font-size:1rem;font-weight:700;cursor:pointer;margin-top:8px}
  .btn-run:hover{background:#2a5080}
  .metric{background:#f7fafc;border-radius:10px;padding:14px;text-align:center;border:1px solid #e2e8f0}
  .metric .val{font-size:1.35rem;font-weight:700;color:#1a3c5e}
  .metric .lbl{font-size:0.72rem;color:#718096;margin-top:4px}
  .highlight{background:#ebf8ff;border:2px solid #1a3c5e}
  img.chart{width:100%;border-radius:8px;margin-top:6px}
  table{width:100%;border-collapse:collapse;font-size:0.84rem}
  th{background:#1a3c5e;color:white;padding:9px 12px;text-align:left}
  td{padding:8px 12px;border-bottom:1px solid #e2e8f0}
  tr:nth-child(even) td{background:#f7fafc}
  .dl-btn{display:inline-block;margin:5px 5px 0 0;padding:9px 20px;background:#2a5080;
          color:white;border-radius:8px;text-decoration:none;font-size:0.88rem;font-weight:600}
  .dl-btn:hover{background:#1a3c5e}
  .info-strip{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin-bottom:20px}
  .info-box{background:#ebf8ff;border:1px solid #bee3f8;border-radius:8px;
            padding:9px 14px;font-size:0.82rem;color:#2c5282;text-align:center}
  .error{background:#fff5f5;border:1px solid #fc8181;border-radius:8px;
         padding:14px;color:#c53030;margin-bottom:14px}
  #spinner{display:none;text-align:center;padding:14px;font-size:0.95rem;
           color:#1a3c5e;font-weight:600}

  /* Coming soon page */
  .coming-soon{text-align:center;padding:80px 20px}
  .coming-soon .icon{font-size:4rem;margin-bottom:16px}
  .coming-soon h2{font-size:1.4rem;color:#1a3c5e;border:none;margin-bottom:8px}
  .coming-soon p{color:#718096;font-size:0.95rem}
</style>
</head>
<body>

<!-- SIDEBAR -->
<aside class="sidebar">
  <div class="sidebar-logo">
    <h2>ББСБ Судалгааны<br>Систем</h2>
    <p>Монголын санхүүгийн шинжилгээ</p>
  </div>
  <nav class="sidebar-nav">
    <div class="nav-section">Үндсэн цэс</div>
    <a href="/" class="nav-item {% if page=='simulation' %}active{% endif %}">
      <span class="icon">📊</span> Монте Карло Симуляц
    </a>
    <a href="/presentation" class="nav-item {% if page=='presentation' %}active{% endif %}">
      <span class="icon">📑</span> Танилцуулга
      <span class="badge">Удахгүй</span>
    </a>
  </nav>
  <div class="sidebar-footer">© 2026 ББСБ Судалгаа</div>
</aside>

<!-- MAIN -->
<div class="main">
  <div class="topbar">
    <div>
      <div class="topbar h1">{{ title }}</div>
      <div class="sub">{{ subtitle }}</div>
    </div>
  </div>
  <div class="content">
    {% block content %}{% endblock %}
  </div>
</div>

</body>
</html>
"""

# ==================================================
# SIMULATION PAGE
# ==================================================
SIM_PAGE = BASE.replace("{% block content %}{% endblock %}", """
<div class="info-strip">
  <div class="info-box">📅 Өгөгдлийн хугацаа: <strong>{{ info.start }} → {{ info.end }}</strong></div>
  <div class="info-box">📊 Нийт улирал: <strong>{{ info.rows }}</strong></div>
  <div class="info-box">🔢 Хувьсагчид: ДНБ · Инфляци · Зээлийн чанар</div>
</div>

<div class="card">
  <h2>⚙️ Симуляцын тохиргоо</h2>
  <form method="POST" onsubmit="document.getElementById('spinner').style.display='block';this.querySelector('button').disabled=true">
    <div class="grid2">
      <div><label>ББСБ-ын хэрэглээний зээлийн өр, орлогын харьцааны доод хязгаар</label>
           <input type="number" name="shock_min" step="0.01" value="{{ f.shock_min }}" required></div>
      <div><label>ББСБ-ын хэрэглээний зээлийн өр, орлогын харьцааны дээд хязгаар</label>
           <input type="number" name="shock_max" step="0.01" value="{{ f.shock_max }}" required></div>
    </div>
    <div class="grid2">
      <div><label>Симуляцын тоо (100–200)</label>
           <input type="number" name="n_sim" min="100" max="200" step="100" value="{{ f.n_sim }}" required></div>
      <div><label>Улирлын тоо (1–4)</label>
           <input type="number" name="n_forecast" min="1" max="4" value="{{ f.n_forecast }}" required></div>
    </div>
    <div class="grid3">
      <div><label>Бодит ДНБ тренд өсөлтийн хувь</label>
           <input type="number" name="g_star" step="0.001" value="{{ f.g_star }}" required></div>
      <div><label>Инфляцийн таргет</label>
           <input type="number" name="pi_star" step="0.001" value="{{ f.pi_star }}" required></div>
      <div><label>Чанаргүй зээлийн зорилтот хувь</label>
           <input type="number" name="npcl_star" step="0.001" value="{{ f.npcl_star }}" required></div>
    </div>
    <button type="submit" class="btn-run">▶️ Симуляц ажиллуулах</button>
    <div id="spinner">⏳ Симуляц тооцоолж байна, түр хүлээнэ үү…</div>
  </form>
</div>

{% if error %}<div class="error">❌ Алдаа: {{ error }}</div>{% endif %}

{% if result %}
<div class="card">
  <h2>🎯 Үр дүн</h2>
  <div class="grid4">
    <div class="metric highlight">
      <div class="val">{{ "%.4f"|format(result.neutral_rate) }}</div>
      <div class="lbl">🏆 ББСБ-ын хэрэглээний зээлийн неутрал өр, орлогын харьцаа</div>
    </div>
    <div class="metric">
      <div class="val">{{ "%.4f"|format(result.Et_gdp) }}</div>
      <div class="lbl">📈 Хүлээгдэж буй Бодит ДНБ өсөлтийн хувь</div>
    </div>
    <div class="metric">
      <div class="val">{{ "%.4f"|format(result.Et_cpi) }}</div>
      <div class="lbl">💹 Инфляци</div>
    </div>
    <div class="metric">
      <div class="val">{{ "%.4f"|format(result.Et_npcl) }}</div>
      <div class="lbl">📉 Чанаргүй хэрэглээний зээлийн хувь</div>
    </div>
  </div>
</div>

<div class="card">
  <h2>📉 ББСБ-ын хэрэглээний зээлийн өр, орлогын харьцаа ба Алдагдлын функц</h2>
  <img class="chart" src="data:image/png;base64,{{ result.chart1 }}">
</div>

<div class="card">
  <h2>📊 Таамаглалын тархалт</h2>
  <img class="chart" src="data:image/png;base64,{{ result.chart2 }}">
</div>

<div class="card">
  <h2>📋 Алдагдал хамгийн бага 10 хувилбар</h2>
  {{ result.table | safe }}
</div>

<div class="card">
  <h2>📥 Үр дүн татаж авах</h2>
  <a class="dl-btn" href="/download/mc_results">⬇️ МК үр дүн (Excel)</a>
  <a class="dl-btn" href="/download/mc_input">⬇️ Бүрэн симуляцын өгөгдөл (Excel)</a>
</div>
{% endif %}
""")

# ==================================================
# PRESENTATION PAGE
# ==================================================
PRES_PAGE = BASE.replace("{% block content %}{% endblock %}", """
<div class="card">
  <div class="coming-soon">
    <div class="icon">📑</div>
    <h2>Танилцуулга</h2>
    <p>Энэ хэсэгт удахгүй танилцуулгын материал байрлах болно.<br>
    PowerPoint болон HTML хэлбэрийн танилцуулга оруулах боломжтой.</p>
  </div>
</div>
""")

# ==================================================
# HELPERS
# ==================================================
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
    model = XGBRegressor(n_estimators=200, learning_rate=0.1, max_depth=5,
                         subsample=0.8, colsample_bytree=0.8,
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

# ==================================================
# ROUTES
# ==================================================
@app.route("/", methods=["GET","POST"])
def simulation():
    df   = load_data()
    info = {"start":str(df.index.min()),"end":str(df.index.max()),"rows":len(df)}
    defaults = dict(shock_min=0.30,shock_max=0.90,n_sim=200,n_forecast=4,
                    g_star=0.058,pi_star=0.060,npcl_star=0.036)
    error, result = None, None

    if request.method=="POST":
        f = {}
        for k in defaults:
            val = request.form[k]
            f[k] = float(val) if k in ("shock_min","shock_max","g_star","pi_star","npcl_star") else int(val)
        if f["shock_min"] >= f["shock_max"]:
            error = "Доод хязгаар нь дээд хязгаараас бага байх ёстой."
        else:
            try:
                mc_df, inp_df, neutral_rate, nr = run_simulation(
                    df, f["shock_min"], f["shock_max"], f["n_sim"], f["n_forecast"],
                    f["g_star"], f["pi_star"], f["npcl_star"])
                _store["mc_df"]  = mc_df
                _store["inp_df"] = inp_df

                fig1,ax=plt.subplots(figsize=(11,4.5))
                ax.scatter(mc_df["DTIbbsb"],mc_df["loss"],alpha=0.3,s=8,color="#4C8BF5")
                ax.axvline(neutral_rate,color="red",linestyle="--",linewidth=2,
                           label=f"Неутрал түвшин = {neutral_rate:.4f}")
                ax.set_xlabel("ББСБ-ын хэрэглээний зээлийн өр, орлогын харьцаа (DTIbbsb)")
                ax.set_ylabel("Тайлорын функцийн утга (Алдагдал)")
                ax.set_title(f"Монте Карло симуляц | Хязгаар [{f['shock_min']:.2f}–{f['shock_max']:.2f}] | N={f['n_sim']}")
                ax.legend(); ax.grid(True,alpha=0.3)
                c1=fig_to_b64(fig1); plt.close(fig1)

                fig2,axes=plt.subplots(1,3,figsize=(14,4))
                for ax2,col,lbl,tgt in zip(axes,["Et_gdp","Et_cpi","Et_npcl"],
                    ["Бодит ДНБ өсөлт","Инфляци","Чанаргүй хэрэглээний зээл"],
                    [f["g_star"],f["pi_star"],f["npcl_star"]]):
                    ax2.hist(mc_df[col],bins=40,color="#4C8BF5",alpha=0.7,edgecolor="white")
                    ax2.axvline(mc_df[col].mean(),color="red",linestyle="--",label=f"Дундаж: {mc_df[col].mean():.4f}")
                    ax2.axvline(tgt,color="green",linestyle="--",label=f"Зорилт: {tgt:.4f}")
                    ax2.set_title(f"Таамаглалын {lbl}"); ax2.legend(fontsize=8); ax2.grid(True,alpha=0.3)
                plt.tight_layout(); c2=fig_to_b64(fig2); plt.close(fig2)

                top10=mc_df.nsmallest(10,"loss").reset_index(drop=True); top10.index+=1
                top10.columns=["DTIbbsb","Хүлээгдэж буй Бодит ДНБ өсөлтийн хувь","Инфляци","Чанаргүй хэрэглээний зээлийн хувь","Алдагдал"]
                tbl=top10.to_html(classes="",border=0,float_format=lambda x:f"{x:.4f}",index=True)

                result=dict(neutral_rate=neutral_rate,Et_gdp=nr["Et_gdp"],Et_cpi=nr["Et_cpi"],
                            Et_npcl=nr["Et_npcl"],chart1=c1,chart2=c2,table=tbl)
                defaults=f
            except Exception as e:
                error=str(e)

    return render_template_string(SIM_PAGE,
        page="simulation",
        title="Монте Карло Симуляц",
        subtitle="Тайлорын функцийн утга хамгийн бага байх неутрал өр, орлогын харьцааг тооцоолно.",
        info=info, f=defaults, error=error, result=result)

@app.route("/presentation")
def presentation():
    return render_template_string(PRES_PAGE,
        page="presentation",
        title="Танилцуулга",
        subtitle="Судалгааны танилцуулга материал")

@app.route("/download/<n>")
def download(n):
    if n=="mc_results" and "mc_df" in _store:
        return send_file(io.BytesIO(to_excel_bytes(_store["mc_df"])),
                         download_name="мк_үр_дүн.xlsx",as_attachment=True,
                         mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    if n=="mc_input" and "inp_df" in _store:
        return send_file(io.BytesIO(to_excel_bytes(_store["inp_df"])),
                         download_name="симуляцын_өгөгдөл.xlsx",as_attachment=True,
                         mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    return "Өгөгдөл байхгүй байна.", 404

if __name__=="__main__":
    port=int(os.environ.get("PORT",5000))
    app.run(host="0.0.0.0",port=port)
