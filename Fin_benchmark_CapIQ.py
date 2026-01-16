import os
import sys
import time
import socket
import subprocess
import webbrowser
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# =============================================================================
# Self-launching Streamlit bootstrap
# =============================================================================
def _running_inside_streamlit() -> bool:
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        if get_script_run_ctx() is not None:
            return True
    except Exception:
        pass
    try:
        from streamlit import runtime
        if runtime.exists():
            return True
    except Exception:
        pass
    return os.environ.get("STREAMLIT_RUN") == "1"


def _wait_for_port(host: str, port: int, timeout: float = 30.0) -> bool:
    start = time.time()
    while time.time() - start < timeout:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.settimeout(0.5)
                sock.connect((host, int(port)))
                return True
        except OSError:
            time.sleep(0.3)
    return False


def _spawn_streamlit():
    """Run `streamlit run` on this file when launched with python."""
    if _running_inside_streamlit():
        main()  # IMPORTANT: call the real UI
        return

    port = os.environ.get("PORT", "8501")
    env = os.environ.copy()
    env["STREAMLIT_RUN"] = "1"
    cmd = [
        sys.executable, "-m", "streamlit", "run", os.path.abspath(__file__),
        "--server.port", port, "--server.headless", "true",
        "--browser.gatherUsageStats=false",
    ]
    try:
        print(f"[INFO] Starting Streamlit on http://localhost:{port} ...")
        proc = subprocess.Popen(cmd, env=env)
        if _wait_for_port("localhost", int(port), timeout=30.0):
            try:
                webbrowser.open_new_tab(f"http://localhost:{port}")
            except Exception:
                pass
        proc.wait()
    except FileNotFoundError:
        print("[WARN] streamlit not found; running inline.")
        os.environ["STREAMLIT_RUN"] = "1"
        main()


# =============================================================================
# App constants
# =============================================================================
DATA_FILE = "Company_Financials_CapIQ.xlsx"
DATA_SHEET = "data"
METRICS_TYPE_SHEET = "metrics_type"

CURRENCY_BY_EXCHANGE = {
    "SGX": "SGD", "NYSE": "USD", "NYSE ARCA": "USD", "NYSE MKT": "USD",
    "BATS": "USD", "LSE": "GBP", "HKEX": "HKD", "Catalist": "SGD"
}

CHIP_CSS = """
<style>
.chip {display:inline-block; padding:3px 10px; border-radius:999px; font-size:12px; font-weight:700; margin-right:6px;}
.chip-green {background:#def7e5; color:#065f46; border:1px solid #34d399;}
.chip-amber {background:#fff3cd; color:#8a6d3b; border:1px solid #fbbf24;}
.chip-red   {background:#fde2e4; color:#7f1d1d; border:1px solid #f87171;}
.kpi {border:1px solid #e5e7eb; border-radius:8px; padding:10px 12px; margin:6px 0; background:#fff;}
.hr {height:1px; background:#eee; border:none; margin:10px 0 6px 0;}
.legend {font-size:12px; color:#4b5563; margin: 4px 0 10px 0;}
</style>
"""


# =============================================================================
# Data helpers
# =============================================================================
@st.cache_data(show_spinner=False)
def load_all_sheets(path: str = DATA_FILE) -> Dict[str, pd.DataFrame]:
    if not os.path.exists(path):
        st.error(f"Missing Excel file: {path} (expected in the current folder).")
        st.stop()
    xl = pd.ExcelFile(path, engine="openpyxl")
    out = {
        DATA_SHEET: xl.parse(DATA_SHEET),
        METRICS_TYPE_SHEET: xl.parse(METRICS_TYPE_SHEET),
    }
    return out


def infer_metric_columns(df: pd.DataFrame) -> List[str]:
    cols = list(df.columns)
    if "FY" not in cols:
        raise ValueError("Expected 'FY' column in data sheet")
    # All columns to the right of FY are interpreted as metric columns
    return cols[cols.index("FY") + 1 :]


def to_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out

@st.cache_data(show_spinner=False)
def compute_percentiles(
    df: pd.DataFrame,
    metric_cols: List[str],
    group_cols: Tuple[str, str, str] = ("EXCHANGE", "INDUSTRY", "FY"),
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      - wide percentiles (MultiIndex columns: (metric, stat))
      - tidy long dataframe with columns [EXCHANGE, INDUSTRY, FY, metric, p25, p50, p75]
    """
    df_num = df[list(group_cols) + list(metric_cols)].copy()
    for m in metric_cols:
        df_num[m] = pd.to_numeric(df_num[m], errors="coerce")

    q = df_num.groupby(list(group_cols))[metric_cols].quantile([0.25, 0.5, 0.75]).unstack(-1)
    new_cols = []
    for metric, qlevel in q.columns:
        tag = {0.25: "p25", 0.5: "p50", 0.75: "p75"}.get(float(qlevel), str(qlevel))
        new_cols.append((metric, tag))
    q.columns = pd.MultiIndex.from_tuples(new_cols, names=["metric", "stat"])

    # Build a tidy view
    rows = []
    for m in metric_cols:
        sub = q[m].reset_index()
        sub.insert(3, "metric", m)
        rows.append(sub)
    tidy = pd.concat(rows, ignore_index=True)
    return q, tidy

@st.cache_data(show_spinner=False)
def slice_company_row_for_fy(df, exchange, industry, fy, company) -> pd.DataFrame:
    mask = (
        (df["EXCHANGE"].astype(str) == exchange) &
        (df["INDUSTRY"].astype(str) == industry) &
        (df["FY"].astype(str) == str(fy))
    )
    df_slice = df.loc[mask]
    if company:
        exact = df_slice[df_slice["ENTITY_NAME"].astype(str).str.strip().str.lower() == company.strip().lower()]
        if not exact.empty:
            return exact.iloc[[0]]
    return df_slice.iloc[[0]] if not df_slice.empty else pd.DataFrame()

@st.cache_data(show_spinner=False)
def yoy_company_series(df, exchange, industry, company) -> pd.DataFrame:
    mask = (
        (df["EXCHANGE"].astype(str) == exchange) &
        (df["INDUSTRY"].astype(str) == industry) &
        (df["ENTITY_NAME"].astype(str).str.strip().str.lower() == company.strip().lower())
    )
    out = df.loc[mask].copy()
    out["FY"] = out["FY"].astype(str)
    return out

# =============================================================================
# Matching & autopopulate helpers 
# =============================================================================

def _safe_take(df: pd.DataFrame, cols: list[str]) -> tuple[pd.DataFrame, list[str]]:
    cols = list(cols)
    missing = [c for c in cols if c not in df.columns]
    if missing:
        return pd.DataFrame(), missing
    return df[cols].copy(), []

def get_company_matches(df: pd.DataFrame, typed: str) -> List[str]:
    if not typed or not str(typed).strip():
        return []
    names = df["ENTITY_NAME"].dropna().astype(str).unique().tolist()
    typed_low = str(typed).strip().lower()
    return [n for n in names if typed_low in n.lower()][:10]


def find_exact_company(df: pd.DataFrame, typed: str) -> pd.DataFrame:
    """Return all rows for the exact case-insensitive company name, else empty df."""
    mask = df["ENTITY_NAME"].astype(str).str.strip().str.lower() == str(typed).strip().lower()
    return df.loc[mask].copy()


def latest_profile_rows(df_company: pd.DataFrame) -> pd.DataFrame:
    """Return rows sorted to have the latest FY first."""
    if df_company.empty:
        return df_company
    df_company = df_company.copy()
    df_company["__FY_"] = pd.to_numeric(df_company["FY"], errors="coerce")
    return df_company.sort_values("__FY_", ascending=False).drop(columns="__FY_", errors="ignore")


def pick_best_profile(df_company: pd.DataFrame) -> Dict[str, str]:
    """Pick the (EXCHANGE, INDUSTRY) from the latest FY row for the company."""
    if df_company.empty:
        return {}
    df_latest = latest_profile_rows(df_company)
    row = df_latest.iloc[0]
    return {"EXCHANGE": str(row.get("EXCHANGE", "")), "INDUSTRY": str(row.get("INDUSTRY", ""))}


def _first_present_column(df: pd.DataFrame, aliases: List[str]) -> str:
    """Find the first column present in df (case-insensitive) from aliases."""
    cols_low = {c.lower(): c for c in df.columns}
    for a in aliases:
        if a.lower() in cols_low:
            return cols_low[a.lower()]
    return ""



def read_key_financials_for(df_company: pd.DataFrame, prefer_fy: str) -> Dict[str, float]:
    if df_company.empty:
        return {}

    dfc = df_company.copy()
    dfc["FY_str"] = dfc["FY"].astype(str)
    if prefer_fy and (dfc["FY_str"] == str(prefer_fy)).any():
        row = dfc.loc[dfc["FY_str"] == str(prefer_fy)].iloc[0]
    else:
        row = latest_profile_rows(dfc).iloc[0]

    # --- Aliases (case-insensitive) updated for your CapIQ columns ---
    # Feel free to extend these lists if you discover more CapIQ names.
    alias_map = {
        "Current Assets":        ["TOTAL_CA", "CURRENT_ASSETS", "Current Assets"],
        "Current Liabilities":   ["TOTAL_CL", "CURRENT_LIABILITIES", "Current Liabilities"],
        "Inventory":             ["INVENTORY", "Inventory"],
        "Operating Cash Flow":   ["TOTAL_OPER_EXPEN", "OPERATING_CASH_FLOW", "Operating Cash Flow", "CFO", "NET_CASH_FROM_OPERATIONS"],
        "Capital Expenditure":   ["CAPEX", "Capital Expenditure", "CAPITAL_EXPENDITURE"],
        "Revenue":               ["TOTAL_REV", "REVENUE", "Revenue", "TOTAL_REVENUE", "SALES"],
        "EBITDA":                ["EBITDA", "Ebitda"],
        "Cost of Revenue":       ["COST_OF_REVENUE", "Cost of Revenue", "COGS"],
        # Optional: surface market cap, kept separate b/c units differ
        "MARKETCAP ($'M)":       ["MARKETCAP ($'M)", "MarketCap_Millions", "MARKETCAP_M"]
    }

    # helper: first matching column name in df (case-insensitive)
    cols_low = {c.lower(): c for c in dfc.columns}
    def pick_col(aliases: list[str]) -> str:
        for a in aliases:
            if a.lower() in cols_low:
                return cols_low[a.lower()]
        return ""

    # Read values
    raw = {}
    for label, aliases in alias_map.items():
        col = pick_col(aliases)
        if not col:
            continue
        v = pd.to_numeric(row.get(col, np.nan), errors="coerce")
        if pd.notna(v):
            raw[label] = float(v)

    # --- Scale: thousands -> units for all standard financials; market cap stays millions ---
    OUT = {}
    for label, v in raw.items():
        if label == "MARKETCAP ($'M)":
            # keep in millions; if you prefer units: OUT["Market Cap"] = v * 1_000_000
            OUT[label] = v
        else:
            OUT[label] = v * 1_000.0    # scale k -> units

    return OUT


def _try_get_percentile_row(pct_wide: pd.DataFrame, exch: str, ind: str, fy: str):
    """Robust access to percentile row by attempting string and numeric FY keys."""
    if pct_wide is None:
        return None
    try:
        return pct_wide.loc[(exch, ind, fy)]
    except Exception:
        pass
    try:
        fy_num = float(fy) if fy is not None else None
        return pct_wide.loc[(exch, ind, fy_num)]
    except Exception:
        return None


# =============================================================================
# Visuals
# =============================================================================
def _classify_bucket(value: float, p25: float, p50: float, p75: float, grade: str) -> str:
    import pandas as _pd
    if _pd.isna(value) or _pd.isna(p25) or _pd.isna(p50) or _pd.isna(p75):
        return "—"
    lower = str(grade).lower().strip() == "lower"  # lower is better
    if lower:
        return "Healthy" if value <= p25 else ("Needs Improvement" if value >= p75 else "Satisfactory")
    else:
        return "Healthy" if value >= p75 else ("Needs Improvement" if value <= p25 else "Satisfactory")


def _chip_class(bucket: str) -> str:
    return {"Healthy": "chip-green", "Satisfactory": "chip-amber", "Needs Improvement": "chip-red"}.get(bucket, "chip-amber")

def _plot_benchmark(
    value, p25, p50, p75,
    grade,
    title="",
    company_name="Company",
    band_thickness=0.16,
    qline_width=1.2,
    cross_size=18,
    # white pill label by default
    label_bgcolor="#FFFFFF",
    label_bordercolor="#1d4ed8",
    label_font_color="#1d4ed8"
):
    import math
    import pandas as _pd
    import numpy as _np
    import plotly.graph_objects as go

    fig = go.Figure()

    # --- normalize
    def f(x):
        try: return float(x)
        except Exception: return float("nan")
    v, q25, q50, q75 = f(value), f(p25), f(p50), f(p75)

    # --- domain (prefer [p25..p75])
    xs = [x for x in [q25, q50, q75, v] if _pd.notna(x) and math.isfinite(x)]
    if not xs:
        xs = [0.0, 1.0]
    if _pd.notna(q25) and _pd.notna(q75) and q75 >= q25:
        base_min, base_max = q25, q75
    else:
        base_min, base_max = min(xs), max(xs)

    span = (base_max - base_min) if base_max != base_min else (abs(base_max) if base_max else 1.0)
    pad = span * 0.25
    x0, x1 = base_min - pad, base_max + pad

    # --- slim band geometry
    y_mid = 0.50
    h = max(0.06, min(0.35, float(band_thickness)))
    y0, y1 = y_mid - h, y_mid + h

    # --- background bands (drawn BELOW traces)
    lower_is_better = str(grade).lower().strip() == "lower"
    def rect(a, b, color):
        a, b = (a, b) if a <= b else (b, a)
        fig.add_shape(
            type="rect", x0=a, x1=b, y0=y0, y1=y1,
            fillcolor=color, line_width=0, layer="below"   # <- keep below data
        )

    _q25 = q25 if _pd.notna(q25) else (x0 + (x1 - x0) * 0.33)
    _q75 = q75 if _pd.notna(q75) else (x0 + (x1 - x0) * 0.66)
    if lower_is_better:
        rect(x0,  _q25, "#def7e5")
        rect(_q25, _q75, "#fff3cd")
        rect(_q75, x1,  "#fde2e4")
    else:
        rect(x0,  _q25, "#fde2e4")
        rect(_q25, _q75, "#fff3cd")
        rect(_q75, x1,  "#def7e5")

    # --- quartile ticks (also BELOW traces)
    def tick(x, color, width):
        if _pd.notna(x):
            fig.add_shape(
                type="line", x0=x, x1=x, y0=y0, y1=y1,
                line=dict(color=color, width=width),
                layer="below"  # <- keep below data
            )
    tick(q25, "#9ca3af", qline_width)
    tick(q50, "#6b7280", qline_width)
    tick(q75, "#374151", qline_width)

    # --- quartile labels (annotations are always above)
    label_y = y1 + 0.05
    def qlabel(x, txt, color):
        if _pd.notna(x):
            fig.add_annotation(
                x=x, y=label_y, text=txt, xanchor="center", yanchor="bottom",
                showarrow=False, font=dict(color=color, size=10)
            )
    qlabel(q25, "p25", "#9ca3af")
    qlabel(q50, "p50", "#6b7280")
    qlabel(q75, "p75", "#374151")

    # --- company marker as a bold X (on TOP of shapes)
    if _pd.notna(v):
        fig.add_trace(
            go.Scatter(
                x=[v], y=[y_mid],
                mode="markers",
                # 'x' is a filled glyph; for a line-only X use 'x-thin' or 'x-open'
                marker=dict(
                    symbol="x",           # try 'x', 'x-thin', or 'x-open'
                    size=cross_size,
                    color="#1d4ed8",      # glyph color
                ),
                name="Company",
                hovertemplate=f"{company_name}<extra></extra>",
                cliponaxis=False
            )
        )
        # white pill label just below the band
        fig.add_annotation(
            x=v, y=y0 - 0.06,
            text=company_name or "Company",
            xanchor="center", yanchor="top",
            showarrow=False,
            bgcolor=label_bgcolor,            # white
            bordercolor=label_bordercolor,    # blue outline
            borderwidth=1,
            borderpad=6,
            font=dict(color=label_font_color, size=12)
        )

    # --- axes & layout
    fig.update_xaxes(range=[x0, x1], showgrid=True, gridcolor="#e5e7eb", zeroline=False, tickmode="auto")
    fig.update_yaxes(visible=False, range=[0, 1])
    fig.update_layout(
        height=110,
        margin=dict(l=6, r=6, t=6, b=12),
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        title=title,
        showlegend=False
    )
    return fig

def _plot_yoy(industry_df: pd.DataFrame, company_df: pd.DataFrame, metric: str, grade: str) -> go.Figure:
    fig = go.Figure()
    if industry_df is not None and not industry_df.empty:
        lower = str(grade).lower().strip() == "lower"
        if lower:
            fig.add_trace(go.Scatter(x=industry_df["FY"], y=industry_df["p25"], name="p25", line=dict(color="green")))
            fig.add_trace(go.Scatter(x=industry_df["FY"], y=industry_df["p50"], name="p50",
                                     line=dict(color="orange", dash="dot")))
            fig.add_trace(go.Scatter(x=industry_df["FY"], y=industry_df["p75"], name="p75", line=dict(color="red")))
        else:
            fig.add_trace(go.Scatter(x=industry_df["FY"], y=industry_df["p25"], name="p25", line=dict(color="red")))
            fig.add_trace(go.Scatter(x=industry_df["FY"], y=industry_df["p50"], name="p50",
                                     line=dict(color="orange", dash="dot")))
            fig.add_trace(go.Scatter(x=industry_df["FY"], y=industry_df["p75"], name="p75", line=dict(color="green")))
    if company_df is not None and not company_df.empty:
        fig.add_trace(go.Scatter(
            x=company_df["FY"], y=company_df[metric], name="Company",
            mode="lines+markers", line=dict(color="blue", width=3)
        ))
    fig.update_layout(height=220, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                      xaxis_title="FY", yaxis_title=metric)
    return fig


# =============================================================================
# GPT stubs (unchanged behavior)
# =============================================================================
def _get_openai_api_key():
    # 1. Try Streamlit secrets
    try:
        key = st.secrets["openai"]["api_key"]
        if key:
            return key
    except Exception:
        pass

    # 2. Try environment variable
    key = os.environ.get("OPENAI_API_KEY")
    if key:
        return key

    return None

def _call_openai(system_prompt: str, user_prompt: str, model: str = None, max_output_tokens: int = 600) -> str:
    key = _get_openai_api_key()
    if not key:
        return ""
    try:
        from openai import OpenAI
        client = OpenAI(api_key=key)
        mdl = (model or os.environ.get("OPENAI_MODEL") or "gpt-5").strip()
        out = client.chat.completions.create(
            model=mdl,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
            max_tokens=max_output_tokens,
        )
        return out.choices[0].message.content
    except Exception:
        return "", f"OpenAI call failed: {e}"


def _build_finhealth_prompt(company, exchange, industry, fy, metrics_rows, metrics_type_name):
    system = (
        "You are a senior financial analyst and internal auditor. "
        "Assess the company's financial health strictly using the metrics provided and their industry percentiles. "
        "Write in plain business English, concise and action-focused. Conclude with 3–5 priority actions."
    )
    lines = []
    for r in metrics_rows:
        lines.append(
            f"- {r['Metrics_Name']} ({r['Metrics_Col']}): value={r['value_str']}  "
            f"p25={r['p25_str']}, p50={r['p50_str']}, p75={r['p75_str']}  "
            f"grade={r['Metrics_Grade']}  bucket={r['bucket']}"
        )
    user = (
        f"Company: {company}  \n"
        f"Exchange: {exchange}  \n"
        f"Industry: {industry}  \n"
        f"FY: {fy}\n"
        f"Metrics type selected: {metrics_type_name}\n" + "\n".join(lines)
    )
    return system, user


def _build_audit_prompt_allmetrics(company, exchange, industry, fy, summary_rows):
    system = (
        "You are an experienced internal auditor. Using all the metrics and benchmark context, "
        "propose the TOP 5 auditable areas with highest risk and business impact. "
        "Avoid external audit or generic compliance steps."
        "Do not use acronyms or abbreviations in your response. Always write out the full term."
        "Be specific about risks, testing steps, and data sources."
    )
    bullets = []
    for r in summary_rows:
        bullets.append(
            f"- {r['Metrics_Name']} ({r['Metrics_Col']}): value={r['value_str']}  "
            f"p25={r['p25_str']}, p50={r['p50_str']}, p75={r['p75_str']}  "
            f"grade={r['Metrics_Grade']}  bucket={r['bucket']}"
        )
    user = (
        f"Company: {company}  \n"
        f"Exchange: {exchange}  \n"
        f"Industry: {industry}  \n"
        f"FY: {fy}\n"
        "Use all metrics below (assume there are 39 metrics). Output exactly 5 audit areas.\n"
        + "\n".join(bullets)
    )
    return system, user


# =============================================================================
# Utility (numeric validation)
# =============================================================================
def _is_number_str(x: Optional[str]) -> bool:
    try:
        sx = "" if x is None else str(x).strip()
        if sx == "":
            return False
        float(sx)
        return True
    except Exception:
        return False


# =============================================================================
# MAIN UI
# =============================================================================
def main():
    st.set_page_config(page_title="Industry Benchmark Analysis Dashboard", layout="wide")
    st.markdown(CHIP_CSS, unsafe_allow_html=True)
    st.title("Industry Benchmark Analysis Dashboard")

    # Load data
    sheets = load_all_sheets()
    data_df = sheets[DATA_SHEET]
    mtype_df = sheets[METRICS_TYPE_SHEET]

    # Validate metrics_type sheet
    required_cols = ["Type", "Metrics_Name", "Metrics_Col", "Metrics_Description", "Metrics_Grade"]
    for c in required_cols:
        if c not in mtype_df.columns:
            st.error(f"'{METRICS_TYPE_SHEET}' sheet missing column: {c}")
            st.stop()

    # Prepare metrics and percentiles
    metric_cols = infer_metric_columns(data_df)
    data_df = to_numeric(data_df, metric_cols)
    pct_wide, pct_tidy = compute_percentiles(data_df, metric_cols)

    # --- Sidebar: Always-visible Company Input (with auto-population) ---
    st.sidebar.header("Company Input")

    # Exchange list (for initial default)
    exch_list = sorted(data_df["EXCHANGE"].dropna().astype(str).unique().tolist())
    default_exch_idx = exch_list.index("SGX") if "SGX" in exch_list else 0

    # Session scaffolding for autofill
    ss = st.session_state
    ss.setdefault("_autofill_company", "")
    ss.setdefault("_autofill_fy", "")
    ss.setdefault("_autofill_exchange", "")
    for k in [
        "current_assets", "current_liabilities", "inventory",
        "operating_cf", "capex", "revenue", "ebitda", "cost_of_revenue"
    ]:
        ss.setdefault(k, "")

    # 1) Company Name (first)
    typed_company = st.sidebar.text_input("Company Name", key="company_name_input")

    # Suggestions (up to 10)
    suggestions = get_company_matches(data_df, typed_company) if typed_company else []
    if typed_company and suggestions and typed_company.strip() not in suggestions:
        st.sidebar.caption("Suggestions: " + ", ".join(suggestions))

    # Find exact company block (case-insensitive)
    df_company = find_exact_company(data_df, typed_company) if typed_company else pd.DataFrame()

    # 2) Exchange (defaults to company’s exchange if exact match, else SGX or first)
    default_exchange = (
        pick_best_profile(df_company).get("EXCHANGE", exch_list[default_exch_idx]) if not df_company.empty
        else exch_list[default_exch_idx]
    )
    exchange = st.sidebar.selectbox("Exchange", exch_list, index=exch_list.index(default_exchange))

    # 3) Industry list depends on selected exchange
    ind_list = sorted(
        data_df.loc[data_df["EXCHANGE"].astype(str) == exchange, "INDUSTRY"].dropna().astype(str).unique().tolist()
    )
    profile_industry = pick_best_profile(df_company).get("INDUSTRY", "") if not df_company.empty else ""
    default_industry = profile_industry if profile_industry in ind_list else (ind_list[0] if ind_list else "")
    industry = st.sidebar.selectbox(
        "Industry",
        ind_list,
        index=(ind_list.index(default_industry) if default_industry in ind_list else 0)
    )

    # 4) Financial Year list
    if not df_company.empty:
        years_arr = sorted(df_company["FY"].dropna().astype(str).unique().tolist())
    else:
        cand = data_df[(data_df["EXCHANGE"].astype(str) == exchange) & (data_df["INDUSTRY"].astype(str) == industry)]
        years_arr = sorted(cand["FY"].dropna().astype(str).unique().tolist())
    fy = st.sidebar.selectbox("Financial Year", years_arr, index=(len(years_arr) - 1 if years_arr else 0), key="fy_select")

    # Currency label by exchange
    ccy = CURRENCY_BY_EXCHANGE.get(exchange, "")

    # Auto-populate financial inputs when (company, fy, exchange) changes and exact match exists
    autofill_trigger = f"{typed_company or ''}|{fy or ''}|{exchange or ''}"
    last_trigger = f"{ss.get('_autofill_company','')}|{ss.get('_autofill_fy','')}|{ss.get('_autofill_exchange','')}"
    if not df_company.empty and fy and autofill_trigger != last_trigger:
        snapshot_vals = read_key_financials_for(df_company, prefer_fy=fy)  # <-- already scaled to units here

        mapping = [
            ("Current Assets",       "current_assets"),
            ("Current Liabilities",  "current_liabilities"),
            ("Inventory",            "inventory"),
            ("Operating Cash Flow",  "operating_cf"),
            ("Capital Expenditure",  "capex"),
            ("Revenue",              "revenue"),
            ("EBITDA",               "ebitda"),
            ("Cost of Revenue",      "cost_of_revenue"),
        ]
        for label, key in mapping:
            if label in snapshot_vals:
                ss[key] = f"{snapshot_vals[label]}"
            else:
                ss.setdefault(key, ss.get(key, ""))

        ss["_autofill_company"]  = typed_company or ""
        ss["_autofill_fy"]       = fy or ""
        ss["_autofill_exchange"] = exchange or ""

    # 5) Key financials — ALWAYS visible text inputs (editable; auto-populated when matched)
    current_assets      = st.sidebar.text_input(f"Current Assets ({ccy})", key="current_assets")
    current_liabilities = st.sidebar.text_input(f"Current Liabilities ({ccy})", key="current_liabilities")
    inventory           = st.sidebar.text_input(f"Inventory ({ccy})", key="inventory")
    operating_cf        = st.sidebar.text_input(f"Operating Cash Flow ({ccy})", key="operating_cf")
    capex               = st.sidebar.text_input(f"Capital Expenditure ({ccy})", key="capex")
    revenue             = st.sidebar.text_input(f"Revenue ({ccy})", key="revenue")
    ebitda              = st.sidebar.text_input(f"EBITDA ({ccy})", key="ebitda")
    cost_of_revenue     = st.sidebar.text_input(f"Cost of Revenue ({ccy})", key="cost_of_revenue")

    # 6) Numeric gating (like your earlier prototype)
    numeric_keys = [
        "current_assets", "current_liabilities", "inventory", "operating_cf",
        "capex", "revenue", "ebitda", "cost_of_revenue"
    ]
    numeric_ok = all(_is_number_str(ss.get(k)) for k in numeric_keys)
    identifiers_ok = bool(exchange and industry and fy and typed_company and typed_company.strip())

    if not numeric_ok:
        st.sidebar.warning("Please complete all financial fields with valid numbers before submitting.")

    # Submit is disabled until all eight financial fields are numeric and identifiers are set
    can_submit = identifiers_ok and numeric_ok
    submit = st.sidebar.button("Submit", type="primary", disabled=not can_submit)

    # --- after numeric gating and when user clicks Submit ---
    if not submit and "selection" not in st.session_state:
        st.info("Fill in the Company Input on the left. All financial fields must be numeric. Click **Submit** to load benchmarks.")
        st.stop()

    if submit:
        st.session_state["selection"] = {
            "company": typed_company.strip(),
            "exchange": exchange,
            "industry": industry,
            "fy": str(fy)  # store as string for consistency
        }
        st.session_state["pct_wide"] = pct_wide
        st.session_state["pct_tidy"] = pct_tidy
        st.session_state["mtype_bm"] = None   # current selection for Benchmarking tab
        st.session_state["mtype_yoy"] = None  # current selection for YoY tab

    sel = st.session_state["selection"]
    company = sel["company"]; exch = sel["exchange"]; ind = sel["industry"]; fy_sel = sel["fy"]
    pct_wide = st.session_state["pct_wide"]; pct_tidy = st.session_state["pct_tidy"]

    # -------------------------------------------------------------------------
    # Body content (after Submit): show Metrics Type first, then benchmarking UI
    # -------------------------------------------------------------------------

    # Freeze choices into session (for consistency across tabs)
    ss["company_name"] = typed_company
    ss["exchange"] = exchange
    ss["industry"] = industry
    ss["fy"] = fy
    ss["pct_wide"] = pct_wide
    ss["pct_tidy"] = pct_tidy

    company = ss["company_name"]
    exch = ss["exchange"]
    ind = ss["industry"]
    fy_sel = ss["fy"]

    st.markdown(f"**Company:** {company} &nbsp;&nbsp; **Exchange:** {exch} &nbsp;&nbsp; "
                f"**Industry:** {ind} &nbsp;&nbsp; **FY:** {fy_sel}")

    tab_bm, tab_yoy, tab_audit = st.tabs(["1.Benchmarking (Selected FY)", "2.YoY Trend", "3.Suggested Audit Areas"])

    # -------------------------------------------------------------------------
    # TAB 1 — Benchmarking (Selected FY)
    # -------------------------------------------------------------------------

    with tab_bm:
        st.markdown(
            '<div class="legend">Legend: '
            '<span class="chip chip-green">Healthy</span>'
            '<span class="chip chip-amber">Satisfactory</span>'
            '<span class="chip chip-red">Needs Improvement</span></div><div class="hr"></div>',
            unsafe_allow_html=True
        )

        type_list = list(mtype_df["Type"].dropna().astype(str).unique())
        # remember last choice just for this tab
        default_idx = 0 if st.session_state.get("mtype_bm") is None else type_list.index(st.session_state["mtype_bm"])
        mtype = st.selectbox("Select a metrics type", type_list, index=default_idx, key="mtype_bm")
        subset = mtype_df[mtype_df["Type"] == mtype].copy()
        st.caption(f"Showing metrics for type: **{mtype}** ({len(subset)} metrics)")

        comp_slice = slice_company_row_for_fy(data_df, exch, ind, fy_sel, company)
        p_row = _try_get_percentile_row(pct_wide, exch, ind, fy_sel)



        grid = st.columns(2)
        assembled_for_llm = []
        skipped_bm = []

        for i, (_, rec) in enumerate(subset.iterrows()):
            m_col   = str(rec["Metrics_Col"]).strip()
            m_name  = str(rec["Metrics_Name"]).strip()
            m_desc  = str(rec["Metrics_Description"]).strip()
            m_grade = str(rec["Metrics_Grade"]).strip()

            val = pd.to_numeric(comp_slice.iloc[0].get(m_col, np.nan), errors="coerce") if not comp_slice.empty else np.nan
            p25 = p50 = p75 = np.nan
            if p_row is not None and (m_col, "p25") in getattr(p_row, "index", []):
                p25 = pd.to_numeric(p_row[(m_col, "p25")], errors="coerce")
                p50 = pd.to_numeric(p_row[(m_col, "p50")], errors="coerce")
                p75 = pd.to_numeric(p_row[(m_col, "p75")], errors="coerce")

            # Skip if nothing to show
            if (pd.isna(val)) and pd.isna(p25) and pd.isna(p50) and pd.isna(p75):
                continue

            bucket = _classify_bucket(val, p25, p50, p75, m_grade)

            with grid[i % 2]:
                # put the entire card into one container so layout stays together
                with st.container():
                    st.markdown(f"**{m_name}**")  # removed column code
                    st.markdown(f'<span class="chip {_chip_class(bucket)}">{bucket}</span>', unsafe_allow_html=True)
                    st.caption(m_desc)

                    fig = _plot_benchmark(
                        val, p25, p50, p75, m_grade,
                        company_name=company,
                        band_thickness=0.16
                    )
                    st.plotly_chart(fig, use_container_width=True, key=f"bm_{m_col}")

                    _fmt = lambda x: "NA" if pd.isna(x) else f"{x:.4g}"
                    st.caption(f"Company={_fmt(val)} · p25={_fmt(p25)} · p50={_fmt(p50)} · p75={_fmt(p75)}")

            # (keep your LLM assembly if you need it)
            assembled_for_llm.append({
                "Metrics_Name": m_name, "Metrics_Col": m_col, "Metrics_Grade": m_grade,
                "value": None if pd.isna(val) else float(val), "value_str": ("NA" if pd.isna(val) else f"{val:.4g}"),
                "p25": None if pd.isna(p25) else float(p25), "p50": None if pd.isna(p50) else float(p50),
                "p75": None if pd.isna(p75) else float(p75),
                "p25_str": ("NA" if pd.isna(p25) else f"{p25:.4g}"),
                "p50_str": ("NA" if pd.isna(p50) else f"{p50:.4g}"),
                "p75_str": ("NA" if pd.isna(p75) else f"{p75:.4g}"),
                "bucket": bucket,
            })

            if skipped_bm:
                with st.expander("Skipped metrics (not available for this FY / industry / company)", expanded=False):
                    st.write(", ".join(skipped_bm))



        with st.expander("Generate analysis (GPT‑5)", expanded=True):
            model = st.text_input("Model", os.environ.get("OPENAI_MODEL", "gpt-5"), key="bm_model")

            generate = st.button("Generate Benchmarking Analysis", type="primary", key="gen_bm_btn")
            if generate:
                # Build prompts (same as before)
                system_prompt, user_prompt = _build_finhealth_prompt(
                    company, exch, ind, str(fy_sel), assembled_for_llm, mtype
                )
                api_key_for_call = _get_openai_api_key()
                if not api_key_for_call:
                    st.error("OpenAI API key is missing. Please set it in your environment or Streamlit secrets.")
                else:
                    with st.spinner("Calling OpenAI and generating suggestions..."):
                        text, err = _call_openai(
                            system_prompt, user_prompt, api_key=api_key_for_call, model=model, max_tokens=600
                        )

                    if err:
                        st.error(err)
                    else:
                        render = (text or "").strip()
                        if not render:
                            st.warning(
                                "No suggestions generated. Try switching to `gpt-4o`, reducing the prompt length, "
                                "or lowering `max_tokens` to stay within the model’s context window."
                            )
                            with st.expander("Debug info"):
                                st.code(f"MODEL: {model}\n\nSYSTEM PROMPT:\n{system_prompt}\n\nUSER PROMPT:\n{user_prompt}")
                        else:
                            st.markdown("##### Benchmarking Analysis")
                            st.markdown(render)
                            st.session_state["ai_bm_suggestions"] = text


    # -------------------------------------------------------------------------
    # TAB 2 — YoY Trend
    # -------------------------------------------------------------------------

    with tab_yoy:
        type_list2 = list(mtype_df["Type"].dropna().astype(str).unique())
        default_idx2 = 0 if st.session_state.get("mtype_yoy") is None else type_list2.index(st.session_state["mtype_yoy"])
        mtype2 = st.selectbox("Select a metrics type for YoY", type_list2, key="mtype_yoy", index=default_idx2)
        subset2 = mtype_df[mtype_df["Type"] == mtype2].copy()
        st.caption(f"Industry percentiles by FY for: **{exch} · {ind}**")


        tidy = pct_tidy[
            (pct_tidy["EXCHANGE"].astype(str) == exch) & (pct_tidy["INDUSTRY"].astype(str) == ind)
        ]

        company_series = yoy_company_series(data_df, exch, ind, company)

        grid = st.columns(2)
        skipped_yoy = []  # <- track omitted metrics

        for i, (_, rec) in enumerate(subset2.iterrows()):
            m_col  = str(rec["Metrics_Col"]).strip()
            m_name = str(rec["Metrics_Name"]).strip()
            m_grade = str(rec["Metrics_Grade"]).strip()

            # Industry percentiles for YoY — this won't KeyError, may be empty though
            ind_df = (
                tidy[tidy["metric"] == m_col][["FY", "p25", "p50", "p75"]]
                .dropna()
                .sort_values("FY")
            )

            # Company series — SAFE: only take the column if present
            comp_df, missing = _safe_take(company_series, ["FY", m_col])
            if missing:
                # Column absent in company data; we can either skip or still show industry-only chart.
                # Since you asked to "omit", we'll skip completely:
                skipped_yoy.append(m_name)
                continue

            comp_df = comp_df.dropna().sort_values("FY")

            # If BOTH are empty, skip chart
            if (ind_df is None or ind_df.empty) and (comp_df is None or comp_df.empty):
                skipped_yoy.append(m_name)
                continue

            with grid[i % 2]:
                st.markdown(f"**{m_name}**")
                fig = _plot_yoy(ind_df, comp_df, m_col, m_grade)
                st.plotly_chart(fig, width="stretch", key=f"yoy_{m_col}")

        if skipped_yoy:
            with st.expander("Skipped YoY metrics (missing in company data / industry percentiles)", expanded=False):
                st.write(", ".join(skipped_yoy))

        with st.expander("Generate YoY analysis (GPT‑5)", expanded=True):
            model = st.text_input("Model (YoY)", os.environ.get("OPENAI_MODEL", "gpt-5"), key="yoy_model")

            generate = st.button("Generate YoY Analysis", type="primary", key="gen_yoy_btn")
            if generate:
                # Assemble YoY snapshot for latest FY context (unchanged logic)
                p_row_latest = _try_get_percentile_row(pct_wide, exch, ind, str(fy_sel))
                comp_last = company_series[company_series["FY"] == str(fy_sel)]
                if comp_last.empty and not company_series.empty:
                    comp_last = company_series.iloc[[-1]]

                assembled = []
                for _, r in subset2.iterrows():
                    c = str(r["Metrics_Col"]).strip()
                    n = str(r["Metrics_Name"]).strip()
                    g = str(r["Metrics_Grade"]).strip()
                    val = pd.to_numeric(comp_last.iloc[0].get(c, np.nan), errors="coerce") if not comp_last.empty else np.nan
                    p25 = p50 = p75 = np.nan
                    if p_row_latest is not None and (c, "p25") in p_row_latest.index:
                        p25 = pd.to_numeric(p_row_latest[(c, "p25")], errors="coerce")
                        p50 = pd.to_numeric(p_row_latest[(c, "p50")], errors="coerce")
                        p75 = pd.to_numeric(p_row_latest[(c, "p75")], errors="coerce")

                    assembled.append({
                        "Metrics_Name": n, "Metrics_Col": c, "Metrics_Grade": g,
                        "value": None if pd.isna(val) else float(val),
                        "value_str": ("NA" if pd.isna(val) else f"{val:.4g}"),
                        "p25": None if pd.isna(p25) else float(p25),
                        "p50": None if pd.isna(p50) else float(p50),
                        "p75": None if pd.isna(p75) else float(p75),
                        "p25_str": ("NA" if pd.isna(p25) else f"{p25:.4g}"),
                        "p50_str": ("NA" if pd.isna(p50) else f"{p50:.4g}"),
                        "p75_str": ("NA" if pd.isna(p75) else f"{p75:.4g}"),
                        "bucket": _classify_bucket(val, p25, p50, p75, g),
                    })

                system_prompt, user_prompt = _build_finhealth_prompt(
                    company, exch, ind, str(fy_sel), assembled, mtype2
                )

                api_key_for_call = _get_openai_api_key()
                if not api_key_for_call:
                    st.error("OpenAI API key is missing. Please set it in your environment or Streamlit secrets.")
                else:
                    with st.spinner("Calling OpenAI and generating suggestions..."):
                        text, err = _call_openai(
                            system_prompt, user_prompt, api_key=api_key_for_call, model=model, max_tokens=600
                        )

                    if err:
                        st.error(err)
                    else:
                        render = (text or "").strip()
                        if not render:
                            st.warning(
                                "No suggestions generated. Try switching to `gpt-4o`, reducing the prompt length, "
                                "or lowering `max_tokens` to stay within the model’s context window."
                            )
                            with st.expander("Debug info"):
                                st.code(f"MODEL: {model}\n\nSYSTEM PROMPT:\n{system_prompt}\n\nUSER PROMPT:\n{user_prompt}")
                        else:
                            st.markdown("##### YoY Analysis")
                            st.markdown(render)
                            st.session_state["ai_yoy_suggestions"] = text


    # -------------------------------------------------------------------------
    # TAB 3 — Suggested Audit Areas (Top 5)
    # -------------------------------------------------------------------------
    with tab_audit: 
        st.subheader("Top 5 Suggested Audit Areas (Consolidated)")
        subset_all = mtype_df.copy()
        subset_all = subset_all.iloc[:39] if len(subset_all) > 39 else subset_all

        sel_mask = (
            (data_df["EXCHANGE"].astype(str) == exch)
            & (data_df["INDUSTRY"].astype(str) == ind)
            & (data_df["FY"].astype(str) == str(fy_sel))
        )
        df_slice = data_df.loc[sel_mask]
        comp_row = df_slice[df_slice["ENTITY_NAME"].astype(str).str.strip().str.lower() == company.strip().lower()]
        if comp_row.empty and not df_slice.empty:
            comp_row = df_slice.iloc[[0]]

        p_row = _try_get_percentile_row(pct_wide, exch, ind, str(fy_sel))

        assembled = []
        for _, r in subset_all.iterrows():
            c = str(r["Metrics_Col"]).strip()
            n = str(r["Metrics_Name"]).strip()
            g = str(r["Metrics_Grade"]).strip()

            val = pd.to_numeric(comp_row.iloc[0].get(c, np.nan), errors="coerce") if not comp_row.empty else np.nan
            p25 = p50 = p75 = np.nan
            if p_row is not None and (c, "p25") in p_row.index:
                p25 = pd.to_numeric(p_row[(c, "p25")], errors="coerce")
                p50 = pd.to_numeric(p_row[(c, "p50")], errors="coerce")
                p75 = pd.to_numeric(p_row[(c, "p75")], errors="coerce")

            assembled.append(
                {
                    "Metrics_Name": n,
                    "Metrics_Col": c,
                    "Metrics_Grade": g,
                    "value": float(val) if pd.notna(val) else None,
                    "value_str": (f"{val:.4g}" if pd.notna(val) else "NA"),
                    "p25": float(p25) if pd.notna(p25) else None,
                    "p50": float(p50) if pd.notna(p50) else None,
                    "p75": float(p75) if pd.notna(p75) else None,
                    "p25_str": (f"{p25:.4g}" if pd.notna(p25) else "NA"),
                    "p50_str": (f"{p50:.4g}" if pd.notna(p50) else "NA"),
                    "p75_str": (f"{p75:.4g}" if pd.notna(p75) else "NA"),
                    "bucket": _classify_bucket(val, p25, p50, p75, g),
                }
            )

        with st.expander("Generate with GPT‑5", expanded=True):
            model = st.text_input("Model (Audit)", os.environ.get("OPENAI_MODEL", "gpt-5"), key="audit_model")

            generate = st.button("Generate Audit Suggestions", type="primary", key="gen_audit_btn")
            if generate:
                system_prompt, user_prompt = _build_audit_prompt_allmetrics(
                    company, exch, ind, str(fy_sel), assembled
                )
                api_key_for_call = _get_openai_api_key()
                if not api_key_for_call:
                    st.error("OpenAI API key is missing. Please set it in your environment or Streamlit secrets.")
                else:
                    with st.spinner("Calling OpenAI and generating suggestions..."):
                        text, err = _call_openai(
                            system_prompt, user_prompt, api_key=api_key_for_call, model=model, max_tokens=600
                        )

                    if err:
                        st.error(err)
                    else:
                        render = (text or "").strip()
                        if not render:
                            st.warning(
                                "No suggestions generated. Try switching to `gpt-4o`, reducing the prompt length, "
                                "or lowering `max_tokens` to stay within the model’s context window."
                            )
                            with st.expander("Debug info"):
                                st.code(f"MODEL: {model}\n\nSYSTEM PROMPT:\n{system_prompt}\n\nUSER PROMPT:\n{user_prompt}")
                        else:
                            st.markdown("##### Suggested Auditable Areas")
                            st.markdown(render)
                            st.session_state["ai_audit_suggestions"] = text



# =============================================================================
# Entrypoint
# =============================================================================
if __name__ == "__main__":
    _spawn_streamlit()

