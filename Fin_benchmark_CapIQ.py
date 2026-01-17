import os
import sys
import time
import socket
import subprocess
import webbrowser
from typing import List, Tuple, Dict, Optional
import math 
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import json
import io
import BytesIO
import re
from difflib import SequenceMatcher
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, ListFlowable, ListItem
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm


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

AUDIT_DB_FILE = "Audit_Work_Program.xlsx"
AUDIT_DB_SHEET = None

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

@st.cache_data(show_spinner=False)
def load_audit_db(path: str = AUDIT_DB_FILE, sheet: str | None = AUDIT_DB_SHEET) -> pd.DataFrame:
    if not os.path.exists(path):
        st.error(f"Missing audit master Excel file: {path}")
        st.stop()
    xl = pd.ExcelFile(path, engine="openpyxl")
    sh = sheet or xl.sheet_names[0]
    df = xl.parse(sh)

    # Normalize expected column names (exact names in your file)
    expected = ["Scope", "Sub-process", "Risk", "Control Description", "Audit Test Steps", "Documents required"]
    missing = [c for c in expected if c not in df.columns]
    if missing:
        st.error(f"Audit DB missing columns: {missing} (expected: {expected})")
        st.stop()

    # Basic cleanup
    for c in expected:
        df[c] = df[c].astype(str).str.strip()
    return df

@st.cache_data(show_spinner=False)
def audit_vocab(df: pd.DataFrame) -> tuple[list[str], dict[str, list[str]]]:
    scopes = sorted(df["Scope"].dropna().unique().tolist())
    subs_by_scope = {s: sorted(df.loc[df["Scope"] == s, "Sub-process"].dropna().unique().tolist()) for s in scopes}
    return scopes, subs_by_scope

# ==============================
# Evidence extraction utilities
# ==============================
try:
    import docx  # python-docx
except Exception:
    docx = None

try:
    import PyPDF2
except Exception:
    PyPDF2 = None

MAX_FILE_CHARS = 60_000          # hard cap per file after parsing
CHUNK_SIZE_CHARS = 2_000         # size for scoring chunks
TOP_K_CHUNKS_PER_FILE = 5        # keep top-k chunks per file
MAX_TOTAL_EXCERPTS = 20          # cross-file cap to limit tokens

def _norm(s: str) -> str:
    s = str(s or "")
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _score_overlap(query: str, text: str) -> float:
    """Simple blended score using sequence similarity and token overlap."""
    from math import sqrt
    q = _norm(query).lower()
    t = _norm(text).lower()
    if not q or not t:
        return 0.0
    seq = SequenceMatcher(None, q, t).ratio()
    q_tokens = set(re.findall(r"[a-z0-9]+", q))
    t_tokens = set(re.findall(r"[a-z0-9]+", t))
    if not q_tokens or not t_tokens:
        return seq
    jacc = len(q_tokens & t_tokens) / max(1, len(q_tokens | t_tokens))
    return 0.6 * seq + 0.4 * jacc

def _chunk_text(text: str, size: int = CHUNK_SIZE_CHARS) -> list[str]:
    text = _norm(text)
    if len(text) <= size:
        return [text]
    chunks = [text[i:i+size] for i in range(0, min(len(text), MAX_FILE_CHARS), size)]
    return chunks

def extract_text_from_upload(uploaded_file) -> tuple[str, dict]:
    """
    Return (text, meta). meta contains basic info for referencing.
    Supported: PDF, DOCX, XLSX/XLS, CSV, TXT.
    """
    import pandas as pd
    name = getattr(uploaded_file, "name", "file")
    meta = {"name": name, "size": getattr(uploaded_file, "size", None), "type": ""}
    raw = uploaded_file.read()
    uploaded_file.seek(0)  # reset

    lower = name.lower()
    text = ""
    try:
        if lower.endswith(".pdf") and PyPDF2 is not None:
            meta["type"] = "pdf"
            reader = PyPDF2.PdfReader(io.BytesIO(raw))
            pages = []
            for i, p in enumerate(reader.pages):
                try:
                    pages.append(p.extract_text() or "")
                except Exception:
                    pages.append("")
            text = "\n\n".join([f"[Page {i+1}] {t}" for i, t in enumerate(pages)])
        elif lower.endswith(".docx") and docx is not None:
            meta["type"] = "docx"
            d = docx.Document(io.BytesIO(raw))
            paras = [p.text for p in d.paragraphs]
            text = "\n".join(paras)
            # tables (simple)
            for tbl in d.tables:
                for r in tbl.rows:
                    cells = [c.text for c in r.cells]
                    if any(_norm(x) for x in cells):
                        text += "\n" + " | ".join(cells)
        elif lower.endswith(".xlsx") or lower.endswith(".xls"):
            meta["type"] = "excel"
            engine = "openpyxl" if lower.endswith(".xlsx") else "xlrd"
            xls = pd.ExcelFile(io.BytesIO(raw), engine=engine)
            parts = []
            for si, sh in enumerate(xls.sheet_names[:3]):
                try:
                    df = xls.parse(sh)
                except Exception:
                    continue
                if df.empty:
                    continue
                head = df.head(15).fillna("").astype(str)
                sample_txt = head.to_csv(index=False)
                parts.append(f"[Sheet {si+1}: {sh}]\n{sample_txt}")
            text = "\n\n".join(parts)
        elif lower.endswith(".csv"):
            meta["type"] = "csv"
            import pandas as pd
            df = pd.read_csv(io.BytesIO(raw))
            if not df.empty:
                text = df.head(50).to_csv(index=False)
        elif lower.endswith(".txt"):
            meta["type"] = "txt"
            text = raw.decode("utf-8", errors="ignore")
        else:
            meta["type"] = "binary"
            text = ""
    except Exception as e:
        meta["error"] = f"parse_error: {e}"
        text = ""

    if len(text) > MAX_FILE_CHARS:
        text = text[:MAX_FILE_CHARS] + "\n...[truncated]..."
    return text, meta

def build_relevant_excerpts(test_steps: str, docs: list[dict]) -> list[dict]:
    q = _norm(test_steps)
    all_excerpts = []
    for d in docs:
        chunks = _chunk_text(d.get("text", ""))
        scored = [(c, _score_overlap(q, c)) for c in chunks]
        scored.sort(key=lambda x: x[1], reverse=True)
        for c, sc in scored[:TOP_K_CHUNKS_PER_FILE]:
            all_excerpts.append({
                "name": d["meta"].get("name"),
                "type": d["meta"].get("type"),
                "excerpt": c,
                "score": float(sc),
            })
    all_excerpts.sort(key=lambda x: x["score"], reverse=True)
    return all_excerpts[:MAX_TOTAL_EXCERPTS]

def _parse_documents_required(cell_value: str) -> list[str]:
    """
    Turn the 'Documents required' cell (Excel column F) into clean labels.
    Supports bullets, dashes, semicolons, and numbered lists.
    """
    raw = str(cell_value or "")
    # Normalize common bullets/delimiters to newline
    raw = raw.replace("•", "\n").replace("‣", "\n").replace("◦", "\n")
    # Split on newlines and semicolons as fallback
    parts = []
    for line in re.split(r"[\n;]+", raw):
        # Trim typical bullets/numbers like '-', '–', '—', '*', '1.', 'a)', etc.
        line = re.sub(r"^\s*[-–—*•\d\.\)]\s*", "", str(line)).strip()
        if line:
            parts.append(line)
    # Deduplicate while preserving order
    seen, out = set(), []
    for p in parts:
        if p not in seen:
            seen.add(p)
            out.append(p)
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

# Clear Tab 1 analysis artifacts
def _clear_tab1_analysis():
    st.session_state.pop("fin_analysis_text", None)
    st.session_state.pop("fin_analysis_pdf_bytes", None)

# Convert simple Markdown text to PDF bytes 
def md_to_pdf_bytes(md_text: str, title: str = "", author: str = "") -> bytes:
    buf = BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        title=title or "Report",
        author=author or "",
        leftMargin=16 * mm,
        rightMargin=16 * mm,
        topMargin=18 * mm,
        bottomMargin=18 * mm,
    )
    styles = getSampleStyleSheet()
    body = styles["BodyText"]
    h2 = styles["Heading2"]
    h3 = styles["Heading3"]

    elements = []
    bullets_acc = []

    def flush_bullets():
        nonlocal bullets_acc, elements
        if bullets_acc:
            items = [ListItem(Paragraph(b, body)) for b in bullets_acc]
            elements.append(ListFlowable(items, bulletType="bullet", start="•"))
            elements.append(Spacer(1, 6))
            bullets_acc = []

    # Normalize newlines; split lines
    for raw in (md_text or "").splitlines():
        line = raw.rstrip()

        # Empty line → paragraph break
        if not line.strip():
            flush_bullets()
            elements.append(Spacer(1, 6))
            continue

        # Headings (#, ##, ###)
        if line.startswith("### "):
            flush_bullets()
            elements.append(Paragraph(line[4:].strip(), h3))
            continue
        if line.startswith("## "):
            flush_bullets()
            elements.append(Paragraph(line[3:].strip(), h2))
            continue
        if line.startswith("# "):
            flush_bullets()
            elements.append(Paragraph(line[2:].strip(), h2))
            continue

        # Bullets (- , * )
        if line.lstrip().startswith("- ") or line.lstrip().startswith("* "):
            bullets_acc.append(line.lstrip()[2:].strip())
            continue

        # Normal paragraph with basic **bold** support
        htmlish = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", line)
        elements.append(Paragraph(htmlish, body))

    flush_bullets()
    doc.build(elements)
    return buf.getvalue()

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
    label_font_color="#1d4ed8",
    domain_mode: str = "auto",   # "auto" | "minmax" | "whisker"
    pad_ratio: float = 0.08      # ~8% padding around chosen domain
):
   
    import pandas as _pd
    import numpy as _np
    import math
    import plotly.graph_objects as go

    fig = go.Figure()

    # --- normalize inputs
    def f(x):
        try:
            return float(x)
        except Exception:
            return float("nan")

    v, q25, q50, q75 = f(value), f(p25), f(p50), f(p75)
    xs = [x for x in [q25, q50, q75, v] if _pd.notna(x) and math.isfinite(x)]
    if not xs:
        xs = [0.0, 1.0]  # fallback

    # --- choose domain so it always includes the company value and percentiles
    # modes:
    #  - "minmax": [min(all), max(all)]
    #  - "whisker": use Tukey whiskers if IQR available; otherwise minmax
    #  - "auto": start with minmax, then expand to whiskers if quartiles exist
    def have_quartiles():
        return _pd.notna(q25) and _pd.notna(q75) and (q75 >= q25)

    min_x = min(xs)
    max_x = max(xs)

    if domain_mode == "minmax":
        pass  # already min/max of available values
    elif domain_mode == "whisker":
        if have_quartiles():
            iqr = q75 - q25
            min_x = min(min_x, q25 - 1.5 * iqr)
            max_x = max(max_x, q75 + 1.5 * iqr)
    else:  # "auto" (default): minmax, expanded by whiskers if present
        if have_quartiles():
            iqr = q75 - q25
            min_x = min(min_x, q25 - 1.5 * iqr)
            max_x = max(max_x, q75 + 1.5 * iqr)

    # guard against degenerate span
    span = max(1e-12, (max_x - min_x))
    pr = max(0.0, float(pad_ratio))
    x0 = min_x - pr * span
    x1 = max_x + pr * span

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
            fillcolor=color, line_width=0, layer="below"
        )

    # default positions if quartiles missing: split the domain roughly into thirds
    _q25 = q25 if _pd.notna(q25) else (x0 + (x1 - x0) * 0.33)
    _q75 = q75 if _pd.notna(q75) else (x0 + (x1 - x0) * 0.66)

    if lower_is_better:
        rect(x0, _q25, "#def7e5")   # healthy (lower)
        rect(_q25, _q75, "#fff3cd") # satisfactory
        rect(_q75, x1, "#fde2e4")   # needs improvement
    else:
        rect(x0, _q25, "#fde2e4")   # needs improvement
        rect(_q25, _q75, "#fff3cd") # satisfactory
        rect(_q75, x1, "#def7e5")   # healthy

    # --- quartile ticks (also BELOW traces)
    def tick(x, color, width):
        if _pd.notna(x):
            fig.add_shape(
                type="line", x0=x, x1=x, y0=y0, y1=y1,
                line=dict(color=color, width=width),
                layer="below"
            )

    tick(q25, "#9ca3af", qline_width)
    tick(q50, "#6b7280", qline_width)
    tick(q75, "#374151", qline_width)

    # --- quartile labels (annotations are above)
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

    # --- company marker (on TOP of shapes)
    if _pd.notna(v):
        fig.add_trace(
            go.Scatter(
                x=[v], y=[y_mid],
                mode="markers",
                marker=dict(symbol="x", size=cross_size, color="#1d4ed8"),
                name="Company",
                hovertemplate=f"{company_name}<extra></extra>",
                cliponaxis=False  # keep marker visible at the edges
            )
        )

        # white pill label just below the band
        # (clamp within axis so it never disappears)
        def _clamp(val, lo, hi):
            try:
                return max(lo, min(hi, float(val)))
            except Exception:
                return (lo + hi) * 0.5

        label_x = _clamp(v, x0, x1)
        fig.add_annotation(
            x=label_x, y=y0 - 0.06,
            text=company_name or "Company",
            xanchor="center", yanchor="top",
            showarrow=False,
            bgcolor=label_bgcolor,
            bordercolor=label_bordercolor,
            borderwidth=1,
            borderpad=6,
            font=dict(color=label_font_color, size=12)
        )

    # --- axes & layout
    fig.update_xaxes(
        range=[x0, x1],
        showgrid=True, gridcolor="#e5e7eb",
        zeroline=False, tickmode="auto"
    )
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

def _extract_text_any(out_obj):
    try:
        # Newer Responses API convenience field
        txt = getattr(out_obj, "output_text", None)
        if txt and str(txt).strip():
            return str(txt).strip()
    except Exception:
        pass

    # Try common shapes
    try:
        # Chat Completions shape
        if hasattr(out_obj, "choices") and out_obj.choices:
            # 1) Standard chat
            msg = getattr(out_obj.choices[0], "message", None)
            if msg and getattr(msg, "content", None):
                return str(msg.content).strip()
            # 2) Responses-like content parts under choices
            content = getattr(out_obj.choices[0], "content", None)
            if isinstance(content, list) and content:
                for part in content:
                    if isinstance(part, dict) and part.get("type") in ("output_text", "text"):
                        if part.get("text"):
                            return str(part["text"]).strip()
    except Exception:
        pass

    # Responses API "output" list shape (content parts)
    try:
        output = getattr(out_obj, "output", None)
        if isinstance(output, list):
            for item in output:
                content = item.get("content") if isinstance(item, dict) else getattr(item, "content", None)
                if isinstance(content, list):
                    for part in content:
                        # part may be dict or object with attributes
                        if isinstance(part, dict):
                            if part.get("type") in ("output_text", "text") and part.get("text"):
                                return str(part["text"]).strip()
                        else:
                            if getattr(part, "type", None) in ("output_text", "text") and getattr(part, "text", None):
                                return str(getattr(part, "text")).strip()
    except Exception:
        pass

    # Final fallback: stringify object (for debugging)
    try:
        return str(out_obj).strip()
    except Exception:
        return ""

def _call_openai(system_prompt, user_prompt, api_key, model=None, max_tokens=800):
        if not api_key:
            return None, "OpenAI API key is not set."

        try:
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            mdl = (model or os.environ.get("OPENAI_MODEL") or "gpt-5").strip()

            # --- Try Responses API first ---
            try:
                out = client.responses.create(
                    model=mdl,
                    input=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user",   "content": user_prompt},
                    ],
                    # Correct for Responses API:
                    response_format={"type": "text"},         # force text section
                    reasoning={"effort": "low"},              # reduce thought token usage
                    max_output_tokens=max_tokens,
                )
                txt = _extract_text_any(out)
                if txt:
                    return txt, None
            except TypeError:
                # Retry without max_output_tokens if SDK build rejects it
                try:
                    out = client.responses.create(
                        model=mdl,
                        input=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user",   "content": user_prompt},
                        ],
                    )
                    txt = _extract_text_any(out)
                    if txt:
                        return txt, None
                except Exception:
                    pass
            except Exception:
                pass  # fall through to Chat Completions

            # --- Fallback: Chat Completions ---
            resp = client.chat.completions.create(
                model=mdl,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_prompt},
                ],
                # Correct for Chat Completions:
                max_tokens=max_tokens,
            )
            txt2 = _extract_text_any(resp)
            return (txt2 if txt2 else None), None

        except Exception as e:
            return None, f"OpenAI call failed: {e}"


# --- Select the worst-performing indicator per metric type ---
def _pick_worst_per_type(summary_rows, mtype_df, max_types=None):

    # Build a map from metric column -> Type (case-insensitive, trimmed)
    col_to_type = {}
    for _, r in mtype_df.iterrows():
        c = str(r.get("Metrics_Col", "")).strip()
        t = str(r.get("Type", "")).strip() or "Uncategorized"
        if c:
            col_to_type[c] = t

    def _to_float(x):
        try:
            return float(str(x).replace(",", "").strip())
        except Exception:
            return float("nan")

    def _bucket_rank(b):
        # Needs Improvement > Satisfactory > Healthy
        order = {"Needs Improvement": 3, "Satisfactory": 2, "Healthy": 1}
        return order.get(str(b).strip(), 0)

    def _badness(row):
        grade = str(row.get("Metrics_Grade", "")).strip().lower()
        v = _to_float(row.get("value", row.get("value_str")))
        p50 = _to_float(row.get("p50", row.get("p50_str")))
        if math.isnan(v) or math.isnan(p50):
            return 0.0
        if grade == "lower":
            # higher than p50 is bad; clamp at 0 for better-than-median
            return max(v - p50, 0.0)
        else:
            # lower than p50 is bad
            return max(p50 - v, 0.0)

    # Group rows by Type
    by_type = {}
    for r in summary_rows:
        c = str(r.get("Metrics_Col", "")).strip()
        t = col_to_type.get(c, "Uncategorized")
        r = dict(r)  # shallow copy
        r["Type"] = t
        by_type.setdefault(t, []).append(r)

    # Pick the worst per Type: prefer Needs Improvement; else worst badness
    worst_per_type = []
    for t, items in by_type.items():
        # 1) try Needs Improvement first
        ni = [x for x in items if str(x.get("bucket","")) == "Needs Improvement"]
        if ni:
            cand = max(ni, key=lambda x: (_bucket_rank(x.get("bucket")), _badness(x)))
        else:
            # 2) otherwise pick the single worst by badness (ties broken by bucket rank)
            cand = max(items, key=lambda x: (_badness(x), _bucket_rank(x.get("bucket"))))
        worst_per_type.append(cand)

    # Rank the chosen types globally: most severe first
    worst_per_type.sort(key=lambda x: (_bucket_rank(x.get("bucket")), _badness(x)), reverse=True)

    # Optional cap
    if isinstance(max_types, int) and max_types > 0:
        worst_per_type = worst_per_type[:max_types]

    return worst_per_type

# --- Financial analysis prompt builder ---
def _build_fin_analysis_prompt(company: str, exchange: str, industry: str, fy: str,
                                     metrics_rows: list[dict], currency: str = "") -> tuple[str, str]:

    system = (
        "You are an experienced financial analyst. Write a concise, management-ready financial analysis. "
        "Use clear, non-technical language, no jargon. Prioritize insights on fraud detection, financial analysis, and internal controls. "
        "Prioritize areas with high risk or anomalies. Do not provide investment advice."
    )

    # Compact table-like lines for the model
    lines = []
    for r in metrics_rows:
        name = r.get("Metrics_Name") or r.get("Metrics_Name")  # tolerate key shape
        val  = r.get("value_str", "NA")
        p25  = r.get("p25_str", "NA")
        p50  = r.get("p50_str", "NA")
        p75  = r.get("p75_str", "NA")
        grade= r.get("Metrics_Grade", "")
        bucket = r.get("bucket", "")
        lines.append(f"- {name}: value={val} {currency} | p25={p25} | p50={p50} | p75={p75} | grade={grade} | bucket={bucket}")

    user = (
        f"Company: {company}\nExchange: {exchange}\nIndustry: {industry}\nFY: {fy}\nCurrency: {currency}\n\n"
        "Metrics (company vs industry percentiles):\n" +
        "\n".join(lines) +
        "\n\nWrite:\n"
        "1) Executive summary (3-5 bullets).\n"
        "2) Key strengths (tie explicitly to metrics and buckets: Healthy).\n"
        "3) Key pressure points (tie explicitly to metrics and buckets: Needs Improvement / Satisfactory).\n"
        "4) Assumptions/limitations (missing or NA values).\n"
        "Avoid acronyms unless already present in metric names. Keep to ~250-350 words. Bold all subheadings."
    )
    return system, user


def _build_audit_prompt(company, exchange, industry, fy, summary_rows, mtype_df,
                                       max_types=3, counts_only=False):
    system = (
        "You are an experienced internal auditor. Your expertise includes fraud detection, financial analysis, and internal controls. "
        "Given company data and industry benchmarks, identify the top 3 control areas for internal audit focus. "
        "Prioritize areas with high risk or anomalies. Avoid external audit or generic compliance steps."
        "Do not use acronyms or abbreviations in your response. Always write out the full term."
    )

    chosen = _pick_worst_per_type(summary_rows, mtype_df, max_types=max_types)
    lines = [
        f"- {r['Metrics_Name']} (grade={r['Metrics_Grade']})"
        for r in chosen]
    metrics_summary = "\n".join(lines)

    user = (
            f"Company: {company}\n"
            f"Industry: {industry}\n"
            f"Year: {fy}\n"
            f"Weakest Financial Metrics: {metrics_summary}\n\n"
            "Suggest internal control areas for audit based on potential anomalies you infer from common risk patterns in the industry that this industry."
            "For each area, include:\n"
            "1. Risk rationale.\n"
            "2. Suggested audit procedures (exceptions/fraud focus).\n"
            "3. Data required (source systems and fields).\n"
            " Bold the title of each priority area."
            "State assumptions if data is missing."
        )
    return system, user


def _convert_suggestions_to_json(readable_text: str, model: str = None) -> dict:
    api_key = _get_openai_api_key()
    if not api_key or not readable_text:
        return {}

    system_prompt = (
        "You are a meticulous converter. Convert the given auditor-readable notes into a JSON object that strictly "
        "follows the 'work_program' schema described below. Do not add fields. Do not include Markdown or prose. "
        "Only output JSON."
    )
    # Give the schema again for reliability (same keys as your mapper expects)
    schema_hint = (
        "Schema:\n"
        "{\n"
        '  "work_program": {\n'
        '    "company": "string", "exchange": "string", "industry": "string", "fy": "string",\n'
        '    "top_areas": [\n'
        '      {\n'
        '        "scope_name": "string",\n'
        '        "scope_keywords": ["string"],\n'
        '        "sub_process_name": "string",\n'
        '        "sub_process_keywords": ["string"],\n'
        '        "risk": "string",\n'
        '        "control_description": "string",\n'
        '        "procedures": ["string"],\n'
        '        "data_required": ["string"]\n'
        '      }\n'
        '    ]\n'
        "  }\n"
        "}"
    )
    user_prompt = (
        f"{schema_hint}\n\n"
        "Convert the following content into the schema. Use best-effort extraction for the three priority areas. "
        "Return ONLY JSON, no backticks:\n\n"
        f"=== INPUT START ===\n{readable_text}\n=== INPUT END ==="
    )

    text, err = _call_openai(
        system_prompt, user_prompt,
        api_key=api_key,
        model=(model or os.environ.get("OPENAI_MODEL", "gpt-5")),
        max_tokens=900
    )
    if err or not text:
        return {}

    try:
        payload = text.strip()
        # Strip any accidental fences or pre/post text
        if payload.startswith("```"):
            payload = payload.strip("`")
            payload = payload[payload.find("{"): payload.rfind("}") + 1]
        return json.loads(payload)
    except Exception:
        return {}


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

# ---- Normalization & tokenization ----
def _norm_text(s: str | None) -> str:
    if s is None:
        return ""
    s = str(s).lower()
    s = re.sub(r"[^a-z0-9\s/\-+&]", " ", s)  # keep some useful symbols
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _tokens(s: str) -> set[str]:
    return set(_norm_text(s).split())

def _bigrams(s: str) -> set[tuple[str, str]]:
    toks = _norm_text(s).split()
    return set(zip(toks, toks[1:])) if len(toks) > 1 else set()

def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / max(1, len(a | b))

def _seq_ratio(a: str, b: str) -> float:
    return SequenceMatcher(None, _norm_text(a), _norm_text(b)).ratio()

# ---- Optional synonym map from an Excel "Synonyms" sheet (Type, From, To) ----
@st.cache_data(show_spinner=False)
def load_synonyms_map(path: str = AUDIT_DB_FILE) -> dict[tuple[str, str], str]:
    try:
        xl = pd.ExcelFile(path, engine="openpyxl")
        if "Synonyms" not in xl.sheet_names:
            return {}
        df = xl.parse("Synonyms")
        # expected columns: Type (Scope/Sub-process), From, To
        out = {}
        for _, r in df.iterrows():
            typ = str(r.get("Type", "")).strip()
            frm = _norm_text(r.get("From", ""))
            to  = str(r.get("To", "")).strip()
            if typ and frm and to:
                out[(typ, frm)] = to
        return out
    except Exception:
        return {}

# ---- Scoring (label + optional keywords vs candidate + context) ----
def _score_label_to_candidate(label: str, candidate: str,
                              extra_keywords: list[str] | None = None,
                              candidate_context: list[str] | None = None) -> float:
    ek = " ".join(extra_keywords or [])
    comp_a = f"{label} {ek}".strip()
    ctx = " ".join(candidate_context or [])
    # base similarities
    s1 = _seq_ratio(comp_a, candidate)
    j1 = _jaccard(_tokens(comp_a), _tokens(candidate))
    j2 = _jaccard(_bigrams(comp_a), _bigrams(candidate))
    # optional context boost
    j_ctx = _jaccard(_tokens(comp_a), _tokens(ctx)) * 0.5 if ctx else 0.0
    # weighted blend (tune to taste)
    score = 0.50 * s1 + 0.30 * j1 + 0.15 * j2 + 0.05 * j_ctx
    return float(score)

# ---- Scope candidate ranking ----
def _rank_scopes(area: dict, audit_df: pd.DataFrame, synonyms: dict | None = None) -> list[tuple[str, float]]:
    label = area.get("scope_name") or area.get("scope") or ""
    kws   = area.get("scope_keywords", []) or []
    if synonyms:
        label = synonyms.get(("Scope", _norm_text(label)), label)

    scopes = sorted(audit_df["Scope"].dropna().unique().tolist())
    # Context per scope: sub-process names + a sample of risk/controls
    ctx_by_scope = {}
    for s in scopes:
        subnames = audit_df.loc[audit_df["Scope"] == s, "Sub-process"].astype(str).unique().tolist()
        risks    = audit_df.loc[audit_df["Scope"] == s, "Risk"].astype(str).tolist()[:10]
        ctrls    = audit_df.loc[audit_df["Scope"] == s, "Control Description"].astype(str).tolist()[:10]
        ctx_by_scope[s] = ["; ".join(subnames), "; ".join(risks), "; ".join(ctrls)]

    scored = []
    for s in scopes:
        scored.append((s, _score_label_to_candidate(label, s, kws, candidate_context=ctx_by_scope[s])))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored

# ---- Sub-process candidate ranking (within a scope) ----
def _rank_subprocess(area: dict, audit_df: pd.DataFrame, scope: str,
                     synonyms: dict | None = None) -> list[tuple[str, float]]:
    label = area.get("sub_process_name") or area.get("sub_process") or ""
    kws   = area.get("sub_process_keywords", []) or []
    if synonyms:
        label = synonyms.get(("Sub-process", _norm_text(label)), label)

    subs = sorted(audit_df.loc[audit_df["Scope"] == scope, "Sub-process"].dropna().unique().tolist())
    ctx_by_sub = {}
    for sp in subs:
        r = audit_df[(audit_df["Scope"] == scope) & (audit_df["Sub-process"] == sp)]
        risk  = r["Risk"].iloc[0] if not r.empty else ""
        ctrl  = r["Control Description"].iloc[0] if not r.empty else ""
        ctx_by_sub[sp] = [str(risk), str(ctrl)]

    scored = []
    for sp in subs:
        scored.append((sp, _score_label_to_candidate(label, sp, kws, candidate_context=ctx_by_sub[sp])))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored

# ---- Full mapping of all top_areas ----
def map_ai_to_master(ai_json: dict, audit_df: pd.DataFrame,
                     auto_threshold: float = 0.82, review_threshold: float = 0.60,
                     synonyms_map: dict | None = None) -> list[dict]:
    areas = (ai_json or {}).get("work_program", {}).get("top_areas", [])
    mappings = []
    for a in areas:
        scope_cands = _rank_scopes(a, audit_df, synonyms_map)
        scope_match, scope_score = scope_cands[0]
        if scope_score < review_threshold:
            mappings.append({
                "area": a,
                "status": "no_match",
                "reason": "scope_below_threshold",
                "scope_candidates": scope_cands[:5],
            })
            continue
        sub_cands = _rank_subprocess(a, audit_df, scope_match, synonyms_map)
        sub_match, sub_score = sub_cands[0]
        status = "auto" if (scope_score >= auto_threshold and sub_score >= auto_threshold) else "review"

        row = audit_df[(audit_df["Scope"] == scope_match) & (audit_df["Sub-process"] == sub_match)].iloc[0]
        mappings.append({
            "area": a,
            "status": status,
            "mapped_scope": scope_match,
            "mapped_sub_process": sub_match,
            "scope_score": float(scope_score),
            "sub_score": float(sub_score),
            "row_index": int(row.name),
            "scope_candidates": scope_cands[:5],
            "sub_candidates": sub_cands[:5],
        })
    return mappings

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

    # --- Footer ---
    st.sidebar.markdown("<hr>", unsafe_allow_html=True)
    st.sidebar.caption("version 3.2 | 2026")

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
        
    tab_bm, tab_yoy, tab_audit, tab_wp, tab_obs = st.tabs(["1.Benchmarking (Selected FY)", "2.YoY Trend", "3.Suggested Audit Areas",
        "4.Audit Work Program", "5.Observations - Summary"
    ])

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
        # Safer default index handling (avoids ValueError if session value not in list)
        if st.session_state.get("mtype_bm") in type_list:
            default_idx = type_list.index(st.session_state["mtype_bm"])
        else:
            default_idx = 0

        mtype = st.selectbox("Select a metrics type", type_list, index=default_idx, key="mtype_bm", on_change=_clear_tab1_analysis)
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
                # skipped_bm.append(m_name)  # uncomment if you want to track skipped
                continue

            bucket = _classify_bucket(val, p25, p50, p75, m_grade)

            with grid[i % 2]:
                with st.container():
                    st.markdown(f"**{m_name}**")
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

            # keep your LLM assembly
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

        st.divider()
        st.subheader("Financial Metrics - Analysis")

        # Either reuse Tab 3's model via session, or keep a unique input here:
        # Option 1 (reuse): fin_model = st.session_state.get("audit_model", os.environ.get("OPENAI_MODEL", "gpt-5"))
        fin_model = st.text_input("Model", os.environ.get("OPENAI_MODEL", "gpt-5"), key="bm_model")

        currency_label = ccy if 'ccy' in locals() else ""

        btn_fin = st.button(
            "Generate Analysis",
            type="primary",
            key="btn_fin_analysis",
            disabled=(len(assembled_for_llm) == 0)
        )

        if btn_fin:
            api_key_fin = _get_openai_api_key()
            if not api_key_fin:
                st.error("OpenAI API key is missing. Set it in Streamlit secrets or OPENAI_API_KEY.")
            else:
                sys_p, usr_p = _build_fin_analysis_prompt(
                    company=company, exchange=exch, industry=ind, fy=str(fy_sel),
                    metrics_rows=assembled_for_llm, currency=currency_label
                )
                with st.spinner("Calling model and drafting financial analysis..."):
                    fin_text, fin_err = _call_openai(
                        sys_p, usr_p, api_key=api_key_fin, model=fin_model, max_tokens=800
                    )
                if fin_err:
                    st.error(f"API Error: {fin_err}")
                elif fin_text:
                    st.session_state["fin_analysis_text"] = fin_text.strip()
                    st.success("Financial analysis generated.")
                else:
                    st.warning("No output received from the model. Try a smaller prompt or different model.")

        if st.session_state.get("fin_analysis_text"):
            text = st.session_state["fin_analysis_text"]
            st.markdown(text)

            # Generate & cache PDF bytes (optional caching)
            if "fin_analysis_pdf_bytes" not in st.session_state:
                st.session_state["fin_analysis_pdf_bytes"] = md_to_pdf_bytes(
                    text,
                    title=f"{company} – Financial Analysis ({fy_sel})",
                    author="Auto-generated by the dashboard"
                )

            st.download_button(
                "Download analysis (.pdf)",
                data=st.session_state["fin_analysis_pdf_bytes"],
                file_name="financial_analysis.pdf",
                mime="application/pdf",
                key="dl_fin_analysis_pdf"
            )

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

    
    # -------------------------------------------------------------------------
    # TAB 3 — Suggested Audit Areas (Top 3)
    # -------------------------------------------------------------------------
    with tab_audit: 
        st.subheader("Suggested Audit Areas")
        st.caption("Analyzes selected company metrics and industry benchmarks, then suggests auditable areas.")
        
        # --- 1. DATA FILTERING (Fixes the NameError) ---
        sel_mask = (
            (data_df["EXCHANGE"].astype(str) == exch)
            & (data_df["INDUSTRY"].astype(str) == ind)
            & (data_df["FY"].astype(str) == str(fy_sel))
        )
        df_slice = data_df.loc[sel_mask]
        
        # Identify the specific company row
        comp_row = df_slice[df_slice["ENTITY_NAME"].astype(str).str.strip().str.lower() == company.strip().lower()]
        if comp_row.empty and not df_slice.empty:
            comp_row = df_slice.iloc[[0]]
            
        # Identify the benchmark/percentile row
        p_row = _try_get_percentile_row(pct_wide, exch, ind, str(fy_sel))

        # --- 2. PREPARE METRICS FOR THE HELPER ---
        all_metrics_to_rank = []
        if not comp_row.empty:
            for _, r in mtype_df.iterrows():
                c = str(r["Metrics_Col"]).strip()
                n = str(r["Metrics_Name"]).strip()
                g = str(r["Metrics_Grade"]).strip()

                val = pd.to_numeric(comp_row.iloc[0].get(c, np.nan), errors="coerce")
                if pd.isna(val):
                    continue

                p25 = p50 = p75 = np.nan
                if p_row is not None and (c, "p25") in p_row.index:
                    p25 = pd.to_numeric(p_row[(c, "p25")], errors="coerce")
                    p50 = pd.to_numeric(p_row[(c, "p50")], errors="coerce")
                    p75 = pd.to_numeric(p_row[(c, "p75")], errors="coerce")

                bucket = _classify_bucket(val, p25, p50, p75, g)
                
                all_metrics_to_rank.append({
                    "Metrics_Name": n,
                    "Metrics_Col": c,
                    "Metrics_Grade": g,
                    "value": float(val),
                    "value_str": f"{val:.4g}",
                    "p50": float(p50) if pd.notna(p50) else None,
                    "bucket": bucket,
                })

        # --- 3. UI AND GENERATION ---
        model = st.text_input("Input Model :", os.environ.get("OPENAI_MODEL", "gpt-5"), key="audit_model")
        generate = st.button("Generate Audit Suggestions", type="primary")

        if generate:
            if not all_metrics_to_rank:
                st.warning("No problematic metrics found to analyze.")
            else:
                system_prompt, user_prompt = _build_audit_prompt(
                    company, exch, ind, str(fy_sel),
                    all_metrics_to_rank, mtype_df, max_types=3
                )
                api_key_for_call = _get_openai_api_key()
                if not api_key_for_call:
                    st.error("OpenAI API key is missing.")
                else:
                    with st.spinner("Calling OpenAI and analyzing risk areas..."):
                        text, err = _call_openai(
                            system_prompt, user_prompt,
                            api_key=api_key_for_call, model=model, max_tokens=800
                        )

                    if err:
                        st.error(f"API Error: {err}")
                    elif text:
                        # 1) Show readable results on-page
                        st.success("Audit suggestions generated!")
                        st.markdown(text)
                        st.session_state["ai_audit_suggestions"] = text  

                        # 2) Convert to JSON silently and store for Tab 4
                        with st.spinner("Structuring suggestions for Tab 4..."):
                            ai_json = _convert_suggestions_to_json(text, model=model)

                        if ai_json and isinstance(ai_json, dict) and ai_json.get("work_program", {}).get("top_areas"):
                            st.session_state["ai_work_program"] = ai_json   
                            # If we had old mappings from a previous run, reset so Tab 4 remaps:
                            st.session_state.pop("ai_mappings", None)
                            st.caption("Structured JSON saved for Tab 4 filtering (not displayed).")
                        else:
                            st.warning(
                                "Could not derive a structured work program from the readable output. "
                                "Tab 4 will not be filtered until a structured result is available."
                            )
                    else:
                        st.warning(
                            "No suggestions generated. Try switching to `gpt-4o`, reducing the prompt length, "
                            "or lowering `max_tokens` to stay within the model’s context window."
                        )

                        if st.session_state.get("ai_audit_suggestions"):
                            st.markdown(st.session_state["ai_audit_suggestions"])
                            # Build once and cache in session
                            if "ai_audit_suggestions_pdf_bytes" not in st.session_state:
                                st.session_state["ai_audit_suggestions_pdf_bytes"] = md_to_pdf_bytes(
                                    st.session_state["ai_audit_suggestions"],
                                    title=f"{company} – Suggested Audit Areas ({fy_sel})",
                                    author="Auto-generated by the dashboard"
                                )

                            st.download_button(
                                "Download suggestions (.pdf)",
                                data=st.session_state["ai_audit_suggestions_pdf_bytes"],
                                file_name="audit_suggestions.pdf",
                                mime="application/pdf",
                                key="dl_audit_suggestions_pdf"
                            )

                with st.expander("Debug info"):
                    st.code(f"MODEL: {model}\n\nSYSTEM PROMPT:\n{system_prompt}\n\nUSER PROMPT:\n{user_prompt}")

    # -------------------------------------------------------------------------
    # TAB 4 — Audit Work Program
    # -------------------------------------------------------------------------

    with tab_wp:
        st.subheader("Audit Work Program — AI-Assisted Audit Testing")
        # ---- Gate: only proceed if Tab 3 suggestions exist ----
        if "ai_work_program" not in st.session_state:
            st.info("Generate suggested audit areas in Tab 3 to view related audit work program.")
            st.stop()

        # Load master and synonyms
        audit_df = load_audit_db()
        synonyms_map = load_synonyms_map()
        scopes, subs_by_scope = audit_vocab(audit_df)

        # ---- Build or reuse AI -> master mappings (restrict to just 3 suggested areas) ----
        ai_json = st.session_state.get("ai_work_program", {})
        mappings = st.session_state.get("ai_mappings")
        if not mappings:
            # Map AI free-text to the master scope/sub-process
            mappings = map_ai_to_master(ai_json, audit_df, auto_threshold=0.82, review_threshold=0.60,
                                        synonyms_map=synonyms_map)
            st.session_state["ai_mappings"] = mappings

        # Keep only successfully matched areas (auto or review)
        matched = [m for m in (mappings or []) if m.get("status") in ("auto", "review")]
        if not matched:
            st.warning("Suggestions are generated, but could not be mapped to the audit master. "
                    "Please refine Tab 3 output or add synonyms in the 'Synonyms' sheet.")
            # Allow full selection as fallback
            filtered_scopes = scopes
            filtered_subs_by_scope = subs_by_scope
        else:
            # Build filtered vocab from the 3 suggested areas
            pairs = {(m["mapped_scope"], m["mapped_sub_process"]) for m in matched
                    if m.get("mapped_scope") and m.get("mapped_sub_process")}
            filtered_scopes = sorted({s for s, _ in pairs})
            filtered_subs_by_scope = {s: sorted({sp for (ss, sp) in pairs if ss == s})
                                    for s in filtered_scopes}

        # ---- Defaults prefer first matched pair ----
        def _pick_defaults():
            if matched:
                s0 = matched[0].get("mapped_scope")
                sp0 = matched[0].get("mapped_sub_process")
                if s0 in filtered_scopes and sp0 in filtered_subs_by_scope.get(s0, []):
                    return s0, sp0
            # Fallback to the first available in the filtered (or full) list
            s = (filtered_scopes[0] if filtered_scopes else (scopes[0] if scopes else ""))
            sp_list = (filtered_subs_by_scope.get(s) if s in filtered_scopes else subs_by_scope.get(s, [])) if s else []
            sp = (sp_list[0] if sp_list else "")
            return s, sp

        d_scope, d_sub = _pick_defaults()

        # ---- Render the filtered selects ----
        if not filtered_scopes:
            st.info("No mappable areas were found. Use the full catalog below.")
            sel_scope = st.selectbox("1) Scope", scopes, index=(scopes.index(d_scope) if d_scope in scopes else 0),
                                    key="wp_scope")
            sel_sub = st.selectbox(
                "2) Sub-process",
                subs_by_scope.get(sel_scope, []),
                index=(subs_by_scope[sel_scope].index(d_sub)
                    if sel_scope in subs_by_scope and d_sub in subs_by_scope[sel_scope] else 0),
                key="wp_subproc"
            )
        else:
            sel_scope = st.selectbox("1) Scope", filtered_scopes,
                                    index=(filtered_scopes.index(d_scope) if d_scope in filtered_scopes else 0),
                                    key="wp_scope")
            sel_sub = st.selectbox(
                "2) Sub-process",
                filtered_subs_by_scope.get(sel_scope, []),
                index=(filtered_subs_by_scope[sel_scope].index(d_sub)
                    if sel_scope in filtered_subs_by_scope and d_sub in filtered_subs_by_scope[sel_scope] else 0),
                key="wp_subproc"
            )

        # ---- Retrieve row and show Risk/Control (unchanged) ----
        row = audit_df[(audit_df["Scope"] == sel_scope) & (audit_df["Sub-process"] == sel_sub)]
        if row.empty:
            st.error("No work program row found for this selection.")
            st.stop()

        rec = row.iloc[0]
        st.markdown("**Risk**")
        st.info(rec["Risk"])
        st.markdown("**Control Description**")
        st.info(rec["Control Description"])

        # ---- Documents required (column F) -> uploaders (use robust parser) ----
        st.markdown("**Documents required**")
        doc_items = _parse_documents_required(rec["Documents required"])
        uploaded = {}
        for i, lab in enumerate(doc_items):
            uploaded[lab] = st.file_uploader(
                f"Upload: {lab}", type=None, accept_multiple_files=False,
                key=f"doc_{sel_scope}_{sel_sub}_{i}"
            )

        st.markdown("---")

        # ---- OpenAI Draft Runner: open files, extract excerpts, evaluate vs test steps ----
        submit_wp = st.button("Run Audit Test Steps & Draft Observations", type="primary", key=f"run_{sel_scope}_{sel_sub}")

        if submit_wp:
            # 1) Save + parse files
            os.makedirs("uploads", exist_ok=True)
            saved = []
            parsed_docs = []
            for label, file in uploaded.items():
                if file is not None:
                    safe_name = f"{int(time.time())}_{company}_{sel_scope}_{sel_sub}_{os.path.basename(file.name)}".replace(" ", "_")
                    path = os.path.join("uploads", safe_name)
                    with open(path, "wb") as f:
                        f.write(file.getbuffer())
                    text, meta = extract_text_from_upload(file)
                    meta["label"] = label
                    parsed_docs.append({"text": text, "meta": meta})
                    saved.append({"label": label, "path": path, "original_name": file.name, "type": meta.get("type", "")})

            # 2) Build concise evidence pack
            evidence = build_relevant_excerpts(rec["Audit Test Steps"], parsed_docs)
            ev_lines = []
            for j, ev in enumerate(evidence, start=1):
                fname = ev["name"]
                ev_lines.append(f"[snippet {j}] file={fname} · score={ev['score']:.2f}\n{ev['excerpt']}\n")
            evidence_block = "\n".join(ev_lines) if ev_lines else "(no snippets extracted)"

            # 3) Call OpenAI (reuses your existing helpers and model setting)
            model = os.environ.get("OPENAI_MODEL", st.session_state.get("audit_model", "gpt-5"))
            api_key = _get_openai_api_key()
            if not api_key:
                st.error("OpenAI API key is missing. Set it in Streamlit secrets or OPENAI_API_KEY.")
                st.stop()

            system_prompt = (
                "You are an experienced internal auditor. You will be given:\n"
                "1) Company context (scope, sub-process, risk, control);\n"
                "2) Audit Test Steps; and\n"
                "3) Evidence excerpts extracted from uploaded documents.\n\n"
                "Task:\n"
                "- Evaluate the evidence against each audit step.\n"
                "- Identify potential observations (exceptions, control gaps, anomalies or missing evidence).\n"
                "- Classify severity (High / Medium / Low).\n"
                "- Propose a concise root cause and recommendation.\n"
                "- Where applicable, cite the evidence by file name and “snippet#”.\n\n"
                "Return ONLY valid JSON of the shape:\n"
                "{\n"
                "  \"observations\": [\n"
                "    {\"observation\":\"...\", \"severity\":\"High|Medium|Low\", \"root_cause\":\"...\", \"recommendation\":\"...\", \"evidence_refs\":[{\"file\":\"...\",\"snippet_id\":1}]}\n"
                "  ]\n"
                "}\n"
                "If there are no issues, return {\"observations\": []}."
            )

            user_prompt = (
                f"Company: {company}\n"
                f"Scope: {sel_scope}\n"
                f"Sub-process: {sel_sub}\n\n"
                f"Risk:\n{rec['Risk']}\n\n"
                f"Control Description:\n{rec['Control Description']}\n\n"
                f"Audit Test Steps:\n{rec['Audit Test Steps']}\n\n"
                f"Evidence Excerpts (top-ranked):\n{evidence_block}\n\n"
                "Draft observations (if any). Use only the JSON schema specified."
            )

            with st.spinner("Calling OpenAI to analyze evidence and draft observations..."):
                text, err = _call_openai(system_prompt, user_prompt, api_key=api_key, model=model, max_tokens=1200)

            # 4) Parse JSON, store for Tab 5
            observations = []
            if err:
                st.error(f"OpenAI call failed: {err}")
            else:
                try:
                    payload = text.strip()
                    if payload.startswith("```"):
                        payload = payload.strip("`")
                        payload = payload[payload.find("{"): payload.rfind("}")+1]
                    out = json.loads(payload)
                    observations = out.get("observations", [])
                    if not isinstance(observations, list):
                        observations = []
                except Exception as e:
                    st.warning(f"Could not parse model JSON: {e}")
                    observations = []

            if observations:
                st.success("Observations drafted.")
                if "audit_observations" not in st.session_state:
                    st.session_state["audit_observations"] = []
                # Map snippet_id -> original file short name; then to saved path
                snippet_to_file = {}
                for j, ev in enumerate(evidence, start=1):
                    snippet_to_file[j] = ev["name"]

                enriched = []
                for o in observations:
                    refs = o.get("evidence_refs", []) or []
                    files_from_refs = []
                    for r in refs:
                        sn = int(r.get("snippet_id", 0))
                        fname = snippet_to_file.get(sn)
                        if fname:
                            match = next((s for s in saved if s["original_name"] == fname), None)
                            files_from_refs.append(match["path"] if match else fname)
                    if not files_from_refs:
                        files_from_refs = [s["path"] for s in saved]  # fallback

                    enriched.append({
                        "company": company, "exchange": exch, "industry": ind, "fy": str(fy_sel),
                        "scope": sel_scope, "sub_process": sel_sub,
                        "observation": o.get("observation", ""),
                        "severity": o.get("severity", ""),
                        "root_cause": o.get("root_cause", ""),
                        "recommendation": o.get("recommendation", ""),
                        "evidence_links": files_from_refs,
                    })
                st.session_state["audit_observations"].extend(enriched)
            else:
                st.info("No issues drafted by the model; consider adding more evidence or refining steps.")


    # -------------------------------------------------------------------------
    # TAB 5 — Observations
    # -------------------------------------------------------------------------

    with tab_obs:
        st.subheader("Audit Observations")
        obs = st.session_state.get("audit_observations", [])
        if not obs:
            st.info("No observations yet. Run a work program in Tab 4.")
        else:
            df_obs = pd.DataFrame(obs)
            st.dataframe(df_obs, use_container_width=True)

            # Export options (CSV / Excel)
            csv = df_obs.to_csv(index=False).encode("utf-8")
            st.download_button("Download CSV", data=csv, file_name="audit_observations.csv", mime="text/csv")

            # Excel export
            from io import BytesIO
            bio = BytesIO()
            with pd.ExcelWriter(bio, engine="openpyxl") as xw:
                df_obs.to_excel(xw, index=False, sheet_name="Observations")
            st.download_button("Download Excel", data=bio.getvalue(), file_name="audit_observations.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")


# =============================================================================
# Entrypoint
# =============================================================================
if __name__ == "__main__":
    _spawn_streamlit()

