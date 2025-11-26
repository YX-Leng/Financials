
import os
import sys
import subprocess
import webbrowser
import time
import socket
from io import BytesIO

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
    if _running_inside_streamlit():
        render_ui()
        return
    port = os.environ.get("PORT", "8501")
    env = os.environ.copy()
    env["STREAMLIT_RUN"] = "1"
    cmd = [
        sys.executable, "-m", "streamlit", "run", os.path.abspath(__file__),
        "--server.port", port,
        "--server.headless", "true",
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
        print("[WARN] Streamlit not found; rendering UI inline.")
        os.environ["STREAMLIT_RUN"] = "1"
        render_ui()

def render_ui():
    import streamlit as st
    import pandas as pd
    import plotly.graph_objects as go

    # --- Session defaults ---
    for key in ("company_name", "industry", "financial_year", "metrics_ready", "metrics"):
        if key not in st.session_state:
            st.session_state[key] = None if key != "metrics_ready" else False

    # --- Page config & theme ---
    st.set_page_config(page_title="Company Benchmarking Dashboard", layout="wide")
    DASHBOARD_CSS = """
    <style>
    .block-container { padding-top: 2.0rem; padding-bottom: 1.6rem; }
    h1, h2, h3, .stMarkdown h1 { margin-top: 0 !important; }
    .metric-card {
      background: #ffffff;
      border: 1px solid #e5e7eb;
      border-radius: 10px;
      padding: 10px 12px;
      color: #0f172a;
      box-shadow: 0 2px 6px rgba(0,0,0,0.06);
      height: 150px;
      display: flex; flex-direction: column; justify-content: flex-start;
      overflow: hidden;
    }
    .metric-title { font-size: 12px; font-weight: 600; color: #334155; margin-bottom: 6px; }
    .metric-value { font-size: 18px; font-weight: 700; color: #0f172a; margin-bottom: 6px; }
    .metric-desc { font-size: 12px; color: #64748b; line-height: 1.35; display: block; margin-top: 4px; }
    .chip { display: inline-block; padding: 3px 8px; border-radius: 999px; font-size: 11px; font-weight: 600; border: 1px solid rgba(0,0,0,0.05); width: fit-content; }
    .chip-red { background: #fde2e4; color: #8b1d29; }
    .chip-amber { background: #fff3cd; color: #8a6d3b; }
    .chip-green { background: #def7e5; color: #166534; }
    .section-title { font-size: 18px; font-weight: 700; margin: 8px 0 10px 0; color: #0f172a; }
    hr.section { border: 0; height: 1px; background: #e5e7eb; margin: 8px 0 16px 0; }
    </style>
    """
    st.markdown(DASHBOARD_CSS, unsafe_allow_html=True)

    # --- Load Data ---
    @st.cache_data
    def load_data():
        file = pd.read_excel("Company_Financials_By_FY.xlsx")
        industry_agg = pd.read_excel(file, sheet_name="IndustryAggregatesRatios")
        analysis = pd.read_excel(file, sheet_name="Analysis")
        return industry_agg, analysis
    industry_agg, analysis = load_data()

    # --- Sidebar: User Inputs ---
    st.sidebar.header("Company Input")
    company_name = st.sidebar.text_input("Company Name")
    financial_year = st.sidebar.selectbox("Financial Year", [2021, 2022, 2023, 2024, 2025])
    industry_list = sorted(industry_agg['Industry'].unique())
    industry = st.sidebar.selectbox("Industry", industry_list)
    current_assets = st.sidebar.text_input("Current Assets (SGD)")
    current_liabilities = st.sidebar.text_input("Current Liabilities (SGD)")
    inventory = st.sidebar.text_input("Inventory (SGD)")
    operating_cf = st.sidebar.text_input("Operating Cash Flow (SGD)")
    capex = st.sidebar.text_input("Capital Expenditure (SGD)")
    revenue = st.sidebar.text_input("Revenue (SGD)")
    ebitda = st.sidebar.text_input("EBITDA (SGD)")
    cost_of_revenue = st.sidebar.text_input("Cost of Revenue (SGD)")

    def is_number(x):
        try: float(x); return True
        except: return False

    all_filled = all([
        company_name.strip(),
        str(financial_year), industry,
        current_assets.strip(), current_liabilities.strip(), inventory.strip(),
        operating_cf.strip(), capex.strip(), revenue.strip(),
        ebitda.strip(), cost_of_revenue.strip()
    ]) and all([
        is_number(current_assets), is_number(current_liabilities), is_number(inventory),
        is_number(operating_cf), is_number(capex), is_number(revenue),
        is_number(ebitda), is_number(cost_of_revenue)
    ])
    submit = st.sidebar.button("Submit", disabled=not all_filled)
    if not all_filled:
        st.sidebar.warning("Please complete all fields with valid numbers before submitting.")

    # --- Helpers ---
    def fmt(metric, v):
        if v is None: return "—"
        if metric in ("EBITDA Margin", "Gross Margin", "FCF Margin", "NWC Margin"):
            return f"{v*100:.1f}%"
        if metric in ("Current Ratio", "Quick Ratio"):
            return f"{v:.2f}×"
        if metric == "Days Inventory on Hand":
            return f"{v:.0f} days"
        return f"{v:.2f}"

    metric_info = {
        "Current Ratio": ("Short-term liquidity — above 1 is generally healthy.", "High is better"),
        "Quick Ratio": ("Liquidity excluding inventory.", "High is better"),
        "EBITDA Margin": ("Operating profitability as % of revenue.", "High is better"),
        "Gross Margin": ("Gross profit as % of revenue.", "High is better"),
        "Days Inventory on Hand": ("Average days inventory is held.", "Low is better"),
        "FCF Margin": ("Free cash flow as % of revenue.", "High is better"),
        "NWC Margin": ("Net working capital as % of revenue.", "Balanced is best"),
    }

    def pct_bucket(p25, p50, p75, value, metric=None):
        if metric == "Days Inventory on Hand":
            # Reverse logic: low = good, high = bad
            if value < p25:
                return "Good"
            elif value > p75:
                return "Bad"
            else:
                return "Average"
        else:
            if value < p25: return "Bad"
            if value < p75: return "Average"
            return "Good"

    def chip_color_by_direction(pct, direction, metric=None):
        mapping = {
            "Good": "chip-green",
            "Average": "chip-amber",
            "Bad": "chip-red"
        }
        return mapping.get(pct, "chip-amber")

    def find_percentile_cols(df, base):
        cols = set(df.columns)
        patterns = [
            (f"{base}__p25", f"{base}__p50", f"{base}__p75"),
            (f"{base.replace(' ', '_')}__p25", f"{base.replace(' ', '_')}__p50", f"{base.replace(' ', '_')}__p75"),
            (f"{base}_p25", f"{base}_p50", f"{base}_p75"),
        ]
        for p25c, p50c, p75c in patterns:
            if p25c in cols and p50c in cols and p75c in cols:
                return p25c, p50c, p75c
        return None, None, None

    def get_benchmark(metric, value):
        row = industry_agg[(industry_agg['Industry'] == industry) & (industry_agg['Financial Year'] == financial_year)]
        if row.empty or value is None:
            return None, None, None, None
        p25c, p50c, p75c = find_percentile_cols(row, metric)
        if not all([p25c, p50c, p75c]):
            return None, None, None, None
        p25, p50, p75 = row[p25c].values[0], row[p50c].values[0], row[p75c].values[0]
        pct = pct_bucket(p25, p50, p75, value, metric)
        return p25, p50, p75, pct

    def plot_benchmark(metric, value, p25, p50, p75):
        import plotly.graph_objects as go
        if any(x is None for x in [p25, p50, p75, value]):
            return go.Figure()
        x_min = min(0, p25, value, p50, p75)
        x_max = max(p75, value, p50)
        pad = (x_max - x_min) * 0.05
        x_min, x_max = x_min - pad, x_max + pad
        fig = go.Figure()
        if metric == "Days Inventory on Hand":
            # Reverse logic: p25 = green (good), p75 = red (bad)
            fig.add_shape(type="rect", x0=x_min, x1=p25, y0=0, y1=1,
                          fillcolor="#def7e5", opacity=0.55, line_width=0)   # Green
            fig.add_shape(type="rect", x0=p25, x1=p75, y0=0, y1=1,
                          fillcolor="#fff3cd", opacity=0.55, line_width=0)   # Amber
            fig.add_shape(type="rect", x0=p75, x1=x_max, y0=0, y1=1,
                          fillcolor="#fde2e4", opacity=0.55, line_width=0)   # Red
        else:
            fig.add_shape(type="rect", x0=x_min, x1=p25, y0=0, y1=1,
                          fillcolor="#fde2e4", opacity=0.55, line_width=0)   # Red
            fig.add_shape(type="rect", x0=p25, x1=p75, y0=0, y1=1,
                          fillcolor="#fff3cd", opacity=0.55, line_width=0)   # Amber
            fig.add_shape(type="rect", x0=p75, x1=x_max, y0=0, y1=1,
                          fillcolor="#def7e5", opacity=0.55, line_width=0)   # Green
        # Company marker
        fig.add_vline(x=value, line_width=3, line_color="#1d4ed8",
                      annotation_text="Company", annotation_position="top right")
        fig.update_xaxes(title=None, range=[x_min, x_max], showgrid=True,
                         gridcolor="#e5e7eb", zeroline=False)
        fig.update_yaxes(visible=False)
        fig.update_layout(height=100, margin=dict(l=6, r=6, t=4, b=4),
                          plot_bgcolor="#ffffff", paper_bgcolor="#ffffff")
        return fig

    # --- YoY helpers (UPDATED) ---
    YEARS = [2021, 2022, 2023, 2024, 2025]
    def get_yoy_benchmark(industry_name, metric):
        cols = set(industry_agg.columns)
        base = metric
        candidates = [
            (f"{base}__p25", f"{base}__p50", f"{base}__p75"),
            (f"{base.replace(' ', '_')}__p25", f"{base.replace(' ', '_')}__p50", f"{base.replace(' ', '_')}__p75"),
            (f"{base}_p25", f"{base}_p50", f"{base}_p75"),
        ]
        p25_col = p50_col = p75_col = None
        for p25c, p50c, p75c in candidates:
            if p25c in cols and p50c in cols and p75c in cols:
                p25_col, p50_col, p75_col = p25c, p50c, p75c
                break
        if not all([p25_col, p50_col, p75_col]):
            return None
        df = industry_agg[industry_agg['Industry'] == industry_name]
        df = df[df['Financial Year'].isin(YEARS)].copy()
        df['YearText'] = df['Financial Year'].astype(str)
        df = df[['Financial Year', 'YearText', p25_col, p50_col, p75_col]].rename(
            columns={p25_col: 'p25', p50_col: 'p50', p75_col: 'p75'}
        ).sort_values('Financial Year')
        return df

    def plot_yoy_trend(df, metric, company_year=None, company_value=None):
        fig = go.Figure()
        if df is None or df.empty:
            fig.update_layout(title=f"{metric} YoY Benchmark (no percentile columns found)", height=240)
            return fig
        # Use categorical x-axis to avoid fractional positions
        if metric == "Days Inventory on Hand":
            # p25 = green (good), p50 = amber, p75 = red (bad)
            fig.add_trace(go.Scatter(x=df['YearText'], y=df['p25'], name='p25', line=dict(color='green')))
            fig.add_trace(go.Scatter(x=df['YearText'], y=df['p50'], name='p50', line=dict(color='orange', dash='dot')))
            fig.add_trace(go.Scatter(x=df['YearText'], y=df['p75'], name='p75', line=dict(color='red')))
        else:
            # p25 = orange, p50 = red, p75 = green
            fig.add_trace(go.Scatter(x=df['YearText'], y=df['p25'], name='p25', line=dict(color='orange')))
            fig.add_trace(go.Scatter(x=df['YearText'], y=df['p50'], name='p50', line=dict(color='red', dash='dot')))
            fig.add_trace(go.Scatter(x=df['YearText'], y=df['p75'], name='p75', line=dict(color='green')))
        if company_year is not None and company_value is not None:
            fig.add_trace(go.Scatter(
                x=[str(company_year)], y=[company_value],
                name='Company', mode='markers',
                marker=dict(size=10, color='blue')
            ))
        fig.update_layout(
            title=f"{metric} YoY Benchmark",
            xaxis_title="Year",
            yaxis_title=metric,
            height=240,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        # Explicit ticks: ensure exactly the five years appear
        fig.update_xaxes(
            type='category',
            tickmode='array',
            tickvals=[str(y) for y in YEARS],
            ticktext=[str(y) for y in YEARS]
        )
        return fig


    # ---- Compute once, persist to session_state ----
    if submit:
        current_assets = float(current_assets)
        current_liabilities = float(current_liabilities)
        inventory = float(inventory)
        operating_cf = float(operating_cf)
        capex = float(capex)
        revenue = float(revenue)
        ebitda = float(ebitda)
        cost_of_revenue = float(cost_of_revenue)

        def calculate_metrics():
            m = {}
            m['Current Ratio'] = current_liabilities and (current_assets / current_liabilities)
            m['Quick Ratio'] = current_liabilities and ((current_assets - inventory) / current_liabilities)
            m['EBITDA Margin'] = revenue and (ebitda / revenue)
            m['Gross Margin'] = revenue and ((revenue - cost_of_revenue) / revenue)
            m['Days Inventory on Hand'] = cost_of_revenue and ((inventory / cost_of_revenue) * 365)
            fcf = operating_cf - capex
            m['FCF Margin'] = revenue and (fcf / revenue)
            nwc = current_assets - current_liabilities
            m['NWC Margin'] = revenue and (nwc / revenue)
            return m

        st.session_state.company_name = company_name
        st.session_state.industry = industry
        st.session_state.financial_year = financial_year
        st.session_state.metrics = calculate_metrics()
        st.session_state.metrics_ready = True

    # --- Top page title (outside tabs so it never gets cut off) ---
    st.title("Company Benchmarking Dashboard")

    # --- Create tabs ---
    tab_bm, tab_yoy = st.tabs(["Benchmarking (Selected Year)", "YoY Trend (2021–2025)"])

    # --- Tab 1: Benchmarking (single-year) ---
    with tab_bm:
        if not st.session_state.metrics_ready:
            st.info("Fill in the inputs and click Submit to see the benchmarking view.")
        else:
            company_name   = st.session_state.company_name
            industry       = st.session_state.industry
            financial_year = st.session_state.financial_year
            metrics        = st.session_state.metrics

            st.markdown(
                f"**Company:** {company_name}  &nbsp;&nbsp;|&nbsp;&nbsp; "
                f"**Year:** {financial_year}  &nbsp;&nbsp;|&nbsp;&nbsp; "
                f"**Industry:** {industry}"
            )
            st.markdown("<hr class='section'/>", unsafe_allow_html=True)

            cards, chart_figs = [], []
            for metric, (desc, direction) in metric_info.items():
                value = metrics.get(metric)
                p25, p50, p75, pct = get_benchmark(metric, value)
                chip_cls = chip_color_by_direction(pct, direction)
                cards.append({
                    "metric": metric, "desc": desc, "direction": direction,
                    "value": value, "p25": p25, "p50": p50, "p75": p75, "pct": pct,
                    "chip_cls": chip_cls
                })
                fig = plot_benchmark(metric, value, p25, p50, p75)
                chart_figs.append((metric, fig))

            # Store chart_figs in session_state for later use if needed
            st.session_state["chart_figs"] = chart_figs

            cols = st.columns([1,1,1,1,1,1,1])
            for c, d in zip(cols, cards):
                c.markdown(
                    f"""
                    <div class="metric-card">
                      <div class="metric-title">{d['metric']}</div>
                      <div class="metric-value">{fmt(d['metric'], d['value'])}</div>
                      <span class="chip {d['chip_cls']}">{d['pct'] or "—"}</span>
                      <span class="metric-desc">{d['desc']}</span>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            st.markdown("<hr class='section'/>", unsafe_allow_html=True)
            st.markdown("<div class='section-title'>Benchmarking visuals</div>", unsafe_allow_html=True)

            grid_cols = st.columns(2)
            for i, d in enumerate(cards):
                with grid_cols[i % 2]:
                    st.caption(f"**{d['metric']}** — p25 / p50 / p75 (marker = company)")
                    fig = chart_figs[i][1]
                    st.plotly_chart(fig, use_container_width=True)

                    row = analysis[(analysis['Industry'] == industry) & (analysis['Metric Name'] == d['metric'])]
                    if not row.empty and d["pct"]:
                        if d["pct"] == "Below 25th":
                            nar = row['25th Percentile Narrative'].values[0]
                        elif d["pct"] in ("25th-50th", "50th-75th"):
                            nar = row['50th Percentile Narrative'].values[0]
                        else:
                            nar = row['75th Percentile Narrative'].values[0]
                        st.markdown(f"<span class='metric-desc'>{nar}</span>", unsafe_allow_html=True)

    # --- Tab 2: YoY Trend (2021–2025) ---
    with tab_yoy:
        if not st.session_state.metrics_ready:
            st.info("Fill in the inputs and click Submit to view YoY trends.")
        else:
            company_name   = st.session_state.company_name
            industry       = st.session_state.industry
            financial_year = st.session_state.financial_year
            metrics        = st.session_state.metrics

            st.markdown("#### Year-on-Year Industry Benchmark Trend (2021–2025)")
            st.caption("p25 / p50 / p75 industry percentiles per metric over 2021–2025. The blue dot marks your company value for the selected year.")

            # Build chart_figs for this tab
            chart_figs = []
            grid_cols = st.columns(2)
            for idx, (metric, (desc, direction)) in enumerate(metric_info.items()):
                df_yoy = get_yoy_benchmark(industry, metric)
                company_val = metrics.get(metric)
                fig = plot_yoy_trend(df_yoy, metric, company_year=financial_year, company_value=company_val)
                chart_figs.append((metric, fig))

                with grid_cols[idx % 2]:
                    st.caption(f"**{metric}** — industry percentiles over time + company marker")
                    st.plotly_chart(fig, use_container_width=True)

            # Store chart_figs in session_state for later use if needed
            st.session_state["yoy_chart_figs"] = chart_figs


# --- Entrypoint ---
if __name__ == "__main__":
    _spawn_streamlit()
