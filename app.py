import os
import yaml
import pandas as pd
import streamlit as st
import streamlit_authenticator as stauth
from functions import backtest_massive as backtest, get_data_massive as get_data
import traceback  
from datetime import datetime


# ---------- Page Config ----------
st.set_page_config(page_title="Dividend Analytics", page_icon="📈", layout="wide")

# --- Helpers ---
def to_plain_dict(obj):
    if hasattr(obj, "items"):
        return {k: to_plain_dict(v) for k, v in obj.items()}
    return obj

# ---------- Load Configuration ----------
if "credentials" in st.secrets:
    credentials = to_plain_dict(st.secrets["credentials"])
    cookie_cfg  = to_plain_dict(st.secrets.get("cookie", {}))
else:
    try:
        with open("config.yaml", "r") as f:
            cfg = yaml.safe_load(f)
        credentials = cfg["credentials"]
        cookie_cfg  = cfg["cookie"]
    except FileNotFoundError:
        st.error("Configuration file not found. Please check secrets or config.yaml.")
        st.stop()

cookie_name  = cookie_cfg.get("name", "dividend_auth")
cookie_key   = cookie_cfg.get("key",  "CHANGE_ME")
cookie_days  = int(cookie_cfg.get("expiry_days", 7))

# ---------- Authenticator ----------
auth = stauth.Authenticate(
    credentials,
    cookie_name,
    cookie_key,
    cookie_days,
)

# ---------- Custom UI Styling ----------
def local_css():
    st.markdown("""
        <style>
        /* Main Container Background */
        .stApp {
            background-color: white;
        }
        
        /* Header Styling */
        .dividend-header {
            font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
            font-weight: 800;
            letter-spacing: 2px;
            color: #1E1E1E;
            text-transform: uppercase;
            margin-top: 10px;
        }

        /* Input Field Styling */
        .stTextInput input, .stNumberInput input, .stDateInput input {
            background-color: #F0F2F6 !important;
            border-radius: 8px !important;
            border: none !important;
        }

        /* TARGET ONLY PRIMARY BUTTON (Analyze) */
        div.stButton > button[kind="primary"] {
            background-color: #FF4B4B !important;
            color: white !important;
            border: none !important;
            padding: 12px 24px !important;
            border-radius: 8px !important;
            font-weight: bold !important;
            width: 100%;
        }
        
        /* Ensure Secondary Button (Reset) keeps Streamlit default look */
        div.stButton > button[kind="secondary"] {
            border-radius: 8px !important;
        }

        /* Sidebar Styling */
        [data-testid="stSidebar"] {
            background-color: #F8F9FB;
            border-right: 1px solid #EDEDED;
        }
        </style>
    """, unsafe_allow_html=True)

# ---------- Protected App Content ----------
def render_app():
    local_css()

    # Session State Initialization
    if "running" not in st.session_state: st.session_state.running = False
    if "summary_df" not in st.session_state: st.session_state.summary_df = None
    if "error_msg" not in st.session_state: st.session_state.error_msg = None
    if "year_disabled" not in st.session_state: st.session_state.year_disabled = False
    if "start_date" not in st.session_state: st.session_state.start_date = None
    if "end_date" not in st.session_state: st.session_state.end_date = None

    # --- Header Section ---
    header_col1, header_col2, header_col3 = st.columns([1, 2, 1])
    with header_col2:
        st.image("./assets/cristo-capital_logo.png", use_container_width=True)
        st.markdown("<h1 class='dividend-header' style='text-align: center;'>DIVIDEND ANALYTICS</h1>", unsafe_allow_html=True)

    st.write("##")

    # --- Input Grid ---
    # Row 1
    r1c1, r1c2 = st.columns(2)
    with r1c1:
        symbol = st.text_input("Stock Symbol", placeholder="e.g. AAPL", key="symbol_input", disabled=st.session_state.running)
    with r1c2:
        recovery_window = st.number_input("Recovery Window (days)", min_value=1, value=5, step=1, disabled=st.session_state.running)

    # Row 2
    r2c1, r2c2, r2c3, r2c4 = st.columns([1, 1, 1, 0.5], vertical_alignment="bottom")
    
    def on_start_date_change():
        st.session_state.year_disabled = st.session_state.start_date is not None

    def on_reset():
        # Clear specific keys
        st.session_state.start_date = None
        st.session_state.end_date = None
        st.session_state.year_disabled = False
        st.session_state.summary_df = None
        st.session_state.error_msg = None
        if "symbol_input" in st.session_state: 
            st.session_state.symbol_input = ""

    with r2c1:
        year = st.number_input("Years of Backtest", min_value=1, max_value=25, value=1, disabled=st.session_state.year_disabled or st.session_state.running)
    with r2c2:
        start_date = st.date_input(
            "Start Date",
            value=st.session_state.start_date,
            key="start_date",
            format="DD/MM/YYYY",
            min_value=MIN_DATE,
            max_value=MAX_DATE,
            on_change=on_start_date_change,
            disabled=st.session_state.running,
        )
        # start_date = st.date_input("Start Date", value=st.session_state.start_date, key="start_date", format="DD/MM/YYYY", on_change=on_start_date_change, disabled=st.session_state.running)
    with r2c3:
        end_date = st.date_input(
            "End Date",
            value=st.session_state.end_date,
            key="end_date",
            format="DD/MM/YYYY",
            min_value=MIN_DATE,
            max_value=MAX_DATE,
            disabled=st.session_state.running,
        )
        # end_date = st.date_input("End Date", value=st.session_state.end_date, key="end_date", format="DD/MM/YYYY", disabled=st.session_state.running)
    with r2c4:
        # Reset is now a standard secondary button
        st.button("Reset", on_click=on_reset, type="secondary", use_container_width=True)

    # Row 3: Action Button
    st.write("##")
    btn_col1, btn_col2, btn_col3 = st.columns([1.5, 1, 1.5])
    with btn_col2:
        # Analyze is the Primary button
        analyze_clicked = st.button("Analyze", type="primary", disabled=st.session_state.running)

    # --- Analysis Logic ---
    if analyze_clicked and not st.session_state.running:
        if not symbol:
            st.warning("Please enter a stock symbol.")
        else:
            st.session_state.running = True
            st.rerun()

    if st.session_state.running:
        progress_bar = st.progress(0, text="Analyzing data...")
        try:
            eff_year = None if st.session_state.year_disabled else year
            intraday_data, per_day, div_data = get_data(
                symbol=symbol, 
                year=eff_year, 
                start=start_date, 
                end=end_date, 
                recovery_window=recovery_window
            )
            start_ts = pd.Timestamp(start_date) if start_date else None
            if div_data.empty or (start_ts and div_data.index.min() > start_ts):
                st.info(
                    f"ℹ️ No dividend events were returned for **{symbol}** from Massive in the selected period. "
                    "This usually means the stock doesn’t pay dividends or the data source has no dividend coverage."
                )
                st.stop()
            
            methods = ["t-1", 't-1_997', "60/40", "70/30"]
            method_names = {
                "t-1": "T-1 Close Full Recovery",
                "t-1_997": "T-1 Close Full Recovery x 0.997",
                "60/40": "Double Tranche 60/40 (T-1C, T-0L)",
                "70/30": "Double Tranche 70/30 (T-1C, T-0L)"
            }
            
            results = []
            for i, m in enumerate(methods):
                progress_bar.progress((i+1)/len(methods), text=f"Processing {method_names[m]}...")
                res = backtest(method=m, per_day=per_day, intraday_data=intraday_data, div_data=div_data)
                res["Method"] = method_names[m]
                results.append(res)
            
            results_df = pd.DataFrame(results).round(2)
            order = ["T-1 Close Full Recovery", "T-1 Close Full Recovery x 0.997", "Double Tranche 70/30 (T-1C, T-0L)", "Double Tranche 60/40 (T-1C, T-0L)"]
            results_df["Method"] = pd.Categorical(results_df["Method"], categories=order, ordered=True)
            st.session_state.summary_df = results_df.sort_values("Method").reset_index(drop=True)
            st.session_state.error_msg = None
            
        # except Exception as e:
        #     st.session_state.error_msg = f"Analysis Error: {str(e)}"
        except Exception as e:
            st.session_state.error_msg = traceback.format_exc()

        finally:
            st.session_state.running = False
            st.rerun()

    # --- Results Display ---
    if st.session_state.error_msg:
        st.error(st.session_state.error_msg)

    if st.session_state.summary_df is not None:
        st.write("---")
        st.subheader("Analysis Summary")
        df_styled = st.session_state.summary_df.copy()
        numeric_cols = df_styled.select_dtypes(include="number").columns
        st.dataframe(df_styled.style.format({col: "{:.2f}" for col in numeric_cols}), use_container_width=True)

# ---------- Login Handling ----------
fields = {"Form name": "Sign in", "Username": "Email", "Password": "Password", "Login": "Sign in"}
auth.login("main", fields=fields)

auth_status = st.session_state.get("authentication_status")
username    = st.session_state.get("username")
name        = st.session_state.get("name")

from datetime import date

MIN_DATE = date(1900, 1, 1)  # or date(1970, 1, 1)
MAX_DATE = datetime.today().date()

if auth_status is False:
    st.error("Invalid email or password.")
elif auth_status is None:
    st.info("Please sign in to access the analytics dashboard.")
elif auth_status:
    with st.sidebar:
        st.write("##")
        st.caption("Signed in as:")
        st.markdown(f"### {name}")
        st.caption(username)
        st.write("---")
        auth.logout("Logout", "sidebar")

    render_app()