import os
import yaml
import streamlit as st
import streamlit_authenticator as stauth

# --- helpers ---
def to_plain_dict(obj):
    """Recursively convert SecretsDict (or any mapping) to a plain dict."""
    if hasattr(obj, "items"):
        return {k: to_plain_dict(v) for k, v in obj.items()}
    return obj

# ---------- Load YAML ----------
# CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")
# with open(CONFIG_PATH, "r") as f:
#     config = yaml.safe_load(f)

# credentials = config["credentials"]
# cookie_cfg = config["cookie"]
if "credentials" in st.secrets:
    credentials = to_plain_dict(st.secrets["credentials"])
    cookie_cfg  = to_plain_dict(st.secrets.get("cookie", {}))
else:
    with open("config.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    credentials = cfg["credentials"]
    cookie_cfg  = cfg["cookie"]

# Safe defaults + type casts
cookie_name  = cookie_cfg.get("name", "dividend_auth")
cookie_key   = cookie_cfg.get("key",  "CHANGE_ME")
cookie_days  = int(cookie_cfg.get("expiry_days", 7))


# ---------- Authenticator ----------
auth = stauth.Authenticate(
    credentials,     # must be a mutable plain dict
    cookie_name,
    cookie_key,
    cookie_days,
)

# ---------- Your app (protected area) ----------
def render_app():
    # st.title("Dividend Analytics 📈")
    # st.write("You are logged in. Put the existing app here.")
    import pandas as pd
    from functions import backtest_new as backtest, get_data_new as get_data
    from pathlib import Path

    st.set_page_config(page_title="Dividend Analytics", page_icon="📈")

    st.markdown(
        """
    <style>
    .st-emotion-cache-1iyw2y2 {
        height: 60px;
        padding-top: 15px !important;
        padding-bottom: 15px !important;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )

    # ---------- Session state init ----------
    if "running" not in st.session_state:
        st.session_state.running = False
    if "summary_df" not in st.session_state:
        st.session_state.summary_df = None
    if "error_msg" not in st.session_state:
        st.session_state.error_msg = None

    st.image("./assets/logo-dark.png", width=350)
    st.title("Dividend Analytics 📈")

    # ---------- Inputs (disabled when running) ----------
    col1, col2 = st.columns(2)

    with col1:
        symbol = st.text_input(
            "Stock Symbol",
            key="symbol",
            placeholder="e.g. AAPL",
            disabled=st.session_state.running,
        )

    with col2:
        recovery_window = st.number_input(
            "Recovery Window (days)",
            min_value=1,
            value=5,
            step=1,
            key="recovery_window",
            disabled=st.session_state.running,
        )

    # Initialize session state
    if "start_date" not in st.session_state:
        st.session_state.start_date = None
    if "end_date" not in st.session_state:
        st.session_state.end_date = None
    if "disable_year" not in st.session_state:
        st.session_state.disable_year = False

    # Initialize session state variables if they don't exist
    if "start_date" not in st.session_state:
        st.session_state.start_date = None
    if "end_date" not in st.session_state:
        st.session_state.end_date = None

    # Callback function for the start date
    def on_start_date_change():
        if st.session_state.start_date is not None:
            st.session_state.year_disabled = True
        else:
            st.session_state.year_disabled = False

    # Callback function for the reset button
    def on_reset():
        st.session_state.start_date = None
        st.session_state.end_date = None
        st.session_state.year_disabled = False

        st.session_state["symbol"] = ""
        st.session_state["recovery_window"] = 5
        st.session_state["year"] = 1

        st.session_state.summary_df = None
        st.session_state.error_msg = None

    # Set the initial state of the year input
    if "year_disabled" not in st.session_state:
        st.session_state.year_disabled = False

    col1, col2, col3, col4 = st.columns(4, vertical_alignment="bottom")

    with col1:
        year = st.number_input(
            "Years of Backtest",
            min_value=1,
            value=1,
            step=1,
            key="year",
            disabled=st.session_state.year_disabled,
        )

    with col2:
        start_date = st.date_input(
            "Start Date",
            value=st.session_state.start_date,
            key="start_date",
            on_change=on_start_date_change,
            format="DD/MM/YYYY",
        )

    with col3:
        end_date = st.date_input(
            "End Date",
            value=st.session_state.end_date,
            key="end_date",
            format="DD/MM/YYYY",
        )

    with col4:
        st.button("Reset", on_click=on_reset)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        clicked = st.button(
            "Analyze",
            key="run_button",
            type="primary",
            use_container_width=True,
            disabled=st.session_state.running,  # already disabled while running
        )

    # ---------- Immediate disable pattern ----------
    # First click: flip running -> True and immediately rerender (button becomes disabled)
    if clicked and not st.session_state.running:
        st.session_state.running = True
        st.session_state.error_msg = None
        st.rerun()

    # ---------- Running block ----------
    if st.session_state.running:
        # Basic validation to avoid locking the UI with bad inputs
        if not symbol:
            st.session_state.running = False
            st.warning("Please enter a stock symbol.")
            st.rerun()

        progress_bar = st.progress(0, text="Starting analysis...")
        method_code = ["t-1", "t-1-15", "t-1+15", "60/40", "70/30"]
        method_name = [
            "T-1 Close Full Recovery",
            "T-1 Close Minus 15 Ticks",
            "T-1 Close Plus 15 Ticks",
            "Double Tranche 60/40 (T-1C, T-0L)",
            "Double Tranche 70/30 (T-1C, T-0L)",
        ]

        method_dict = dict(zip(method_code, method_name))
        methods = ["t-1+15", "t-1", "t-1-15", "70/30", "60/40"]
        if st.session_state.year_disabled:
            year = None
        results = []

        try:
            data = get_data(
                symbol=symbol,
                year=year,
                recovery_window=recovery_window,
                start=start_date,
                end=end_date,
            )
            for i, method in enumerate(methods, 1):
                progress_bar.progress(
                    i / len(methods), text=f"Calculating {method} method"
                )
                res = backtest(
                    symbol=symbol,
                    method=method,
                    recovery_window=recovery_window,
                    data=data,
                )
                results.append(res)
            results_df = pd.DataFrame(results).round(2)

            order = [
                "T-1 Close Plus 15 Ticks",
                "T-1 Close Full Recovery",
                "T-1 Close Minus 15 Ticks",
                "Double Tranche 70/30 (T-1C, T-0L)",
                "Double Tranche 60/40 (T-1C, T-0L)",
            ]

            results_df["Method"] = results_df["Method"].apply(lambda x: method_dict[x])
            df = results_df.copy()
            df["Method"] = pd.Categorical(df["Method"], categories=order, ordered=True)
            df = df.sort_values("Method").reset_index(drop=True)
            df.index = df.index + 1

            # results_df["Method"] = [
            #     "T-1 Close Full Recovery",
            #     "T-1 Close Minus 15 Ticks",
            #     "T-1 Close Plus 15 Ticks",
            #     "Double Tranche 60/40 (T-1C, T-0L)",
            #     "Double Tranche 70/30 (T-1C, T-0L)",
            # ]
            # st.session_state.summary_df = pd.DataFrame(results)

            # results_df = results_df.sort_index().reset_index(drop=True)
            # st.session_state.summary_df = results_df
            st.session_state.summary_df = df

        except Exception as e:

            # if "date" not in pd.DataFrame(results):
            #     st.session_state.error_msg = f"Error: Tikcer not available"
            # else:
            st.session_state.error_msg = f"Error: {e}"
            st.session_state.summary_df = None

        finally:
            progress_bar.empty()
            st.session_state.running = False
            st.rerun()  # Rerender with button re-enabled and results visible

    # ---------- Results / Errors ----------
    if st.session_state.error_msg:
        # st.dataframe(st.session_state.summary_df)
        st.error(st.session_state.error_msg)

    if st.session_state.summary_df is not None and len(st.session_state.summary_df) > 0:
        st.subheader("Results")
        df = st.session_state.summary_df.copy()

        # Pick only numeric columns
        numeric_cols = df.select_dtypes(include="number").columns

        # Apply formatting only to those
        st.dataframe(
            df.style.format({col: "{:.2f}" for col in numeric_cols}),
            use_container_width=True,
        )
        # st.dataframe(
        #     st.session_state.summary_df.style.format("{:.2f}"), use_container_width=True
        # )


# ---------- Page + Login ----------
st.set_page_config(page_title="Dividend Analytics", page_icon="📈")

# name, auth_status, username = auth.login("Sign in", "main")

fields = {
    "Form name": "Sign in",
    "Username": "Email",
    "Password": "Password",
    "Login": "Sign in",
}

name, auth_status, username = auth.login(
    "main", fields=fields  # location  # new required dict
)

if auth_status is False:
    st.error("Invalid email or password.")
elif auth_status is None:
    st.info("Please enter your email and password to continue.")
elif auth_status:
    # Sidebar user panel
    with st.sidebar:
        st.caption("Signed in as:")
        st.subheader(credentials["usernames"][username]["name"])
        st.text(username)  # email as username
        auth.logout("Logout", "sidebar")

        # Optional: self-serve password reset (persists to YAML)
        with st.expander("Account"):
            try:
                if auth.reset_password(username=username, location="main"):
                    st.success("Password updated.")
                    # Persist the new hash back to YAML:
                    # `auth.credentials` now contains updated hashes
                    config["credentials"] = auth.credentials
                    try:
                        with open(CONFIG_PATH, "w") as f:
                            yaml.safe_dump(config, f, sort_keys=False)
                        st.caption("New password saved to config.yaml.")
                    except Exception as e:
                        st.warning(
                            f"Password changed for this session. Could not write config.yaml: {e}"
                        )
            except Exception as e:
                st.warning(f"Password reset not available: {e}")

    render_app()
