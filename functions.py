import pandas as pd
import requests
from datetime import datetime, time, timedelta
import numpy as np
from matplotlib import rcParams
from collections import Counter
from massive import RESTClient

MASSIVE_API_KEY = 'D914R_H0qcJGRGduOrUgVCzMZ5jln5T2'

import seaborn as sns
import matplotlib.pyplot as plt

sns.set_style("darkgrid")

custom_style = {
    "font.family": "Avenir",
}

rcParams.update(custom_style)

import warnings

warnings.filterwarnings("ignore")

API_KEY = "oKA3BVZgLfafjoDY5g5QXif92Pb5Z2s4"


def year_delta(year):
    today = datetime.today().date()
    one_year_before = today.replace(year=today.year - year)
    return one_year_before


def value_counts_list(
    items,
    *,
    normalize=False,
    sort=True,
    ascending=False,
    dropna=True,
    topn=None,
    na_label="<NA>",
):
    """pd.value_counts-like for Python iterables, returns list of (value, count or %)"""
    NA = object()

    def is_na(x):
        try:
            return x is None or x != x
        except Exception:
            return False

    counts = Counter()
    for x in items:
        if is_na(x):
            if dropna:
                continue
            x = NA
        counts[x] += 1

    pairs = list(counts.items())

    if sort:
        pairs.sort(key=lambda kv: kv[1], reverse=not ascending)

    if topn is not None:
        pairs = pairs[:topn]

    if normalize:
        denom = sum(c for _, c in pairs) or 1
        pairs = [(k, round((c / denom) * 100, 2)) for k, c in pairs]  
    else:
        pairs = [(k, round(c, 2)) for k, c in pairs]  

    out = [(na_label if k is NA else k, v) for k, v in pairs]
    return out

# --- NEW: Granular 7-block session ---
def get_trading_session(timestamp):
    if isinstance(timestamp, str):
        dt = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
        input_time = dt.time()
    else:
        input_time = pd.to_datetime(timestamp).time()

    sessions = {
        "T0 Rec 9:30 to 10:00": (time(9, 30), time(10, 0)),
        "T0 Rec 10:00 to 11:00": (time(10, 0), time(11, 0)),
        "T0 Rec 11:00 to 12:00": (time(11, 0), time(12, 0)),
        "T0 Rec 12:00 to 1:00": (time(12, 0), time(13, 0)),
        "T0 Rec 1:00 to 2:00": (time(13, 0), time(14, 0)),
        "T0 Rec 2:00 to 3:00": (time(14, 0), time(15, 0)),
        "T0 Rec 3:00 to 4:00": (time(15, 0), time(16, 1)), # Extended slightly past 4PM to catch exact 16:00:00 closures
    }

    for session, (start, end) in sessions.items():
        if start <= input_time < end:
            return session

    return "Outside Market Hours"


def get_hist_massive(symbol, start, API_KEY=MASSIVE_API_KEY, multiplier=1, end=None, timespan="minute"):
    client = RESTClient(API_KEY)
    if end is None:
        end = datetime.now()

    aggs = list(
        client.list_aggs(
            ticker=symbol,
            multiplier=multiplier,
            timespan=timespan,
            from_=start,
            to=end,
            adjusted=True,
            sort="asc",
            limit=50000,
        )
    )
    if not aggs:
        raise ValueError(f"No aggregate data returned for {symbol}")

    df = pd.DataFrame(aggs)
    if "timestamp" not in df.columns:
        raise KeyError("timestamp missing from Massive response; cannot build intraday data")

    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    intraday_data = df.set_index("timestamp")[["open", "high", "low", "close"]].sort_index()
    
    def intraday_adjustments(day_df):
        # NEW: Stop trimming the first 15 rows. The client explicitly wants opening minutes to count.
        trimmed = day_df 
        if trimmed.empty:
            return pd.Series(
                {"open": pd.NA, "close": pd.NA, "low": pd.NA, "high": pd.NA}
            )
        return pd.Series(
            {
                "open": trimmed.open.iloc[0],
                "close": trimmed.close.iloc[-1],
                "low": trimmed.low.min(),
                "high": trimmed.high.max(),
            }
        )

    per_day = (
        intraday_data
        .groupby(intraday_data.index.normalize())
        .apply(intraday_adjustments)
    )
    
    return intraday_data, per_day

def get_div_massive(symbol, API_KEY=MASSIVE_API_KEY):
    client = RESTClient(API_KEY)
    dividends = list(
        client.list_dividends(
            ticker=symbol, order="asc", limit=1000, sort="ex_dividend_date"
        )
    )

    if not dividends:
        df_empty = pd.DataFrame({"is_dividend": []})
        df_empty.index = pd.to_datetime([])
        df_empty.index.name = "ex_dividend_date"
        return df_empty

    df = (
        pd.DataFrame(dividends)
        .set_index("ex_dividend_date")[["id"]]
        .rename(columns={"id": "is_dividend"})
    )
    df.index = pd.to_datetime(df.index)
    return df

def get_data_massive(symbol, year, start=None, end=None, recovery_window=5):
    
    if year:
        start = year_delta(year)
        end = datetime.today().date()
    
    intraday_data, per_day = get_hist_massive(symbol=symbol, start=start, end=end)
    div_data = get_div_massive(symbol=symbol)
    per_day['entry_date'] = pd.to_datetime(per_day.index.to_series().shift(1))
    per_day["entry_price_t_1"] = per_day["close"].shift(1)
    per_day["entry_price_997"] = 0.997 * per_day["close"].shift(1)
    per_day["entry_price_6040"] = 0.6 * per_day["close"].shift(1) + 0.4 * per_day["low"]
    per_day["entry_price_7030"] = 0.7 * per_day["close"].shift(1) + 0.3 * per_day["low"]
    
    per_day['entry_price_1005'] = 1.005 * per_day["close"].shift(1)
    per_day['entry_price_10033'] = 1.0033 * per_day["close"].shift(1)
    per_day['entry_price_0999'] = 0.999 * per_day["close"].shift(1)
    per_day['entry_price_0997'] = 0.997 * per_day["close"].shift(1)
    
    per_day["end_window_date"] = per_day.index.to_series().shift(-recovery_window)
    per_day["prev_two_days"] = pd.to_datetime(per_day.index.to_series().shift(2))
    
    return intraday_data, per_day, div_data


def backtest_massive(method, per_day, intraday_data, div_data):
    events = pd.merge(per_day, div_data, left_index=True, right_index=True, how='left').dropna(subset=['is_dividend', 'end_window_date'])
    
    def get_recovery_time_massive(events):
        recovery_times = []
        for event_date, event in events.iterrows():
            entry_price = event[entry_price_pairs[method]]
            start_date = event_date.date()
            end_date = event['end_window_date'].date()
            window_date = intraday_data.loc[str(start_date):str(end_date)]
            window_date['is_recover'] = window_date.close > entry_price
            recovery_time = (window_date.resample('D').sum().mean().round(2).is_recover)
            recovery_times.append(recovery_time)
        return np.round(np.mean(recovery_times), 2)

    processed_events = 0
    failed_events = 0

    is_recover_list = []
    recovery_days_list = []
    overrun_list = []
    days_to_overrun_list = []
    trading_session_list = []
    cummulative_recovery_list = []
    loss_delta_pct_list = []
    days_to_loss_delta_list = []
    
    entry_price_pairs = {
        "t-1": "entry_price_t_1",
        't-1_997': 'entry_price_997',
        "60/40": "entry_price_6040",
        "70/30": "entry_price_7030",
        "t-1_1005": "entry_price_1005",
        "t-1_10033": "entry_price_10033",
        "t-1_0999": "entry_price_0999",
        "t-1_0997": "entry_price_0997"
    }

    for event_date, event in events.iterrows():
        entry_price = event[entry_price_pairs[method]]
        end_window = event["end_window_date"]

        window_data = per_day.loc[event_date:end_window].copy()
        if window_data.empty:
            continue

        processed_events += 1

        highs = pd.to_numeric(window_data["high"], errors="coerce")
        entry = pd.to_numeric(entry_price, errors="coerce")
        window_data["is_recover"] = highs > entry
        is_recover = bool(window_data["is_recover"].any())
        
        # --- NEW CLIENT REQUEST: FIRST TOUCH T0 RECOVERY ---
        # Scan intraday data specifically on T0 (Ex-Div Date)
        t0_date_str = str(event_date.date())
        try:
            t0_intraday = intraday_data.loc[t0_date_str]
            # Client specific wording: "equalled the recovery benchmark or higher"
            t0_recoveries = t0_intraday[t0_intraday['high'] >= entry]
            
            if not t0_recoveries.empty:
                # Get the absolute FIRST timestamp that meets criteria
                first_touch_ts = t0_recoveries.index[0] 
                trading_session = get_trading_session(first_touch_ts)
                trading_session_list.append(trading_session)
        except KeyError:
            # Passes gracefully if Massiv doesn't have intraday data for this specific date
            pass

        # --- Standard window evaluations continue ---
        if is_recover:
            recovery_days = window_data.reset_index().is_recover.idxmax()

            peak_ts = highs.idxmax()
            peak_date = peak_ts.date()
            peak_price = highs.max()

            if pd.isna(entry) or entry == 0 or pd.isna(peak_price):
                overrun = np.nan
            else:
                overrun = np.round(100 * (peak_price - entry) / entry, 2)

            days_to_overrun = window_data.reset_index().high.idxmax()

            # The old overrun session capture was removed here because we 
            # are exclusively using the First Touch T0 method above for our categories now.

        else:
            failed_events += 1
            recovery_days = np.nan
            overrun = np.nan
            days_to_overrun = np.nan

            best_pos = int(highs.reset_index(drop=True).idxmax()) if highs.notna().any() else None
            days_to_loss_delta = float(best_pos) if best_pos is not None else np.nan
            days_to_loss_delta_list.append(days_to_loss_delta)

            best_price = highs.max()

            if pd.isna(entry) or entry == 0 or pd.isna(best_price):
                loss_delta_pct = np.nan
            else:
                loss_delta_pct = round(100 * (entry - best_price) / entry, 2)
                loss_delta_pct = max(loss_delta_pct, 0.0) 

            loss_delta_pct_list.append(loss_delta_pct)
        
        first5_highs = pd.to_numeric(window_data["high"].head(5), errors="coerce")
        is_rec_days = (first5_highs > entry).fillna(False).tolist()

        if len(is_rec_days) < 5:
            is_rec_days.extend([False] * (5 - len(is_rec_days)))
        else:
            is_rec_days = is_rec_days[:5]

        cummulative_recovery_list.append(is_rec_days)

        is_recover_list.append(is_recover)
        recovery_days_list.append(recovery_days)
        overrun_list.append(overrun)
        days_to_overrun_list.append(days_to_overrun)

    avg_loss_delta_pct = (
        np.round(np.nanmean(loss_delta_pct_list), 2)
        if len(loss_delta_pct_list) > 0 else np.nan
    )
    
    avg_days_to_loss_delta = (
        np.round(np.nanmean(days_to_loss_delta_list), 2)
        if len(days_to_loss_delta_list) > 0 else np.nan
    )


    total_events = processed_events
    rec_events = int(np.nansum(is_recover_list))

    recovery_pct = np.round(
        100 * rec_events / total_events if total_events > 0 else 0, 2
    )
    avg_recovery_days = np.nanmean(recovery_days_list)
    avg_overrun = np.nanmean(overrun_list)
    med_overrun = np.nanmedian(overrun_list)
    avg_days_to_overrun = np.nanmean(days_to_overrun_list)
    avg_recovery_times = get_recovery_time_massive(events)

    # Filtering the new valid columns
    desired_sessions = {
        "T0 Rec 9:30 to 10:00",
        "T0 Rec 10:00 to 11:00",
        "T0 Rec 11:00 to 12:00",
        "T0 Rec 12:00 to 1:00",
        "T0 Rec 1:00 to 2:00",
        "T0 Rec 2:00 to 3:00",
        "T0 Rec 3:00 to 4:00"
    }
    filtered_sessions = [s for s in trading_session_list if s in desired_sessions]

    dict_trading_session = dict(
        value_counts_list(
            filtered_sessions,
            normalize=True,
            sort=True,
            ascending=False,
            dropna=True,
            topn=None,
            na_label="<NA>",
        )
    )

    order = [
        "T0 Rec 9:30 to 10:00",
        "T0 Rec 10:00 to 11:00",
        "T0 Rec 11:00 to 12:00",
        "T0 Rec 12:00 to 1:00",
        "T0 Rec 1:00 to 2:00",
        "T0 Rec 2:00 to 3:00",
        "T0 Rec 3:00 to 4:00"
    ]

    final_trading_session = {
        f"{sess}": dict_trading_session.get(sess, 0.0) for sess in order
    }

    result = {
        "Method": method,
        "Total Events": total_events,
        "Recovery %": recovery_pct,
        "Avg Days to Recover": avg_recovery_days,
        "Avg Minutes Above Recovery": avg_recovery_times,
        "Median Overrun %": med_overrun,
        "Avg Days to Overrun": avg_days_to_overrun,
        "Loss Delta %": avg_loss_delta_pct,
        "Avg Days To Loss Delta": avg_days_to_loss_delta,
    }
    
    result.update(final_trading_session)

    cummulative_recovery_df = pd.DataFrame(cummulative_recovery_list).mean()
    cummulative_recovery_df.index = list(map(lambda x: f'T{x} Rec %', cummulative_recovery_df.index.values))
    cummulative_recovery_dict = cummulative_recovery_df.mul(100).round(2).to_dict()
    
    result.update(cummulative_recovery_dict)

    return result