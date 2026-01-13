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
        # Treat None and NaN (incl. numpy.nan) as NA
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
        pairs = [(k, round((c / denom) * 100, 2)) for k, c in pairs]  # percentage
    else:
        pairs = [
            (k, round(c, 2)) for k, c in pairs
        ]  # keep raw counts but round (optional)

    out = [(na_label if k is NA else k, v) for k, v in pairs]
    return out


def get_hist(symbol, start, end=""):
    df = (
        pd.DataFrame(
            requests.get(
                f"https://financialmodelingprep.com/stable/historical-price-eod/full?symbol={symbol}&from={start}&to={end}&apikey={API_KEY}"
            ).json()
        )
        .set_index("date")
        .drop("symbol", axis=1)
    )
    # df  = pd.DataFrame(requests.get(f'https://financialmodelingprep.com/stable/historical-chart/5min?symbol={symbol}&from={start}&to={end}&apikey={API_KEY}').json()).set_index('date').drop('symbol', axis = 1)
    df.index = pd.to_datetime(df.index)
    return df


def get_div(symbol, start, end=""):
    df = pd.DataFrame(
        requests.get(
            f"https://financialmodelingprep.com/api/v3/historical-price-full/stock_dividend/{symbol}?apikey={API_KEY}"
        ).json()["historical"]
    ).set_index("date")
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    df = df.loc[start:end]
    return df


def get_data(symbol, start, end="", recovery_window=1):
    hist_data = get_hist(symbol, start, end)  # <-- your API call
    div_data = get_div(symbol, start, end)  # <-- your API call
    df = pd.merge(hist_data, div_data, left_index=True, right_index=True, how="outer")
    df["entry_date"] = pd.to_datetime(df.index.to_series().shift(1))
    df["entry_price_t_1"] = df["close"].shift(1)
    # df["entry_price_t_1_min15"] = df["close"].shift(1) - 0.15
    # df["entry_price_t_1_plus15"] = df["close"].shift(1) + 0.15
    df["entry_price_6040"] = 0.6 * df["close"].shift(1) + 0.4 * df["open"]
    df["entry_price_7030"] = 0.7 * df["close"].shift(1) + 0.3 * df["open"]

    # # Entry price depending on method
    # if method == "t-1":
    #     df["entry_date"] = pd.to_datetime(df.index.to_series().shift(1))
    #     df["entry_price"] = df["close"].shift(1)
    # elif method == "t-1-15":
    #     df["entry_date"] = pd.to_datetime(df.index.to_series().shift(1))
    #     df["entry_price"] = df["close"].shift(1) - 0.15
    # elif method == "t-1+15":
    #     df["entry_date"] = pd.to_datetime(df.index.to_series().shift(1))
    #     df["entry_price"] = df["close"].shift(1) + 0.15
    # elif method == "60/40":
    #     df["entry_date"] = pd.to_datetime(df.index.to_series().shift(1))
    #     df["entry_price"] = 0.6 * df["close"].shift(1) + 0.4 * df["open"]
    # elif method == "70/30":
    #     df["entry_date"] = pd.to_datetime(df.index.to_series().shift(1))
    #     df["entry_price"] = 0.7 * df["close"].shift(1) + 0.3 * df["open"]

    # Add recovery window boundaries
    df["end_window_date"] = df.index.to_series().shift(-recovery_window)
    df["prev_two_days"] = pd.to_datetime(df.index.to_series().shift(2))
    return df


def get_trading_session(timestamp_str):
    dt = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
    input_time = dt.time()

    sessions = {
        "Overrun on Opening %": (time(9, 30), time(10, 30)),
        "Overrun on Late Morning %": (time(10, 30), time(12, 0)),
        "Overrun on Early Afternoon %": (time(12, 0), time(14, 0)),
        "Overrun on Closing %": (time(14, 0), time(16, 0)),
    }

    for session, (start, end) in sessions.items():
        if start <= input_time < end:
            return session

    return "Outside Market Hours"


def plot_event(
    event,
    data,
    symbol,
    method,
    recovery_window,
    event_date=None,
    entry_price=None,
    is_recover=None,
    recovery_days=None,
    overrun=None,
    days_to_overrun=None,
):
    entry_price_pairs = {
        "t-1": "entry_price_t_1",
        "t-1-15": "entry_price_t_1_min15",
        "t-1+15": "entry_price_t_1",
        "60/40": "entry_price_6040",
        "70/30": "entry_price_7030",
    }
    window_data = data.loc[event.name : event["end_window_date"]]
    event_plot = data.loc[event["prev_two_days"] : event["end_window_date"]]

    fig, ax = plt.subplots(figsize=(12, 7))
    plt.title(f"{symbol}; {method}; {recovery_window}-day window")

    plt.plot(event_plot.close, label="Close", marker="o")
    plt.plot(event_plot.high, label="High", marker="o")
    # plt.axhline(y=event.entry_price, color="g", linestyle="--", label="entry price")
    plt.axhline(
        y=event[entry_price_pairs[method]],
        color="g",
        linestyle="--",
        label="entry price",
    )
    plt.axvline(x=event.name, color="r", linestyle="-.", label="ex-dividend")
    plt.axvline(x=event.entry_date, color="r", linestyle="dotted", label="entry date")
    if is_recover:
        plt.axvline(
            x=window_data.high.idxmax(), linestyle="-.", color="g", label="overrun"
        )

    text_str = (
        f"Event Date: {event_date}\n"
        f"Entry Price: {entry_price:.2f}\n"
        f"Recovered: {is_recover}\n"
        f"Recovery Days: {recovery_days}\n"
        f"Overrun: {overrun}%\n"
        f"Days to Overrun: {days_to_overrun}"
    )

    ax.text(
        0.5,
        0.5,
        text_str,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        horizontalalignment="center",
        bbox=dict(facecolor="white", alpha=0.7, edgecolor="black"),
    )

    plt.legend()
    plt.show()


def get_time(peak_day):
    peak_time = peak_day.high.idxmax()
    trading_session = get_trading_session(peak_time)
    return peak_time


def backtest(symbol, method, recovery_window, plot=True, year=1, start=None, end=None):

    def get_recovery_time(events, symbol=symbol):
        recovery_times = []
        for event_date in events.index:
            event = events.loc[event_date]
            # entry_price = event.entry_price
            event[entry_price_pairs[method]]

            # window_data = data.loc[event.name : event.end_window_date].copy()
            intraday_data = pd.DataFrame(
                requests.get(
                    f'https://financialmodelingprep.com/stable/historical-chart/1min?symbol={symbol}&from={event.name.strftime("%Y-%m-%d")}&to={event.end_window_date.date().strftime("%Y-%m-%d")}&apikey={API_KEY}'
                ).json()
            )
            intraday_data.index = pd.to_datetime(intraday_data.date).rename("index")
            intraday_data = intraday_data.sort_index()
            intraday_data.date = pd.to_datetime(intraday_data.date).dt.date
            intraday_data["is_recover"] = intraday_data.close > entry_price
            recovery_time = (
                intraday_data.groupby(by="date")["is_recover"].sum().mean().round(2)
            )
            recovery_times.append(recovery_time)
        return np.round(np.mean(recovery_times), 2)

    if year:
        start = year_delta(year)
        end = datetime.today().date()

    data = get_data(
        symbol=symbol,
        start=start,
        end=end,
        recovery_window=recovery_window,
        # method=method,
    )
    events = data.dropna(subset=["dividend", "end_window_date"])
    for event_date in events.index:
        temp = pd.DataFrame(
            requests.get(
                f"https://financialmodelingprep.com/stable/historical-chart/15min?symbol={symbol}&from={event_date.date()}&to={event_date.date()}&apikey={API_KEY}"
            ).json()
        )
        temp["date"] = pd.to_datetime(temp.date)
        temp = temp.set_index("date").sort_index()
        temp_new = temp.iloc[1:].reset_index(drop=True)

        open = temp_new.open.iloc[0]
        close = temp_new.close.iloc[-1]
        low = temp_new.low.min()
        high = temp_new.high.max()

        events.loc[event_date, "open"] = open
        events.loc[event_date, "close"] = close
        events.loc[event_date, "low"] = low
        events.loc[event_date, "high"] = high

    is_recover_list = []
    recovery_days_list = []
    overrun_list = []
    days_to_overrun_list = []
    trading_session_list = []

    entry_price_pairs = {
        "t-1": "entry_price_t_1",
        "t-1-15": "entry_price_t_1_min15",
        "t-1+15": "entry_price_t_1",
        "60/40": "entry_price_6040",
        "70/30": "entry_price_7030",
    }

    for event_date in events.index:
        event = events.loc[event_date]
        # entry_price = event.entry_price
        entry_price = event[entry_price_pairs[method]]

        # Window slice (copy to avoid SettingWithCopyWarning)
        window_data = data.loc[event.name : event["end_window_date"]].copy()

        # Recovery test = High ≥ Entry
        window_data["is_recover"] = window_data["high"] >= entry_price
        is_recover = window_data["is_recover"].any()

        if is_recover:
            # First recovery day
            # recovery_idx = window_data[window_data['is_recover']].index[0]
            # recovery_days = (recovery_idx - event_date).days
            recovery_days = window_data.reset_index().is_recover.idxmax()

            # Overrun peak
            # peak_idx = window_data['high'].idxmax()
            # overrun = round(100 * (window_data.loc[peak_idx, 'high'] - entry_price) / entry_price, 2)
            # days_to_overrun = (peak_idx - recovery_idx).days
            # overrun

            peak_date = window_data.high.idxmax().date()
            peak_price = window_data.high.max()
            # peak_day = pd.DataFrame(requests.get(f'https://financialmodelingprep.com/stable/historical-chart/5min?symbol={symbol}&from={peak_date}&apikey={API_KEY}').json()).set_index('date')
            # peak_time = peak_day.high.idxmax()
            # print(peak_time)
            # print(peak_price)
            overrun = np.round(100 * (peak_price - entry_price) / entry_price, 2)

            # days to overrun
            days_to_overrun = window_data.reset_index().high.idxmax()
            peak_day_data = pd.DataFrame(
                requests.get(
                    f"https://financialmodelingprep.com/stable/historical-chart/1hour?symbol={symbol}&from={peak_date}&to={peak_date}&apikey={API_KEY}"
                ).json()
            ).set_index("date")
            # print(peak_day_data)
            peak_time = peak_day_data.high.idxmax()
            trading_session = get_trading_session(peak_time)
            trading_session_list.append(trading_session)

        else:
            recovery_days = np.nan
            overrun = np.nan
            days_to_overrun = np.nan

        # Store results
        is_recover_list.append(is_recover)
        recovery_days_list.append(recovery_days)
        overrun_list.append(overrun)
        days_to_overrun_list.append(days_to_overrun)

        # Plot individual event
        if plot:
            plot_event(
                event,
                data,
                symbol,
                method,
                recovery_window,
                event_date=event_date.date(),
                entry_price=entry_price,
                is_recover=is_recover,
                recovery_days=recovery_days,
                overrun=overrun,
                days_to_overrun=days_to_overrun,
            )

    # ---- Summary ----
    # print("\nSummary:")
    total_events = len(events)
    rec_events = sum(is_recover_list)

    recovery_pct = 100 * rec_events / total_events if total_events > 0 else 0
    avg_recovery_days = np.nanmean(recovery_days_list)
    avg_overrun = np.nanmean(overrun_list)
    avg_days_to_overrun = np.nanmean(days_to_overrun_list)
    avg_recovery_times = get_recovery_time(events)

    # print(f"Total Events: {total_events}")
    # print(f"Recovered: {rec_events}")
    # print(f"Recovery %: {100 * rec_events / total_events:.2f}%")
    # print(f"Avg Days to Recover: {np.nanmean(recovery_days_list):.2f}")
    # print(f"Avg Overrun %: {np.nanmean(overrun_list):.2f}%")
    # print(f"Avg Days to Overrun: {np.nanmean(days_to_overrun_list):.2f}")
    # print(
    #     value_counts_list(
    #         trading_session_list,
    #         normalize=True,
    #         sort=True,
    #         ascending=False,
    #         dropna=True,
    #         topn=None,
    #         na_label="<NA>",
    #     )
    # )

    dict_trading_session = dict(
        value_counts_list(
            trading_session_list,
            normalize=True,
            sort=True,
            ascending=False,
            dropna=True,
            topn=None,
            na_label="<NA>",
        )
    )
    order = ["Opening", "Late Morning", "Early Afternoon", "Closing"]

    # dict_trading_session looks like {"Opening": 42.0, "Closing": 10.0, ...}
    final_trading_session = {
        f"Overrun on {sess} %": dict_trading_session.get(sess, np.nan) for sess in order
    }
    # print()

    result = {
        "Method": method,
        "Total Events": total_events,
        "Recovery %": recovery_pct,
        "Avg Days to Recover": avg_recovery_days,
        "Avg Time Above Recovery (in Minutes)": avg_recovery_times,
        "Avg Overrun %": avg_overrun,
        "Avg Days to Overrun": avg_days_to_overrun,
    }
    result.update(final_trading_session)
    return result
    # return (recovery_pct, avg_recovery_days, avg_overrun, avg_days_to_overrun)


def backtest_all(symbol, recovery_window, plot=True, year=1):
    # methods = ["t-1", "t-1-15", "t-1+15", "60/40", "70/30"]
    methods = ["t-1+15", "t-1", "t-1-15", "70/30", "60/40"]
    summary = []

    for method in methods:
        print(
            f"\n=== Backtest for {symbol} using {method} with {recovery_window}-day window ==="
        )
        result = backtest(
            symbol=symbol,
            method=method,
            recovery_window=recovery_window,
            plot=plot,
            year=year,
        )
        summary.append(result)
    return summary


def get_data_new(symbol, year, recovery_window=1, start=None, end=None):

    if year:
        start = year_delta(year)
        end = datetime.today().date()

    hist_data = get_hist(symbol, start, end)  # <-- your API call
    div_data = get_div(symbol, start, end)  # <-- your API call
    df = pd.merge(hist_data, div_data, left_index=True, right_index=True, how="outer")
    df["entry_date"] = pd.to_datetime(df.index.to_series().shift(1))
    df["entry_price_t_1"] = df["close"].shift(1)
    # df["entry_price_t_1_min15"] = df["close"].shift(1) - 0.15
    # df["entry_price_t_1_plus15"] = df["close"].shift(1) + 0.15
    df["entry_price_6040"] = 0.6 * df["close"].shift(1) + 0.4 * df["low"]
    df["entry_price_7030"] = 0.7 * df["close"].shift(1) + 0.3 * df["low"]

    # Add recovery window boundaries
    df["end_window_date"] = df.index.to_series().shift(-recovery_window)
    df["prev_two_days"] = pd.to_datetime(df.index.to_series().shift(2))

    events = df.dropna(subset=["dividend", "end_window_date"])
    for event_date in events.index:
        try:
            temp = pd.DataFrame(
                requests.get(
                    f"https://financialmodelingprep.com/stable/historical-chart/15min?symbol={symbol}&from={event_date.date()}&to={event_date.date()}&apikey={API_KEY}"
                ).json()
            )
            temp["date"] = pd.to_datetime(temp.date)
            temp = temp.set_index("date").sort_index()
            temp_new = temp.iloc[1:].reset_index(drop=True)

            open = temp_new.open.iloc[0]
            close = temp_new.close.iloc[-1]
            low = temp_new.low.min()
            high = temp_new.high.max()

            df.loc[event_date, "open"] = open
            df.loc[event_date, "close"] = close
            df.loc[event_date, "low"] = low
            df.loc[event_date, "high"] = high
        except:
            continue
    return df


def backtest_new(symbol, method, recovery_window, data, plot=False):

    def get_recovery_time(events, symbol=symbol):
        recovery_times = []
        for event_date in events.index:
            event = events.loc[event_date]
            # entry_price = event.entry_price
            entry_price = event[entry_price_pairs[method]]

            # window_data = data.loc[event.name : event.end_window_date].copy()
            intraday_data = pd.DataFrame(
                requests.get(
                    f'https://financialmodelingprep.com/stable/historical-chart/30min?symbol={symbol}&from={event.name.strftime("%Y-%m-%d")}&to={event.end_window_date.date().strftime("%Y-%m-%d")}&apikey={API_KEY}'
                ).json()
            )
            intraday_data.index = pd.to_datetime(intraday_data.date).rename("index")
            intraday_data = intraday_data.sort_index()
            intraday_data.date = pd.to_datetime(intraday_data.date).dt.date
            intraday_data["is_recover"] = intraday_data.close > entry_price
            recovery_time = np.round(
                (intraday_data.groupby(by="date")["is_recover"].sum().mul(30).mean()), 2
            )
            recovery_times.append(recovery_time)
        return np.round(np.mean(recovery_times), 2)


    events = data.dropna(subset=["dividend", "end_window_date"])

    is_recover_list = []
    recovery_days_list = []
    overrun_list = []
    days_to_overrun_list = []
    trading_session_list = []
    cummulative_recovery_list = []
    

    entry_price_pairs = {
        "t-1": "entry_price_t_1",
        # "t-1-15": "entry_price_t_1_min15",
        # "t-1+15": "entry_price_t_1_plus15",
        "60/40": "entry_price_6040",
        "70/30": "entry_price_7030",
    }

    for event_date in events.index:
        event = events.loc[event_date]
        entry_price = event[entry_price_pairs[method]]

        # Window slice (copy to avoid SettingWithCopyWarning)
        window_data = data.loc[event.name : event["end_window_date"]].copy()

        # Recovery test = High ≥ Entry
        window_data["is_recover"] = window_data["high"] > entry_price
        is_recover = window_data["is_recover"].any()

        if is_recover:
            # First recovery day
            # recovery_idx = window_data[window_data['is_recover']].index[0]
            # recovery_days = (recovery_idx - event_date).days
            recovery_days = window_data.reset_index().is_recover.idxmax()

            # Overrun peak
            # peak_idx = window_data['high'].idxmax()
            # overrun = round(100 * (window_data.loc[peak_idx, 'high'] - entry_price) / entry_price, 2)
            # days_to_overrun = (peak_idx - recovery_idx).days
            # overrun

            peak_date = window_data.high.idxmax().date()
            peak_price = window_data.high.max()
            # peak_day = pd.DataFrame(requests.get(f'https://financialmodelingprep.com/stable/historical-chart/5min?symbol={symbol}&from={peak_date}&apikey={API_KEY}').json()).set_index('date')
            # peak_time = peak_day.high.idxmax()
            # print(peak_time)
            # print(peak_price)
            overrun = np.round(100 * (peak_price - entry_price) / entry_price, 2)

            # days to overrun
            days_to_overrun = window_data.reset_index().high.idxmax()
            peak_day_data = pd.DataFrame(
                requests.get(
                    f"https://financialmodelingprep.com/stable/historical-chart/1hour?symbol={symbol}&from={peak_date}&to={peak_date}&apikey={API_KEY}"
                ).json()
            ).set_index("date")
            # print(peak_day_data)
            peak_time = peak_day_data.high.idxmax()
            trading_session = get_trading_session(peak_time)
            trading_session_list.append(trading_session)

        else:
            recovery_days = np.nan
            overrun = np.nan
            days_to_overrun = np.nan

        is_rec_days = [window_data.iloc[i]['high'] > entry_price for i in range(5)]
        cummulative_recovery_list.append(is_rec_days)
        
        # Store results
        is_recover_list.append(is_recover)
        recovery_days_list.append(recovery_days)
        overrun_list.append(overrun)
        days_to_overrun_list.append(days_to_overrun)

        # Plot individual event
        if plot:
            plot_event(
                event,
                data,
                symbol,
                method,
                recovery_window,
                event_date=event_date.date(),
                entry_price=entry_price,
                is_recover=is_recover,
                recovery_days=recovery_days,
                overrun=overrun,
                days_to_overrun=days_to_overrun,
            )

    # ---- Summary ----
    # print("\nSummary:")
    total_events = len(events)
    rec_events = sum(is_recover_list)

    recovery_pct = np.round(
        100 * rec_events / total_events if total_events > 0 else 0, 2
    )
    avg_recovery_days = np.nanmean(recovery_days_list)
    avg_overrun = np.nanmean(overrun_list)
    med_overrun = np.nanmedian(overrun_list)
    avg_days_to_overrun = np.nanmean(days_to_overrun_list)
    avg_recovery_times = get_recovery_time(events)

    # print(f"Total Events: {total_events}")
    # print(f"Recovered: {rec_events}")
    # print(f"Recovery %: {100 * rec_events / total_events:.2f}%")
    # print(f"Avg Days to Recover: {np.nanmean(recovery_days_list):.2f}")
    # print(f"Avg Overrun %: {np.nanmean(overrun_list):.2f}%")
    # print(f"Avg Days to Overrun: {np.nanmean(days_to_overrun_list):.2f}")
    # print(
    #     value_counts_list(
    #         trading_session_list,
    #         normalize=True,
    #         sort=True,
    #         ascending=False,
    #         dropna=True,
    #         topn=None,
    #         na_label="<NA>",
    #     )
    # )

    dict_trading_session = dict(
        value_counts_list(
            trading_session_list,
            normalize=True,
            sort=True,
            ascending=False,
            dropna=True,
            topn=None,
            na_label="<NA>",
        )
    )
    # order = ["Opening", "Late Morning", "Early Afternoon", "Closing"]
    order = [
        "Overrun on Opening %",
        "Overrun on Late Morning %",
        "Overrun on Early Afternoon %",
        "Overrun on Closing %",
    ]

    # dict_trading_session looks like {"Opening": 42.0, "Closing": 10.0, ...}
    final_trading_session = {
        f"{sess}": dict_trading_session.get(sess, np.nan) for sess in order
    }
    # print()

    result = {
        "Method": method,
        "Total Events": total_events,
        "Recovery %": recovery_pct,
        "Avg Days to Recover": avg_recovery_days,
        "Avg Minutes Above Recovery": avg_recovery_times,
        "Median Overrun %": med_overrun,
        "Avg Days to Overrun": avg_days_to_overrun,
    }
    result.update(final_trading_session)

    cummulative_recovery_df = pd.DataFrame(cummulative_recovery_list).mean()
    cummulative_recovery_df.index = list(map(lambda x: f'T{x} Rec %', cummulative_recovery_df.index.values))
    cummulative_recovery_dict = cummulative_recovery_df.mul(100).round(2).to_dict()
    
    result.update(cummulative_recovery_dict)

    return result
    # return (recovery_pct, avg_recovery_days, avg_overrun, avg_days_to_overrun)



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
        trimmed = day_df.iloc[15:]  # drop first 15 rows
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

    # group by date from the intraday index
    per_day = (
        intraday_data
        .groupby(intraday_data.index.normalize())
        .apply(intraday_adjustments)
    )
    
    
    # daily_data = (
    #     intraday_data.resample("D")
    #     .agg({"open": "first", "high": "max", "low": "min", "close": "last"})
    #     .dropna()
    # )
    return intraday_data, per_day



def get_div_massive(symbol, API_KEY=MASSIVE_API_KEY):
    client = RESTClient(API_KEY)
    dividends = list(
        client.list_dividends(
            ticker=symbol, order="asc", limit=1000, sort="ex_dividend_date"
        )
    )
    if not dividends:
        raise ValueError(f"No dividend data returned for {symbol}")

    df = pd.DataFrame(dividends).set_index('ex_dividend_date')[['id']].rename(columns={'id': 'is_dividend'})
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
            window_date = intraday_data.loc[start_date:end_date]
            window_date['is_recover'] = window_date.close > entry_price
            recovery_time = (window_date.resample('D').sum().mean().round(2).is_recover)
            recovery_times.append(recovery_time)
        return np.round(np.mean(recovery_times), 2)


    is_recover_list = []
    recovery_days_list = []
    overrun_list = []
    days_to_overrun_list = []
    trading_session_list = []
    cummulative_recovery_list = []
    

    entry_price_pairs = {
        "t-1": "entry_price_t_1",
        't-1_997': 'entry_price_997',
        "60/40": "entry_price_6040",
        "70/30": "entry_price_7030",
    }

    for event_date, event in events.iterrows():
        entry_price = event[entry_price_pairs[method]]
        end_window = event["end_window_date"]

        # Window slice: respect recovery window end date
        window_data = per_day.loc[event_date:end_window].copy()
        if window_data.empty:
            continue

        # Recovery test = High ≥ Entry
        window_data["is_recover"] = window_data["high"] > entry_price
        is_recover = window_data["is_recover"].any()

        if is_recover:
            recovery_days = window_data.reset_index().is_recover.idxmax()
            peak_ts = window_data.high.idxmax()
            peak_date = peak_ts.date()
            peak_price = window_data.high.max()
            overrun = np.round(100 * (peak_price - entry_price) / entry_price, 2)

            days_to_overrun = window_data.reset_index().high.idxmax()
            # slice intraday data by date string to get that day's bars
            peak_day_data = intraday_data.loc[str(peak_date)]
            peak_time = peak_day_data.high.idxmax()
            trading_session = get_trading_session(str(peak_time))
            trading_session_list.append(trading_session)

        else:
            recovery_days = np.nan
            overrun = np.nan
            days_to_overrun = np.nan

        highs = window_data['high'].head(5)
        is_rec_days = (highs > entry_price).tolist()
        # pad to always keep 5 trading-day slots (T0-T4) for consistent recovery columns
        if len(is_rec_days) < 5:
            is_rec_days.extend([False] * (5 - len(is_rec_days)))
        else:
            is_rec_days = is_rec_days[:5]
        cummulative_recovery_list.append(is_rec_days)
        
        # Store results
        is_recover_list.append(is_recover)
        recovery_days_list.append(recovery_days)
        overrun_list.append(overrun)
        days_to_overrun_list.append(days_to_overrun)


    # ---- Summary ----
    total_events = len(events)
    rec_events = sum(is_recover_list)

    recovery_pct = np.round(
        100 * rec_events / total_events if total_events > 0 else 0, 2
    )
    avg_recovery_days = np.nanmean(recovery_days_list)
    avg_overrun = np.nanmean(overrun_list)
    med_overrun = np.nanmedian(overrun_list)
    avg_days_to_overrun = np.nanmean(days_to_overrun_list)
    avg_recovery_times = get_recovery_time_massive(events)


    desired_sessions = {
        "Overrun on Opening %",
        "Overrun on Late Morning %",
        "Overrun on Early Afternoon %",
        "Overrun on Closing %",
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
        "Overrun on Opening %",
        "Overrun on Late Morning %",
        "Overrun on Early Afternoon %",
        "Overrun on Closing %",
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
    }
    result.update(final_trading_session)

    cummulative_recovery_df = pd.DataFrame(cummulative_recovery_list).mean()
    cummulative_recovery_df.index = list(map(lambda x: f'T{x} Rec %', cummulative_recovery_df.index.values))
    cummulative_recovery_dict = cummulative_recovery_df.mul(100).round(2).to_dict()
    
    result.update(cummulative_recovery_dict)

    return result
