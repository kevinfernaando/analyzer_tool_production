# ai_tools.py
from __future__ import annotations
import os, json, traceback, requests
from typing import List, Dict, Callable, Any, Optional
from dataclasses import dataclass
from functools import partial
from collections import Counter
from datetime import datetime, date, time

import numpy as np
import pandas as pd
from openai import OpenAI

# ===== Utilities =====

def _json_default(o):
    """Safe fallback serializer for OpenAI tool outputs and pandas/numpy types."""
    # pandas / python datetimes
    if isinstance(o, (pd.Timestamp, datetime, date, time)):
        return o.isoformat()
    # numpy numbers
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else float(o)
    # sets -> lists
    if isinstance(o, set):
        return list(o)
    # final fallback for objects or other non-serializable types
    return str(o)


def json_sanitize(obj):
    if isinstance(obj, dict):
        return {k: json_sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [json_sanitize(v) for v in obj]
    if isinstance(obj, (pd.Timestamp, datetime, date, time)):
        return obj.isoformat()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return None if np.isnan(obj) else float(obj)
    return obj

def df_compact(df: pd.DataFrame, max_rows=50) -> dict:
    d = df.copy()
    if not isinstance(d.index, pd.RangeIndex) or d.index.name is not None:
        d = d.reset_index()
    for c in d.columns:
        s = d[c]
        if pd.api.types.is_datetime64_any_dtype(s) or pd.api.types.is_timedelta64_dtype(s):
            d[c] = s.apply(lambda v: None if pd.isna(v) else pd.to_datetime(v).isoformat())
        else:
            d[c] = s.where(pd.notnull(s), None)
    return {
        "__type__": "dataframe",
        "shape": list(df.shape),
        "columns": [str(c) for c in df.columns],
        "head_rows": d.head(max_rows).to_dict(orient="records"),
        "note": f"Showing first {min(max_rows, len(d))} rows only."
    }

def prepare_context(obj, max_rows_per_df=50) -> str:
    def make_jsonable(x):
        if isinstance(x, pd.DataFrame): return df_compact(x, max_rows_per_df)
        if isinstance(x, pd.Series):    return {"__type__":"series","name":str(x.name),"values": x.reset_index().to_dict(orient="records")}
        if isinstance(x, (pd.Timestamp, pd.Timedelta, np.datetime64, datetime, time)):
            try: return pd.to_datetime(x).isoformat()
            except Exception: return str(x)
        if isinstance(x, (np.integer,)):  return int(x)
        if isinstance(x, (np.floating,)): return None if np.isnan(x) else float(x)
        if isinstance(x, (list, tuple, set)): return [make_jsonable(v) for v in x]
        if isinstance(x, dict): return {str(k): make_jsonable(v) for k, v in x.items()}
        try:
            if pd.isna(x): return None
        except Exception:
            pass
        return x
    return json.dumps(make_jsonable(obj), indent=2, ensure_ascii=False)

# ===== Your helpers =====
def get_trading_session(timestamp_str: str) -> str:
    dt = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
    t = dt.time()
    sessions = {
        "Overrun on Opening %": (time(9,30), time(10,30)),
        "Overrun on Late Morning %": (time(10,30), time(12,0)),
        "Overrun on Early Afternoon %": (time(12,0), time(14,0)),
        "Overrun on Closing %": (time(14,0), time(16,0)),
    }
    for label, (a,b) in sessions.items():
        if a <= t < b: return label
    return "Outside Market Hours"

def value_counts_list(
    items, *, normalize=False, sort=True, ascending=False, dropna=True, topn=None, na_label="<NA>",
):
    NA = object()
    def is_na(x):
        try: return x is None or x != x
        except Exception: return False
    from collections import Counter
    counts = Counter()
    for x in items:
        if is_na(x):
            if dropna: continue
            x = NA
        counts[x] += 1
    pairs = list(counts.items())
    if sort: pairs.sort(key=lambda kv: kv[1], reverse=not ascending)
    if topn is not None: pairs = pairs[:topn]
    if normalize:
        denom = sum(c for _, c in pairs) or 1
        pairs = [(k, round((c/denom)*100, 2)) for k,c in pairs]
    else:
        pairs = [(k, round(c, 2)) for k,c in pairs]
    return [(na_label if k is NA else k, v) for k,v in pairs]

# ===== FMP tools =====
def get_hist(symbol: str, start: str, end: str = "", *, api_key: str | None = None) -> dict:
    api_key = api_key or os.getenv("FMP_API_KEY")
    if not api_key:
        return {"error": "Missing FMP_API_KEY. Set env var FMP_API_KEY or pass api_key=..."}
    if not symbol or not isinstance(symbol, str):
        return {"error": "symbol must be a non-empty string"}
    if not start or not isinstance(start, str):
        return {"error": "start must be a non-empty YYYY-MM-DD string"}

    # default "until now"
    end = end or date.today().isoformat()

    url = ("https://financialmodelingprep.com/stable/historical-price-eod/full"
           f"?symbol={symbol}&from={start}&to={end}&apikey={api_key}")
    r = requests.get(url, timeout=30)
    try:
        data = r.json()
        df = pd.DataFrame(data).set_index("date").drop(columns=["symbol"], errors="ignore")
        df.index = pd.to_datetime(df.index, errors="coerce")
        df = df.dropna().sort_index(ascending=False)

        # compact, JSON-safe preview (convert datetime index column)
        preview = df.head(5).reset_index().copy()
        for c in preview.columns:
            if np.issubdtype(preview[c].dtype, np.datetime64):
                preview[c] = preview[c].dt.isoformat()
        preview = preview.to_dict(orient="records")

        result = {
            "symbol": symbol.upper(),
            "start": start,
            "end": end,
            "rows": int(len(df)),
            "columns": list(map(str, df.columns)),
            "head": preview,
            "note": "Compact preview. Expand in your app if you need full rows."
        }
        # ensure fully JSON-safe before returning
        return json_sanitize(result)
    except Exception as e:
        return {"error": f"Fetch/parse error: {e}", "status": r.status_code if hasattr(r, 'status_code') else None, "raw": r.text[:500] if hasattr(r, 'text') else None}

def get_div(symbol: str, start: str, end: str = "", *, api_key: str | None = None) -> dict:
    api_key = api_key or os.getenv("FMP_API_KEY")
    if not api_key:
        return {"error": "Missing FMP_API_KEY. Set env var FMP_API_KEY or pass api_key=..."}
    if not symbol or not isinstance(symbol, str):
        return {"error": "symbol must be a non-empty string"}
    if not start or not isinstance(start, str):
        return {"error": "start must be a non-empty YYYY-MM-DD string"}

    end = end or date.today().isoformat()

    url = f"https://financialmodelingprep.com/api/v3/historical-price-full/stock_dividend/{symbol}?apikey={api_key}"
    r = requests.get(url, timeout=30)
    try:
        data = r.json()
        if "historical" not in data:
            return {"error": "Unexpected API response", "raw": data}

        df = pd.DataFrame(data["historical"])
        if df.empty or "date" not in df.columns:
            return {"error": "No dividend data", "symbol": symbol.upper(), "start": start, "end": end}

        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"]).sort_values("date")
        mask = (df["date"] >= pd.to_datetime(start)) & (df["date"] <= pd.to_datetime(end))
        df = df.loc[mask]

        preview = df.head(5).copy()
        for c in preview.columns:
            if np.issubdtype(preview[c].dtype, np.datetime64):
                preview[c] = preview[c].dt.isoformat()
        preview = preview.to_dict(orient="records")

        result = {
            "symbol": symbol.upper(),
            "start": start,
            "end": end,
            "rows": int(len(df)),
            "columns": list(map(str, df.columns)),
            "head": preview,
            "note": "Compact preview of dividend data."
        }
        return json_sanitize(result)
    except Exception as e:
        return {"error": f"Fetch/parse error: {e}", "status": r.status_code if hasattr(r, 'status_code') else None, "raw": r.text[:500] if hasattr(r, 'text') else None}

# ===== Backtest tool (wrapper) =====
ALLOWED_METHODS = {"t-1", "t-1-15", "t-1+15", "60/40", "70/30"}

def backtest_new_tool(
    symbol: str,
    method: str,
    recovery_window: int,
    *,
    data: pd.DataFrame,
    api_key: str,
    get_trading_session: Callable[[str], str],
    value_counts_list: Callable[..., list],
    plot: bool = False,
) -> dict:
    method = method.strip()
    if method not in ALLOWED_METHODS:
        return {"error": f"Invalid method '{method}'. Allowed: {sorted(ALLOWED_METHODS)}"}
    if not isinstance(recovery_window, int) or recovery_window <= 0:
        return {"error": "recovery_window must be a positive integer."}
    if not isinstance(symbol, str) or not symbol:
        return {"error": "symbol must be a non-empty string."}

    entry_price_pairs = {
        "t-1": "entry_price_t_1",
        "t-1-15": "entry_price_t_1_min15",
        "t-1+15": "entry_price_t_1_plus15",
        "60/40": "entry_price_6040",
        "70/30": "entry_price_7030",
    }

    def get_recovery_time(events, symbol=symbol):
        times = []
        for idx in events.index:
            event = events.loc[idx]
            entry_price = event[entry_price_pairs[method]]
            r = requests.get(
                "https://financialmodelingprep.com/stable/historical-chart/30min"
                f"?symbol={symbol}&from={event.name:%Y-%m-%d}"
                f"&to={event.end_window_date.date():%Y-%m-%d}&apikey={api_key}"
            )
            intraday = pd.DataFrame(r.json())
            if intraday.empty or "date" not in intraday or "close" not in intraday: continue
            intraday.index = pd.to_datetime(intraday.date)
            intraday = intraday.sort_index()
            intraday["date"] = intraday.index.date
            intraday["is_recover"] = intraday["close"] > entry_price
            minutes = np.round(intraday.groupby("date")["is_recover"].sum().mul(30).mean(), 2)
            times.append(minutes)
        return float(np.round(np.nanmean(times), 2)) if times else float("nan")

    events = data.dropna(subset=["dividend", "end_window_date"])
    is_rec, rec_days, overrun, dto, sessions, cum = [], [], [], [], [], []

    for idx in events.index:
        event = events.loc[idx]
        entry_price = event[entry_price_pairs[method]]
        try:
            window = data.loc[event.name:event["end_window_date"]].copy()
        except Exception:
            continue
        window["is_recover"] = window["high"] > entry_price
        got = bool(window["is_recover"].any())

        if got:
            rec_day = int(window.reset_index().is_recover.idxmax())
            peak_idx = window["high"].idxmax()
            peak_date = peak_idx.date()
            peak_price = float(window.loc[peak_idx, "high"])
            over = float(np.round(100 * (peak_price - entry_price) / entry_price, 2))
            dto_i = int(window.reset_index().high.idxmax())

            r = requests.get(
                "https://financialmodelingprep.com/stable/historical-chart/1hour"
                f"?symbol={symbol}&from={peak_date}&to={peak_date}&apikey={api_key}"
            )
            pday = pd.DataFrame(r.json())
            if not pday.empty and {"high","date"} <= set(pday.columns):
                pday = pday.set_index("date")
                peak_time = str(pday["high"].idxmax())
                session = get_trading_session(peak_time)
            else:
                session = None
            sessions.append(session)
        else:
            rec_day, over, dto_i = np.nan, np.nan, np.nan

        n = min(5, len(window))
        flags = [bool(window.iloc[i]["high"] > entry_price) for i in range(n)]
        if n < 5: flags += [False] * (5 - n)

        is_rec.append(got); rec_days.append(rec_day); overrun.append(over); dto.append(dto_i); cum.append(flags)

    total = int(len(events))
    rec_n = int(sum(is_rec))
    recovery_pct = float(np.round(100 * rec_n / total, 2) if total else 0.0)
    avg_rec_days = float(np.nanmean(rec_days)) if rec_days else float("nan")
    med_overrun = float(np.nanmedian(overrun)) if overrun else float("nan")
    avg_dto = float(np.nanmean(dto)) if dto else float("nan")
    avg_minutes_above = get_recovery_time(events)

    sess_pairs = value_counts_list(sessions, normalize=True, sort=True, ascending=False, dropna=True, na_label="<NA>")
    sess = dict(sess_pairs)
    session_map = {
        "Overrun on Opening %": sess.get("Overrun on Opening %", np.nan),
        "Overrun on Late Morning %": sess.get("Overrun on Late Morning %", np.nan),
        "Overrun on Early Afternoon %": sess.get("Overrun on Early Afternoon %", np.nan),
        "Overrun on Closing %": sess.get("Overrun on Closing %", np.nan),
    }

    cum_df = pd.DataFrame(cum).mean()
    cum_df.index = [f"T{i} Rec %" for i in range(len(cum_df))]
    cum_dict = cum_df.mul(100).round(2).to_dict()

    out = {
        "Method": method,
        "Symbol": symbol.upper(),
        "Total Events": total,
        "Recovery %": recovery_pct,
        "Avg Days to Recover": avg_rec_days,
        "Avg Minutes Above Recovery": avg_minutes_above,
        "Median Overrun %": med_overrun,
        "Avg Days to Overrun": avg_dto,
        **session_map,
        **cum_dict,
    }
    return json_sanitize(out)

# ===== Tool spec + chat with tool calling =====
@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: Dict[str, Any]
    fn: Callable[..., Any]
    def as_openai_tool(self) -> Dict[str, Any]:
        return {"type": "function", "function": {
            "name": self.name, "description": self.description, "parameters": self.parameters
        }}

class GroundedChatSession:
    def __init__(
        self,
        context_text: str = "",
        model: str = "gpt-4o-mini",
        system_preface: str = (
            "Use the CONTEXT if it answers the question. "
            "If not, CALL TOOLS with the right arguments."
        ),
        max_tokens: int = 1200,
        temperature: float = 0.2,
        api_key: Optional[str] = None,
        tools: Optional[List[ToolSpec]] = None,
    ):
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._system_preface = system_preface
        self._context_text = context_text or ""
        self.messages: List[Dict[str, str]] = [{"role": "system", "content": self._build_system_msg()}]
        self._tools: Dict[str, ToolSpec] = {}
        if tools:
            for t in tools: self.add_tool(t)

    def _build_system_msg(self) -> str:
        return f"{self._system_preface}\n\n=== BEGIN CONTEXT ===\n{self._context_text}\n=== END CONTEXT ==="

    def set_context(self, context_text: str):
        self._context_text = context_text or ""
        self.messages[0]["content"] = self._build_system_msg()

    def append_context(self, extra_text: str):
        self._context_text = (self._context_text + "\n" + (extra_text or "")).strip()
        self.messages[0]["content"] = self._build_system_msg()

    def add_tool(self, tool: ToolSpec) -> None:
        if tool.name in self._tools: raise ValueError(f"Tool '{tool.name}' already registered.")
        self._tools[tool.name] = tool

    def list_tool_specs(self) -> List[Dict[str, Any]]:
        return [t.as_openai_tool() for t in self._tools.values()]

    def ask(self, user_text: str) -> str:
        self.messages.append({"role": "user", "content": user_text})
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=self.messages,
            tools=self.list_tool_specs() if self._tools else None,
            tool_choice="auto" if self._tools else "none",
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        msg = resp.choices[0].message
        tool_calls = getattr(msg, "tool_calls", None)

        if not tool_calls:
            final = msg.content or ""
            self.messages.append({"role": "assistant", "content": final})
            return final

        # execute requested tools
        tool_msgs = []
        for tc in tool_calls:
            name = tc.function.name
            arg_str = tc.function.arguments or "{}"
            try:
                args = json.loads(arg_str)
            except Exception:
                args = {"__raw__": arg_str}
            if name not in self._tools:
                out = {"error": f"Unknown tool '{name}'."}
            else:
                try:
                    out = self._tools[name].fn(**(args if isinstance(args, dict) else {"__raw__": args}))
                except Exception as e:
                    out = {"error": f"Exception in tool '{name}': {e}", "traceback": traceback.format_exc(limit=3)}
            tool_msgs.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "name": name,
                "content": json.dumps(out, default=_json_default)  # JSON-safe dump
            })


        follow = self.client.chat.completions.create(
            model=self.model,
            messages=self.messages + [{"role": "assistant", "content": msg.content, "tool_calls": tool_calls}] + tool_msgs,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        final = follow.choices[0].message.content or ""
        self.messages.append({"role": "assistant", "content": final})
        return final

    def reset(self, new_context_text: Optional[str] = None):
        if new_context_text is not None: self._context_text = new_context_text
        self.messages = [{"role": "system", "content": self._build_system_msg()}]

# ===== Tool factories =====
def make_get_hist_tool(api_key: Optional[str] = None) -> ToolSpec:
    ak = api_key or os.getenv("FMP_API_KEY")
    return ToolSpec(
        name="get_hist",
        description="Fetch historical end-of-day prices from FMP.",
        parameters={
            "type": "object",
            "properties": {
                "symbol": {"type": "string"},
                "start": {"type": "string", "description": "YYYY-MM-DD"},
                "end": {"type": "string", "description": "YYYY-MM-DD (optional)"},
            },
            "required": ["symbol", "start"],
        },
        fn=partial(get_hist, api_key=ak),
    )

def make_get_div_tool(api_key: Optional[str] = None) -> ToolSpec:
    ak = api_key or os.getenv("FMP_API_KEY")
    return ToolSpec(
        name="get_div",
        description="Fetch historical dividend data from FMP.",
        parameters={
            "type": "object",
            "properties": {
                "symbol": {"type": "string"},
                "start": {"type": "string", "description": "YYYY-MM-DD"},
                "end": {"type": "string", "description": "YYYY-MM-DD (optional)"},
            },
            "required": ["symbol", "start"],
        },
        fn=partial(get_div, api_key=ak),
    )

def make_backtest_tool(data: pd.DataFrame, api_key: Optional[str] = None, plot: bool = False) -> ToolSpec:
    ak = api_key or os.getenv("FMP_API_KEY")
    bound = partial(
        backtest_new_tool,
        data=data,
        api_key=ak,
        get_trading_session=get_trading_session,
        value_counts_list=value_counts_list,
        plot=plot,
    )
    return ToolSpec(
        name="backtest_new",
        description=("Run dividend recovery backtest for a symbol/method on the preloaded dataset; "
                     "returns Recovery %, Avg Days to Recover, Overrun %, session distribution, and T0..T4 recovery rates."),
        parameters={
            "type": "object",
            "properties": {
                "symbol": {"type": "string"},
                "method": {"type": "string", "enum": ["t-1","t-1-15","t-1+15","60/40","70/30"]},
                "recovery_window": {"type": "integer"},
            },
            "required": ["symbol", "method", "recovery_window"],
        },
        fn=bound,
    )
