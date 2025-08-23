# bot.py
# Bot avanzado (paper trading + opción de ejecutar real con botón seguro)
# - Conecta a Binance (lee API keys desde variables de entorno)
# - Genera señales con ML ligero y reglas de acumulación
# - No envía órdenes reales salvo que EXECUTE_REAL=true y SIMULATE_TRADES=false
# - Panel web mínimo con token protegido para activar/desactivar EXECUTE_REAL
# Requerimientos: python-binance, pandas, numpy, scikit-learn, flask

import os, json, time, threading, pickle, traceback
from datetime import datetime, timedelta
from functools import wraps

import numpy as np
import pandas as pd
from binance.client import Client
from flask import Flask, request, jsonify, abort

# ---------------- CONFIG (ajustar si quieres) ----------------
CFG = {
    "SYMBOLS": ["BTCUSDT", "ETHUSDT"],
    "INTERVAL": "1h",
    "LOOKBACK": 800,
    "TRAIN_DAYS": 90,
    "HORIZON_BARS": 1,
    "RETRAIN_HOURS": 24,
    "PARTIAL_BUY_FRAC": 0.25,
    "PARTIAL_SELL_FRAC": 0.25,
    "MAX_POS_PCT": 0.9,
    "MIN_CASH_PCT": 0.05,
    "ML_BUY_PROB": 0.60,
    "ML_SELL_PROB": 0.60,
    "COOLDOWN_MIN": 10,
    "VERIFY_AFTER_N_BARS": 1,
    "CAPITAL_INIT": 100.0
}

# ---------------- Environment / Secrets (SET IN RENDER) ----------------
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY", "").strip() or None
BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET", "").strip() or None
MANAGEMENT_TOKEN = os.getenv("MANAGEMENT_TOKEN", "changeme")  # token para proteger el toggle
# Flags
SIMULATE_TRADES = os.getenv("SIMULATE_TRADES", "True").lower() in ("1","true","yes")
# Nota: EXECUTE_REAL se controla por endpoint y por variable de entorno inicial
EXECUTE_REAL = os.getenv("EXECUTE_REAL", "False").lower() in ("1","true","yes")

# ---------------- Cliente Binance (solo inicializa si hay claves, aunque para klines no son necesarias) ----------------
if BINANCE_API_KEY and BINANCE_API_SECRET:
    client = Client(BINANCE_API_KEY, BINANCE_API_SECRET)
else:
    client = Client()  # client público (lectura)
print("Binance client ready. SIMULATE_TRADES=", SIMULATE_TRADES, "EXECUTE_REAL=", EXECUTE_REAL)

# ---------------- Utilities ----------------
def _load_json(path, default):
    try:
        if os.path.exists(path):
            with open(path,"r") as f:
                return json.load(f)
    except Exception:
        pass
    return default

def _save_json(path, obj):
    with open(path,"w") as f:
        json.dump(obj, f, indent=2, default=str)

# ---------------- Simple indicators (no TA-Lib) ----------------
def ema(series, period): return series.ewm(span=period, adjust=False).mean()
def rsi(series, period=14):
    delta = series.diff()
    up = np.where(delta>0, delta, 0.0)
    dn = np.where(delta<0, -delta, 0.0)
    roll_up = pd.Series(up, index=series.index).ewm(alpha=1/period, adjust=False).mean()
    roll_dn = pd.Series(dn, index=series.index).ewm(alpha=1/period, adjust=False).mean()
    rs = roll_up/(roll_dn+1e-12)
    return 100 - (100/(1+rs))
def atr(high, low, close, period=14):
    prev_close = close.shift(1)
    tr = pd.concat([(high-low),(high-prev_close).abs(),(low-prev_close).abs()],axis=1).max(axis=1)
    return tr.rolling(period).mean()

def add_features(df):
    df = df.copy()
    df["ema_fast"] = ema(df["close"], 21)
    df["ema_slow"] = ema(df["close"], 55)
    df["rsi"] = rsi(df["close"], 14)
    df["atr"] = atr(df["high"], df["low"], df["close"], 14)
    df["atrp"] = df["atr"]/(df["close"]+1e-12)
    df["dist_ema"] = (df["close"]-df["ema_slow"])/(df["atr"]+1e-12)
    df["ret_1"] = df["close"].pct_change(1)
    df["vol_chg"] = df["volume"].pct_change(3)
    return df

# ---------------- Fetch klines ----------------
def fetch_klines(symbol, interval, limit=500, start_ms=None, end_ms=None):
    if start_ms is None and end_ms is None:
        raw = client.get_klines(symbol=symbol, interval=interval, limit=limit)
    else:
        raw = client.get_klines(symbol=symbol, interval=interval, startTime=start_ms, endTime=end_ms, limit=limit)
    if not raw:
        return pd.DataFrame()
    cols = ["open_time","open","high","low","close","volume","close_time","a","b","c","d","e"]
    df = pd.DataFrame(raw, columns=cols)
    df["time_open"] = pd.to_datetime(df["open_time"].astype(np.int64), unit="ms").dt.tz_localize(None)
    for c in ["open","high","low","close","volume"]:
        df[c] = df[c].astype(float)
    return df[["time_open","open","high","low","close","volume"]]

# ---------------- Simple ML loader/trainer (lightweight) ----------------
from sklearn.experimental import enable_hist_gradient_boosting  # noqa
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split

def make_labels(df, horizon=1):
    df = df.copy()
    df["future_close"] = df["close"].shift(-horizon)
    df["label_up"] = (df["future_close"] > df["close"]).astype(int)
    return df.dropna()

def train_ml(symbol, model_path):
    end = datetime.utcnow()
    start = end - timedelta(days=CFG["TRAIN_DAYS"])
    start_ms = int(start.timestamp()*1000)
    end_ms = int(end.timestamp()*1000)
    df = fetch_klines(symbol, CFG["INTERVAL"], limit=1000, start_ms=start_ms, end_ms=end_ms)
    if df.empty or len(df) < 200:
        raise RuntimeError("Hist insufficient for train")
    df = add_features(df).dropna()
    df = make_labels(df, CFG["HORIZON_BARS"])
    features = ["ema_fast","ema_slow","rsi","atr","atrp","dist_ema","ret_1","vol_chg","close","volume"]
    X = df[features].values
    y = df["label_up"].values
    Xtr, Xval, ytr, yval = train_test_split(X,y,test_size=0.2, shuffle=False)
    m = HistGradientBoostingClassifier(max_depth=6, learning_rate=0.08, max_iter=300)
    m.fit(Xtr,ytr)
    acc = m.score(Xval,yval)
    with open(model_path,"wb") as f:
        pickle.dump({"model":m,"features":features}, f)
    return acc

def load_ml(model_path, symbol):
    if os.path.exists(model_path):
        with open(model_path,"rb") as f:
            p = pickle.load(f)
        return p["model"], p["features"]
    # fallback: try to train
    acc = train_ml(symbol, model_path)
    m,f = load_ml(model_path, symbol)
    return m,f

# ---------------- Paper trading state helpers ----------------
def ensure_state(path, capital_init):
    state = _load_json(path,{})
    state.setdefault("cash_usdt", float(capital_init))
    state.setdefault("asset_qty", 0.0)
    state.setdefault("avg_cost", 0.0)
    state.setdefault("last_buy_price", None)
    state.setdefault("last_sell_ref", None)
    state.setdefault("last_signal_time", None)
    state.setdefault("equity_history", [])
    state.setdefault("last_seen_close_time", None)
    state.setdefault("last_retrain_time", None)
    return state

def mark_to_market(state, price):
    return state["cash_usdt"] + state["asset_qty"]*price

def buy_partial(state, price):
    equity = mark_to_market(state, price)
    max_pos = equity * CFG["MAX_POS_PCT"]
    pos_value = state["asset_qty"]*price
    if pos_value >= max_pos:
        return 0.0, "pos at max"
    buy_usdt = state["cash_usdt"] * CFG["PARTIAL_BUY_FRAC"]
    room = max_pos - pos_value
    buy_usdt = min(buy_usdt, room)
    if buy_usdt < 5.0:
        return 0.0, "too small buy_usdt"
    qty = buy_usdt/price
    new_avg = (state["avg_cost"]*state["asset_qty"] + qty*price)/(state["asset_qty"]+qty)
    state["asset_qty"] += qty
    state["cash_usdt"] -= buy_usdt
    state["avg_cost"] = new_avg
    state["last_buy_price"] = price
    return qty, f"bought {qty:.8f} @ {price:.2f}"

def sell_partial(state, price):
    if state["asset_qty"] <= 0:
        return 0.0, "no pos"
    qty = state["asset_qty"] * CFG["PARTIAL_SELL_FRAC"]
    if qty*price < 5.0:
        return 0.0, "small sell"
    state["asset_qty"] -= qty
    state["cash_usdt"] += qty*price
    state["last_sell_ref"] = price
    return qty, f"sold {qty:.8f} @ {price:.2f}"

# ---------------- SIGNAL RULES ----------------
def compute_risk_row(r):
    base = 0.0
    base += min(10, 180*float(r.get("atrp",0.0)))
    rsi_v = float(r.get("rsi",50.0))
    if rsi_v > 70: base += 2.0
    if rsi_v < 30: base -= 1.0
    if float(r.get("dist_ema",0.0)) < -2.0: base += 2.0
    base = max(1.0, min(10.0, base))
    return int(round(base))

def decide_signal_rules(r):
    price = float(r["close"]); ema_s = float(r["ema_slow"]); atr_v = float(r["atr"]); rsi_v = float(r["rsi"])
    buy_cond  = (rsi_v < 30) or (price < ema_s - CFG["atr_k_buy"]*atr_v if "atr_k_buy" in CFG else (price < ema_s - 1.5*atr_v))
    sell_cond = (rsi_v > 70) or (price > ema_s + CFG.get("atr_k_sell",1.5)*atr_v)
    if buy_cond and not sell_cond:
        return "BUY", f"RSI={rsi_v:.1f} or below EMA-ATR"
    if sell_cond and not buy_cond:
        return "SELL", f"RSI={rsi_v:.1f} or above EMA+ATR"
    return "HOLD", "no edge"

# ---------------- ORDER (guarded) ----------------
def place_order_real(symbol, side, qty):
    global SIMULATE_TRADES, EXECUTE_REAL
    if SIMULATE_TRADES:
        return {"sim":"ok"}
    if not EXECUTE_REAL:
        return {"skipped":"EXECUTE_REAL false"}
    # require keys present
    if not BINANCE_API_KEY or not BINANCE_API_SECRET:
        raise RuntimeError("Missing Binance keys in env")
    if side=="BUY":
        return client.order_market_buy(symbol=symbol, quantity=qty)
    else:
        return client.order_market_sell(symbol=symbol, quantity=qty)

# ---------------- Predictions log helpers ----------------
def append_prediction(path, entry):
    log = _load_json(path, [])
    log.append(entry)
    _save_json(path, log)

def verify_pending(pred_path, now_time, current_close_time, current_price):
    log = _load_json(pred_path, [])
    changed = False
    for r in log:
        if r.get("success") is not None:
            continue
        pred_close_time = pd.to_datetime(r["bar_close_time"]).tz_localize(None)
        if current_close_time >= pred_close_time:
            if r["decision"]=="BUY":
                r["actual_next_move"] = "up" if current_price>r["entry_price"] else "down"
                r["success"] = (current_price>r["entry_price"])
            elif r["decision"]=="SELL":
                r["actual_next_move"] = "down" if current_price<r["entry_price"] else "up"
                r["success"] = (current_price<r["entry_price"])
            else:
                r["actual_next_move"]="flat"; r["success"]=None
            r["verified_at"]=now_time.isoformat(); changed=True
    if changed:
        _save_json(pred_path, log)

# ---------------- Live loop ----------------
def live_loop(symbol):
    model_path = f"model_{symbol}.pkl"
    state_path = f"state_{symbol}.json"
    pred_path  = f"preds_{symbol}.json"
    state = ensure_state(state_path, CFG["CAPITAL_INIT"])

    # load or train model
    try:
        model, features = load_ml(model_path, symbol)
    except Exception as e:
        print(f"[{symbol}] ML load failed: {e}")
        model, features = None, None

    interval_minutes = {"1h":60,"4h":240,"1d":1440}.get(CFG["INTERVAL"],60)
    verify_offset = CFG["VERIFY_AFTER_N_BARS"]*interval_minutes

    print(f"[{symbol}] started loop. Simulate={SIMULATE_TRADES} ExecReal={EXECUTE_REAL}")

    while True:
        try:
            now = pd.Timestamp.utcnow().to_pydatetime().replace(tzinfo=None)
            df = fetch_klines(symbol, CFG["INTERVAL"], limit=CFG["LOOKBACK"])
            if df.empty or len(df) < 60:
                time.sleep(10); continue
            df = add_features(df).dropna()
            # use penultimate closed bar
            r = df.iloc[-2]
            last_close_time = r["time_open"] + timedelta(minutes=interval_minutes)
            # avoid reprocessing same bar
            if state.get("last_seen_close_time") and last_close_time.isoformat() <= state["last_seen_close_time"]:
                time.sleep(5); continue
            state["last_seen_close_time"] = last_close_time.isoformat()

            decision, plan = decide_signal_rules(r)
            # ML filter
            if model is not None:
                feat_vec = np.array([r[f] for f in features], dtype=float).reshape(1,-1)
                prob_up = float(model.predict_proba(feat_vec)[0][1])
                prob_down = 1.0 - prob_up
            else:
                prob_up, prob_down = 0.5, 0.5

            if decision=="BUY" and prob_up < CFG["ML_BUY_PROB"]:
                plan += f" | ML filt buy ({prob_up:.2f})"; decision="HOLD"
            if decision=="SELL" and prob_down < CFG["ML_SELL_PROB"]:
                plan += f" | ML filt sell ({prob_down:.2f})"; decision="HOLD"

            price = float(r["close"])
            risk = compute_risk_row(r)
            equity = mark_to_market(state, price)

            # cooldown
            if state.get("last_signal_time"):
                last_sig = pd.to_datetime(state["last_signal_time"]).tz_localize(None)
                if (now - last_sig).total_seconds() < CFG["COOLDOWN_MIN"]*60:
                    decision="HOLD"; plan=f"Cooldown {CFG['COOLDOWN_MIN']}m"

            # DCA / ladder logic (simple)
            if decision=="HOLD" and state["asset_qty"]>0 and state["last_buy_price"]:
                drop = (state["last_buy_price"] - price)/max(state["last_buy_price"],1e-9)
                if drop >= CFG["dca_step_pct"] if "dca_step_pct" in CFG else 0.03:
                    decision="BUY"; plan=f"DCA drop {drop*100:.1f}%"

            # ladder sell by gain
            if decision=="HOLD" and state["asset_qty"]>0:
                ref = state.get("last_sell_ref") or state.get("avg_cost") or price
                gain = (price - ref)/max(ref,1e-9)
                if gain >= CFG.get("sell_step_gain",0.05):
                    decision="SELL"; plan=f"Take partial gain {gain*100:.1f}%"

            # execute paper / optionally real
            filled_text = "-"
            qty = 0.0
            if decision=="BUY":
                min_cash = equity*CFG["MIN_CASH_PCT"]
                if state["cash_usdt"] <= min_cash:
                    filled_text = "no cash"
                else:
                    qty, filled_text = buy_partial(state, price)
                    if qty>0:
                        state["last_signal_time"] = now.isoformat()
                        # optional: send real order
                        try:
                            res = place_order_real(symbol,"BUY",round(qty,8))
                            if not SIMULATE_TRADES:
                                filled_text += f" | real_order:{getattr(res,'get',lambda k:res)('orderId','ok')}"
                        except Exception as e:
                            filled_text += f" | real_err:{e}"
            elif decision=="SELL":
                qty, filled_text = sell_partial(state, price)
                if qty>0:
                    state["last_signal_time"] = now.isoformat()
                    try:
                        res = place_order_real(symbol,"SELL",round(qty,8))
                        if not SIMULATE_TRADES:
                            filled_text += f" | real_order:{getattr(res,'get',lambda k:res)('orderId','ok')}"
                    except Exception as e:
                        filled_text += f" | real_err:{e}"

            # persist prediction
            pred = {
                "time": now.isoformat(),
                "bar_close_time": (last_close_time + timedelta(minutes=verify_offset)).isoformat(),
                "symbol": symbol,
                "decision": decision,
                "plan": plan,
                "entry_price": price,
                "prob_up": round(prob_up,4),
                "prob_down": round(prob_down,4),
                "risk": risk,
                "equity": equity,
                "success": None
            }
            append_prediction(pred_path, pred)
            verify_pending(pred_path, now, last_close_time, price)

            # save state
            equity_now = mark_to_market(state, price)
            state["equity_history"].append([now.isoformat(), equity_now])
            _save_json(state_path, state)

            print(f"[{symbol}] {now} | P={price:.2f} | Risk={risk} | ProbUp={prob_up:.2f} | {decision} | {filled_text} | Equity={equity_now:.2f}")

            # periodic retrain
            last_rt = state.get("last_retrain_time")
            need_rt = False
            if last_rt is None:
                need_rt = True
            else:
                last_rt = pd.to_datetime(last_rt).tz_localize(None)
                need_rt = (now - last_rt).total_seconds() > CFG["RETRAIN_HOURS"]*3600
            if need_rt:
                try:
                    train_ml(symbol, f"model_{symbol}.pkl")
                    state["last_retrain_time"] = now.isoformat()
                    _save_json(state_path, state)
                    model, features = load_ml(f"model_{symbol}.pkl", symbol)
                    print(f"[{symbol}] retrained and loaded model.")
                except Exception as e:
                    print(f"[{symbol}] retrain failed: {e}\n{traceback.format_exc()}")

            time.sleep(5)  # short heartbeat; loop processes only on new closed bar due to last_seen_close_time
        except KeyboardInterrupt:
            print("Interrupted by user."); break
        except Exception as e:
            print(f"Error in loop {symbol}: {e}\n{traceback.format_exc()}")
            time.sleep(10)

# ---------------- Minimal Flask management API ----------------
app = Flask(__name__)

def require_token(fn):
    @wraps(fn)
    def wrapper(*a, **kw):
        token = request.headers.get("X-MGMT-TOKEN") or request.args.get("token")
        if token != MANAGEMENT_TOKEN:
            return abort(403)
        return fn(*a, **kw)
    return wrapper

@app.route("/status")
def status():
    return jsonify({"simulate": SIMULATE_TRADES, "execute_real": EXECUTE_REAL, "symbols": CFG["SYMBOLS"]})

@app.route("/toggle_execute", methods=["POST"])
@require_token
def toggle_execute():
    global EXECUTE_REAL
    body = request.json or {}
    val = body.get("execute")
    if val is None:
        EXECUTE_REAL = not EXECUTE_REAL
    else:
        EXECUTE_REAL = bool(val)
    return jsonify({"execute_real": EXECUTE_REAL})

# ---------------- Start threads + Flask ----------------
def start_bot_threads():
    threads = []
    for s in CFG["SYMBOLS"]:
        t = threading.Thread(target=live_loop, args=(s,), daemon=True)
        t.start()
        threads.append(t)
    return threads

if __name__ == "__main__":
    # launch workers
    start_bot_threads()
    # run flask (Render expects a web server)
    port = int(os.getenv("PORT", "8000"))
    app.run(host="0.0.0.0", port=port)