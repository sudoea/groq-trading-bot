"""
Live Groq -> Bitget Trading Bot (v1 - Render Ready)
Works with DEMO and LIVE trading on Bitget Futures
- Groq AI signal generation
- ATR-based SL/TP
- Telegram alerts
- CSV logging
- 24/7 cloud deployment

Requirements:
pip install python-dotenv pandas numpy requests
"""

import csv
import html
import json
import os
import re
import requests
from datetime import datetime, timezone

from dotenv import load_dotenv
import pandas as pd
import numpy as np
from time import sleep


load_dotenv()


# =============== CONFIGURATION ===============
# Bitget Settings
BITGET_API_KEY = os.getenv("BITGET_API_KEY", "")
BITGET_SECRET_KEY = os.getenv("BITGET_SECRET_KEY", "")
BITGET_PASSPHRASE = os.getenv("BITGET_PASSPHRASE", "")
BITGET_SYMBOL = os.getenv("BITGET_SYMBOL", "BTCUSDT")
BITGET_MARGIN_MODE = os.getenv("BITGET_MARGIN_MODE", "cross")
BITGET_DEMO_MODE = os.getenv("BITGET_DEMO_MODE", "true").lower() in ("1", "true", "yes")
BITGET_LEVERAGE = int(os.getenv("BITGET_LEVERAGE", "1"))

BITGET_URL = (
    "https://api.bitget.com/v2/mix/orders/place-order"
    if not BITGET_DEMO_MODE
    else "https://api.bitget.com/v2/mix/orders/place-order"  # Demo uses same URL with X-SIMULATED header
)

# Trading Settings
CANDLES_TO_FETCH = int(os.getenv("CANDLES_TO_FETCH", "100"))
LOT_SIZE = float(os.getenv("LOT_SIZE", "0.01"))
MIN_CONFIDENCE = float(os.getenv("MIN_CONFIDENCE", "70.0"))

# Groq Settings
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_ENDPOINT = os.getenv(
    "GROQ_ENDPOINT",
    "https://api.groq.com/openai/v1/chat/completions"
)
GROQ_TEMP = float(os.getenv("GROQ_TEMP", "0.05"))
GROQ_MAX_TOKENS = int(os.getenv("GROQ_MAX_TOKENS", "200"))
GROQ_RETRY_COUNT = int(os.getenv("GROQ_RETRY_COUNT", "3"))
GROQ_RETRY_DELAY = float(os.getenv("GROQ_RETRY_DELAY", "2.0"))

# Telegram Settings
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

# Bot Settings
DRY_RUN = os.getenv("DRY_RUN", "false").lower() in ("1", "true", "yes")
LOOP_INTERVAL = int(os.getenv("LOOP_INTERVAL", "300"))

# ATR Settings
USE_ATR_LEVELS = os.getenv("USE_ATR_LEVELS", "true").lower() in ("1", "true", "yes")
ATR_PERIOD = int(os.getenv("ATR_PERIOD", "14"))
ATR_SL_MULTIPLIER = float(os.getenv("ATR_SL_MULTIPLIER", "2.0"))
ATR_TP_MULTIPLIER = float(os.getenv("ATR_TP_MULTIPLIER", "3.0"))

LOGFILE = os.getenv("TRADE_LOG_FILE", "trade_log.csv")


# =============== VALIDATION ===============
def validate_config():
    """Validate configuration on startup"""
    errors = []

    if not BITGET_API_KEY or not BITGET_SECRET_KEY or not BITGET_PASSPHRASE:
        errors.append("Bitget API credentials missing (KEY, SECRET, PASSPHRASE)")

    if not GROQ_API_KEY or not GROQ_API_KEY.startswith("gsk_"):
        errors.append("GROQ_API_KEY missing or invalid (must start with 'gsk_')")

    if not TELEGRAM_TOKEN:
        print("⚠️  WARNING: TELEGRAM_TOKEN not set - notifications disabled")

    if MIN_CONFIDENCE < 0 or MIN_CONFIDENCE > 100:
        errors.append(f"MIN_CONFIDENCE must be 0-100, got {MIN_CONFIDENCE}")

    if LOT_SIZE <= 0:
        errors.append(f"LOT_SIZE must be positive, got {LOT_SIZE}")

    if errors:
        for err in errors:
            print(f"❌ CONFIG ERROR: {err}")
        raise ValueError("Invalid configuration")

    print("✅ Configuration validated successfully")
    print(f"✅ Bitget: {BITGET_SYMBOL} | Mode: {'DEMO' if BITGET_DEMO_MODE else 'LIVE'}")
    print(f"✅ Leverage: {BITGET_LEVERAGE}x | Margin: {BITGET_MARGIN_MODE}")
    if USE_ATR_LEVELS:
        print(f"✅ ATR-based SL/TP enabled (period={ATR_PERIOD})")
    return True


# =============== HELPERS ===============
def send_telegram(text: str, retry_count: int = 2) -> bool:
    """Send a Telegram message with retry logic"""
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("⚠️  Telegram not configured - skipping notification")
        return False

    safe = html.escape(text)
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"

    for attempt in range(1, retry_count + 1):
        try:
            r = requests.get(
                url,
                params={
                    "chat_id": TELEGRAM_CHAT_ID,
                    "text": safe,
                    "parse_mode": "HTML"
                },
                timeout=15
            )
            if r.ok:
                print(f"✅ Telegram sent (attempt {attempt})")
                return True
            else:
                print(f"⚠️  Telegram HTTP {r.status_code}: {r.text[:200]}")
        except Exception as e:
            print(f"⚠️  Telegram exception (attempt {attempt}): {type(e).__name__}: {e}")

        if attempt < retry_count:
            sleep(1)

    return False


def log_trade(row: dict):
    """Append a row to CSV logfile"""
    header = [
        "timestamp", "symbol", "signal", "confidence", "entry", "tp", "sl",
        "lot", "executed", "result", "notes"
    ]
    exists = os.path.exists(LOGFILE)

    try:
        with open(LOGFILE, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=header)
            if not exists:
                writer.writeheader()

            for key in header:
                if key not in row:
                    row[key] = ""

            writer.writerow(row)
        print(f"✅ Trade logged to {LOGFILE}")
    except Exception as e:
        print(f"❌ Failed to write log: {e}")


# =============== CANDLE DATA (Using Public API) ===============
def fetch_bitget_candles(symbol: str, period: str = "15m", count: int = 100):
    """Fetch candles from Bitget public API"""
    try:
        # Bitget public candles endpoint (no auth needed)
        url = f"https://api.bitget.com/v2/mix/market/candles"

        params = {
            "symbol": symbol,
            "granularity": period,
            "limit": count
        }

        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()

        data = resp.json()

        if not data.get("data"):
            print("❌ No candle data from Bitget")
            return None

        # Parse candles
        candles = []
        for candle in data["data"]:
            # Bitget format: [time, open, high, low, close, volume, quote_asset_volume]
            candles.append({
                "time": int(candle[0]) // 1000,  # Convert to seconds
                "open": float(candle[1]),
                "high": float(candle[2]),
                "low": float(candle[3]),
                "close": float(candle[4]),
                "volume": float(candle[5])
            })

        df = pd.DataFrame(candles)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        df = df[["time", "open", "high", "low", "close", "volume"]].set_index("time")

        print(f"✅ Fetched {len(df)} candles for {symbol}")
        return df

    except Exception as e:
        print(f"❌ Failed to fetch candles: {type(e).__name__}: {e}")
        return None


# =============== ATR CALCULATION ===============
def calculate_atr(df: pd.DataFrame, period: int = 14) -> float:
    """Calculate Average True Range (ATR)"""
    try:
        if len(df) < period:
            return 0.0

        df_calc = df.copy()
        df_calc['tr0'] = abs(df_calc['high'] - df_calc['low'])
        df_calc['tr1'] = abs(df_calc['high'] - df_calc['close'].shift())
        df_calc['tr2'] = abs(df_calc['low'] - df_calc['close'].shift())
        df_calc['tr'] = df_calc[['tr0', 'tr1', 'tr2']].max(axis=1)

        atr = df_calc['tr'].rolling(window=period).mean().iloc[-1]

        if np.isnan(atr):
            return 0.0

        print(f"📊 ATR ({period}-period): ${atr:,.2f}")
        return atr

    except Exception as e:
        print(f"❌ ATR calculation failed: {e}")
        return 0.0


def calculate_atr_levels(entry_price: float, signal: str, atr: float):
    """Calculate SL/TP based on ATR"""
    if atr <= 0:
        return {
            "stop_loss": entry_price * 0.99 if signal == "BUY" else entry_price * 1.01,
            "take_profit": entry_price * 1.01 if signal == "BUY" else entry_price * 0.99,
            "atr_based": False
        }

    if signal == "BUY":
        sl = entry_price - (atr * ATR_SL_MULTIPLIER)
        tp = entry_price + (atr * ATR_TP_MULTIPLIER)
    else:
        sl = entry_price + (atr * ATR_SL_MULTIPLIER)
        tp = entry_price - (atr * ATR_TP_MULTIPLIER)

    print(
        f"🎯 ATR Levels: SL=${sl:,.2f}, TP=${tp:,.2f} "
        f"(ATR*{ATR_SL_MULTIPLIER}/${ATR_TP_MULTIPLIER})"
    )

    return {
        "stop_loss": sl,
        "take_profit": tp,
        "atr_based": True,
        "atr_value": atr
    }


# =============== GROQ API ===============
def is_valid_json_response(text: str) -> bool:
    """Check if response contains valid JSON"""
    if not text or len(text) < 50:
        return False

    valid_chars = sum(1 for c in text if c.isalnum() or c in '{}[]:," .-')
    validity_ratio = valid_chars / len(text)

    if validity_ratio < 0.6:
        print(f"⚠️  Response validity: {validity_ratio*100:.1f}% (too much garbage)")
        return False

    if '{' not in text or '}' not in text:
        return False

    return True


def call_groq_model(candles_df, atr_value: float = 0.0, retry_count: int = None):
    """Call Groq API for signal generation"""
    if retry_count is None:
        retry_count = GROQ_RETRY_COUNT

    if not GROQ_API_KEY or not GROQ_API_KEY.startswith("gsk_"):
        print("❌ GROQ_API_KEY missing or invalid")
        return None

    snippet = candles_df[
        ["open", "high", "low", "close"]
    ].tail(10).reset_index().to_dict(orient="records")

    system_msg = (
        "You are a trading signal AI. Return ONLY valid JSON with: "
        '{"signal":"BUY" or "SELL" or "HOLD", "confidence":0-100, '
        '"entry_price":number, "take_profit":number, "stop_loss":number, '
        '"comment":"max 20 chars"}. No other text, no markdown.'
    )
    user_msg = f"{BITGET_SYMBOL} M15 last 10 candles: {json.dumps(snippet, default=str)}"

    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg}
        ],
        "temperature": GROQ_TEMP,
        "max_tokens": GROQ_MAX_TOKENS
    }

    for attempt in range(1, retry_count + 1):
        try:
            print(f"📡 Groq request (attempt {attempt}/{retry_count})...")
            resp = requests.post(
                GROQ_ENDPOINT,
                headers={
                    "Authorization": f"Bearer {GROQ_API_KEY}",
                    "Content-Type": "application/json"
                },
                json=payload,
                timeout=30
            )

            print(f"📥 Groq response: HTTP {resp.status_code}")

            if resp.status_code == 401:
                print("❌ Groq auth failed - invalid API key")
                return None

            if resp.status_code == 429:
                print(f"⚠️  Groq rate limited (attempt {attempt})")
                if attempt < retry_count:
                    sleep(GROQ_RETRY_DELAY * attempt)
                    continue
                return None

            resp.raise_for_status()

            try:
                data = resp.json()
            except Exception as e:
                print(f"❌ Failed to parse Groq JSON response: {e}")
                if attempt < retry_count:
                    sleep(GROQ_RETRY_DELAY)
                    continue
                return None

            content = None
            if (isinstance(data, dict) and "choices" in data and
                    len(data["choices"]) > 0):
                choice = data["choices"][0]
                if isinstance(choice.get("message"), dict):
                    content = choice["message"].get("content")
                elif isinstance(choice.get("text"), str):
                    content = choice["text"]

            if not content:
                print(f"❌ No content from Groq")
                if attempt < retry_count:
                    sleep(GROQ_RETRY_DELAY)
                    continue
                return None

            print(f"📄 Groq response content length: {len(content)}")

            if not is_valid_json_response(content):
                print("⚠️  Response appears corrupted")
                if attempt < retry_count:
                    print(f"Retrying...")
                    sleep(GROQ_RETRY_DELAY)
                    continue
                print("❌ Groq returning corrupted responses - using fallback")
                return None

            m = re.search(r"(\{[\s\S]*\})", content)
            if not m:
                print("❌ No JSON block found in Groq output")
                if attempt < retry_count:
                    sleep(GROQ_RETRY_DELAY)
                    continue
                return None

            json_str = m.group(1)

            try:
                parsed = json.loads(json_str)
            except Exception as e:
                print(f"❌ Failed to parse extracted JSON: {e}")
                if attempt < retry_count:
                    sleep(GROQ_RETRY_DELAY)
                    continue
                return None

            required_keys = {
                "signal", "confidence", "entry_price",
                "take_profit", "stop_loss"
            }
            missing = required_keys - set(parsed.keys())
            if missing:
                print(f"❌ Missing keys in AI response: {missing}")
                if attempt < retry_count:
                    sleep(GROQ_RETRY_DELAY)
                    continue
                return None

            try:
                signal = str(parsed["signal"]).upper()
                confidence = float(parsed["confidence"])
                entry_price = float(parsed["entry_price"])
                take_profit = float(parsed["take_profit"])
                stop_loss = float(parsed["stop_loss"])

                if signal not in ("BUY", "SELL", "HOLD"):
                    print(f"❌ Invalid signal: {signal}")
                    if attempt < retry_count:
                        sleep(GROQ_RETRY_DELAY)
                        continue
                    return None

                if not (0 <= confidence <= 100):
                    print(f"❌ Confidence out of range: {confidence}")
                    if attempt < retry_count:
                        sleep(GROQ_RETRY_DELAY)
                        continue
                    return None

                if entry_price <= 0 or take_profit <= 0 or stop_loss <= 0:
                    print("❌ Invalid prices (must be positive)")
                    if attempt < retry_count:
                        sleep(GROQ_RETRY_DELAY)
                        continue
                    return None

                if stop_loss == entry_price or take_profit == entry_price:
                    print("❌ SL/TP cannot equal entry price")
                    if attempt < retry_count:
                        sleep(GROQ_RETRY_DELAY)
                        continue
                    return None

            except (ValueError, TypeError) as e:
                print(f"❌ Failed to parse AI values: {e}")
                if attempt < retry_count:
                    sleep(GROQ_RETRY_DELAY)
                    continue
                return None

            print(f"✅ Groq response valid: {signal} conf={confidence}%")
            return parsed

        except requests.exceptions.Timeout:
            print(f"⚠️  Groq timeout (attempt {attempt})")
            if attempt < retry_count:
                sleep(GROQ_RETRY_DELAY)
        except requests.exceptions.ConnectionError as e:
            print(f"⚠️  Groq connection error (attempt {attempt}): {e}")
            if attempt < retry_count:
                sleep(GROQ_RETRY_DELAY)
        except Exception as e:
            print(f"❌ Groq request failed: {type(e).__name__}: {e}")
            if attempt < retry_count:
                sleep(GROQ_RETRY_DELAY)
            continue

    print(f"❌ Groq API failed after {retry_count} attempts")
    return None


# =============== BITGET TRADING ===============
def place_bitget_order(signal: str, entry: float, tp: float, sl: float, lot: float) -> dict:
    """Place order on Bitget Futures"""
    result = {
        "success": False,
        "order_id": None,
        "error": None,
        "details": ""
    }

    try:
        if signal == "BUY":
            side = "buy"
            position_side = "long"
        elif signal == "SELL":
            side = "sell"
            position_side = "short"
        else:
            result["error"] = "Invalid signal"
            return result

        # Build order request
        request_body = {
            "symbol": BITGET_SYMBOL,
            "productType": "mix",  # Futures
            "side": side,
            "positionSide": position_side,
            "orderType": "market",
            "size": str(lot),
            "marginMode": BITGET_MARGIN_MODE,
            "leverage": str(BITGET_LEVERAGE)
        }

        headers = {
            "Authorization": f"Bearer {BITGET_API_KEY}",
            "Content-Type": "application/json"
        }

        # Add demo header if demo mode
        if BITGET_DEMO_MODE:
            headers["X-SIMULATED"] = "1"
            print("📝 DEMO MODE - Trading simulated")

        print(f"📤 Sending Bitget order: {request_body}")

        resp = requests.post(
            BITGET_URL,
            json=request_body,
            headers=headers,
            timeout=10
        )

        print(f"📥 Bitget response: HTTP {resp.status_code}")

        if resp.status_code != 200:
            result["error"] = f"Bitget HTTP {resp.status_code}: {resp.text[:200]}"
            return result

        data = resp.json()

        if data.get("code") != "00000":
            result["error"] = f"Bitget error: {data.get('msg', 'Unknown error')}"
            return result

        order_id = data.get("data", {}).get("orderId")
        if not order_id:
            result["error"] = "No order ID returned"
            return result

        result["success"] = True
        result["order_id"] = order_id
        result["details"] = (
            f"Bitget order {order_id} placed\n"
            f"Symbol: {BITGET_SYMBOL}\n"
            f"Side: {side.upper()}\n"
            f"Size: {lot}\n"
            f"Leverage: {BITGET_LEVERAGE}x"
        )
        print(f"✅ Order executed: {result['details']}")
        return result

    except Exception as e:
        result["error"] = f"Exception: {type(e).__name__}: {e}"
        print(f"❌ Order placement exception: {result['error']}")
        return result


# =============== MAIN ===============
def main(execute_trades: bool = True):
    print("\n" + "="*60)
    print(
        f"Live Groq -> Bitget Bot - "
        f"{datetime.now(timezone.utc).isoformat()}"
    )
    print(f"Symbol: {BITGET_SYMBOL} | Mode: {'DEMO' if BITGET_DEMO_MODE else 'LIVE'}")
    print("="*60 + "\n")

    try:
        validate_config()
    except ValueError as e:
        print(f"❌ Startup failed: {e}")
        return

    # Fetch candles
    df = fetch_bitget_candles(BITGET_SYMBOL, "15m", CANDLES_TO_FETCH)
    if df is None or df.empty:
        send_telegram("❌ Failed to fetch candles")
        return

    last_price = float(df['close'].iloc[-1])
    print(f"Last price: ${last_price:,.2f}")

    # Calculate ATR
    atr = calculate_atr(df, ATR_PERIOD)

    # Call Groq
    ai_raw = call_groq_model(df, atr)

    # If Groq fails, use ATR-based fallback
    if not ai_raw:
        print("\n⚠️  Groq API failed - using ATR-based fallback signal")

        close_prices = df['close'].tail(5).values
        if len(close_prices) > 1:
            trend = "BUY" if close_prices[-1] > close_prices[0] else "SELL"
        else:
            trend = "HOLD"

        signal = trend
        confidence = 50.0
        entry = last_price

        atr_levels = calculate_atr_levels(entry, signal, atr)
        tp = atr_levels["take_profit"]
        sl = atr_levels["stop_loss"]
        comment = "Fallback-ATR"

        print(f"🔄 Fallback Signal: {signal} conf={confidence}% (ATR-based)")
    else:
        try:
            signal = str(ai_raw.get("signal", "HOLD")).upper()
            confidence = float(ai_raw.get("confidence", 0))
            entry = float(ai_raw.get("entry_price", last_price))
            tp = float(ai_raw.get("take_profit", entry * 1.01))
            sl = float(ai_raw.get("stop_loss", entry * 0.99))
            comment = str(ai_raw.get("comment", ""))

            # Override with ATR if enabled
            if USE_ATR_LEVELS and atr > 0:
                atr_levels = calculate_atr_levels(entry, signal, atr)
                tp = atr_levels["take_profit"]
                sl = atr_levels["stop_loss"]
                comment += " [ATR]"

        except (ValueError, TypeError) as e:
            print(f"❌ Failed to parse AI response: {e}")
            send_telegram(f"❌ Invalid AI response: {e}")
            return

    print(
        f"🤖 AI Decision: {signal} | Confidence: {confidence}% | "
        f"Entry: ${entry:,.2f} | TP: ${tp:,.2f} | SL: ${sl:,.2f}"
    )

    log_entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "symbol": BITGET_SYMBOL,
        "signal": signal,
        "confidence": confidence,
        "entry": entry,
        "tp": tp,
        "sl": sl,
        "lot": LOT_SIZE,
        "executed": False,
        "result": "",
        "notes": ""
    }

    notify_text = (
        f"<b>{BITGET_SYMBOL} M15 Signal</b>\n"
        f"Mode: {'DEMO' if BITGET_DEMO_MODE else 'LIVE'}\n"
        f"Signal: <b>{signal}</b>\n"
        f"Confidence: {confidence}%\n"
        f"Entry: ${entry:,.2f}\n"
        f"TP: ${tp:,.2f}\n"
        f"SL: ${sl:,.2f}\n"
        f"ATR: ${atr:,.2f}\n"
        f"Leverage: {BITGET_LEVERAGE}x"
    )
    send_telegram(notify_text)

    # Validation
    if entry <= 0 or tp <= 0 or sl <= 0:
        log_entry["notes"] = "Invalid trade levels"
        log_trade(log_entry)
        return

    if DRY_RUN or not execute_trades:
        print("⏭️  Execution skipped (DRY_RUN or flag=false)")
        log_entry["notes"] = "DRY_RUN mode"
        log_trade(log_entry)
        return

    if signal not in ("BUY", "SELL"):
        print(f"⏭️  Signal is {signal}, skipping execution")
        log_entry["notes"] = f"Signal is {signal}"
        log_trade(log_entry)
        return

    if confidence < MIN_CONFIDENCE:
        print(
            f"⏭️  Confidence {confidence}% < {MIN_CONFIDENCE}%, "
            "skipping execution"
        )
        log_entry["notes"] = f"Low confidence: {confidence}%"
        log_trade(log_entry)
        return

    # Execute trade
    print("\n🔥 EXECUTING TRADE...")
    order_result = place_bitget_order(signal, entry, tp, sl, LOT_SIZE)

    if order_result["success"]:
        log_entry["executed"] = True
        log_entry["result"] = "SUCCESS"
        log_entry["notes"] = order_result["details"]
        log_trade(log_entry)

        msg = f"✅ <b>Order Executed</b>\n{order_result['details']}"
        send_telegram(msg)
        print("✅ Trade executed successfully")

    else:
        log_entry["executed"] = False
        log_entry["result"] = "FAILED"
        log_entry["notes"] = order_result["error"]
        log_trade(log_entry)

        msg = f"❌ <b>Order Failed</b>\n{order_result['error']}"
        send_telegram(msg)
        print(f"❌ Trade execution failed: {order_result['error']}")


# =============== CONTINUOUS LOOP ===============
if __name__ == "__main__":
    import time as time_module

    print(f"\n{'='*60}")
    print(f"🤖 STARTING CONTINUOUS LOOP")
    print(f"Runs every {LOOP_INTERVAL} seconds ({LOOP_INTERVAL/60:.1f} minutes)")
    print(f"Press Ctrl+C to stop")
    print(f"{'='*60}\n")

    loop_count = 0
    while True:
        try:
            loop_count += 1
            print(f"\n📊 Loop #{loop_count} - {datetime.now(timezone.utc).isoformat()}")
            print("-" * 60)

            main(execute_trades=True)

            print(f"✅ Loop #{loop_count} completed")
            print(f"⏳ Next run in {LOOP_INTERVAL} seconds ({LOOP_INTERVAL/60:.1f} minutes)...")

            time_module.sleep(LOOP_INTERVAL)

        except KeyboardInterrupt:
            print(f"\n\n{'='*60}")
            print(f"⏹️  BOT STOPPED BY USER")
            print(f"Total loops executed: {loop_count}")
            print(f"{'='*60}\n")
            break
        except Exception as e:
            print(f"\n❌ Loop error: {type(e).__name__}: {e}")
            print(f"⚠️  Waiting {LOOP_INTERVAL} seconds before retry...")
            time_module.sleep(LOOP_INTERVAL)
