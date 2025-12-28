import os
import requests
import json
import time
import math
import pandas as pd
import pandas_ta as ta
from binance.client import Client
from binance.enums import *
from dotenv import load_dotenv

load_dotenv()

# Client Configuration
BASE_OLLAMA_URL = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
OLLAMA_GENERATE_URL = f"{BASE_OLLAMA_URL}/api/generate"
OLLAMA_TAGS_URL = f"{BASE_OLLAMA_URL}/api/tags"
OLLAMA_PULL_URL = f"{BASE_OLLAMA_URL}/api/pull"
MODEL_NAME = "llama3"

BINANCE_API_KEY = os.getenv('BINANCE_API_KEY')
BINANCE_API_SECRET = os.getenv('BINANCE_API_SECRET')
BRAIN_INTERVAL_SECONDS = int(os.getenv('BRAIN_INTERVAL_SECONDS', '12'))


def pull_model(model_name):
    print(f"Downloading/Verifying model '{model_name}'... (this may take several minutes)")
    try:
        payload = {"name": model_name}
        with requests.post(OLLAMA_PULL_URL, json=payload, stream=True, timeout=None) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if line:
                    data = json.loads(line)
                    status = data.get('status', '')
                    total = data.get('total', 0)
                    completed = data.get('completed', 0)
                    if total > 0:
                        percent = (completed / total) * 100
                        print(f"\rStatus: {status} | Progress: {percent:.2f}%", end="", flush=True)
                    else:
                        print(f"\rStatus: {status}", end="", flush=True)
            print(f"\nModel '{model_name}' is ready!")
            return True
    except Exception as e:
        print(f"\nERROR downloading model: {e}")
        return False

def check_ollama_status():
    print(f"Checking Ollama connection at {BASE_OLLAMA_URL}...", flush=True)
    try:
        r = requests.get(OLLAMA_TAGS_URL, timeout=5)
        r.raise_for_status()
        models = r.json().get('models', [])
        if any(MODEL_NAME in m['name'] for m in models):
            print(f"OK: Ollama is active and model '{MODEL_NAME}' is available.", flush=True)
            return True
        return pull_model(MODEL_NAME)
    except Exception as e:
        print(f"ERROR connecting to Ollama: {e}", flush=True)
        return False

def get_binance_client():
    if not BINANCE_API_KEY or not BINANCE_API_SECRET or BINANCE_API_KEY == 'your_testnet_api_key_here':
        print("Warning: Real Binance keys not found or template keys detected in .env file. Running in SIMULATED mode.")
        return None
    try:
        # Add a timeout to all requests made by the client
        requests_params = {'timeout': 10}
        return Client(BINANCE_API_KEY, BINANCE_API_SECRET, testnet=True, requests_params=requests_params)
    except Exception as e:
        print(f"Error connecting to Binance: {e}")
        return None

def get_wallet_info(client, symbol):
    global mock_portfolio
    # SIMULATED MODE
    if not client:
        asset = symbol.replace('USDT', '')
        # In simulated mode, we'll use a fallback price to avoid network issues.
        current_price = 45000.0 # Fallback price
            
        usdt_balance = mock_portfolio["usdt"]
        asset_balance = mock_portfolio["asset"]
        total_usd = usdt_balance + (asset_balance * current_price)
        
        balances = [
            {"asset": "USDT", "free": usdt_balance, "usd_val": usdt_balance},
            {"asset": asset, "free": asset_balance, "usd_val": asset_balance * current_price}
        ]
        
        text = f"--- SIMULATED PORTFOLIO ---\nTOTAL WORTH: {total_usd:.2f} USD\n"
        for b in balances:
             if b['free'] > 0: text += f"{b['asset']}: {b['free']:.6f} (${b['usd_val']:.2f})\n"

        return {
            "balances": balances,
            "total_usd": total_usd,
            "text": text
        }
        
    try:
        # REAL MODE: Optimized to fetch only relevant balances
        symbol_info = client.get_symbol_info(symbol)
        base_asset = symbol_info['baseAsset']
        quote_asset = symbol_info['quoteAsset']
        
        # Fetch all tickers in one call to find prices efficiently
        all_tickers = client.get_all_tickers()
        prices = {t['symbol']: float(t['price']) for t in all_tickers}

        # Fetch all account balances once
        account = client.get_account()
        
        total_usd = 0.0
        displayed_balances = []

        # Calculate total portfolio value from all assets
        for b in account['balances']:
            free = float(b['free'])
            locked = float(b['locked'])
            total = free + locked
            asset_name = b['asset']

            if total > 0.00000001:
                usd_val = 0.0
                if asset_name in ['USDT', 'BUSD', 'USDC']:
                    usd_val = total
                else:
                    # Find price from the pre-fetched tickers list
                    usd_val = total * prices.get(f"{asset_name}USDT", 0)
                
                if usd_val > 0.01:
                    total_usd += usd_val

                # Only add the assets we're trading to the display list
                if asset_name == base_asset or asset_name == quote_asset:
                    displayed_balances.append({
                        "asset": asset_name,
                        "free": free,
                        "usd_val": usd_val
                    })

        text = f"--- REAL PORTFOLIO ---\nTOTAL WORTH (ALL ASSETS): ${total_usd:.2f} USD\n-- Relevant Balances --\n"
        if not displayed_balances:
            text += "No relevant asset balances found.\n"
        else:
            # Sort to keep a consistent order (e.g., quote asset first)
            displayed_balances.sort(key=lambda x: x['asset'] != quote_asset)
            for b in displayed_balances:
                text += f"{b['asset']}: {b['free']:.6f} (${b['usd_val']:.2f})\n"

        return {
            "balances": displayed_balances,
            "total_usd": total_usd,
            "text": text
        }

    except Exception as e:
        print(f"Error fetching real wallet: {e}", flush=True)
        return {
            "balances": [{"asset": "USDT", "free": 0.0, "usd_val": 0.0}],
            "total_usd": 0.0,
            "text": f"WALLET ERROR: {e}"
        }

def get_market_summary(client, symbol):
    if not client:
        return "Market data unavailable"
    try:
        klines = client.get_klines(symbol=symbol, interval=Client.KLINE_INTERVAL_1HOUR, limit=100)
        df = pd.DataFrame(klines, columns=['ts', 'open', 'high', 'low', 'close', 'vol', 'close_ts', 'qav', 'num_trades', 'taker_base', 'taker_quote', 'ignore'])
        df['close'] = df['close'].astype(float)
        
        # Indicators
        df['RSI'] = ta.rsi(df['close'], length=14)
        df['EMA_20'] = ta.ema(df['close'], length=20)
        df['EMA_50'] = ta.ema(df['close'], length=50)
        
        # ADDED: Bollinger Bands
        bbands = ta.bbands(df['close'], length=20, std=2)
        if bbands is not None:
            df = pd.concat([df, bbands], axis=1)
        
        # ADDED: MACD
        macd = ta.macd(df['close'])
        if macd is not None:
            df = pd.concat([df, macd], axis=1)
        
        last = df.iloc[-1]
        prev = df.iloc[-2]
        
        # Robust Column Discovery
        def get_col(df, prefix):
            cols = [c for c in df.columns if c.startswith(prefix)]
            return cols[0] if cols else None

        bbu_col = get_col(df, 'BBU')
        bbl_col = get_col(df, 'BBL')
        bbm_col = get_col(df, 'BBM')
        macd_col = get_col(df, 'MACD_12_26_9')
        signal_col = get_col(df, 'MACDs_12_26_9')
        
        price = last['close']
        rsi = last['RSI'] if 'RSI' in last else 50
        ema20 = last['EMA_20'] if 'EMA_20' in last else price
        ema50 = last['EMA_50'] if 'EMA_50' in last else price
        
        # Indicator Values with extraction
        curr_macd = last[macd_col] if macd_col else 0
        curr_signal = last[signal_col] if signal_col else 0
        curr_bbu = last[bbu_col] if bbu_col else price * 1.02
        curr_bbl = last[bbl_col] if bbl_col else price * 0.98
        curr_bbm = last[bbm_col] if bbm_col else price
        
        # Aggressive Trend Detection
        trend = "STRONG BULLISH" if price > ema20 > ema50 and curr_macd > curr_signal else \
                "STRONG BEARISH" if price < ema20 < ema50 and curr_macd < curr_signal else \
                "SIDEWAYS/STABLE"
        
        volatility = "HIGH" if (curr_bbu - curr_bbl) / curr_bbm > 0.05 else "LOW"
        
        summary = (
            f"--- Aggressive Market Snapshot ({symbol}) ---\n"
            f"Price: {price} | RSI: {rsi:.2f}\n"
            f"MACD: {curr_macd:.2f} | Signal: {curr_signal:.2f}\n"
            f"BBands: Upper {curr_bbu:.2f} / Lower {curr_bbl:.2f}\n"
            f"Trend: {trend} | Volatility: {volatility}\n"
            f"1h Price Change: {((last['close'] - prev['close']) / prev['close'] * 100):.2f}%"
        )
        return summary
    except Exception as e:
        return f"Error calculating technical indicators: {e}"

def ask_ai_opinion(market_summary, wallet_text):
    # Dynamic Context-Aware Prompt
    prompt = f"""
    ROLE: You are an Autonomous Hedge Fund Manager. 
    You must REGULATE your persona and your trade intensity based on the market context provided.
    
    SYSTEM STATE:
    {wallet_text}
    
    DATA FEED:
    {market_summary}
    
    TASKS & RULES:
    1. CONTEXT ANALYSIS: Determine if the market is trending, ranging, or volatile.
    2. PERSONA REGULATION: Choose a persona (AGGRESSIVE, CONSERVATIVE, HEDGER, SCALPER).
    3. BALANCE AWARENESS (CRITICAL):
       - If USDT is below $15.0, DO NOT SUGGEST A BUY. You should either SELL a portion of your holdings to get USDT or HOLD.
       - If you have zero holdings of an asset, you cannot SELL it.
       - Your goal is to keep a healthy balance of USDT for future opportunities.
    4. QUANTITY REGULATION: Decide WHAT PERCENTAGE of your available balance to use (0.1 to 1.0).
    5. SPECULATION BIAS: You are in 'TRY HARD' mode. Minimize 'HOLD' decisions unless absolutely necessary. 
    
    RESPONSE FORMAT (JSON ONLY):
    {{
        "reasoning": "Explain why this persona and quantity are best for this specific context",
        "decision": "BUY", "SELL", or "HOLD",
        "quantity_pct": 0.5
    }}
    """
    
    print(f"Running context-aware analysis with {MODEL_NAME}...")
    try:
        options = {
            "temperature": 0.7,
            "num_predict": 300,
            "num_ctx": 4096
        }
        payload = {
            "model": MODEL_NAME, 
            "prompt": prompt, 
            "stream": False, 
            "format": "json",
            "options": options
        }
        r = requests.post(OLLAMA_GENERATE_URL, json=payload, timeout=70)
        r.raise_for_status()
        
        res = r.json().get('response', '')
        
        data = json.loads(res)
        persona = data.get('persona', 'BALANCED').upper()
        decision = data.get('decision', 'HOLD').upper()
        quantity = float(data.get('quantity_pct', 1.0))
        reasoning = data.get('reasoning', '')
        
        print(f"\n--- AI CONTEXT ANALYSIS ---\nPERSONA: {persona}\nQUANTITY: {quantity*100:.0f}%\nREASONING: {reasoning}\n--------------------------")
        return decision, reasoning, persona, quantity
    except Exception as e:
        print(f"ERROR: AI Brain failed to regulate: {e}")
        return "HOLD", str(e), "ERROR", 0.0

def _get_lot_size_precision(step_size_str: str) -> int:
    """Calculates the number of decimal places for a given step size."""
    if '.' in step_size_str:
        # Find the position of the first '1' after the decimal point
        try:
            return step_size_str.index('1') - step_size_str.index('.')
        except ValueError:
             # Handle cases like "0.00000000"
            return len(step_size_str.split('.')[1])
    return 0

def _format_quantity(quantity: float, precision: int) -> float:
    """Formats a quantity to a specific precision by flooring it."""
    if precision == 0:
        return math.floor(quantity)
    factor = 10**precision
    return math.floor(quantity * factor) / factor

def execute_trade(client, symbol, decision, quantity_pct=1.0):
    global trade_history, mock_portfolio
    asset = symbol.replace('USDT', '')
    q_pct = max(0.1, min(1.0, quantity_pct))
    
    # 1. Get current price for calculations
    price = 45000.0 # Use a fallback price
    try:
        if client: # If in REAL mode
            price = float(client.get_symbol_ticker(symbol=symbol)['price'])
    except Exception as e:
        print(f"Warning: Could not fetch real-time price for trade. Using fallback. Error: {e}", flush=True)
    
    if not client: # If in SIMULATED mode
        print(f"SIMULATED TRADE: {decision} @ {price}")
        if decision == "BUY":
            buy_val = mock_portfolio["usdt"] * q_pct
            if buy_val > 1.0:
                mock_portfolio["usdt"] -= buy_val
                mock_portfolio["asset"] += (buy_val / price)
                trade_history.append({"time": time.strftime('%H:%M:%S'), "type": "BUY", "price": price, "amount": buy_val})
        elif decision == "SELL":
            sell_qty = mock_portfolio["asset"] * q_pct
            if sell_qty > 0.000001:
                mock_portfolio["asset"] -= sell_qty
                mock_portfolio["usdt"] += (sell_qty * price)
                trade_history.append({"time": time.strftime('%H:%M:%S'), "type": "SELL", "price": price, "amount": sell_qty * price})
        return

    try:
        if decision == "BUY":
            usdt_balance = float(client.get_asset_balance(asset='USDT')['free'])
            buy_amount_usdt = usdt_balance * q_pct
            
            print(f"DEBUG: Attempting BUY. Available USDT: {usdt_balance}, Calc Amount: {buy_amount_usdt}", flush=True)
            if buy_amount_usdt >= 10.0:
                print(f"Executing BUY order using {q_pct*100:.0f}% of USDT (${buy_amount_usdt:.2f})...", flush=True)
                order = client.create_order(symbol=symbol, side=SIDE_BUY, type=ORDER_TYPE_MARKET, quoteOrderQty=round(buy_amount_usdt, 2))
                print(f"BUY Success: {order['orderId']}", flush=True)
                trade_history.append({"time": time.strftime('%H:%M:%S'), "type": "BUY", "price": price, "amount": buy_amount_usdt})
            else:
                print(f"SKIPPED BUY: Amount ${buy_amount_usdt:.2f} is below Binance minimum ($10). Check Testnet balance.", flush=True)
                
        elif decision == "SELL":
            asset_balance = float(client.get_asset_balance(asset=asset)['free'])
            sell_amount_asset = asset_balance * q_pct
            sell_val_usd = sell_amount_asset * price
            
            print(f"DEBUG: Attempting SELL. Available {asset}: {asset_balance}, Value: ${sell_val_usd:.2f}", flush=True)
            if sell_val_usd >= 10.0:
                # --- FIX: Adhere to LOT_SIZE filter ---
                try:
                    symbol_info = client.get_symbol_info(symbol)
                    lot_size_filter = next((f for f in symbol_info['filters'] if f['filterType'] == 'LOT_SIZE'), None)
                    
                    if not lot_size_filter:
                        raise Exception(f"Could not find LOT_SIZE filter for {symbol}")

                    step_size_str = lot_size_filter['stepSize']
                    precision = _get_lot_size_precision(step_size_str)
                    
                    adjusted_quantity = _format_quantity(sell_amount_asset, precision)
                    
                    print(f"DEBUG: Original Qty: {sell_amount_asset}, Adjusted Qty (precision {precision}): {adjusted_quantity}", flush=True)

                    if adjusted_quantity <= 0:
                         print(f"SKIPPED SELL: Adjusted quantity {adjusted_quantity} is too small after applying LOT_SIZE filter.", flush=True)
                         return

                    print(f"Executing SELL order using {q_pct*100:.0f}% of holdings (~${sell_val_usd:.2f})...", flush=True)
                    order = client.create_order(symbol=symbol, side=SIDE_SELL, type=ORDER_TYPE_MARKET, quantity=adjusted_quantity)
                    print(f"SELL Success: {order['orderId']}", flush=True)
                    trade_history.append({"time": time.strftime('%H:%M:%S'), "type": "SELL", "price": price, "amount": sell_val_usd})

                except Exception as e:
                    print(f"ERROR during SELL order pre-check: {e}", flush=True)
                # --- END FIX ---
            else:
                print(f"SKIPPED SELL: Value ${sell_val_usd:.2f} is below Binance minimum ($10).", flush=True)
        else:
            print(f"BRAIN DECISION: {decision}. No action taken.", flush=True)
    except Exception as e:
        print(f"ERROR executing regulated trade: {e}", flush=True)

def get_live_ticker(client, symbol):
    if not client:
        return {"price": "0.00", "high24h": "0.00", "low24h": "0.00", "volume": "0.00", "price_change_pct": "0.00"}
    try:
        ticker = client.get_ticker(symbol=symbol)
        return {
            "price": ticker['lastPrice'],
            "high24h": ticker['highPrice'],
            "low24h": ticker['lowPrice'],
            "volume": ticker['volume'],
            "price_change_pct": ticker['priceChangePercent']
        }
    except Exception as e:
        print(f"Error fetching live ticker: {e}")
        return {}

# Globals for the dual-loop and history
last_brain_run = 0
current_decision = "HOLD"
current_reasoning = "Initializing Brain..."
current_persona = "BALANCED"
current_quantity = 0.0
initial_worth = None
trade_history = []  # List of {"time": str, "type": str, "price": float, "amount": float}
mock_portfolio = {"usdt": 1000.0, "asset": 0.0, "asset_name": "BTC"} # Mock for paper trading

# History tracking
price_history = []  # List of {"time": str, "price": float}
profit_history = [] # List of {"time": str, "worth": float}

def print_status_update(market_summary, wallet_info, decision, reasoning, persona="NEUTRAL", quantity=0.0, live_data=None):
    global price_history, profit_history, initial_worth
    try:
        now = time.strftime('%H:%M:%S')
        
        current_price = float(live_data.get('price', 0)) if live_data else 0
        total_worth = float(wallet_info.get('total_usd', 0))
        
        if initial_worth is None:
            initial_worth = total_worth
            
        if current_price > 0:
            price_history.append({"time": now, "price": current_price})
        profit_history.append({"time": now, "worth": total_worth})
        
        if len(price_history) > 100: price_history.pop(0)
        if len(profit_history) > 100: profit_history.pop(0)

        connection_mode = "REAL" if os.getenv('BINANCE_API_KEY') and "REAL" in wallet_info.get('text', '') else "SIMULATED"
        symbol = os.getenv('SYMBOL', 'BTCUSDT')

        print("\n" + "="*50)
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] - TRADING BOT STATUS ({connection_mode} MODE)")
        print("="*50)
        print(f"SYMBOL: {symbol}")
        if live_data:
            price_change_pct_str = live_data.get('price_change_pct', '0.00')
            try:
                price_change_pct = float(price_change_pct_str)
                price_change_str = f"{price_change_pct:.2f}%"
            except (ValueError, TypeError):
                price_change_str = "N/A"
            print(f"Live Price: {live_data.get('price', 'N/A')} | 24h Change: {price_change_str}")
        print(f"Total Worth: ${total_worth:.2f} | Initial Worth: ${initial_worth:.2f}")
        profit = total_worth - initial_worth
        profit_pct = (profit / initial_worth * 100) if initial_worth > 0 else 0
        print(f"Profit/Loss: ${profit:.2f} ({profit_pct:.2f}%)")
        print("-"*50)
        print(wallet_info['text'])
        print(market_summary)
        print("-"*50)
        print(f"AI DECISION: [{decision}]")
        print(f"REASONING: {reasoning}")
        print(f"PERSONA: {persona} | QUANTITY: {quantity*100:.0f}%")
        print("="*50)
        
        if trade_history:
            print("\n--- RECENT TRADES ---")
            for trade in reversed(trade_history[-5:]):
                print(f"{trade['time']} | {trade['type']} | Amount: ${trade['amount']:.2f} @ Price: {trade['price']:.2f}")
            print("---"*7)

    except Exception as e:
        print(f"Error printing status update: {e}")


def run_trading_cycle(force_brain=False):
    global last_brain_run, current_decision, current_reasoning, current_persona, current_quantity, initial_worth
    
    symbol = os.getenv('SYMBOL', 'BTCUSDT')
    client = get_binance_client()
    
    # 1. ALWAYS Get live data (Fast)
    live_data = get_live_ticker(client, symbol)
    wallet_info = get_wallet_info(client, symbol)
    
    if initial_worth is None:
        initial_worth = float(wallet_info.get('total_usd', 0))
    
    # 2. Extract Market Summary
    market_summary = get_market_summary(client, symbol)
    
    # 3. THROTTLE: AI Brain every 12s
    current_time = time.time()
    if force_brain or (current_time - last_brain_run) >= BRAIN_INTERVAL_SECONDS:
        print(f"\n--- BRAIN RE-EVALUATION ({time.strftime('%H:%M:%S')}) ---")
        decision, reasoning, persona, quantity = ask_ai_opinion(market_summary, wallet_info['text'])
        
        current_decision = decision
        current_reasoning = reasoning
        current_persona = persona
        current_quantity = quantity
        last_brain_run = current_time
        
        if current_decision != "HOLD":
            execute_trade(client, symbol, current_decision, current_quantity)
    
    # 4. Print status update to terminal
    print_status_update(market_summary, wallet_info, current_decision, current_reasoning, current_persona, current_quantity, live_data)

def main_loop():
    print("Starting AI Trading Terminal V4.0 (Terminal Mode)...", flush=True)

    # Status check (can be slow)
    if not check_ollama_status():
        print("\nAborting: Ollama is not ready.", flush=True)
        return

    # Initial run
    run_trading_cycle(force_brain=True)
    
    while True:
        try:
            run_trading_cycle()
        except Exception as e:
            print(f"Loop Error: {e}", flush=True)
        
        time.sleep(2) # 2s price update interval

if __name__ == "__main__":
    main_loop()
