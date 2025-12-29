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

def get_multi_timeframe_analysis(client, symbol):
    if not client:
        return "Market data unavailable", {}
    
    timeframes = {
        "15m": Client.KLINE_INTERVAL_15MINUTE,
        "1h": Client.KLINE_INTERVAL_1HOUR,
        "4h": Client.KLINE_INTERVAL_4HOUR
    }
    
    full_summary = "--- Multi-Timeframe Analysis ---"
    analysis_data = {}
    
    for tf_name, tf_interval in timeframes.items():
        try:
            klines = client.get_klines(symbol=symbol, interval=tf_interval, limit=100)
            df = pd.DataFrame(klines, columns=['ts', 'open', 'high', 'low', 'close', 'vol', 'close_ts', 'qav', 'num_trades', 'taker_base', 'taker_quote', 'ignore'])
            df['close'] = df['close'].astype(float)
            
            # Indicators
            df['RSI'] = ta.rsi(df['close'], length=14)
            df['EMA_20'] = ta.ema(df['close'], length=20)
            df['EMA_50'] = ta.ema(df['close'], length=50)
            
            last = df.iloc[-1]
            price = last['close']
            rsi = last['RSI'] if 'RSI' in last else 50
            ema20 = last['EMA_20'] if 'EMA_20' in last else price
            ema50 = last['EMA_50'] if 'EMA_50' in last else price
            
            trend = "BULLISH" if price > ema20 > ema50 else "BEARISH" if price < ema20 < ema50 else "SIDEWAYS"
            
            full_summary += f"\n[{tf_name} TF] Trend: {trend} | RSI: {rsi:.2f}"
            analysis_data[tf_name] = {"rsi": rsi}
            
        except Exception as e:
            full_summary += f"\n[{tf_name} TF] Error: {e}"
            
    return full_summary, analysis_data

def get_order_book_snapshot(client, symbol):
    if not client:
        return "Order book data unavailable"
    try:
        depth = client.get_order_book(symbol=symbol, limit=10)
        
        bid_volume = sum([float(bid[1]) for bid in depth['bids']])
        ask_volume = sum([float(ask[1]) for bid in depth['asks']])
        
        ratio = bid_volume / ask_volume if ask_volume > 0 else float('inf')
        
        pressure = "BUY" if ratio > 1.1 else "SELL" if ratio < 0.9 else "NEUTRAL"
        
        return f"Bid/Ask Volume Ratio: {ratio:.2f}:1 | Immediate Pressure: {pressure}"
    except Exception as e:
        return f"Could not get order book: {e}"

def get_fear_and_greed_index():
    try:
        r = requests.get("https://api.alternative.me/fng/?limit=1", timeout=10)
        r.raise_for_status()
        data = r.json()['data'][0]
        value = int(data['value'])
        classification = data['value_classification']
        return f"Value: {value} ({classification})"
    except Exception as e:
        return f"Could not fetch: {e}"

def ask_ai_opinion(market_summary, wallet_text, trade_history, order_book, sentiment, live_data):
    trade_summary = "No recent trades."
    if trade_history:
        trade_summary = "Recent Trades:\n"
        for trade in trade_history[-5:]:
            trade_summary += f"- {trade['type']} @ ${trade['price']:.2f} for ${trade['amount']:.2f}\n"

    live_data_summary = f"Live Price: {live_data.get('price', 'N/A')}"

    prompt = f"""
    ROLE: You are a sophisticated, multi-dimensional crypto trading analyst. Your goal is to maximize profit by synthesizing multiple data points into a coherent trading strategy.

    You must follow a strict Chain-of-Thought reasoning process.

    --- DATA STREAMS ---
    1. CURRENT PORTFOLIO:
    {wallet_text}
    {trade_summary}

    2. LIVE DATA (Current Moment):
    - {live_data_summary}
    - Order Book Snapshot: {order_book}
    
    3. MARKET SENTIMENT:
    - Fear & Greed Index: {sentiment}

    4. TECHNICAL ANALYSIS:
    {market_summary}

    --- REASONING PROCESS (Chain-of-Thought) ---
    1.  **Synthesize Macro View (4h):** What is the major trend direction on the 4-hour chart? Are we in a larger uptrend or downtrend?
    2.  **Identify Session Trend (1h):** What is the trend for the current trading session on the 1-hour chart? Does it align with the macro view?
    3.  **Pinpoint Entry/Exit (15m):** What is the immediate, short-term trend on the 15-minute chart? Is it overbought (RSI > 70) or oversold (RSI < 30)?
    4.  **Check for Divergences:** Compare the timeframes. Is there a divergence (e.g., 4h is bullish, but 15m is bearish and overbought)? Also, compare live price action to indicators.
    5.  **Factor in Sentiment & Order Book:** How do the Fear & Greed Index and the order book pressure influence the decision? Does strong buying pressure justify entering an overbought market? Does extreme greed suggest a correction is coming?
    6.  **Propose & Critique Plan:** Based on the synthesis of all data, propose a primary action. Then, critique it. What are the risks? What is the alternative? (e.g., "The macro trend is bullish, but the 15m chart is overbought and F&G is at 'Extreme Greed'. Buying now is risky. A better plan might be to wait for a pullback to the 15m 20-EMA.").
    7.  **Final Decision:** State your final, synthesized decision (BUY, SELL, or HOLD) and the quantity percentage.

    --- CRITICAL RULES ---
    - **Minimum Trade:** Must be at least $10 USDT.
    - **BUY Rule:** DO NOT BUY if USDT balance is < $15.
    - **SELL Rule:** DO NOT SELL if you have zero holdings of the asset.

    --- RESPONSE FORMAT (JSON ONLY) ---
    {{
      "reasoning": "CHAIN-OF-THOUGHT ANALYSIS: [Your full, step-by-step reasoning process here, following the 7 steps above]",
      "decision": "BUY", "SELL", or "HOLD",
      "quantity_pct": 0.5
    }}
    """
    
    print(f"Running multi-dimensional analysis with {MODEL_NAME}...")
    try:
        options = {
            "temperature": 0.7,
            "num_predict": 700,
            "num_ctx": 4096
        }
        payload = {
            "model": MODEL_NAME, 
            "prompt": prompt, 
            "stream": False, 
            "format": "json",
            "options": options
        }
        r = requests.post(OLLAMA_GENERATE_URL, json=payload, timeout=120)
        r.raise_for_status()
        
        res = r.json().get('response', '')
        
        data = json.loads(res)
        decision = data.get('decision', 'HOLD').upper()
        quantity = float(data.get('quantity_pct', 1.0))
        reasoning = data.get('reasoning', '')
        
        return decision, reasoning, "MULTI_DIM", quantity
    except Exception as e:
        print(f"ERROR: AI Brain failed to regulate: {e}")
        return "HOLD", str(e), "ERROR", 0.0

def _get_lot_size_precision(step_size_str: str) -> int:
    """Calculates the number of decimal places for a given step size."""
    if '.' in step_size_str:
        try:
            return step_size_str.index('1') - step_size_str.index('.')
        except ValueError:
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
    
    price = 45000.0
    try:
        if client:
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
            
            if buy_amount_usdt >= 10.0:
                print(f"Executing BUY order using {q_pct*100:.0f}% of USDT (${buy_amount_usdt:.2f})...", flush=True)
                order = client.create_order(symbol=symbol, side=SIDE_BUY, type=ORDER_TYPE_MARKET, quoteOrderQty=round(buy_amount_usdt, 2))
                print(f"BUY Success: {order['orderId']}", flush=True)
                trade_history.append({"time": time.strftime('%H:%M:%S'), "type": "BUY", "price": price, "amount": buy_amount_usdt})
            else:
                print(f"SKIPPED BUY: Amount ${buy_amount_usdt:.2f} is below Binance minimum ($10).", flush=True)
                
        elif decision == "SELL":
            asset_balance = float(client.get_asset_balance(asset=asset)['free'])
            sell_amount_asset = asset_balance * q_pct
            sell_val_usd = sell_amount_asset * price
            
            if sell_val_usd >= 10.0:
                try:
                    symbol_info = client.get_symbol_info(symbol)
                    lot_size_filter = next((f for f in symbol_info['filters'] if f['filterType'] == 'LOT_SIZE'), None)
                    
                    if not lot_size_filter:
                        raise Exception(f"Could not find LOT_SIZE filter for {symbol}")

                    step_size_str = lot_size_filter['stepSize']
                    precision = _get_lot_size_precision(step_size_str)
                    
                    adjusted_quantity = _format_quantity(sell_amount_asset, precision)
                    
                    if adjusted_quantity <= 0:
                         print(f"SKIPPED SELL: Adjusted quantity {adjusted_quantity} is too small.", flush=True)
                         return

                    print(f"Executing SELL order using {q_pct*100:.0f}% of holdings (~${sell_val_usd:.2f})...", flush=True)
                    order = client.create_order(symbol=symbol, side=SIDE_SELL, type=ORDER_TYPE_MARKET, quantity=adjusted_quantity)
                    print(f"SELL Success: {order['orderId']}", flush=True)
                    trade_history.append({"time": time.strftime('%H:%M:%S'), "type": "SELL", "price": price, "amount": sell_val_usd})
                except Exception as e:
                    print(f"ERROR during SELL order pre-check: {e}", flush=True)
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

last_brain_run = 0
current_decision = "HOLD"
current_reasoning = "Initializing Brain..."
current_persona = "BALANCED"
current_quantity = 0.0
initial_worth = None
trade_history = []
mock_portfolio = {"usdt": 1000.0, "asset": 0.0, "asset_name": "BTC"}
price_history = []
profit_history = []

def print_status_update(market_summary, wallet_info, decision, reasoning, persona, quantity, live_data, order_book, sentiment):
    global price_history, profit_history, initial_worth
    try:
        now = time.strftime('%H:%M:%S')
        
        current_price = float(live_data.get('price', 0)) if live_data else 0
        total_worth = float(wallet_info.get('total_usd', 0))
        
        if initial_worth is None: initial_worth = total_worth
            
        if current_price > 0: price_history.append({"time": now, "price": current_price})
        profit_history.append({"time": now, "worth": total_worth})
        
        if len(price_history) > 100: price_history.pop(0)
        if len(profit_history) > 100: profit_history.pop(0)

        connection_mode = "REAL" if os.getenv('BINANCE_API_KEY') else "SIMULATED"
        symbol = os.getenv('SYMBOL', 'BTCUSDT')

        # This is the main print block
        print("\n" + "="*80)
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] - TRADING BOT STATUS ({connection_mode} MODE)")
        print("="*80)
        print(f"SYMBOL: {symbol} | Live Price: ${live_data.get('price', 'N/A')} | 24h Change: {live_data.get('price_change_pct', 'N/A')}% ")
        profit = total_worth - initial_worth
        profit_pct = (profit / initial_worth * 100) if initial_worth > 0 else 0
        print(f"Total Worth: ${total_worth:.2f} | P/L: ${profit:.2f} ({profit_pct:.2f}%)")
        print(wallet_info['text'])
        print(f"--- SENTIMENT & ORDER BOOK ---\n{sentiment}\n{order_book}\n")
        print(market_summary)
        
        # AI decision is printed separately in run_trading_cycle if it runs
        
        if trade_history:
            print("\n--- RECENT TRADES ---")
            for trade in reversed(trade_history[-5:]):
                print(f"{trade['time']} | {trade['type']:<4} | Amount: ${trade['amount']:<10.2f} @ Price: ${trade['price']:.2f}")
            print("-" * 80)

    except Exception as e:
        print(f"Error printing status update: {e}")

def run_trading_cycle(force_brain=False):
    global last_brain_run, current_decision, current_reasoning, current_persona, current_quantity, initial_worth
    
    symbol = os.getenv('SYMBOL', 'BTCUSDT')
    client = get_binance_client()
    
    # 1. Fetch all data first
    live_data = get_live_ticker(client, symbol)
    wallet_info = get_wallet_info(client, symbol)
    if initial_worth is None: initial_worth = float(wallet_info.get('total_usd', 0))
    
    market_summary, market_analysis_data = get_multi_timeframe_analysis(client, symbol)
    order_book = get_order_book_snapshot(client, symbol)
    sentiment = f"Fear & Greed Index: {get_fear_and_greed_index()}"
    
    # 2. Print the current state BEFORE making a decision
    print_status_update(market_summary, wallet_info, current_decision, current_reasoning, current_persona, current_quantity, live_data, order_book, sentiment)

    # 3. Decide if it's time for the brain to run
    current_time = time.time()
    if force_brain or (current_time - last_brain_run) >= BRAIN_INTERVAL_SECONDS:
        print(f"\n--- BRAIN RE-EVALUATION ({time.strftime('%H:%M:%S')}) ---")
        decision, reasoning, persona, quantity = ask_ai_opinion(market_summary, wallet_info['text'], trade_history, order_book, sentiment, live_data)
        
        # --- GUARDRAILS & DYNAMIC RISK ---
        usdt_balance = next((b['free'] for b in wallet_info.get('balances', []) if b['asset'] == 'USDT'), 0.0)
        base_asset_balance = next((b['free'] for b in wallet_info.get('balances', []) if b['asset'] == symbol.replace('USDT', '')), 0.0)

        if decision == "BUY":
            if usdt_balance < 10.0:
                print(f"!!! AI OVERRIDE: Insufficient USDT for minimum trade. Forcing HOLD. !!!", flush=True)
                decision = "HOLD"
            else: # Dynamic Risk Scaling
                if market_analysis_data.get('15m', {}).get('rsi', 50) > 75:
                    print(f"!!! RISK SCALING: 15m RSI is over 75 (overbought). Reducing quantity. !!!", flush=True)
                    quantity = min(quantity, 0.1) # Cap quantity at 10%

        if decision == "SELL" and base_asset_balance <= 0:
            print(f"!!! AI OVERRIDE: Zero asset balance to sell. Forcing HOLD. !!!", flush=True)
            decision = "HOLD"
        # --- END GUARDRAILS ---

        # Print the AI's thoughts and final decision
        print("\n" + "-"*80)
        print("--- AI ANALYSIS & DECISION ---")
        print(f"{reasoning}")
        print(f"\nFINAL DECISION: [{decision}] | QUANTITY: {quantity*100:.0f}%")
        print("-" * 80)

        current_decision, current_reasoning, current_persona, current_quantity = decision, reasoning, persona, quantity
        last_brain_run = current_time
        
        if current_decision != "HOLD":
            execute_trade(client, symbol, current_decision, current_quantity)
    


def main_loop():
    print("Starting AI Trading Terminal V5.1 (Synchronized Logging)...", flush=True)

    if not check_ollama_status():
        print("\nAborting: Ollama is not ready.", flush=True)
        return

    run_trading_cycle(force_brain=True)
    
    while True:
        try:
            run_trading_cycle()
        except Exception as e:
            print(f"Loop Error: {e}", flush=True)
        
        time.sleep(2)

if __name__ == "__main__":
    main_loop()
