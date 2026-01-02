import os
import requests
import json
import time
import math
import pandas as pd
import pandas_ta as ta
import sqlite3
import datetime
import traceback
from binance.client import Client
from binance.enums import *
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

# --- CONFIGURATION ---
BASE_OLLAMA_URL = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
OLLAMA_GENERATE_URL = f"{BASE_OLLAMA_URL}/api/generate"
OLLAMA_TAGS_URL = f"{BASE_OLLAMA_URL}/api/tags"
OLLAMA_PULL_URL = f"{BASE_OLLAMA_URL}/api/pull"
MODEL_NAME = "llama3"

BINANCE_API_KEY = os.getenv('BINANCE_API_KEY')
BINANCE_API_SECRET = os.getenv('BINANCE_API_SECRET')
BRAIN_INTERVAL_SECONDS = int(os.getenv('BRAIN_INTERVAL_SECONDS', '12'))
DB_FILE = '/app/shared/trading_decisions_v2.db'
DB_MAX_SIZE_MB = int(os.getenv('DB_MAX_SIZE_MB', '500'))

CLEAN_RUN_ENABLED = os.getenv('CLEAN_RUN_ENABLED', 'true').lower() == 'true'
CLEAR_TRACKED_SYMBOLS_ON_CLEAN_RUN = os.getenv('CLEAR_TRACKED_SYMBOLS_ON_CLEAN_RUN', 'true').lower() == 'true'

# --- GLOBALS ---
last_brain_run = 0
current_decision = "HOLD"
current_reasoning = "Initializing Brain..."
current_persona = "BALANCED"
current_quantity = 0.0
initial_worth = None
mock_portfolio = {"usdt": 200.0, "asset": 0.0, "asset_name": "BTC"}
price_history = []
profit_history = []
last_prune_date = None
KLINE_CACHE = {}
KLINE_CACHE_TTL_SECONDS = 300 # 5 minutes

global_time_offset_ms = 0
original_time_time = time.time


# --- CORE FUNCTIONS ---

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

def init_db():
    try:
        with sqlite3.connect(DB_FILE) as conn:
            c = conn.cursor()
            c.execute('''
                CREATE TABLE IF NOT EXISTS decisions_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    symbol TEXT, ai_decision TEXT, ai_quantity_pct REAL, ai_reasoning TEXT,
                    actual_action TEXT, actual_quantity REAL, actual_price REAL,
                    post_trade_usdt_balance REAL, post_trade_asset_balance REAL,
                    post_trade_total_worth REAL, context_snapshot TEXT,
                    trade_id INTEGER, profit REAL
                )
            ''')
            c.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON decisions_log (timestamp)")
            c.execute('''
                CREATE TABLE IF NOT EXISTS tracked_symbols (
                    symbol TEXT PRIMARY KEY, added_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    status TEXT DEFAULT 'active'
                )
            ''')
        print("Database initialized successfully.")
    except Exception as e:
        print(f"Error initializing database: {e}")

def get_db_size():
    if os.path.exists(DB_FILE):
        return os.path.getsize(DB_FILE) / (1024 * 1024)
    return 0

def prune_database():
    try:
        if get_db_size() <= DB_MAX_SIZE_MB:
            return
        with sqlite3.connect(DB_FILE) as conn:
            c = conn.cursor()
            print(f"Database size exceeds {DB_MAX_SIZE_MB} MB. Pruning...", flush=True)
            while get_db_size() > DB_MAX_SIZE_MB * 0.9:
                c.execute("DELETE FROM decisions_log WHERE id IN (SELECT id FROM decisions_log ORDER BY timestamp ASC LIMIT 100)")
                conn.commit()
                if c.rowcount == 0: break
            print(f"Database pruned. New size: {get_db_size():.2f} MB.", flush=True)
    except Exception as e:
        print(f"Error pruning database: {e}", flush=True)

def add_tracked_symbol(symbol):
    if not symbol or not symbol.strip():
        print(f"Warning: Attempted to add an empty or invalid symbol '{symbol}'. Skipping.")
        return
    try:
        with sqlite3.connect(DB_FILE) as conn:
            c = conn.cursor()
            c.execute("INSERT OR IGNORE INTO tracked_symbols (symbol, status) VALUES (?, ?)", (symbol, 'active'))
        print(f"Symbol {symbol} added to tracked_symbols.", flush=True)
    except Exception as e:
        print(f"Error adding tracked symbol: {e}", flush=True)

def remove_tracked_symbol(symbol):
    try:
        with sqlite3.connect(DB_FILE) as conn:
            c = conn.cursor()
            c.execute("UPDATE tracked_symbols SET status = 'inactive' WHERE symbol = ?", (symbol,))
        print(f"Symbol {symbol} marked as inactive.", flush=True)
    except Exception as e:
        print(f"Error removing tracked symbol: {e}", flush=True)

def clear_tracked_symbols_table():
    try:
        with sqlite3.connect(DB_FILE) as conn:
            c = conn.cursor()
            c.execute("DELETE FROM tracked_symbols")
        print("Tracked symbols table cleared.")
    except Exception as e:
        print(f"Error clearing tracked symbols table: {e}")

def get_active_tracked_symbols():
    try:
        with sqlite3.connect(DB_FILE) as conn:
            c = conn.cursor()
            c.execute("SELECT symbol FROM tracked_symbols WHERE status = 'active'")
            symbols = [row[0] for row in c.fetchall()]
        
        valid_symbols = [s for s in symbols if s and s.strip() and s not in ['None', '.']]
        if len(valid_symbols) != len(symbols):
            invalid = list(set(symbols) - set(valid_symbols))
            print(f"Filtered invalid symbols: {invalid}.")
        return valid_symbols
    except Exception as e:
        print(f"Error getting active tracked symbols: {e}", flush=True)
        return []

def log_decision_and_trade(symbol, ai_decision, ai_quantity_pct, ai_reasoning, trade_executed_info, wallet_info, context_snapshot):
    try:
        with sqlite3.connect(DB_FILE) as conn:
            c = conn.cursor()
            trade_info = trade_executed_info or {}
            context_to_store = json.dumps(context_snapshot) if trade_info.get('action') in ["BUY", "SELL"] else None
            
            usdt_balance = next((float(b['free']) for b in wallet_info.get('balances', []) if b['asset'] == 'USDT'), 0.0)
            asset_balance = next((float(b['free']) for b in wallet_info.get('balances', []) if b['asset'] == symbol.replace('USDT', '')), 0.0)
            
            c.execute('''
                INSERT INTO decisions_log (symbol, ai_decision, ai_quantity_pct, ai_reasoning, actual_action, actual_quantity, actual_price, post_trade_usdt_balance, post_trade_asset_balance, post_trade_total_worth, context_snapshot)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                symbol, ai_decision, ai_quantity_pct, ai_reasoning,
                trade_info.get('action', 'NONE'), trade_info.get('amount', 0.0), trade_info.get('price', 0.0),
                usdt_balance, asset_balance, wallet_info.get('total_usd', 0.0),
                context_to_store
            ))
            return c.lastrowid
    except Exception as e:
        print(f"Error logging decision to database: {e}", flush=True)
        return None

def get_recent_trades_summary():
    try:
        with sqlite3.connect(DB_FILE) as conn:
            conn.row_factory = sqlite3.Row
            c = conn.cursor()
            c.execute("SELECT timestamp, actual_action, actual_quantity, actual_price, symbol FROM decisions_log WHERE actual_action IN ('BUY', 'SELL') ORDER BY timestamp DESC LIMIT 5")
            trades = c.fetchall()
            if not trades: return "No recent trades found in database."
            
            summary = "--- RECENT TRADES (from DB) ---"
            for trade in trades:
                trade_dict = dict(trade)
                summary += f"{trade_dict['timestamp'].split(' ')[1].split('.')[0]} | {trade_dict['actual_action']:<4} | {trade_dict['actual_quantity']:.6f} {trade_dict['symbol'].replace('USDT', '')} @ Price: ${trade_dict['actual_price']:.2f}\n"
            return summary
    except Exception as e:
        print(f"Error getting recent trades summary: {e}")
        return "Could not retrieve recent trades."

def check_ollama_status():
    print(f"Checking Ollama connection at {BASE_OLLAMA_URL}...", flush=True)
    try:
        r = requests.get(OLLAMA_TAGS_URL, timeout=5)
        r.raise_for_status()
        if any(MODEL_NAME in m.get('name', '') for m in r.json().get('models', [])):
            print(f"OK: Ollama is active and model '{MODEL_NAME}' is available.", flush=True)
            return True
        return pull_model(MODEL_NAME)
    except Exception as e:
        print(f"ERROR connecting to Ollama: {e}", flush=True)
        return False

def get_binance_client():
    global global_time_offset_ms
    # --- DEBUGGING ---
    print(f"DEBUG: BINANCE_API_KEY loaded: {BINANCE_API_KEY[:5]}..." if BINANCE_API_KEY else "Not loaded")
    print(f"DEBUG: BINANCE_API_SECRET loaded: {'*' * 5}..." if BINANCE_API_SECRET else "Not loaded")
    # --- END DEBUGGING ---
    if not BINANCE_API_KEY or not BINANCE_API_SECRET or BINANCE_API_KEY == 'your_testnet_api_key_here':
        print("Warning: Binance keys not found. Running in mock mode.")
        return None
    try:
        # Create a temporary client to get server time
        temp_client = Client(BINANCE_API_KEY, BINANCE_API_SECRET, testnet=True)
        server_time = temp_client.get_server_time()
        local_time = int(original_time_time() * 1000) # Use original time.time()
        global_time_offset_ms = server_time['serverTime'] - local_time
        print(f"Calculated global time offset with Binance server: {global_time_offset_ms}ms")

        # Monkey-patch time.time()
        time.time = get_adjusted_time
        print("time.time() has been monkey-patched for Binance synchronization.")

        return Client(BINANCE_API_KEY, BINANCE_API_SECRET, testnet=True, requests_params={'timeout': 10})
    except Exception as e:
        print(f"Error connecting to Binance: {e}", flush=True)
        return None

def get_wallet_info(client, symbols):
    if not client: # Mock mode
        return { "balances": [], "total_usd": 0.0, "trading_pair_worth": {}, "text": "MOCK MODE: No client.", "detailed_balances_list": [] }
        
    try:
        account = client.get_account(recvWindow=60000)
        all_tickers = client.get_all_tickers()
        prices = {t['symbol']: float(t['price']) for t in all_tickers}
        
        all_balances = {b['asset']: float(b['free']) for b in account['balances'] if float(b['free']) > 0.000001}
        
        relevant_assets = set()
        for s in symbols:
            base_asset = s.replace('USDT', '')
            if base_asset:
                relevant_assets.add(base_asset)
        relevant_assets.add('USDT')

        filtered_balances_for_display = {asset: balance for asset, balance in all_balances.items() if asset in relevant_assets}

        total_usd = 0
        detailed_balances = []
        for asset, balance in filtered_balances_for_display.items():
            price = 1.0 if asset in ['USDT', 'BUSD', 'USDC', 'TUSD'] else prices.get(f"{asset}USDT", 0)
            usd_val = balance * price
            if usd_val > 0.01:
                total_usd += usd_val
            detailed_balances.append({"asset": asset, "balance": balance, "usd_value": usd_val, "price": price})

        text = f"--- REAL PORTFOLIO (TRACKED ASSETS) ---\nTOTAL WORTH: ${total_usd:.2f} USD\n"
        for item in sorted(detailed_balances, key=lambda x: x['usd_value'], reverse=True):
             if item['usd_value'] > 1.0:
                text += f"{item['asset']}: {item['balance']:.6f} (${item['usd_value']:.2f})\n"

        trading_pair_worth = {s: filtered_balances_for_display.get(s.replace('USDT', ''), 0.0) * prices.get(s, 0.0) for s in symbols}

        return {
            "balances": account['balances'],
            "total_usd": total_usd,
            "trading_pair_worth": trading_pair_worth,
            "text": text,
            "detailed_balances_list": detailed_balances
        }
    except Exception as e:
        print(f"Error fetching real wallet: {e}", flush=True)
        return { "balances": [], "total_usd": 0.0, "trading_pair_worth": {}, "text": f"WALLET ERROR: {e}", "detailed_balances_list": [] }

def get_multi_timeframe_analysis(client, symbol):
    global KLINE_CACHE
    if not client: return "Market data unavailable", {}
    timeframes = {
        "15m": Client.KLINE_INTERVAL_15MINUTE,
        "1h": Client.KLINE_INTERVAL_1HOUR,
        "4h": Client.KLINE_INTERVAL_4HOUR,
        "1d": Client.KLINE_INTERVAL_1DAY,
        "1w": Client.KLINE_INTERVAL_1WEEK
    }
    
    full_summary = "--- Multi-Timeframe Analysis ---"
    analysis_data = {}
    
    for tf_name, tf_interval in timeframes.items():
        try:
            cache_key = f"{symbol}_{tf_name}"
            now = time.time()
            
            if cache_key in KLINE_CACHE and (now - KLINE_CACHE[cache_key]['timestamp']) < KLINE_CACHE_TTL_SECONDS:
                klines = KLINE_CACHE[cache_key]['data']
            else:
                klines = client.get_klines(symbol=symbol, interval=tf_interval, limit=100)
                KLINE_CACHE[cache_key] = {'timestamp': now, 'data': klines}

            df = pd.DataFrame(klines, columns=['ts', 'open', 'high', 'low', 'close', 'vol', 'close_ts', 'qav', 'num_trades', 'taker_base', 'taker_quote', 'ignore'])
            df['close'] = df['close'].astype(float)
            df['RSI'] = ta.rsi(df['close'], length=14)
            df['EMA_20'] = ta.ema(df['close'], length=20)
            df['EMA_50'] = ta.ema(df['close'], length=50)
            last = df.iloc[-1]
            price = last['close']
            rsi = last['RSI'] if pd.notna(last['RSI']) else 50
            ema20 = last['EMA_20'] if pd.notna(last['EMA_20']) else price
            ema50 = last['EMA_50'] if pd.notna(last['EMA_50']) else price
            trend = "BULLISH" if price > ema20 > ema50 else "BEARISH" if price < ema20 < ema50 else "SIDEWAYS"
            full_summary += f"\n[{tf_name} TF] Trend: {trend} | RSI: {rsi:.2f}"
            analysis_data[tf_name] = {"rsi": rsi}
        except Exception as e:
            full_summary += f"\n[{tf_name} TF] Error: {e}"
            
    return full_summary, analysis_data

def get_order_book_snapshot(client, symbol):
    if not client: return "Order book data unavailable"
    try:
        depth = client.get_order_book(symbol=symbol, limit=10)
        bid_volume = sum([float(bid[1]) for bid in depth['bids']])
        ask_volume = sum([float(a[1]) for a in depth['asks']])
        ratio = bid_volume / ask_volume if ask_volume > 0 else float('inf')
        pressure = "BUY" if ratio > 1.1 else "SELL" if ratio < 0.9 else "NEUTRAL"
        return f"Bid/Ask Volume Ratio: {ratio:.2f}:1 | Immediate Pressure: {pressure}"
    except Exception as e:
        return f"Could not get order book: {e}"

def ask_ai_opinion(current_tracked_symbols, market_summary, full_wallet_info, order_book, live_data, trade_summary):
    wallet_text = full_wallet_info.get('text', '')
    usdt_balance = next((float(b['free']) for b in full_wallet_info.get('balances', []) if b['asset'] == 'USDT'), 0.0)
    total_portfolio_worth = full_wallet_info.get('total_usd', 0.0)

    live_data_summary = f"Live Price: {live_data.get('price', 'N/A')}"
    
    tracked_symbols_str = ", ".join(current_tracked_symbols)

    knowledge_base = """--- ADVANCED TRADING KNOWLEDGE BASE ---
**Quantitative & Algorithmic Strategies:**
- **Predictive Modeling (ML):** Use Machine Learning to analyze historical data and predict future price movements.
- **Sentiment Analysis (AI):** Process news and social media to gauge market sentiment.
- **Mean Reversion:** Assume that asset prices will revert to their historical average.
- **Trend Following/Momentum:** Identify and trade in the direction of strong trends.
**Risk Management:**
- Always use Stop-Loss and Take-Profit orders.
- Diversify your portfolio.
- Use proper position sizing (e.g., risk 1-2% of portfolio per trade).
"""

    detailed_balances_list = full_wallet_info.get('detailed_balances_list', [])
    detailed_balances_prompt_section = "--- DETAILED ASSET BREAKDOWN ---"
    for item in detailed_balances_list:
        detailed_balances_prompt_section += f"- {item['asset']}: {item['balance']:.6f} (${item['usd_value']:.2f})\n"

    prompt = f"""
    ROLE: You are a sophisticated crypto trading analyst. Your goal is to maximize profit.
    You are tracking: {tracked_symbols_str}.
    Based on your analysis, decide to BUY, SELL, or HOLD for the *current* symbol.

    {knowledge_base}

    --- DATA STREAMS ---
    1. CURRENT PORTFOLIO:
    {wallet_text}
    - Total Portfolio Worth: ${total_portfolio_worth:.2f} USD
    - Available USDT for trading: ${usdt_balance:.2f}
    
    {detailed_balances_prompt_section}
    
    {trade_summary}

    2. TECHNICAL ANALYSIS:
    {market_summary}

    --- REASONING PROCESS (Chain-of-Thought) ---
    1.  **Assess Portfolio & Market:** Analyze market conditions and your portfolio.
    2.  **Assess Current Symbol:** Based on all data, select a strategy from the knowledge base for the current symbol.
    3.  **Propose & Critique Plan:** Propose a primary action (BUY, SELL, or HOLD). Critique your plan.
    4.  **Final Decision:** State your final, synthesized decision.

    --- CRITICAL RULES ---
    - **Minimum Trade:** Must be at least $10 USDT.
    - **BUY Rule:** DO NOT BUY if USDT balance is < $15.
    - **SELL Rule:** DO NOT SELL if you have zero holdings of the asset.

    --- RESPONSE FORMAT (JSON ONLY) ---
    {{
      "reasoning": "CHAIN-OF-THOUGHT ANALYSIS...",
      "decision": "BUY",
      "quantity_pct": 0.5
    }}
    """
    
    print(f"Running multi-dimensional analysis with {MODEL_NAME}...")
    try:
        payload = {
            "model": MODEL_NAME, "prompt": prompt, "stream": False, "format": "json",
            "options": { "temperature": 0.7, "num_predict": 700, "num_ctx": 8192 }
        }
        r = requests.post(OLLAMA_GENERATE_URL, json=payload, timeout=120)
        r.raise_for_status()
        
        data = r.json().get('response', '{}')
        data = json.loads(data)

        decision = data.get('decision', 'HOLD').upper()
        quantity_pct_val = data.get('quantity_pct')
        if quantity_pct_val is None:
            quantity = 0.0
        else:
            try:
                quantity = float(quantity_pct_val)
            except (ValueError, TypeError):
                quantity = 0.0
        reasoning = data.get('reasoning', '')
        
        return decision, reasoning, "MULTI_DIM", quantity, None, data.get('add_symbols', []), data.get('remove_symbols', [])
        
    except Exception as e:
        print(f"ERROR: AI Brain failed to regulate: {e}", flush=True)
        traceback.print_exc()
        return "HOLD", str(e), "ERROR", 0.0, None, [], []

def _get_lot_size_precision(step_size_str: str) -> int:
    return len(step_size_str.split('.')[1].split('1')[0]) if '.' in step_size_str else 0

def _format_quantity(quantity: float, precision: int) -> float:
    factor = 10**precision
    return math.floor(quantity * factor) / factor

def get_adjusted_time():
    return original_time_time() + (global_time_offset_ms / 1000.0)

def execute_trade(client, symbol, decision, quantity_pct=1.0):
    if not client: 
        print("Cannot execute trade without a client (mock mode).")
        return None

    asset = symbol.replace('USDT', '')
    q_pct = max(0.1, min(1.0, quantity_pct))
    
    try:
        price = float(client.get_symbol_ticker(symbol=symbol)['price'])
    except Exception as e:
        print(f"Warning: Could not fetch real-time price for trade. Error: {e}", flush=True)
        return None

    trade_executed_info = None
    try:
        if decision == "BUY":
            usdt_balance = float(client.get_asset_balance(asset='USDT')['free'])
            buy_amount_usdt = usdt_balance * q_pct
            
            if buy_amount_usdt >= 10.0:
                print(f"Executing BUY order for {symbol} using {q_pct*100:.0f}% of USDT (${buy_amount_usdt:.2f})...", flush=True)
                order = client.create_order(symbol=symbol, side=SIDE_BUY, type=ORDER_TYPE_MARKET, quoteOrderQty=round(buy_amount_usdt, 2))
                print(f"BUY Success: {order['orderId']}", flush=True)
                
                bought_quantity = float(order['executedQty'])
                trade_executed_info = {'action': "BUY", 'amount': bought_quantity, 'price': price, 'orderId': order['orderId'], 'currency': asset, 'usdt_amount': buy_amount_usdt}
            else:
                print(f"SKIPPED BUY: Amount ${buy_amount_usdt:.2f} is below Binance minimum ($10).", flush=True)
                
        elif decision == "SELL":
            asset_balance = float(client.get_asset_balance(asset=asset)['free'])
            sell_amount_asset = asset_balance * q_pct
            sell_val_usdt = sell_amount_asset * price
            
            if sell_val_usdt >= 10.0:
                symbol_info = client.get_symbol_info(symbol)
                lot_size_filter = next((f for f in symbol_info['filters'] if f['filterType'] == 'LOT_SIZE'), None)
                if not lot_size_filter:
                    raise Exception(f"Could not find LOT_SIZE filter for {symbol}")

                precision = _get_lot_size_precision(lot_size_filter['stepSize'])
                adjusted_quantity = _format_quantity(sell_amount_asset, precision)
                
                if adjusted_quantity > 0:
                    print(f"Executing SELL order for {adjusted_quantity:.6f} {asset}...", flush=True)
                    order = client.create_order(symbol=symbol, side=SIDE_SELL, type=ORDER_TYPE_MARKET, quantity=adjusted_quantity)
                    print(f"SELL Success: {order['orderId']}", flush=True)
                    trade_executed_info = {'action': "SELL", 'amount': adjusted_quantity, 'price': price, 'orderId': order['orderId'], 'currency': asset}
                else:
                    print(f"SKIPPED SELL: Adjusted quantity {adjusted_quantity} is too small.", flush=True)
            else:
                print(f"SKIPPED SELL: Value ${sell_val_usdt:.2f} is below Binance minimum ($10).", flush=True)
        else:
            print(f"BRAIN DECISION: {decision}. No action taken.", flush=True)
    except Exception as e:
        print(f"ERROR executing trade: {e}", flush=True)
        traceback.print_exc()
    
    return trade_executed_info

def print_status_update(symbol, market_summary, wallet_info, ai_decision, ai_reasoning, ai_quantity, live_data, order_book, is_brain_run, trade_executed_info):
    global initial_worth
    try:
        total_portfolio_worth = wallet_info.get('total_usd', 0.0)
        if initial_worth is None:
            initial_worth = total_portfolio_worth
            print(f"Initial portfolio worth set to: ${initial_worth:.2f}")

        session_profit = total_portfolio_worth - initial_worth
        session_profit_pct = (session_profit / initial_worth * 100) if initial_worth and initial_worth > 0 else 0
        
        print("\n" + "="*80)
        print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] - TRADING BOT STATUS")
        print(f"SYMBOL: {symbol} | Live Price: ${live_data.get('price', 'N/A')} | 24h Change: {live_data.get('price_change_pct', 'N/A')}% ")
        print(f"TOTAL PORTFOLIO WORTH: ${total_portfolio_worth:.2f} | Session P/L: ${session_profit:.2f} ({session_profit_pct:.2f}%)")
        print(wallet_info.get('text', ''))
        print(f"--- ORDER BOOK ---\n{order_book}\n")
        print(market_summary)
        
        if is_brain_run:
            print("\n" + "-"*80 + "\n--- AI ANALYSIS & DECISION ---")
            print(f"{ai_reasoning}")
            print(f"\nFINAL DECISION: [{ai_decision}] | QUANTITY: {ai_quantity*100:.0f}%")
            print("-" * 80)
        
    except Exception as e:
        print(f"Error printing status update: {e}", flush=True)

def get_live_ticker(client, symbol):
    if not client:
        return {"price": "0.00", "high24h": "0.00", "low24h": "0.00", "volume": "0.00", "price_change_pct": "0.00"}
    try:
        ticker = client.get_ticker(symbol=symbol)
        return {
            "price": ticker['lastPrice'], "high24h": ticker['highPrice'],
            "low24h": ticker['lowPrice'], "volume": ticker['volume'],
            "price_change_pct": ticker['priceChangePercent']
        }
    except Exception as e:
        print(f"Error fetching live ticker for {symbol}: {e}", flush=True)
        return {}

def prepare_real_wallet_for_clean_run(client, tracked_symbols):
    if not client:
        print("Cannot prepare wallet: client not available.")
        return
    print("--- PREPARING REAL WALLET FOR CLEAN RUN ---")
    
    base_assets_to_clear = {s.replace('USDT', '') for s in tracked_symbols}
    print(f"Will attempt to sell the following assets to USDT: {list(base_assets_to_clear)}")

    try:
        account = client.get_account(recvWindow=60000)
        all_tickers = client.get_all_tickers()
        prices = {t['symbol']: float(t['price']) for t in all_tickers}

        for balance in account.get('balances', []):
            asset = balance['asset']
            free = float(balance['free'])
            
            if asset in base_assets_to_clear and free > 0:
                symbol = f"{asset}USDT"
                if symbol in prices:
                    try:
                        symbol_info = client.get_symbol_info(symbol)
                        
                        min_notional_filter = next((f for f in symbol_info['filters'] if f['filterType'] == 'MIN_NOTIONAL'), None)
                        if not min_notional_filter:
                            print(f"Warning: MIN_NOTIONAL filter not found for {symbol}. Assuming 0.0 and proceeding.")
                            min_notional = 0.0
                        else:
                            min_notional = float(min_notional_filter['minNotional'])
                        
                        if free * prices[symbol] > min_notional:
                            lot_size_filter = next((f for f in symbol_info['filters'] if f['filterType'] == 'LOT_SIZE'), None)
                            if not lot_size_filter:
                                print(f"Warning: LOT_SIZE filter not found for {symbol}. Skipping sell.")
                                continue
                            
                            precision = _get_lot_size_precision(lot_size_filter['stepSize'])
                            quantity_to_sell = _format_quantity(free, precision)
                            
                            if quantity_to_sell > 0:
                                print(f"Executing cleanup sell: {quantity_to_sell} of {asset}...")
                                client.create_order(symbol=symbol, side=SIDE_SELL, type=ORDER_TYPE_MARKET, quantity=quantity_to_sell)
                                print(f"SOLD {quantity_to_sell} {asset}.")
                        else:
                            print(f"Balance of {asset} is below minimum notional value to trade. Skipping.")
                    except Exception as e:
                        print(f"Could not sell {asset} during cleanup: {e}")
                        traceback.print_exc()
                else:
                    print(f"Could not find a USDT market for {asset}. Skipping cleanup.")
    except Exception as e:
        print(f"Error preparing wallet for clean run: {e}")

    # Adjust USDT balance to 200
    try:
        print("--- Adjusting USDT balance to 200 ---")
        usdt_balance = float(client.get_asset_balance(asset='USDT')['free'])
        
        if usdt_balance > 200:
            surplus_usdt = usdt_balance - 200
            if surplus_usdt > 10: # Only park if surplus is significant (above min trade size)
                print(f"Surplus of {surplus_usdt:.2f} USDT detected. Parking it in TUSD.")
                try:
                    # To park USDT, we buy TUSD. The pair is TUSDUSDT. We place a BUY order for TUSD.
                    client.create_order(symbol='TUSDUSDT', side=SIDE_BUY, type=ORDER_TYPE_MARKET, quoteOrderQty=round(surplus_usdt, 2))
                    print("Successfully parked surplus USDT in TUSD.")
                except Exception as e:
                    print(f"Could not park surplus USDT: {e}")
            else:
                print(f"Surplus USDT (${surplus_usdt:.2f}) is not significant enough to park.")
        elif usdt_balance < 200:
            print(f"Warning: Starting USDT balance is ${usdt_balance:.2f}, which is less than the desired 200 USDT.")
    except Exception as e:
        print(f"Error adjusting USDT balance: {e}")

    print("--- WALLET PREPARATION FINISHED ---")

def run_trading_cycle(client, current_symbol, all_tracked_symbols, wallet_info, force_brain=False):
    global last_brain_run, current_decision, current_reasoning, current_quantity
    
    live_data = get_live_ticker(client, current_symbol)
    market_summary, _ = get_multi_timeframe_analysis(client, current_symbol)
    order_book = get_order_book_snapshot(client, current_symbol)
    trade_summary = get_recent_trades_summary()
    
    ai_decision, ai_reasoning, _, ai_quantity, _, add_syms, remove_syms = ask_ai_opinion(all_tracked_symbols, market_summary, wallet_info, order_book, live_data, trade_summary)
    
    trade_executed_info = execute_trade(client, current_symbol, ai_decision, ai_quantity)
    
    if trade_executed_info:
        wallet_info = get_wallet_info(client, all_tracked_symbols)

    print_status_update(current_symbol, market_summary, wallet_info, ai_decision, ai_reasoning, ai_quantity, live_data, order_book, True, trade_executed_info)
    
    log_decision_and_trade(current_symbol, ai_decision, ai_quantity, ai_reasoning, trade_executed_info, wallet_info, {})

    # Return symbol recommendations to main_loop
    return add_syms, remove_syms

def main_loop():
    global last_prune_date
    print("Starting AI Trading Terminal...", flush=True)
    init_db()

    if not check_ollama_status():
        print("\nAborting: Ollama is not ready.", flush=True)
        return
        
    if CLEAN_RUN_ENABLED and CLEAR_TRACKED_SYMBOLS_ON_CLEAN_RUN:
        print("Clearing tracked symbols table for clean run...")
        clear_tracked_symbols_table()

    tracked_symbols = get_active_tracked_symbols()
    if not tracked_symbols:
        print("No active symbols in database. Initializing from .env file...")
        initial_symbols_str = os.getenv('SYMBOLS', 'BTCUSDT,ETHUSDT')
        initial_symbols = [s.strip() for s in initial_symbols_str.split(',')]
        for s in initial_symbols:
            add_tracked_symbol(s)
        tracked_symbols = get_active_tracked_symbols()
        print(f"Initialized tracked symbols from .env: {tracked_symbols}")

    client = get_binance_client()
    if CLEAN_RUN_ENABLED:
        prepare_real_wallet_for_clean_run(client, tracked_symbols)

    while True:
        try:
            today = datetime.date.today()
            if last_prune_date != today:
                prune_database()
                last_prune_date = today

            current_tracked_symbols = get_active_tracked_symbols()
            if not current_tracked_symbols:
                print("No active symbols to trade. Waiting...", flush=True)
                time.sleep(BRAIN_INTERVAL_SECONDS)
                continue

            full_wallet_info = get_wallet_info(client, current_tracked_symbols)

            for symbol in current_tracked_symbols:
                add_syms, remove_syms = run_trading_cycle(client, symbol, current_tracked_symbols, full_wallet_info)
                
                # Handle symbol recommendations
                for s in add_syms:
                    if s not in current_tracked_symbols:
                        add_tracked_symbol(s)
                for s in remove_syms:
                    if s in current_tracked_symbols:
                        remove_tracked_symbol(s)
            
            print(f"\n{get_recent_trades_summary()}")
            print("-" * 80)
            
        except Exception as e:
            print(f"Main loop error: {e}", flush=True)
        
        time.sleep(BRAIN_INTERVAL_SECONDS)

if __name__ == "__main__":
    main_loop()