import os
import requests
import json
import time
import math
import pandas as pd
import pandas_ta as ta
import sqlite3
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
DB_FILE = '/app/shared/trading_decisions_v2.db'
DB_MAX_SIZE_MB = int(os.getenv('DB_MAX_SIZE_MB', '500')) # Max 500MB for the database file


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
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        c.execute('''
            CREATE TABLE IF NOT EXISTS decisions_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                symbol TEXT,
                ai_decision TEXT,
                ai_quantity_pct REAL,
                ai_reasoning TEXT,
                actual_action TEXT,
                actual_quantity REAL,
                actual_price REAL,
                post_trade_usdt_balance REAL,
                post_trade_asset_balance REAL,
                post_trade_total_worth REAL,
                context_snapshot TEXT,
                trade_id INTEGER,
                profit REAL
            )
        ''')
        # Create indexes to improve query performance
        c.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON decisions_log (timestamp)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_symbol ON decisions_log (symbol)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_actual_action ON decisions_log (actual_action)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_trade_id ON decisions_log (trade_id)")

        c.execute('''
            CREATE TABLE IF NOT EXISTS tracked_symbols (
                symbol TEXT PRIMARY KEY,
                added_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                status TEXT DEFAULT 'active'
            )
        ''')
        
        conn.commit()
        conn.close()
        print("Database initialized successfully.")
    except Exception as e:
        print(f"Error initializing database: {e}")

def get_db_size():
    if os.path.exists(DB_FILE):
        return os.path.getsize(DB_FILE) / (1024 * 1024) # Size in MB
    return 0

def prune_database():
    try:
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        
        current_db_size = get_db_size()
        if current_db_size <= DB_MAX_SIZE_MB:
            return

        print(f"Database size {current_db_size:.2f} MB exceeds {DB_MAX_SIZE_MB} MB. Pruning old records...", flush=True)

        # Keep deleting oldest records in chunks until size is manageable
        while get_db_size() > DB_MAX_SIZE_MB * 0.9: # Prune down to 90% of max size
            c.execute("DELETE FROM decisions_log WHERE id IN (SELECT id FROM decisions_log ORDER BY timestamp ASC LIMIT 100)")
            conn.commit()
            if c.rowcount == 0: # No more rows to delete
                break
        
        print(f"Database pruned. New size: {get_db_size():.2f} MB.", flush=True)
            
    except Exception as e:
        print(f"Error pruning database: {e}", flush=True)

def add_tracked_symbol(symbol):
    try:
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        c.execute("INSERT OR IGNORE INTO tracked_symbols (symbol, status) VALUES (?, ?)", (symbol, 'active'))
        conn.commit()
        conn.close()
        print(f"Symbol {symbol} added to tracked_symbols.", flush=True)
    except Exception as e:
        print(f"Error adding tracked symbol: {e}", flush=True)

def remove_tracked_symbol(symbol):
    try:
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        c.execute("UPDATE tracked_symbols SET status = 'inactive' WHERE symbol = ?", (symbol,))
        conn.commit()
        conn.close()
        print(f"Symbol {symbol} marked as inactive in tracked_symbols.", flush=True)
    except Exception as e:
        print(f"Error removing tracked symbol: {e}", flush=True)

def get_active_tracked_symbols():
    try:
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        c.execute("SELECT symbol FROM tracked_symbols WHERE status = 'active'")
        symbols = [row[0] for row in c.fetchall()]
        conn.close()
        return symbols
    except Exception as e:
        print(f"Error getting active tracked symbols: {e}", flush=True)
        return []

def log_decision_and_trade(symbol, ai_decision, ai_quantity_pct, ai_reasoning, trade_executed_info, wallet_info, context_snapshot):
    try:
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()

        actual_action = "NONE"
        actual_quantity = 0.0
        actual_price = 0.0
        if trade_executed_info:
            actual_action = trade_executed_info.get('action', 'NONE')
            actual_quantity = trade_executed_info.get('amount', 0.0)
            actual_price = trade_executed_info.get('price', 0.0)

        usdt_balance = next((b['free'] for b in wallet_info.get('balances', []) if b['asset'] == 'USDT'), 0.0)
        asset_name = symbol.replace('USDT', '')
        asset_balance = next((b['free'] for b in wallet_info.get('balances', []) if b['asset'] == asset_name), 0.0)
        total_worth = wallet_info.get('total_usd', 0.0)

        # Only store the context snapshot for BUY and SELL decisions to save space
        context_to_store = json.dumps(context_snapshot) if actual_action in ["BUY", "SELL"] else None

        cursor = c.execute('''
            INSERT INTO decisions_log (
                symbol, ai_decision, ai_quantity_pct, ai_reasoning,
                actual_action, actual_quantity, actual_price,
                post_trade_usdt_balance, post_trade_asset_balance, post_trade_total_worth,
                context_snapshot, trade_id, profit
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            symbol, ai_decision, ai_quantity_pct, ai_reasoning,
            actual_action, actual_quantity, actual_price,
            usdt_balance, asset_balance, total_worth,
            context_to_store,
            None, # trade_id
            None # profit
        ))
        
        last_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return last_id
    except Exception as e:
        print(f"Error logging decision to database: {e}", flush=True)
        return None


def calculate_and_log_profit(sell_trade_log_id):
    try:
        conn = sqlite3.connect(DB_FILE)
        conn.row_factory = sqlite3.Row
        c = conn.cursor()

        # Get the SELL trade
        c.execute("SELECT * FROM decisions_log WHERE id = ?", (sell_trade_log_id,))
        sell_trade = c.fetchone()
        if not sell_trade or sell_trade['actual_action'] != 'SELL':
            conn.close()
            return

        # Find the last unmatched BUY trade
        c.execute("""
            SELECT * FROM decisions_log 
            WHERE actual_action = 'BUY' AND trade_id IS NULL AND timestamp < ?
            ORDER BY timestamp DESC
            LIMIT 1
        """, (sell_trade['timestamp'],))
        buy_trade = c.fetchone()

        if buy_trade:
            # Simple profit calculation (assuming the amounts correspond)
            profit = sell_trade['actual_quantity'] - buy_trade['actual_quantity']
            
            # Create a new trade_id
            c.execute("SELECT MAX(trade_id) FROM decisions_log")
            max_trade_id = c.fetchone()[0]
            new_trade_id = (max_trade_id or 0) + 1
            
            # Update both BUY and SELL logs with the trade_id and profit
            c.execute("UPDATE decisions_log SET trade_id = ?, profit = ? WHERE id = ?", (new_trade_id, profit, sell_trade_log_id))
            c.execute("UPDATE decisions_log SET trade_id = ?, profit = ? WHERE id = ?", (new_trade_id, profit, buy_trade['id']))
            
            conn.commit()
            print(f"Profit calculation complete. Trade {new_trade_id} resulted in a profit of {profit:.2f}", flush=True)

        conn.close()
    except Exception as e:
        print(f"Error calculating profit: {e}", flush=True)

def get_past_performance_summary():
    try:
        conn = sqlite3.connect(DB_FILE)
        conn.row_factory = sqlite3.Row
        c = conn.cursor()

        c.execute("""
            SELECT * FROM decisions_log
            WHERE trade_id IS NOT NULL
            ORDER BY timestamp DESC
            LIMIT 10
        """)
        trades = c.fetchall()
        conn.close()

        if not trades:
            return "No completed trades found yet."

        summary = """--- SUMMARY OF LAST COMPLETED TRADES ---
"""
        
        processed_trades = {}
        for trade in trades:
            if trade['trade_id'] not in processed_trades:
                processed_trades[trade['trade_id']] = {}

            processed_trades[trade['trade_id']][trade['actual_action']] = trade

        for trade_id, actions in sorted(processed_trades.items(), key=lambda item: item[0], reverse=True)[:5]:
            if 'BUY' in actions and 'SELL' in actions:
                buy_trade = actions['BUY']
                sell_trade = actions['SELL']
                profit = sell_trade['profit']
                summary += f"Trade #{trade_id}: BUY @ ${buy_trade['actual_price']:.2f}, SELL @ ${sell_trade['actual_price']:.2f}, Profit: ${profit:.2f}\n"

        return summary
    except Exception as e:
        print(f"Error getting past performance summary: {e}", flush=True)
        return "Could not retrieve past performance."

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
        print(f"Error connecting to Binance: {e}", flush=True)
        return None

def get_wallet_info(client, symbols):
    global mock_portfolio
    # SIMULATED MODE
    if not client:
        # In simulated mode, we'll just mock a simple portfolio for the first symbol
        symbol = symbols[0]
        asset = symbol.replace('USDT', '')
        
        # Get live price even in simulated mode for accurate valuation
        live_ticker_data = get_live_ticker(client, symbol) # client will be None here, handled by get_live_ticker
        current_price = float(live_ticker_data.get('price', '45000.0')) # Use actual live price or fallback
            
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
            "trading_pair_worth": {symbol: total_usd},
            "text": text
        }
        
    try:
        # REAL MODE
        all_tickers = client.get_all_tickers()
        prices = {t['symbol']: float(t['price']) for t in all_tickers}
        account = client.get_account()
        
        total_usd = 0.0
        trading_pair_worth = {s: 0.0 for s in symbols}
        all_balances = {}

        # First, build a dictionary of all balances
        for b in account['balances']:
            free = float(b['free'])
            locked = float(b['locked'])
            total = free + locked
            asset_name = b['asset']
            if total > 0.00000001:
                all_balances[asset_name] = total

        # Calculate total worth and trading pair worth
        detailed_balances = []
        total_usd = 0.0
        trading_pair_worth = {s: 0.0 for s in symbols}

        # Filter balances to only include assets relevant to tracked symbols
        relevant_assets = set()
        for s in symbols: # e.g., 'BTCUSDT'
            # Assuming symbols are like 'BTCUSDT', extract 'BTC'
            base_asset = s.replace('USDT', '') 
            if base_asset: # Ensure it's not empty, e.g., for USDT itself
                relevant_assets.add(base_asset)
            relevant_assets.add('USDT') # USDT is always relevant as quote asset

        # Calculate total worth and trading pair worth ONLY for relevant assets
        for asset, total_balance in all_balances.items():
            if asset not in relevant_assets: # Skip assets not relevant to tracked pairs
                continue

            usd_val = 0.0
            if asset in ['USDT', 'BUSD', 'USDC']: # Quote assets
                usd_val = total_balance
                detailed_balances.append({"asset": asset, "balance": total_balance, "usd_value": usd_val})
            else: # Base assets (that are in relevant_assets)
                price_key = f"{asset}USDT"
                asset_price = prices.get(price_key, 0)
                usd_val = total_balance * asset_price
                if asset_price == 0 and total_balance > 0.01: # Warn only for significant balances
                    print(f"Warning: Price for {asset} ({price_key}) not found. Skipping from total_usd calculation.", flush=True)
                detailed_balances.append({"asset": asset, "balance": total_balance, "usd_value": usd_val, "price": asset_price})
            
            if usd_val > 0.01:
                total_usd += usd_val
            
            # Check which trading pair this asset belongs to
            for s in symbols:
                # Check if the asset is the base asset of the symbol or USDT
                if asset == s.replace('USDT', '') or asset == 'USDT':
                    trading_pair_worth[s] += usd_val
        
        # Build the text output
        text = f"--- REAL PORTFOLIO ---\nTOTAL WORTH (ALL ASSETS): ${total_usd:.2f} USD\n"
        for symbol, worth in trading_pair_worth.items():
            text += f"-- {symbol} Worth: ${worth:.2f} --\n"
            symbol_info = client.get_symbol_info(symbol)
            base_asset = symbol_info['baseAsset']
            quote_asset = symbol_info['quoteAsset']
            base_asset_balance = all_balances.get(base_asset, 0.0)
            quote_asset_balance = all_balances.get(quote_asset, 0.0)
            text += f"{base_asset}: {base_asset_balance:.6f} (${base_asset_balance * prices.get(symbol, 0):.2f})\n"
            text += f"{quote_asset}: {quote_asset_balance:.6f} (${quote_asset_balance:.2f})\n"


        return {
            "balances": account['balances'],
            "total_usd": total_usd,
            "trading_pair_worth": trading_pair_worth,
            "text": text,
            "detailed_balances_list": detailed_balances # NEW: Detailed list for AI
        }

    except Exception as e:
        print(f"Error fetching real wallet: {e}", flush=True)
        return {
            "balances": [],
            "total_usd": 0.0,
            "trading_pair_worth": {},
            "text": f"WALLET ERROR: {e}",
            "detailed_balances_list": [] # ADDED THIS LINE
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
        ask_volume = sum([float(a[1]) for a in depth['asks']])
        
        ratio = bid_volume / ask_volume if ask_volume > 0 else float('inf')
        
        pressure = "BUY" if ratio > 1.1 else "SELL" if ratio < 0.9 else "NEUTRAL"
        
        return f"Bid/Ask Volume Ratio: {ratio:.2f}:1 | Immediate Pressure: {pressure}"
    except Exception as e:
        return f"Could not get order book: {e}"

def ask_ai_opinion(current_tracked_symbols, market_summary, full_wallet_info, trade_history, order_book, live_data, dynamic_knowledge=""):
    wallet_text = full_wallet_info['text']
    usdt_balance = next((float(b['free']) for b in full_wallet_info.get('balances', []) if b['asset'] == 'USDT'), 0.0)
    total_portfolio_worth = full_wallet_info['total_usd']
    trade_summary = "No recent trades."
    if trade_history:
        trade_summary = "Recent Trades:\n"
        for trade in trade_history[-5:]:
            trade_summary += f"- {trade['type']} @ ${trade['price']:.2f} for ${trade['amount']:.2f}\n"

    live_data_summary = f"Live Price: {live_data.get('price', 'N/A')}"
    past_performance_summary = get_past_performance_summary()

    tracked_symbols_str = ", ".join(current_tracked_symbols)

    knowledge_base = """--- ADVANCED TRADING KNOWLEDGE BASE ---
**Quantitative & Algorithmic Strategies:**
- **Predictive Modeling (ML):** Use Machine Learning to analyze historical data and predict future price movements.
- **Sentiment Analysis (AI):** Process news and social media to gauge market sentiment.
- **Mean Reversion:** Assume that asset prices will revert to their historical average. Typically involves buying when the asset is significantly below its average (oversold) expecting a bounce, or selling when significantly above (overbought) expecting a pull-back. Focus on finding undervalued entry points.
- **Trend Following/Momentum:** Identify and trade in the direction of strong trends.
- **Pairs Trading:** Trade two highly correlated assets when their price ratio deviates from its average.

**Arbitrage Strategies:**
- **Cross-Exchange Arbitrage:** Exploit price differences for the same asset across different exchanges.
- **Triangular Arbitrage:** Exploit price differences between three different cryptocurrency pairs on a single exchange.
- **Spot-Futures Arbitrage:** Exploit price differences between the spot price and the futures contract price.
- **Market Making:** Profit from the bid-ask spread by placing both buy and sell orders.

**Advanced Technical Analysis:**
- **Elliott Wave Theory:** Identify recurring long-term price patterns related to investor psychology.
- **Blockchain Analysis:** Analyze on-chain data (e.g., transaction volumes, wallet movements) to gain insights.

**Risk Management:**
- Always use Stop-Loss and Take-Profit orders.
- Diversify your portfolio.
- Avoid high leverage.
- Maintain emotional discipline.
- Use proper position sizing (e.g., risk 1-2% of portfolio per trade).
"""

    detailed_balances_list = full_wallet_info['detailed_balances_list']
    
    detailed_balances_prompt_section = "--- DETAILED ASSET BREAKDOWN ---\n"
    for item in detailed_balances_list:
        detailed_balances_prompt_section += f"- {item['asset']}: {item['balance']:.6f} (${item['usd_value']:.2f})"
        if item.get('price'):
            detailed_balances_prompt_section += f" @ ${item['price']:.2f}/{item['asset']}"
        detailed_balances_prompt_section += "\n"


    prompt = f"""
    ROLE: You are a sophisticated, self-improving crypto trading analyst. Your primary goal is to maximize profit. You are currently tracking the following symbols: {tracked_symbols_str}. Based on your analysis, you can decide to BUY, SELL, HOLD, or recommend changes to the list of tracked symbols.

    You must follow a strict Chain-of-Thought reasoning process.

    {knowledge_base}

    --- DYNAMIC KNOWLEDGE (from recent web search) ---
    {dynamic_knowledge}

    --- DATA STREAMS ---
    1. CURRENT PORTFOLIO:
    {wallet_text}
    - Total Portfolio Worth: ${total_portfolio_worth:.2f} USD (This value needs verification against individual assets)
    - Available USDT for trading: ${usdt_balance:.2f}
    
    {detailed_balances_prompt_section}
    
    {trade_summary}

    2. PAST PERFORMANCE:
    {past_performance_summary}


    4. TECHNICAL ANALYSIS:
    {market_summary}

    --- REASONING PROCESS (Chain-of-Thought) ---
    1.  **Assess Portfolio & Market:** Analyze the overall market conditions and your current portfolio, considering all tracked symbols.
    2.  **Evaluate Tracked Symbols:** Review the performance and potential of currently tracked symbols. Should any be removed or new ones added for better diversification or profit opportunities?
    3.  **Formulate Symbol Recommendations (if any):** If you recommend adding or removing symbols, justify these changes.
    4.  **Assess Situation & Select Strategy:** For the current symbol, based on all available data (past performance, market analysis, etc.), select the most appropriate strategy from the ADVANCED TRADING KNOWLEDGE BASE. Justify your choice.
    5.  **Analyze Data Through the Lens of the Chosen Strategy:** Apply the selected strategy to the available data streams for the current symbol.
    6.  **Propose & Critique Plan:** Based on your analysis, propose a primary action (BUY, SELL, or HOLD) for the current symbol. Critique your plan, considering the risks and potential rewards.
    7.  **Final Decision:** State your final, synthesized decision for the current symbol, and any recommendations for tracked symbols.

    --- CRITICAL RULES ---
    - **Minimum Trade:** Must be at least $10 USDT.
    - **BUY Rule:** DO NOT BUY if USDT balance is < $15.
    - **SELL Rule:** DO NOT SELL if you have zero holdings of the asset.

    --- RESPONSE FORMAT (JSON ONLY) ---
    {{
      "reasoning": "CHAIN-OF-THOUGHT ANALYSIS: [Your full, step-by-step reasoning process here. This must be a single JSON-escaped string, without nested JSON objects or unescaped newlines/quotes.]",
      "decision": "BUY", "SELL", or "HOLD",
      "quantity_pct": 0.5,
      "add_symbols": ["BTCUSDT", "ETHUSDT"],
      "remove_symbols": ["LTCUSDT"]
    }}
    """
    
    print(f"Running multi-dimensional analysis with {MODEL_NAME}...")
    try:
        options = {
            "temperature": 0.7,
            "num_predict": 700,
            "num_ctx": 8192
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
        
        try:
            data = json.loads(res)
        except json.JSONDecodeError as e:
            print(f"ERROR: AI returned invalid JSON: {e}", flush=True)
            print(f"Invalid JSON received: {res}", flush=True)
            return "HOLD", "Invalid JSON response from AI", "ERROR", 0.0, None, [], []

        decision = data.get('decision', 'HOLD').upper()
        
        quantity_pct_str = data.get('quantity_pct', '0.0')
        if isinstance(quantity_pct_str, (int, float)):
            quantity = float(quantity_pct_str)
        elif isinstance(quantity_pct_str, str) and (quantity_pct_str.replace('.', '', 1).isdigit() or (quantity_pct_str.startswith('-') and quantity_pct_str[1:].replace('.', '', 1).isdigit())):
            quantity = float(quantity_pct_str)
        else:
            quantity = 0.0

        reasoning = data.get('reasoning', '')
        add_symbols = data.get('add_symbols', [])
        remove_symbols = data.get('remove_symbols', [])
        
        return decision, reasoning, "MULTI_DIM", quantity, None, add_symbols, remove_symbols
        
    except Exception as e:
        print(f"ERROR: AI Brain failed to regulate: {e}", flush=True)
        return "HOLD", str(e), "ERROR", 0.0, None, [], []

def perform_web_search_and_summarize(search_query):
    print(f"AI decided to search for: {search_query}", flush=True)
    # This is a placeholder for the web search and summarization logic.
    # We will implement this in the next steps.
    return "Web search is not yet implemented."


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
        print(f"SIMULATED TRADE: {decision} @ {price}", flush=True)
        if decision == "BUY":
            buy_val = mock_portfolio["usdt"] * q_pct
            if buy_val > 1.0:
                mock_portfolio["usdt"] -= buy_val
                mock_portfolio["asset"] += (buy_val / price)
                # For simulated BUY, store the quantity of asset bought and USDT amount spent
                trade_history.append({"time": time.strftime('%H:%M:%S'), "type": "BUY", "price": price, "amount": (buy_val / price), "currency": asset, "usdt_amount": buy_val})
        elif decision == "SELL":
            sell_qty = mock_portfolio["asset"] * q_pct
            if sell_qty > 0.000001:
                mock_portfolio["asset"] -= sell_qty
                mock_portfolio["usdt"] += (sell_qty * price)
                # For simulated SELL, amount is the quantity of the asset sold (e.g., BTC)
                trade_history.append({"time": time.strftime('%H:%M:%S'), "type": "SELL", "price": price, "amount": sell_qty, "currency": asset})
        return None # Return None in simulated mode for trade_executed_info

    trade_executed_info = None
    try:
        if decision == "BUY":
            usdt_balance = float(client.get_asset_balance(asset='USDT')['free'])
            buy_amount_usdt = usdt_balance * q_pct
            base_asset_name = symbol.replace('USDT', '')
            
            if buy_amount_usdt >= 10.0:
                print(f"Executing BUY order using {q_pct*100:.0f}% of USDT (${buy_amount_usdt:.2f})...", flush=True)
                order = client.create_order(symbol=symbol, side=SIDE_BUY, type=ORDER_TYPE_MARKET, quoteOrderQty=round(buy_amount_usdt, 2))
                print(f"BUY Success: {order['orderId']}", flush=True)
                
                # Calculate quantity of base asset bought
                bought_quantity = float(order['executedQty']) # This is the actual quantity received
                
                trade_info = {"time": time.strftime('%H:%M:%S'), "type": "BUY", "price": price, "amount": bought_quantity, "currency": base_asset_name, "usdt_amount": buy_amount_usdt}
                trade_history.append(trade_info)
                trade_executed_info = {'action': "BUY", 'amount': bought_quantity, 'price': price, 'orderId': order['orderId'], 'currency': base_asset_name, 'usdt_amount': buy_amount_usdt}
            else:
                print(f"SKIPPED BUY: Amount ${buy_amount_usdt:.2f} is below Binance minimum ($10). Check Testnet balance.", flush=True)
                
        elif decision == "SELL":
            base_asset_name = symbol.replace('USDT', '')
            asset_balance = float(client.get_asset_balance(asset=base_asset_name)['free'])
            sell_amount_asset = asset_balance * q_pct
            sell_val_usdt = sell_amount_asset * price # Value in USDT
            
            if sell_val_usdt >= 10.0:
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
                         return None # Return None if skipped

                    print(f"Executing SELL order using {q_pct*100:.0f}% of holdings ({adjusted_quantity:.6f} {base_asset_name})...", flush=True)
                    order = client.create_order(symbol=symbol, side=SIDE_SELL, type=ORDER_TYPE_MARKET, quantity=adjusted_quantity)
                    print(f"SELL Success: {order['orderId']}", flush=True)
                    # For real SELL, amount is the quantity of the asset sold (e.g., BTC)
                    trade_info = {"time": time.strftime('%H:%M:%S'), "type": "SELL", "price": price, "amount": adjusted_quantity, "currency": base_asset_name}
                    trade_history.append(trade_info)
                    trade_executed_info = {'action': "SELL", 'amount': adjusted_quantity, 'price': price, 'orderId': order['orderId'], 'currency': base_asset_name}
                except Exception as e:
                    print(f"ERROR during SELL order pre-check: {e}", flush=True)
            else:
                print(f"SKIPPED SELL: Value ${sell_val_usdt:.2f} is below Binance minimum ($10).", flush=True)
        else:
            print(f"BRAIN DECISION: {decision}. No action taken.", flush=True)
    except Exception as e:
        print(f"ERROR executing regulated trade: {e}", flush=True)
    
    return trade_executed_info

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
        print(f"Error fetching live ticker: {e}", flush=True)
        return {}

last_brain_run = 0
current_decision = "HOLD"
current_reasoning = "Initializing Brain..."
current_persona = "BALANCED"
current_quantity = 0.0
initial_worth = None
trade_history = []
mock_portfolio = {"usdt": 48.525301, "asset": 0.614980, "asset_name": "BTC"}
price_history = []
profit_history = []
last_prune_date = None

def print_status_update(symbol, market_summary, wallet_info, ai_decision, ai_reasoning, ai_persona, ai_quantity, live_data, order_book, is_brain_run, trade_executed_info=None):
    global price_history, profit_history, initial_worth
    try:
        now = time.strftime('%H:%M:%S')
        
        current_price = float(live_data.get('price', 0)) if live_data else 0
        total_worth = float(wallet_info.get('trading_pair_worth', 0))
        
        if initial_worth is None: initial_worth = total_worth
            
        if current_price > 0: price_history.append({"time": now, "price": current_price})
        profit_history.append({"time": now, "worth": total_worth})
        
        if len(price_history) > 100: price_history.pop(0)
        if len(profit_history) > 100: profit_history.pop(0)

        connection_mode = "REAL" if os.getenv('BINANCE_API_KEY') else "SIMULATED"

        print("\n" + "="*80)
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] - TRADING BOT STATUS ({connection_mode} MODE)")
        print("="*80)
        print(f"SYMBOL: {symbol} | Live Price: ${live_data.get('price', 'N/A')} | 24h Change: {live_data.get('price_change_pct', 'N/A')}% ")
        profit = total_worth - initial_worth
        profit_pct = (profit / initial_worth * 100) if initial_worth > 0 else 0
        print(f"Trading Pair Worth: ${total_worth:.2f} | Session P/L: ${profit:.2f} ({profit_pct:.2f}%)")
        print(wallet_info['text'])
        print(f"--- ORDER BOOK ---\n{order_book}\n")
        print(market_summary)
        
        if is_brain_run: # This condition prints AI decision if it ran in this cycle
            print("\n" + "-"*80)
            print("--- AI ANALYSIS & DECISION ---")
            print(f"{ai_reasoning}")
            print(f"\nFINAL DECISION: [{ai_decision}] | QUANTITY: {ai_quantity*100:.0f}%")
            print("-" * 80)
        else: # If brain didn't run, report last known decision summary
            print("\n" + "-"*80)
            print("--- LAST AI DECISION (Brain not re-evaluated this cycle) ---")
            print(f"DECISION: [{ai_decision}] | QUANTITY: {ai_quantity*100:.0f}%")
            print("-" * 80)
        
        if trade_executed_info:
            print(f"\n--- TRADE EXECUTED ---")
            if trade_executed_info['action'] == "BUY":
                print(f"ACTION: {trade_executed_info['action']} {trade_executed_info['amount']:.6f} {trade_executed_info['currency']} WITH ${trade_executed_info['usdt_amount']:.2f} USDT @ PRICE: ${trade_executed_info['price']:.2f}")
            elif trade_executed_info['action'] == "SELL":
                print(f"ACTION: {trade_executed_info['action']} {trade_executed_info['amount']:.6f} {trade_executed_info['currency']} FOR ${trade_executed_info['amount'] * trade_executed_info['price']:.2f} USDT @ PRICE: ${trade_executed_info['price']:.2f}")
            print(f"ORDER ID: {trade_executed_info['orderId']}")
            print("-" * 80)

        if trade_history:
            print("\n--- RECENT TRADES ---")
            for trade in reversed(trade_history[-5:]):
                if trade['type'] == "BUY":
                    print(f"{trade['time']} | {trade['type']:<4} | {trade['amount']:.6f} {trade['currency']} WITH ${trade['usdt_amount']:.2f} USDT @ Price: ${trade['price']:.2f}")
                elif trade['type'] == "SELL":
                    print(f"{trade['time']} | {trade['type']:<4} | {trade['amount']:.6f} {trade['currency']} FOR ${trade['amount'] * trade['price']:.2f} USDT @ Price: ${trade['price']:.2f}")
            print("-" * 80)

    except Exception as e:
        print(f"Error printing status update: {e}", flush=True)


def run_trading_cycle(current_symbol, all_tracked_symbols, wallet_info, force_brain=False):
    global last_brain_run, current_decision, current_reasoning, current_persona, current_quantity, initial_worth
    
    client = get_binance_client()
    
    # --- Fetch market data for the current_symbol ---
    live_data = get_live_ticker(client, current_symbol)
    market_summary, market_analysis_data = get_multi_timeframe_analysis(client, current_symbol)
    order_book = get_order_book_snapshot(client, current_symbol)

    # --- AI Decision Logic ---
    current_time = time.time()
    is_brain_run_this_cycle = force_brain or (current_time - last_brain_run) >= BRAIN_INTERVAL_SECONDS
    
    # Initialize variables for the AI's decision, these will be updated if AI runs
    decision = current_decision # Use globals as starting point
    reasoning = current_reasoning
    persona = current_persona
    quantity = current_quantity
    
    add_symbols_recommended = []
    remove_symbols_recommended = []

    MAX_RETRIES = 3
    ai_call_succeeded = False

    if is_brain_run_this_cycle:
        for attempt in range(MAX_RETRIES):
            try:
                print(f"\n--- BRAIN RE-EVALUATION (Attempt {attempt+1}/{MAX_RETRIES}) ({time.strftime('%H:%M:%S')}) for {current_symbol} ---", flush=True)
                ai_decision, ai_reasoning, ai_persona, ai_quantity, search_query, ai_add_symbols, ai_remove_symbols = ask_ai_opinion(
                    all_tracked_symbols, market_summary, wallet_info, trade_history, order_book, live_data
                )
                
                # If successful, assign to local variables and break
                decision = ai_decision
                reasoning = ai_reasoning
                persona = ai_persona
                quantity = ai_quantity
                add_symbols_recommended = ai_add_symbols
                remove_symbols_recommended = ai_remove_symbols
                ai_call_succeeded = True
                break
            except Exception as e:
                print(f"Error during AI re-evaluation (Attempt {attempt+1}/{MAX_RETRIES}): {e}", flush=True)
                if attempt < MAX_RETRIES - 1:
                    time.sleep(5) # Wait before retrying
                else:
                    # If all retries fail, fall back to HOLD and log the error in current_reasoning
                    decision = "HOLD"
                    reasoning = f"AI Brain failed after {MAX_RETRIES} retries: {e}"
                    persona = "ERROR"
                    quantity = 0.0
                    print(f"!!! AI Brain failed after {MAX_RETRIES} retries. Forcing HOLD. !!!", flush=True)
                    # add_symbols_recommended and remove_symbols_recommended remain empty
    
    # Update global state for next cycle if brain ran this cycle
    if is_brain_run_this_cycle and ai_call_succeeded:
        current_decision = decision
        current_reasoning = reasoning
        current_persona = persona
        current_quantity = quantity
        last_brain_run = current_time # Only update last_brain_run if AI call was successful

    # Use the local 'decision', 'reasoning', 'quantity' (potentially updated by AI or fallback) for reporting and guardrails
    decision_for_report = decision
    reasoning_for_report = reasoning
    persona_for_report = persona
    quantity_for_report = quantity
    
    trade_executed_info = None # Initialize to None before potential execution

    # --- GUARDRAILS & DYNAMIC RISK (for the current_symbol) ---
    usdt_balance = float(next((b['free'] for b in wallet_info.get('balances', []) if b['asset'] == 'USDT'), 0.0))
    base_asset = current_symbol.replace('USDT', '')
    base_asset_balance = float(next((b['free'] for b in wallet_info.get('balances', []) if b['asset'] == base_asset), 0.0))

    if decision_for_report == "BUY":
        if usdt_balance < 10.0:
            print(f"!!! AI OVERRIDE: AI decided to BUY {current_symbol} with insufficient USDT (${usdt_balance:.2f}). Forcing HOLD. !!!", flush=True)
            decision_for_report = "HOLD" # Override the decision for report
            reasoning_for_report += " (OVERRIDDEN: Insufficient USDT to meet minimum trade size of $10.)"
        else: # Dynamic Risk Scaling based on 15m RSI
            # Need to extract 15m RSI from chosen_symbol_market_summary
            rsi_15m = 50 # Default
            # Re-fetching single symbol market data to get analysis_data
            _, chosen_market_analysis_data = get_multi_timeframe_analysis(client, current_symbol)

            if chosen_market_analysis_data.get('15m', {}).get('rsi', 50) > 75:
                print(f"!!! RISK SCALING: 15m RSI for {current_symbol} is over 75 (overbought). Reducing quantity. !!!", flush=True)
                quantity_for_report = min(quantity_for_report, 0.1) # Cap quantity at 10%

    elif decision_for_report == "SELL": # Use elif to ensure only one decision branch is taken
        if base_asset_balance <= 0:
            print(f"!!! AI OVERRIDE: AI decided to SELL {current_symbol} with zero {base_asset} balance. Forcing HOLD. !!!", flush=True)
            decision_for_report = "HOLD" # Override the decision for report
            reasoning_for_report += f" (OVERRIDDEN: Zero {base_asset} balance to sell.)"
    # --- END GUARDRAILS ---

    if decision_for_report != "HOLD":
        trade_executed_info = execute_trade(client, current_symbol, decision_for_report, quantity_for_report)
    
    # --- Final Status Report for this cycle ---
    current_symbol_wallet_text = f"--- REAL PORTFOLIO ---\nTOTAL WORTH (ALL ASSETS): ${wallet_info['total_usd']:.2f} USD\n"
    if current_symbol in wallet_info['trading_pair_worth']:
        current_symbol_wallet_text += f"-- {current_symbol} Worth: ${wallet_info['trading_pair_worth'][current_symbol]:.2f} --\n"
        symbol_info = client.get_symbol_info(current_symbol) # This might be an expensive call, consider moving up if client is real
        base_asset = symbol_info['baseAsset']
        quote_asset = symbol_info['quoteAsset']
        
        base_balance_obj = next((b for b in wallet_info['balances'] if b['asset'] == base_asset), None)
        quote_balance_obj = next((b for b in wallet_info['balances'] if b['asset'] == quote_asset), None)
        
        if base_balance_obj:
            current_symbol_wallet_text += f"{float(base_balance_obj['free']):.6f} {base_asset} (${float(base_balance_obj['free']) * float(live_data.get('price', 0)):.2f})\n"
        if quote_balance_obj:
            current_symbol_wallet_text += f"{float(quote_balance_obj['free']):.6f} {quote_asset} (${float(quote_balance_obj['free']):.2f})\n"
        
    print_status_update(current_symbol, market_summary, {"text": current_symbol_wallet_text, "trading_pair_worth": wallet_info['trading_pair_worth'].get(current_symbol, 0.0)}, decision_for_report, reasoning_for_report, persona_for_report, quantity_for_report, live_data, order_book, is_brain_run_this_cycle, trade_executed_info)
    
    # --- Log decision and context to database ---
    context_snapshot = { "market_summary": market_summary, "order_book": order_book, "live_data": live_data, "wallet_info": wallet_info, "trade_history": trade_history }
    last_log_id = log_decision_and_trade(current_symbol, decision_for_report, quantity_for_report, str(reasoning_for_report), trade_executed_info, wallet_info, context_snapshot)
    
    if last_log_id and trade_executed_info and trade_executed_info.get('action') == 'SELL':
        calculate_and_log_profit(last_log_id)
        
    return add_symbols_recommended, remove_symbols_recommended


def main_loop():
    global last_prune_date
    print("Starting AI Trading Terminal V5.3 (Final Synchronous Workflow)...", flush=True)
    init_db()

    if not check_ollama_status():
        print("\nAborting: Ollama is not ready.", flush=True)
        return

    # Initialize from DB or .env
    symbols = get_active_tracked_symbols()
    if not symbols:
        initial_symbols_str = os.getenv('SYMBOLS', 'BTCUSDT')
        initial_symbols = [s.strip() for s in initial_symbols_str.split(',')]
        for s in initial_symbols:
            add_tracked_symbol(s)
        symbols = get_active_tracked_symbols()
        print(f"Initialized tracked symbols from .env: {symbols}", flush=True)
    
    # The initial force_brain run will now be handled inside the main loop
    
    while True:
        try:
            today = time.strftime('%Y-%m-%d')
            if last_prune_date != today:
                prune_database()
                last_prune_date = today

            current_tracked_symbols = get_active_tracked_symbols() # Always get the freshest list
            if not current_tracked_symbols:
                print("No active symbols to trade. Waiting for new symbols...", flush=True)
                time.sleep(BRAIN_INTERVAL_SECONDS)
                continue

            # Initialize full_wallet_info to a safe default before it's potentially used
            full_wallet_info = {
                "balances": [],
                "total_usd": 0.0,
                "trading_pair_worth": {},
                "text": "WALLET NOT INITIALIZED",
                "detailed_balances_list": []
            }

            client = get_binance_client() # Get client once per loop iteration
            full_wallet_info = get_wallet_info(client, current_tracked_symbols)

            # Run trading cycle for each symbol sequentially
            for symbol in current_tracked_symbols:
                add_syms, remove_syms = run_trading_cycle(symbol, current_tracked_symbols, full_wallet_info, force_brain=True) # force_brain=True for initial pass
                
                for s in add_syms:
                    if s not in current_tracked_symbols:
                        add_tracked_symbol(s)
                for s in remove_syms:
                    if s in current_tracked_symbols:
                        remove_tracked_symbol(s)

        except Exception as e:
            print(f"Loop Error: {e}", flush=True)
        
        time.sleep(BRAIN_INTERVAL_SECONDS)

if __name__ == "__main__":
    main_loop()