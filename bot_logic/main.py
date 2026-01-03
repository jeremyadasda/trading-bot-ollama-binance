import os
import time
import datetime
import traceback
from dotenv import load_dotenv, find_dotenv
from binance.client import Client

from database import DatabaseHandler
from market_data import MarketDataHandler
from strategy import AIStrategy
from execution import TradeExecutor

# Load Env
load_dotenv(find_dotenv())

# --- CONFIGURATION ---
BINANCE_API_KEY = os.getenv('BINANCE_API_KEY')
BINANCE_API_SECRET = os.getenv('BINANCE_API_SECRET')
BRAIN_INTERVAL_SECONDS = int(os.getenv('BRAIN_INTERVAL_SECONDS', '12'))
CLEAN_RUN_ENABLED = os.getenv('CLEAN_RUN_ENABLED', 'true').lower() == 'true'
CLEAR_TRACKED_SYMBOLS_ON_CLEAN_RUN = os.getenv('CLEAR_TRACKED_SYMBOLS_ON_CLEAN_RUN', 'true').lower() == 'true'
MODEL_NAME = "llama3"

# --- GLOBALS ---
global_time_offset_ms = 0
original_time_time = time.time

def get_adjusted_time():
    return original_time_time() + (global_time_offset_ms / 1000.0)

def init_binance_client():
    global global_time_offset_ms
    if not BINANCE_API_KEY or not BINANCE_API_SECRET:
        print("Warning: Binance keys not found. Running in MOCK MODE.")
        return None

    try:
        temp_client = Client(BINANCE_API_KEY, BINANCE_API_SECRET, testnet=True)
        server_time = temp_client.get_server_time()
        local_time = int(original_time_time() * 1000)
        global_time_offset_ms = (server_time['serverTime'] - local_time) - 2000 # Subtract 2s buffer to avoid "ahead of server" errors
        print(f"Time offset: {global_time_offset_ms}ms (including safety buffer)")
        
        time.time = get_adjusted_time # Monkey-patch
        return Client(BINANCE_API_KEY, BINANCE_API_SECRET, testnet=True, requests_params={'timeout': 10})
    except Exception as e:
        print(f"Error connecting to Binance: {e}")
        return None

def print_status_update(symbol, market_summary, wallet_info, ai_decision, ai_reasoning, ai_quantity, live_data, order_book, trade_executed_info, session_profit=0.0, session_profit_pct=0.0, trades_summary=""):
    try:
        total_worth = wallet_info.get('total_usd', 0.0)
        print("\n" + "="*80)
        print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] - TRADING BOT STATUS")
        print(f"SYMBOL: {symbol} | Live Price: ${live_data.get('price', 'N/A')} | 24h Change: {live_data.get('price_change_pct', 'N/A')}% ")
        print(f"TOTAL PORTFOLIO WORTH: ${total_worth:.2f} | Session P/L: ${session_profit:.2f} ({session_profit_pct:.2f}%)")
        print(wallet_info.get('text', ''))
        print(f"{trades_summary}")
        print(f"--- ORDER BOOK ---\n{order_book}\n")
        print(f"{market_summary}")
        
        print("\n" + "-"*80 + "\n--- AI ANALYSIS & DECISION ---")
        print(f"{ai_reasoning}")
        print(f"\nFINAL DECISION: [{ai_decision}] | QUANTITY: {ai_quantity*100:.0f}%")
        if trade_executed_info:
             print(f"TRADE EXECUTED: {trade_executed_info}")
        print("-" * 80)
        
    except Exception as e:
        print(f"Error printing status update: {e}")

def main_loop():
    print("Starting Modular AI Trading Terminal...", flush=True)

    # 1. Initialize DB
    try:
        db = DatabaseHandler()
    except Exception as e:
        print(f"CRITICAL: Database connection failed. {e}")
        return

    # 2. Initialize Strategy (Ollama)
    ai = AIStrategy(MODEL_NAME)
    if not ai.ensure_model_available():
        print("Aborting: AI Model not available.")
        return

    # 3. Clean Run Logic
    if CLEAN_RUN_ENABLED:
        if CLEAR_TRACKED_SYMBOLS_ON_CLEAN_RUN:
            # We also reset the DB tables now to ensure schema issues are cleared on clean runs
            db.reset_tables()
        else:
            db.clear_tracked_symbols()

    tracked_symbols = db.get_active_tracked_symbols()
    if not tracked_symbols:
        print("No active symbols. Initializing from .env...")
        initial = os.getenv('SYMBOLS', 'BTCUSDT,ETHUSDT')
        for s in initial.split(','):
            db.add_tracked_symbol(s.strip())
        tracked_symbols = db.get_active_tracked_symbols()

    # 4. Initialize Binance & Sub-handlers
    client = init_binance_client()
    market = MarketDataHandler(client)
    executor = TradeExecutor(client)

    # Prepare Wallet for Clean Run (Liquidate and Reset to 200 USDT)
    if CLEAN_RUN_ENABLED:
        executor.prepare_real_wallet_for_clean_run(tracked_symbols)

    initial_worth = None

    while True:
        try:
            # Prune DB daily (simple check)
            # db.prune_database() # Optional, call strictly if needed

            current_tracked_symbols = db.get_active_tracked_symbols()
            if not current_tracked_symbols:
                print("No symbols to trade. Sleeping...")
                time.sleep(BRAIN_INTERVAL_SECONDS)
                continue

            wallet_info = executor.get_wallet_info(current_tracked_symbols)
            
            # Track Session Profit
            total_worth = wallet_info.get('total_usd', 0.0)
            if initial_worth is None: initial_worth = total_worth
            profit = total_worth - initial_worth
            profit_pct = (profit/initial_worth*100) if initial_worth else 0


            # A. Prepare Comprehensive Portfolio Data
            portfolio_summary = ""
            for s in current_tracked_symbols:
                summary, _ = market.get_multi_timeframe_analysis(s)
                portfolio_summary += f"\n{summary}\n"

            trades_summary = db.get_recent_trades_summary()

            # Trade Cycle
            for symbol in current_tracked_symbols:
                # B. Per-Symbol Real-time Data
                live_data = market.get_live_ticker(symbol)
                order_book = market.get_order_book_snapshot(symbol)

                # C. AI Analysis (Now with full portfolio context)
                decision, reasoning, quantity, add_syms, remove_syms, kb_update = ai.ask_ai_opinion(
                    current_tracked_symbols, portfolio_summary, wallet_info, order_book, live_data, trades_summary
                )
                
                if kb_update:
                    ai.update_knowledge_base(kb_update)

                # C. Execution
                trade_info = executor.execute_trade(symbol, decision, quantity)

                # D. Logging
                if trade_info:
                    # Update wallet if trade happened
                    wallet_info = executor.get_wallet_info(current_tracked_symbols)

                # Log decision first to ensure DB is up to date
                db.log_decision(symbol, decision, quantity, reasoning, trade_info, wallet_info, {})

                # If a trade occurred, refresh the summary so it appears in the status immediately
                if trade_info:
                     trades_summary = db.get_recent_trades_summary()

                print_status_update(
                    symbol, portfolio_summary, wallet_info, decision, reasoning, quantity, live_data, order_book, trade_info, profit, profit_pct, trades_summary
                )

                # E. Symbol Management
                for s in add_syms: 
                    if not s or not isinstance(s, str) or not s.strip(): continue # Filter invalid
                    s = s.strip().upper()
                    
                    # CRITICAL: Prevent re-adding (and re-cleaning) symbols we are already tracking
                    if s in current_tracked_symbols:
                        continue

                    print(f"AI RECOMMENDATION: ADD {s}")
                    trade_info_cleanup = executor.liquidate_symbol(s) # Zero it first
                    
                    if trade_info_cleanup:
                        # Log the liquidation for visibility
                        db.log_decision(s, "REMOVE", 1.0, f"Initial cleanup before adding {s} to tracking.", trade_info_cleanup, wallet_info, {})
                        
                        liquidated_val = trade_info_cleanup['usdt_amount']
                        # User Request: If we have existing funds (e.g. 100 SOL), "clean" it by buying INDEPENDENT currency (USDC).
                        # This segregates it from Trading Capital (USDT) and Profit Savings (TUSD).
                        print(f"Parking ${liquidated_val:.2f} of pre-existing {s} into USDC (Capital Exclusion).")
                        executor.park_asset(liquidated_val, target_asset='USDC')
                        # We do NOT adjust initial_worth. This money is ejected from the system.
                        
                    db.add_tracked_symbol(s)
                    current_tracked_symbols.append(s) # Update local list for immediate visibility
                    
                for s in remove_syms: 
                    if not s or not isinstance(s, str) or not s.strip(): continue
                    s = s.strip().upper()
                    
                    print(f"AI RECOMMENDATION: REMOVE {s}")
                    # Just exit position to USDT. Do NOT park it.
                    trade_info_remove = executor.liquidate_symbol(s) 
                    if trade_info_remove:
                        db.log_decision(s, "REMOVE", 1.0, f"AI Recommendation: Remove {s} from tracking.", trade_info_remove, wallet_info, {})
                    
                    db.remove_tracked_symbol(s)
                    if s in current_tracked_symbols: current_tracked_symbols.remove(s) # Update local list
                    
                # Update wallet info immediately to reflect new assets for next iteration/logging
                if add_syms or remove_syms:
                     wallet_info = executor.get_wallet_info(current_tracked_symbols)

        except Exception as e:
            print(f"Main Loop Error: {e}")
            traceback.print_exc()

        time.sleep(BRAIN_INTERVAL_SECONDS)

if __name__ == "__main__":
    main_loop()