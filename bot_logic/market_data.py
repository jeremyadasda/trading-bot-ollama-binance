import pandas as pd
import pandas_ta as ta
from binance.client import Client
import time
import os
import json

class MarketDataHandler:
    def __init__(self, client):
        self.client = client
        self.kline_cache = {}
        self.kline_cache_ttl = 300  # 5 minutes
        self.stats_file = "bot_logic/global_stats.json"
        self._load_global_stats()

    def get_structural_analysis(self, symbol, interval, limit=100):
        # 1. Fetch Klines
        try:
            klines = self.client.get_klines(symbol=symbol, interval=interval, limit=limit)
            if not klines: return "No structural data"
            df = pd.DataFrame(klines, columns=['ts', 'open', 'high', 'low', 'close', 'vol', 'close_ts', 'qav', 'num_trades', 'taker_base', 'taker_quote', 'ignore'])
            df['close'] = df['close'].astype(float)
            df['high'] = df['high'].astype(float)
            df['low'] = df['low'].astype(float)
            prices = df['close'].tolist()
        except: return "Error fetching structural data"

        # 2. PAA (Piecewise Aggregate Approximation) - Downsample to 20 segments
        m = 20
        n = len(prices)
        paa = []
        if n >= m:
            step = n / m
            for i in range(m):
                segment = prices[int(i*step):int((i+1)*step)]
                paa.append(sum(segment) / len(segment))
        else:
            paa = prices

        # Normalize PAA for AI (0-100 scale)
        p_min, p_max = min(paa), max(paa)
        norm_paa = [int((p - p_min) / (p_max - p_min) * 100) if p_max > p_min else 50 for p in paa]

        # 3. ZigZag (Pivot Point Detection) - 2% threshold
        threshold = 0.02 
        pivots = []
        if n > 2:
            last_p = prices[0]
            last_type = None # 'H' or 'L'
            pivots.append(("START", last_p))
            
            for i in range(1, n):
                curr_p = prices[i]
                diff = (curr_p - last_p) / last_p
                
                if abs(diff) >= threshold:
                    pivots.append(("DOT", curr_p))
                    last_p = curr_p
            pivots.append(("END", prices[-1]))

        # 4. ASCII Sparkline
        spark = self._generate_sparkline(norm_paa)
        
        # 5. Format for AI
        summary = f"SHAPE: {spark}\n"
        summary += f"NORMALIZED PRICE MAP (0-100): {norm_paa}\n"
        summary += f"PIVOTS: {' -> '.join([f'${p[1]:.2f}' for p in pivots[-6:]])} (Last 6 swings)"
        return summary

    def _generate_sparkline(self, sequence):
        # Simple 3-level ASCII sparkline
        chars = [' ', '_', '-', '^']
        # Map 0-100 to 0-3
        return "".join([chars[min(3, int(v/25))] for v in sequence])

    def _load_global_stats(self):
        if os.path.exists(self.stats_file):
            try:
                with open(self.stats_file, 'r') as f:
                    self.global_stats = json.load(f)
            except:
                self.global_stats = {}
        else:
            self.global_stats = {}

    def _save_global_stats(self):
        try:
            os.makedirs(os.path.dirname(self.stats_file), exist_ok=True)
            with open(self.stats_file, 'w') as f:
                json.dump(self.global_stats, f, indent=4)
        except:
            pass

    def _fetch_global_founding_price(self, symbol):
        # Static map of founding/ICO prices for major coins
        # To avoid listing date limitations on exchanges
        founding_prices = {
            "BTCUSDT": 0.0009,   # April 2010
            "ETHUSDT": 0.31,     # July 2014 ICO
            "SOLUSDT": 0.22,     # April 2020 ICO
            "BNBUSDT": 0.15,     # July 2017 ICO
            "ADAUSDT": 0.0024,   # Oct 2017 Launch
            "XRPUSDT": 0.005,    # 2013 Launch
            "DOTUSDT": 2.90,     # Aug 2020 Launch
            "LINKUSDT": 0.11,    # Sept 2017 ICO
            "MATICUSDT": 0.0026, # April 2019 IEO
            "DOGEUSDT": 0.0002,  # Dec 2013 Launch
        }
        return founding_prices.get(symbol)

    def get_live_ticker(self, symbol):
        if not self.client: return {"price": 0.0}
        try:
            ticker = self.client.get_symbol_ticker(symbol=symbol)
            return {"price": float(ticker['price'])}
        except Exception as e:
            print(f"Error getting live ticker for {symbol}: {e}")
            return {"price": 0.0}

    def get_multi_timeframe_analysis(self, symbol):
        if not self.client: return "Market data unavailable", {}
        
        timeframes = {
            "15m": Client.KLINE_INTERVAL_15MINUTE,
            "1h": Client.KLINE_INTERVAL_1HOUR,
            "4h": Client.KLINE_INTERVAL_4HOUR,
            "1d": Client.KLINE_INTERVAL_1DAY,
            "1w": Client.KLINE_INTERVAL_1WEEK
        }
        
        full_summary = f"--- [{symbol}] Multi-Timeframe Analysis ---"
        analysis_data = {}
        
        for tf_name, tf_interval in timeframes.items():
            try:
                cache_key = f"{symbol}_{tf_name}"
                now = time.time()
                
                if cache_key in self.kline_cache and (now - self.kline_cache[cache_key]['timestamp']) < self.kline_cache_ttl:
                    klines = self.kline_cache[cache_key]['data']
                else:
                    klines = self.client.get_klines(symbol=symbol, interval=tf_interval, limit=100)
                    self.kline_cache[cache_key] = {'timestamp': now, 'data': klines}

                df = pd.DataFrame(klines, columns=['ts', 'open', 'high', 'low', 'close', 'vol', 'close_ts', 'qav', 'num_trades', 'taker_base', 'taker_quote', 'ignore'])
                df['close'] = df['close'].astype(float)
                
                # Indicators
                df['RSI'] = ta.rsi(df['close'], length=14)
                df['EMA_20'] = ta.ema(df['close'], length=20)
                df['EMA_50'] = ta.ema(df['close'], length=50)
                
                last = df.iloc[-1]
                price = last['close']
                rsi = last['RSI'] if pd.notna(last['RSI']) else 50
                ema20 = last['EMA_20'] if pd.notna(last['EMA_20']) else price
                ema50 = last['EMA_50'] if pd.notna(last['EMA_50']) else price
                
                trend = "BULL" if price > ema20 > ema50 else "BEAR" if price < ema20 < ema50 else "SIDE"
                full_summary += f"\n{tf_name}: {trend} (RSI {rsi:.0f})"
                analysis_data[tf_name] = {"rsi": rsi, "trend": trend, "close": price}

            except Exception as e:
                full_summary += f"\n{tf_name}: Err {e}"

        # 1 Year & Lifetime Analysis (Special Logic using 1M)
        try:
             # Fetch latest monthly data. limit=1000 covers ~83 years.
             klines_1m = self.client.get_klines(symbol=symbol, interval=Client.KLINE_INTERVAL_1MONTH, limit=1000)
             
             if not klines_1m:
                 full_summary += "\nLIFETIME: No Data"
             else:
                 df_1m = pd.DataFrame(klines_1m, columns=['ts', 'open', 'high', 'low', 'close', 'vol', 'close_ts', 'qav', 'num_trades', 'taker_base', 'taker_quote', 'ignore'])
                 df_1m['close'] = df_1m['close'].astype(float)
                 current_price = float(df_1m.iloc[-1]['close'])

                 # 1. Lifetime Analysis (Global vs Binance)
                 binance_start_price = float(df_1m.iloc[0]['open']) if float(df_1m.iloc[0]['open']) > 0 else float(df_1m.iloc[0]['close'])
                 
                 founding_price = self._fetch_global_founding_price(symbol)
                 if founding_price:
                      lifetime_change = ((current_price - founding_price) / founding_price) * 100
                      full_summary += f"\nLIFETIME (Real World): BULL ({lifetime_change:+.0f}%) [Calculated from ICO/founding price of ${founding_price}]"
                 else:
                      lifetime_change = ((current_price - binance_start_price) / binance_start_price) * 100
                      lifetime_trend = "BULL" if current_price > binance_start_price else "BEAR"
                      full_summary += f"\nLIFETIME (Exchange): {lifetime_trend} ({lifetime_change:+.0f}%) [Since first candle on Binance]"

                 # 2. 1 Year Analysis (Last 12 months)
                 price_1y_ago = None
                 if len(df_1m) >= 12:
                     price_1y_ago = float(df_1m.iloc[-12]['close'])
                 else:
                     # Fallback to Weekly Data (approx 52 weeks)
                     try:
                         klines_1w = self.client.get_klines(symbol=symbol, interval=Client.KLINE_INTERVAL_1WEEK, limit=53)
                         if len(klines_1w) >= 52:
                             price_1y_ago = float(klines_1w[0][4]) # Close of 52 weeks ago
                     except: pass

                 if price_1y_ago:
                     change_1y = ((current_price - price_1y_ago) / price_1y_ago) * 100
                     trend_1y = "BULL" if change_1y > 0 else "BEAR"
                     full_summary += f"\n1Y: {trend_1y} ({change_1y:+.0f}%)"
                 else:
                     since_start_change = ((current_price - binance_start_price) / binance_start_price) * 100
                     full_summary += f"\n1Y: N/A ({since_start_change:+.1f}% since listing {len(df_1m)}m ago)"

             # 3. Add Structural "Visual" Context (Daily/Weekly based on timeframe)
             structural_context = self.get_structural_analysis(symbol, Client.KLINE_INTERVAL_1DAY, limit=60) 
             full_summary += f"\n\n--- [VIRTUAL GRAPH: LAST 60 DAYS] ---\n{structural_context}"

        except Exception as e:
            full_summary += f"\nLONG TERM: Err {e}"
                
        return full_summary, analysis_data

    def get_ml_features(self, symbol, interval=Client.KLINE_INTERVAL_1HOUR, limit=500):
        """Generates a technical feature set for ML models."""
        try:
            klines = self.client.get_klines(symbol=symbol, interval=interval, limit=limit)
            if not klines: return None
            
            df = pd.DataFrame(klines, columns=['ts', 'open', 'high', 'low', 'close', 'vol', 'close_ts', 'qav', 'num_trades', 'taker_base', 'taker_quote', 'ignore'])
            df[['open', 'high', 'low', 'close', 'vol']] = df[['open', 'high', 'low', 'close', 'vol']].astype(float)
            
            # 1. Technical Indicators
            df['RSI'] = ta.rsi(df['close'], length=14)
            df['ATR'] = ta.atr(df['high'], df['low'], df['close'], length=14)
            
            # MACD
            macd = ta.macd(df['close'])
            df = pd.concat([df, macd], axis=1)
            
            # Bollinger Bands
            bbands = ta.bbands(df['close'], length=20, std=2)
            df = pd.concat([df, bbands], axis=1)
            
            # EMAs
            df['EMA_20'] = ta.ema(df['close'], length=20)
            df['EMA_50'] = ta.ema(df['close'], length=50)
            df['EMA_200'] = ta.ema(df['close'], length=200)
            
            # 2. Percentage Changes (Momentum)
            df['roc_1'] = df['close'].pct_change(1)
            df['roc_5'] = df['close'].pct_change(5)
            df['roc_10'] = df['close'].pct_change(10)
            
            # 3. Volatility (Rolling Std)
            df['volatility_20'] = df['roc_1'].rolling(20).std()
            
            # 4. Target Labeling (For Training Fallback)
            # Future return over next 4 candles (e.g. 4 hours if interval is 1h)
            df['target_return'] = df['close'].shift(-4) / df['close'] - 1
            df['target'] = (df['target_return'] > 0.01).astype(int) # 1 if > 1% gain
            
            # Clean up NaNs
            df = df.dropna()
            return df
        except Exception as e:
            print(f"Error generating ML features for {symbol}: {e}")
            return None

    def get_order_book_snapshot(self, symbol):
        if not self.client: return "Order book data unavailable"
        try:
            depth = self.client.get_order_book(symbol=symbol, limit=10)
            bid_volume = sum([float(bid[1]) for bid in depth['bids']])
            ask_volume = sum([float(a[1]) for a in depth['asks']])
            ratio = bid_volume / ask_volume if ask_volume > 0 else float('inf')
            pressure = "BUY" if ratio > 1.1 else "SELL" if ratio < 0.9 else "NEUTRAL"
            return f"Bid/Ask Volume Ratio: {ratio:.2f}:1 | Immediate Pressure: {pressure}"
        except Exception as e:
            return f"Could not get order book: {e}"
