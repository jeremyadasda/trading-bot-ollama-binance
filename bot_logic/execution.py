import math
import traceback
from binance.enums import *

class TradeExecutor:
    def __init__(self, client):
        self.client = client

    def _get_lot_size_precision(self, step_size_str: str) -> int:
        return len(step_size_str.split('.')[1].split('1')[0]) if '.' in step_size_str else 0

    def _format_quantity(self, quantity: float, precision: int) -> float:
        factor = 10**precision
        return math.floor(quantity * factor) / factor

    def _get_symbol_rules(self, symbol):
        """Fetch LOT_SIZE and NOTIONAL rules for a symbol."""
        try:
            info = self.client.get_symbol_info(symbol)
            if not info: return None
            
            lot_filter = next((f for f in info['filters'] if f['filterType'] == 'LOT_SIZE'), None)
            notional_filter = next((f for f in info['filters'] if f['filterType'] in ['NOTIONAL', 'MIN_NOTIONAL']), None)
            
            rules = {
                'precision': self._get_lot_size_precision(lot_filter['stepSize']) if lot_filter else 8,
                'min_qty': float(lot_filter['minQty']) if lot_filter else 0.0,
                'min_notional': float(notional_filter.get('minNotional', notional_filter.get('notional', 5.5))) if notional_filter else 5.5
            }
            return rules
        except:
            return None

    def execute_trade(self, symbol, decision, quantity_pct=1.0):
        if not self.client:
            print("Cannot execute trade without a client (mock mode).")
            return None

        asset = symbol.replace('USDT', '')
        q_pct = max(0.1, min(1.0, quantity_pct))
        
        try:
            price = float(self.client.get_symbol_ticker(symbol=symbol)['price'])
        except Exception as e:
            print(f"Warning: Could not fetch real-time price for trade. Error: {e}")
            return None

        trade_executed_info = None
        try:
            if decision == "BUY":
                usdt_balance = float(self.client.get_asset_balance(asset='USDT')['free'])
                buy_amount_usdt = usdt_balance * q_pct
                
                # Safety buffer for fees/fluctuations if using full balance
                if q_pct > 0.99:
                     buy_amount_usdt *= 0.995 # Leave 0.5% buffer
                
                rules = self._get_symbol_rules(symbol)
                min_buy = rules['min_notional'] if rules else 10.0
                
                if buy_amount_usdt >= min_buy:
                    print(f"Executing BUY order for {symbol} using {q_pct*100:.0f}% of USDT (${buy_amount_usdt:.2f})...")
                    order = self.client.create_order(symbol=symbol, side=SIDE_BUY, type=ORDER_TYPE_MARKET, quoteOrderQty=round(buy_amount_usdt, 2))
                    print(f"BUY Success: {order['orderId']}")
                    
                    bought_quantity = float(order['executedQty'])
                    trade_executed_info = {'action': "BUY", 'amount': bought_quantity, 'price': price, 'orderId': order['orderId'], 'currency': asset, 'usdt_amount': buy_amount_usdt}
                else:
                    print(f"SKIPPED BUY: Amount ${buy_amount_usdt:.2f} is below minimum (${min_buy}).")
                    
            elif decision == "SELL":
                asset_balance = float(self.client.get_asset_balance(asset=asset)['free'])
                sell_amount_asset = asset_balance * q_pct
                sell_val_usdt = sell_amount_asset * price
                
                rules = self._get_symbol_rules(symbol)
                min_sell = rules['min_notional'] if rules else 10.0

                if sell_val_usdt >= min_sell:
                    precision = rules['precision'] if rules else 8
                    adjusted_quantity = self._format_quantity(sell_amount_asset, precision)
                    
                    if adjusted_quantity > 0:
                        print(f"Executing SELL order for {adjusted_quantity:.6f} {asset}...")
                        order = self.client.create_order(symbol=symbol, side=SIDE_SELL, type=ORDER_TYPE_MARKET, quantity=adjusted_quantity)
                        print(f"SELL Success: {order['orderId']}")
                        trade_executed_info = {'action': "SELL", 'amount': adjusted_quantity, 'price': price, 'orderId': order['orderId'], 'currency': asset}
                    else:
                        print(f"SKIPPED SELL: Adjusted quantity {adjusted_quantity} is too small.")
                else:
                    print(f"SKIPPED SELL: Value ${sell_val_usdt:.2f} is below minimum (${min_sell}).")
            else:
                print(f"BRAIN DECISION: {decision}. No action taken.")
        except Exception as e:
            print(f"ERROR executing trade: {e}")
            traceback.print_exc()
        
        return trade_executed_info

    def get_wallet_info(self, symbols):
        if not self.client: 
            return { "balances": [], "total_usd": 0.0, "trading_pair_worth": {}, "text": "MOCK MODE: No client.", "detailed_balances_list": [] }
            
        try:
            account = self.client.get_account(recvWindow=60000)
            all_tickers = self.client.get_all_tickers()
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
            print(f"Error fetching real wallet: {e}")
            return { "balances": [], "total_usd": 0.0, "trading_pair_worth": {}, "text": f"WALLET ERROR: {e}", "detailed_balances_list": [] }

    def prepare_real_wallet_for_clean_run(self, tracked_symbols):
        if not self.client:
            print("Cannot prepare wallet: client not available.")
            return

        print("--- PREPARING REAL WALLET FOR CLEAN RUN ---")
        
        # 1. Liquidate actively tracked symbols first
        for s in tracked_symbols:
             self.liquidate_symbol(s) # Note: We could log these too if user wants total history

        # 2. Scavenge all OTHER side-assets to USDT (including TUSD, FDUSD, etc.)
        # This simplifies the process and consolidates all capital to USDT.
        try:
            print("Consolidating side-assets to USDT...")
            account = self.client.get_account(recvWindow=60000)
            all_tickers = self.client.get_all_tickers()
            prices = {t['symbol']: float(t['price']) for t in all_tickers}
            
            # Stables/Side-assets we want to liquidate into USDT
            # EXCEPT USDC which is our "Safe Zone" for excluded capital
            side_assets_to_clean = {'TUSD', 'FDUSD', 'BUSD', 'DAI', 'DOT', 'LTC', 'ADA', 'XRP', 'SOL', 'ETH', 'BTC'}
            
            for b in account.get('balances', []):
                asset = b['asset']
                free = float(b['free'])
                if asset in side_assets_to_clean and free > 0.0:
                    pair = f"{asset}USDT"
                    if pair in prices:
                        self.liquidate_symbol(pair)

            # 3. Final Balance Adjustment to $200
            fresh_usdt = float(self.client.get_asset_balance(asset='USDT')['free'])
            print(f"Current USDT Balance: ${fresh_usdt:.2f}")

            if fresh_usdt < 200:
                print(f"Still below $200. No more side-assets found to scavenge.")
            elif fresh_usdt > 200:
                surplus = fresh_usdt - 200
                if surplus > 1.0:
                    self.park_asset(surplus, target_asset='USDC')
                else:
                    print(f"Balance is ${fresh_usdt:.2f} (Within $1 tolerance of $200).")
        except Exception as e:
            print(f"Error consolidating wallet: {e}")

        print("--- WALLET PREPARATION FINISHED ---")
                


        print("--- WALLET PREPARATION FINISHED ---")

    def liquidate_symbol(self, symbol):
        """Liquidates the ENTIRE balance of a symbol into USDT. Returns trade_info dict."""
        if not self.client: return None
        asset = symbol.replace('USDT', '')
        try:
            balance = float(self.client.get_asset_balance(asset=asset)['free'])
            if balance <= 0: return None

            rules = self._get_symbol_rules(symbol)
            if not rules: return None

            price = float(self.client.get_symbol_ticker(symbol=symbol)['price'])
            usdt_val = balance * price
            
            if usdt_val < rules['min_notional']:
                print(f"Skipping {asset}: Value ${usdt_val:.2f} < Min Notional ${rules['min_notional']:.2f}")
                return None

            qty = self._format_quantity(balance, rules['precision'])
            if qty < rules['min_qty']:
                print(f"Skipping {asset}: Qty {qty} < Min Qty {rules['min_qty']}")
                return None

            print(f"Liquidating {qty} {asset}...")
            order = self.client.create_order(symbol=symbol, side=SIDE_SELL, type=ORDER_TYPE_MARKET, quantity=qty)
            
            print(f"Liquidation Success: {order['orderId']}")
            return {
                'action': "SELL", 
                'amount': qty, 
                'price': price, 
                'orderId': order['orderId'], 
                'currency': asset,
                'usdt_amount': qty * price
            }
        except Exception as e:
            print(f"Error liquidating {symbol}: {e}")
        return None

        return 0.0
    


    def park_asset(self, usdt_amount, target_asset='USDC'):
        if not self.client or usdt_amount <= 0: return
        
        target_symbol = f"{target_asset}USDT"
        rules = self._get_symbol_rules(target_symbol)
        
        # Check Notional
        if not rules or usdt_amount < rules['min_notional']:
            # If we are parking, and it's too small, we just leave it as USDT.
            # No error, just a log.
            print(f"Surplus ${usdt_amount:.2f} too small to park into {target_asset} (Min Notional ${rules['min_notional'] if rules else '??'}). Keeping as liquid USDT.")
            return

        print(f"Parking ${usdt_amount:.2f} USDT into {target_asset}...")
        try:
            self.client.create_order(symbol=target_symbol, side=SIDE_BUY, type=ORDER_TYPE_MARKET, quoteOrderQty=round(usdt_amount, 2))
            print(f"Parking successful.")
        except Exception as e:
            print(f"Error parking asset to {target_asset}: {e}")
