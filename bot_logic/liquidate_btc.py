import os
import time
from binance.client import Client
from dotenv import load_dotenv

def sell_all_btc():
    print("--- 🚀 LIQUIDATING BTC FOR USDT TESTING ---")
    load_dotenv()
    
    api_key = os.getenv('BINANCE_API_KEY')
    api_secret = os.getenv('BINANCE_API_SECRET')
    symbol = os.getenv('SYMBOL', 'BTCUSDT')
    
    if not api_key or not api_secret:
        print("ERROR: API keys not found in .env")
        return

    client = Client(api_key, api_secret, testnet=True)
    
    try:
        # 1. Check BTC Balance
        asset = symbol.replace('USDT', '')
        asset_balance = float(client.get_asset_balance(asset=asset)['free'])
        print(f"You have: {asset_balance} {asset}")
        
        if asset_balance < 0.0001:
            print("❌ Nothing to sell! Your BTC balance is too low.")
            return

        # 2. Market Sell EVERYTHING
        print(f"Selling ALL {asset} to get USDT...")
        
        # Round down to avoid 'precision' errors
        order = client.create_order(
            symbol=symbol,
            side=Client.SIDE_SELL,
            type=Client.ORDER_TYPE_MARKET,
            quantity=round(asset_balance, 5) 
        )
        
        print(f"✅ SUCCESS! Sold holdings for USDT.")
        print(f"Order ID: {order['orderId']}")
        
        # 3. Final Balance Check
        time.sleep(1)
        new_usdt = float(client.get_asset_balance(asset='USDT')['free'])
        print(f"\n💰 NEW BALANCE: ${new_usdt:,.2f} USDT")
        print("The bot now has plenty of 'fuel' for high-frequency trading!")
        
    except Exception as e:
        print(f"\n❌ FAILED: {e}")

if __name__ == "__main__":
    sell_all_btc()
