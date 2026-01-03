from ddgs import DDGS
import time

class SentimentSentinel:
    def __init__(self, ai_strategy):
        self.ai = ai_strategy
        self.ddgs = DDGS()

    def analyze_symbol_sentiment(self, symbol):
        """Fetches recent news for a symbol and returns a sentiment score (-1 to 1)."""
        # Remove USDT from symbol for better search results
        query_asset = symbol.replace("USDT", "")
        query = f"{query_asset} crypto news today"
        
        print(f"Sentinel fetching sentiment for {symbol}...")
        try:
            results = self.ddgs.text(query, max_results=5)
            if not results:
                return 0.0 # Neutral if no data
            
            headlines = "\n".join([f"- {r.get('title')}: {r.get('body')[:200]}" for r in results])
            
            # Use AI to score
            score = self.ai.get_sentiment_score(symbol, headlines)
            return score
        except Exception as e:
            print(f"Sentiment analysis error for {symbol}: {e}")
            return 0.0
