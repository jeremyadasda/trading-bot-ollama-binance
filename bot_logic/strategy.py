import os
import json
import requests
import datetime
import traceback

class AIStrategy:
    def __init__(self, model_name="llama3"):
        self.base_ollama_url = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
        self.generate_url = f"{self.base_ollama_url}/api/generate"
        self.tags_url = f"{self.base_ollama_url}/api/tags"
        self.pull_url = f"{self.base_ollama_url}/api/pull"
        self.model_name = model_name
        self.kb_path = 'bot_logic/knowledge_base.md'

    def ensure_model_available(self):
        print(f"Checking Ollama connection at {self.base_ollama_url}...")
        try:
            r = requests.get(self.tags_url, timeout=5)
            r.raise_for_status()
            models = r.json().get('models', [])
            if any(self.model_name in m.get('name', '') for m in models):
                print(f"OK: Model '{self.model_name}' is available.")
                return True
            return self.pull_model(self.model_name)
        except Exception as e:
            print(f"ERROR connecting to Ollama: {e}")
            return False

    def pull_model(self, model_name):
        print(f"Downloading/Verifying model '{model_name}'...")
        try:
            payload = {"name": model_name}
            with requests.post(self.pull_url, json=payload, stream=True, timeout=None) as r:
                r.raise_for_status()
                for line in r.iter_lines():
                    if line:
                        data = json.loads(line)
                        status = data.get('status', '')
                        print(f"\rStatus: {status}", end="", flush=True)
                print(f"\nModel '{model_name}' is ready!")
                return True
        except Exception as e:
            print(f"\nERROR downloading model: {e}")
            return False

    def update_knowledge_base(self, new_insight):
        if not new_insight or len(new_insight) < 5: return
        
        # Filter generic non-updates and hallucinations
        boilerplate = ["none needed", "no update", "optional. a new rule", "a new rule or insight", "optional material"]
        if any(b in new_insight.lower() for b in boilerplate):
            return

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d")
        entry = f"- **[Auto-Learned {timestamp}]:** {new_insight}"
        
        try:
            # Ensure the directory exists
            os.makedirs(os.path.dirname(self.kb_path), exist_ok=True)

            # Read existing content for deduplication
            content = ""
            if os.path.exists(self.kb_path):
                with open(self.kb_path, "r", encoding='utf-8') as f:
                    content = f.read()
            
            # Simple Deduplication: Check if the exact insight string already exists
            if new_insight.strip() in content:
                print(f"Skipping duplicate knowledge update: {new_insight[:50]}...")
                return

            # Append the new entry
            with open(self.kb_path, "a", encoding='utf-8') as f:
                f.write(f"\n{entry}")
            print(f"Knowledge Base Updated: {new_insight[:50]}...")
            
        except Exception as e:
            print(f"Could not update Knowledge Base: {e}")

    def ask_ai_opinion(self, current_tracked_symbols, market_summary, full_wallet_info, order_book, live_data, trade_summary, ml_score=0.5, thought_buffer=None, sentiment_score=0.0):
        # ... (rest of the setup logic remains same)
        wallet_text = full_wallet_info.get('text', '')
        usdt_balance = next((float(b['free']) for b in full_wallet_info.get('balances', []) if b['asset'] == 'USDT'), 0.0)
        total_portfolio_worth = full_wallet_info.get('total_usd', 0.0)
        
        tracked_symbols_str = ", ".join(current_tracked_symbols)
        detailed_balances_list = full_wallet_info.get('detailed_balances_list', [])
        
        detailed_balances_prompt_section = "--- DETAILED ASSET BREAKDOWN ---"
        for item in detailed_balances_list:
            detailed_balances_prompt_section += f"- {item['asset']}: {item['balance']:.6f} (${item['usd_value']:.2f})\n"

        try:
            with open(self.kb_path, 'r', encoding='utf-8') as f:
                knowledge_base = f.read()
        except Exception:
            knowledge_base = "Knowledge base file not found. Using default strategies."

        # Format Scores
        ml_insight = f"ML Statistical Confidence: {ml_score*100:.1f}% "
        sent_insight = "BULLISH" if sentiment_score > 0.3 else "BEARISH" if sentiment_score < -0.3 else "NEUTRAL"
        sent_val = f"{sentiment_score:+.2f} ({sent_insight})"

        # Recursive Prompt Buffer Injection
        recursive_history = ""
        if thought_buffer and len(thought_buffer) > 0:
            recursive_history = "\n--- RECURSIVE REASONING CHAIN (PREVIOUS CYCLES) ---\n"
            for i, thought in enumerate(thought_buffer):
                recursive_history += f"Cycle {i+1}: {thought}\n"
            recursive_history += "\nCRITICAL: Use your previous thoughts to refine or pivot your final decision. Do not ignore them.\n"

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

    2. TECHNICAL ANALYSIS (ML + Structural):
    {market_summary}
    Live Price: {live_data.get('price', 'N/A')}
    Order Book: {order_book}
    ---
    HYBRID INTELLIGENCE:
    [ML MODEL SCORE]: {ml_insight}
    [SENTIMENT SCORE]: {sent_val} (Range -1 to +1)
    
    Note: Weight statistical and sentiment scores against your structural analysis.

    {recursive_history}

    --- REASONING PROCESS (Chain-of-Thought) ---
    1.  **Assess Portfolio & Market:** Analyze market conditions and your portfolio.
    2.  **Evaluate Current Symbol:** Using data streams, determine if this is a strong setup.
    3.  **Cross-Check:** Compare ML Score and Sentiment with Structural Analysis.
    4.  **Meta-Review:** If you have previous thoughts, how have the new data points changed your outlook?
    5.  **Final Verdict:** Decide on an action and quantity.

    RESPONSE FORMAT (JSON ONLY):
    {{
      "reasoning": "Your analysis...",
      "decision": "BUY",
      "quantity_pct": 0.0,
      "thinking_cycles_requested": 0,
      "knowledge_update": "A single, specific new insight or strategy you've learned. If NO new insight, use null.",
      "add_symbols": ["SOLUSDT"],
      "remove_symbols": [] 
    }}
    
    DEEP THINKING RULE:
    If you are unsure or want to observe more data before acting, set "thinking_cycles_requested" to 1-5 (each is 12s). 
    
    IMPORTANT: Do NOT include placeholders like 'Optional' or 'Optional material' in the knowledge_update. If you have nothing new to add, the field MUST be null.
    """
        
        print(f"Running hybrid analysis with {self.model_name}...")
        try:
            payload = {
                "model": self.model_name, "prompt": prompt, "stream": False, "format": "json",
                "options": { "temperature": 0.7, "num_predict": 700, "num_ctx": 8192 }
            }
            r = requests.post(self.generate_url, json=payload, timeout=120)
            r.raise_for_status()
            
            data = r.json().get('response', '{}')
            data_json = json.loads(data)

            decision = (data_json.get('decision') or 'HOLD').upper()
            
            # Robust quantity parsing
            raw_qty = data_json.get('quantity_pct')
            quantity = float(raw_qty) if raw_qty is not None else 0.0
            
            reasoning = data_json.get('reasoning') or 'No reasoning provided.'
            kb_update = data_json.get('knowledge_update')
            
            raw_add = data_json.get('add_symbols')
            add_syms = raw_add if isinstance(raw_add, list) else []
            
            raw_remove = data_json.get('remove_symbols')
            remove_syms = raw_remove if isinstance(raw_remove, list) else []

            thinking_cycles = int(data_json.get('thinking_cycles_requested', 0))
            thinking_cycles = max(0, min(5, thinking_cycles))

            return decision, reasoning, quantity, add_syms, remove_syms, kb_update, thinking_cycles
            
        except Exception as e:
            print(f"ERROR: AI Brain failed to regulate: {e}")
            traceback.print_exc()
            return "HOLD", str(e), 0.0, [], [], None, 0

    def get_sentiment_score(self, symbol, headlines):
        """Asks AI to score the sentiment of a list of headlines."""
        prompt = f"""
    ROLE: Financial Sentiment Analyst.
    TASK: Analyze the following headlines for {symbol} and provide a sentiment score.
    
    HEADLINES:
    {headlines}
    
    RULES:
    1. Score must be between -1.0 (Extreme Bearish/Fear) and +1.0 (Extreme Bullish/Greed).
    2. 0.0 is Neutral.
    3. Return ONLY the number. No text.
    
    SCORE:
    """
        try:
            payload = {
                "model": self.model_name, "prompt": prompt, "stream": False,
                "options": { "temperature": 0.0, "num_predict": 10 }
            }
            r = requests.post(self.generate_url, json=payload, timeout=30)
            r.raise_for_status()
            
            score_text = r.json().get('response', '0.0').strip()
            # Extract only the first number found
            import re
            match = re.search(r"(-?\d+\.?\d*)", score_text)
            if match:
                return float(match.group(1))
            return 0.0
        except Exception as e:
            print(f"Error scoring sentiment: {e}")
            return 0.0
