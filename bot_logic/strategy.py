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

    def ask_ai_opinion(self, current_tracked_symbols, market_summary, full_wallet_info, order_book, live_data, trade_summary):
        # ... (rest of the function setup)
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
    Live Price: {live_data.get('price', 'N/A')}
    Order Book: {order_book}

    --- REASONING PROCESS (Chain-of-Thought) ---
    1.  **Assess Portfolio & Market:** Analyze market conditions and your portfolio.
    2.  **Assess Current Symbol:** Select a strategy.
    3.  **Self-Correction:** If you see a pattern that contradicts the Knowledge Base or requires a new rule, formulate a "Knowledge Update".
    4.  **Final Decision:** State your decision.

    --- RESPONSE FORMAT (JSON ONLY) ---
    {{
      "reasoning": "Your analysis...",
      "decision": "BUY",
      "quantity_pct": 0.0,
      "knowledge_update": "A single, specific new insight or strategy you've learned. If NO new insight, use null.",
      "add_symbols": ["SOLUSDT"],
      "remove_symbols": [] 
    }}
    
    IMPORTANT: Do NOT include placeholders like 'Optional' or 'Optional material' in the knowledge_update. If you have nothing new to add, the field MUST be null.
    """
        
        print(f"Running multi-dimensional analysis with {self.model_name}...")
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
            
            # Robust quantity parsing (handles null or missing)
            raw_qty = data_json.get('quantity_pct')
            quantity = float(raw_qty) if raw_qty is not None else 0.0
            
            reasoning = data_json.get('reasoning') or 'No reasoning provided.'
            kb_update = data_json.get('knowledge_update')
            
            # Extract symbol recommendations (handles null or missing)
            raw_add = data_json.get('add_symbols')
            add_syms = raw_add if isinstance(raw_add, list) else []
            
            raw_remove = data_json.get('remove_symbols')
            remove_syms = raw_remove if isinstance(raw_remove, list) else []

            return decision, reasoning, quantity, add_syms, remove_syms, kb_update
            
        except Exception as e:
            print(f"ERROR: AI Brain failed to regulate: {e}")
            traceback.print_exc()
            return "HOLD", str(e), 0.0, [], [], None
