from ddgs import DDGS
import requests
import json
import time

class ResearchAgent:
    def __init__(self, ai_strategy):
        self.ai = ai_strategy
        self.ddgs = DDGS()

    def conduct_study_session(self):
        """Perform a research cycle to find and ingest high-standard trading expertise."""
        queries = [
            "advanced crypto trading risk management strategies pdf",
            "professional swing trading rules price action",
            "institutional crypto trading execution tactics",
            "market psychology and trading discipline expertise"
        ]
        
        print("Starting Autonomous Research Study Session...")
        all_insights = []
        
        for query in queries:
            try:
                print(f"Searching for: {query}")
                results = self.ddgs.text(query, max_results=3)
                
                for res in results:
                    title = res.get('title')
                    body = res.get('body')
                    href = res.get('href')
                    
                    if not body: continue
                    
                    print(f"Analyzing expert source: {title}")
                    insight = self.ai.summarize_expertise(title, body, href)
                    if insight and insight.lower() != "null" and "optional" not in insight.lower():
                        all_insights.append(insight)
                        # To avoid spamming the KB, we ingest only the best 1 per query
                        break 
                
                time.sleep(2) # Rate limiting
            except Exception as e:
                print(f"Search error for '{query}': {e}")

        if all_insights:
            print(f"Research complete. Ingesting {len(all_insights)} new professional rules.")
            for insight in all_insights:
                self.ai.update_knowledge_base(insight)
            return True
        
        print("Study session complete. No new high-standard insights found.")
        return False
