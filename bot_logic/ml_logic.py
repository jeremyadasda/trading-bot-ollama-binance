import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pickle
import os

class MLPredictor:
    def __init__(self, model_path="bot_logic/trading_model.pkl"):
        self.model_path = model_path
        self.model = None
        self._load_model()

    def _load_model(self):
        if os.path.exists(self.model_path):
            try:
                with open(self.model_path, 'rb') as f:
                    self.model = pickle.load(f)
                print("ML Model loaded successfully.")
            except Exception as e:
                print(f"Error loading ML model: {e}")
                self.model = None

    def _save_model(self):
        try:
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            with open(self.model_path, 'wb') as f:
                pickle.dump(self.model, f)
            print("ML Model saved successfully.")
        except Exception as e:
            print(f"Error saving ML model: {e}")

    def train_on_data(self, df):
        """Trains a model on the provided feature DataFrame."""
        if df is None or len(df) < 50:
            print("Insufficient data for training ML model.")
            return False

        # Define features (avoiding targets and raw prices)
        features = [col for col in df.columns if col not in ['ts', 'open', 'high', 'low', 'close', 'vol', 'close_ts', 'qav', 'num_trades', 'taker_base', 'taker_quote', 'ignore', 'target', 'target_return']]
        
        X = df[features]
        y = df['target']

        print(f"Training ML model with {len(X)} samples and {len(features)} features...")
        self.model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
        self.model.fit(X, y)
        self._save_model()
        return True

    def predict_confidence(self, latest_df):
        """Predicts the probability of the target (Price up > 1% in 4h)."""
        if self.model is None:
            return 0.5 # Neutral
        
        try:
            features = [col for col in latest_df.columns if col not in ['ts', 'open', 'high', 'low', 'close', 'vol', 'close_ts', 'qav', 'num_trades', 'taker_base', 'taker_quote', 'ignore', 'target', 'target_return']]
            last_row = latest_df[features].tail(1)
            
            # Probability of class 1 (Bullish)
            probs = self.model.predict_proba(last_row)[0]
            bull_prob = probs[1]
            return bull_prob
        except Exception as e:
            print(f"ML Prediction Error: {e}")
            return 0.5
