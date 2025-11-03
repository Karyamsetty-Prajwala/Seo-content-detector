# streamlit_app/utils/scorer.py
import pickle
from pathlib import Path
import numpy as np

class QualityScorer:
    """
    Loads a scikit-learn model if present; otherwise uses rule-based labels.
    Model should accept features in order: ['wordcount','sentence_count','flesch_reading_ease','avg_sentence_length','lexical_diversity']
    """
    def __init__(self, model_path=None):
        self.model = None
        if model_path is not None and Path(model_path).exists():
            try:
                with open(model_path, "rb") as f:
                    self.model = pickle.load(f)
            except Exception:
                self.model = None

    def _rule_label(self, features):
        wc = features.get("wordcount", 0)
        readability = features.get("flesch_reading_ease", 0)
        if (wc > 1500) and (50 <= readability <= 70):
            return "High", None
        if (wc < 500) or (readability < 30):
            return "Low", None
        return "Medium", None

    def predict_label(self, features):
        """Return (label_str, probs_dict_or_None)"""
        if self.model is None:
            label, _ = self._rule_label(features)
            return label, None
        # construct feature vector in expected order
        feat_order = ['wordcount','sentence_count','flesch_reading_ease','avg_sentence_length','lexical_diversity']
        try:
            x = np.array([[features.get(k, 0) for k in feat_order]])
            pred = self.model.predict(x)[0]
            probs = None
            if hasattr(self.model, "predict_proba"):
                p = self.model.predict_proba(x)[0]
                probs = {str(c): float(p_i) for c,p_i in zip(self.model.classes_, p)}
            return str(pred), probs
        except Exception:
            # fallback to rule
            return self._rule_label(features)
