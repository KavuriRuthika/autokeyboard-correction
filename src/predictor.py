"""
Unified Smart Keyboard Predictor Engine
Coordinates Autocorrect, N-Gram LM, and LSTM LM.
Determines in-word completion vs next-word anticipation based on typing state.
"""

import os
import time
from typing import List, Dict, Any, Optional
from src.preprocessing import TextTokenizer, tokenize_sentence
from src.autocorrect import AutocorrectEngine
from src.ngram_model import NGramLanguageModel
from src.lstm_model import LSTMNextWordModel

class SmartKeyboardPredictor:
    """
    Central orchestration engine for the AI Smart Keyboard.
    """
    def __init__(self, models_dir: str = "models", data_dir: str = "data/processed"):
        self.models_dir = models_dir
        self.data_dir = data_dir
        
        self.tokenizer: Optional[TextTokenizer] = None
        self.autocorrect: Optional[AutocorrectEngine] = None
        self.ngram_model: Optional[NGramLanguageModel] = None
        self.lstm_model: Optional[LSTMNextWordModel] = None
        
        self.load_components()

    def load_components(self):
        """Load all trained models and dictionary resources."""
        # 1. Autocorrect Dictionary
        freq_path = os.path.join(self.data_dir, "word_frequencies.json")
        if os.path.exists(freq_path):
            self.autocorrect = AutocorrectEngine(freq_path)
            
        # 2. Tokenizer
        tok_path = os.path.join(self.models_dir, "tokenizer.json")
        if os.path.exists(tok_path):
            self.tokenizer = TextTokenizer.load(tok_path)
            
        # 3. N-Gram Model
        ngram_path = os.path.join(self.models_dir, "ngram_model.pkl")
        if os.path.exists(ngram_path):
            self.ngram_model = NGramLanguageModel.load(ngram_path)
            
        # 4. LSTM Model
        lstm_path = os.path.join(self.models_dir, "lstm_model.keras")
        if os.path.exists(lstm_path) and self.tokenizer is not None:
            self.lstm_model = LSTMNextWordModel.load(lstm_path, self.tokenizer)

    def predict(
        self,
        full_text: str,
        engine: str = "lstm",
        top_k: int = 5,
        temperature: float = 1.0,
        enable_autocorrect: bool = True
    ) -> Dict[str, Any]:
        """
        Produce predictions based on current text input and state.
        
        Rules:
        - If input ends in whitespace (user finished a word): anticipate NEXT WORD.
        - If input ends with letters (in the middle of a word): AUTOCORRECT / COMPLETE current token.
        """
        start_time = time.perf_counter()
        
        if not full_text:
            # Default starter words for empty text
            starters = ["the", "i", "we", "please", "what", "how"]
            return {
                "mode": "next_word",
                "model_used": "baseline",
                "active_token": "",
                "context": "",
                "latency_ms": round((time.perf_counter() - start_time) * 1000, 2),
                "predictions": [{"word": w, "confidence": round(100 / len(starters), 1), "type": "starter"} for w in starters[:top_k]]
            }

        # Check if cursor is immediately after space or inside a word
        is_trailing_space = full_text.endswith(" ")
        
        # If user is in the middle of typing a word (e.g. "I am go")
        if not is_trailing_space and enable_autocorrect and self.autocorrect:
            # Extract last word being typed
            words = full_text.split()
            current_typed = words[-1] if words else ""
            context_text = " ".join(words[:-1])
            
            # Run autocorrect, shortcut expansion & prefix completion
            corrections = self.autocorrect.correct(current_typed, top_k=top_k)
            latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
            
            is_shortcut = bool(corrections and corrections[0].get("type") == "shortcut")
            mode = "shortcut" if is_shortcut else "autocorrect"
            model_used = "shortcut_expander" if is_shortcut else "q_bayesian_autocorrect"
            
            line_correction = self.autocorrect.correct_sentence(full_text.strip()) if self.autocorrect else None
            return {
                "mode": mode,
                "model_used": model_used,
                "active_token": current_typed,
                "context": context_text,
                "latency_ms": latency_ms,
                "predictions": corrections,
                "line_correction": line_correction
            }
            
        # User finished a word and pressed space -> NEXT WORD PREDICTION
        context_text = full_text.strip()
        tokens = tokenize_sentence(context_text)
        
        # Normalize any misspelled tokens in the context window
        if self.autocorrect and tokens:
            normalized_tokens = []
            for t in tokens:
                if t in self.autocorrect.vocab_set:
                    normalized_tokens.append(t)
                else:
                    corrs = self.autocorrect.correct(t, top_k=1)
                    if corrs and corrs[0]["type"] == "correction" and corrs[0]["confidence"] > 45:
                        normalized_tokens.append(corrs[0]["word"])
                    else:
                        normalized_tokens.append(t)
            context_text = " ".join(normalized_tokens)

        # High-frequency natural conversational collocation shortcuts
        COMMON_COLOCATIONS = {
            "good": ["morning", "afternoon", "evening", "luck", "job"],
            "have a": ["great", "nice", "good", "wonderful", "safe"],
            "see you": ["soon", "tomorrow", "later", "there", "again"],
            "it is a": ["great", "pleasure", "good", "matter", "very"],
            "how are you": ["doing", "today", "feeling", "with"],
            "what is your": ["name", "email", "opinion", "favorite", "plan"],
            "thank you": ["so", "very", "for", "much"],
            "can you": ["please", "help", "explain", "send", "confirm"],
            "i want to": ["learn", "know", "see", "make", "thank"],
            "let me": ["know", "check", "see", "take", "help"],
            "for your": ["time", "help", "support", "patience", "feedback"],
            "thank you for your": ["time", "help", "support", "patience", "feedback"],
            "thank you very much for your": ["time", "help", "support", "patience", "feedback"],
            "thanks for your": ["help", "time", "support", "feedback", "response"],
            "looking forward to": ["hearing", "seeing", "working", "meeting", "your"],
            "please let me know": ["if", "what", "how", "when", "your"],
            "please find": ["attached", "enclosed", "below", "the", "my"],
            "i would like to": ["thank", "know", "see", "ask", "share"],
            "the weather is": ["very", "nice", "quite", "warm", "cold"]
        }

        # Check for direct conversational match first
        lower_ctx = context_text.lower()
        direct_match = None
        for key in sorted(COMMON_COLOCATIONS.keys(), key=lambda k: len(k.split()), reverse=True):
            if lower_ctx.endswith(key):
                direct_match = COMMON_COLOCATIONS[key]
                break

        # Check if the preceding word was an expanded shortcut term with domain-specific related words
        if not direct_match and tokens and self.autocorrect and hasattr(self.autocorrect, 'shortcut_engine'):
            last_word = tokens[-1].lower()
            for sc_key, sc_info in self.autocorrect.shortcut_engine.shortcuts.items():
                if sc_info["expansion"].lower() == last_word or sc_info["expansion"].lower().endswith(" " + last_word):
                    direct_match = sc_info.get("related", [])
                    break

        predictions = []
        model_used = engine.lower()
        
        if model_used == "ngram" and self.ngram_model:
            predictions = self.ngram_model.predict_next_words(context_text, top_k=top_k)
        elif model_used == "lstm" and self.lstm_model:
            predictions = self.lstm_model.predict_next_words(context_text, top_k=top_k, temperature=temperature)
        elif model_used == "ensemble" and self.ngram_model and self.lstm_model:
            # Blend predictions from both models
            ng_preds = {p["word"]: p["probability"] for p in self.ngram_model.predict_next_words(context_text, top_k=top_k*2)}
            ls_preds = {p["word"]: p["probability"] for p in self.lstm_model.predict_next_words(context_text, top_k=top_k*2, temperature=temperature)}
            
            all_words = set(ng_preds.keys()).union(set(ls_preds.keys()))
            blended = []
            for w in all_words:
                p_ng = ng_preds.get(w, 0.0)
                p_ls = ls_preds.get(w, 0.0)
                p_comb = 0.45 * p_ng + 0.55 * p_ls
                blended.append({"word": w, "probability": p_comb})
                
            blended.sort(key=lambda x: x["probability"], reverse=True)
            top_blended = blended[:top_k]
            tot_p = sum(x["probability"] for x in top_blended) or 1.0
            for x in top_blended:
                x["confidence"] = round((x["probability"] / tot_p) * 100, 1)
                x["probability"] = round(x["probability"], 4)
            predictions = top_blended
        else:
            if self.lstm_model:
                predictions = self.lstm_model.predict_next_words(context_text, top_k=top_k, temperature=temperature)
                model_used = "lstm"
            elif self.ngram_model:
                predictions = self.ngram_model.predict_next_words(context_text, top_k=top_k)
                model_used = "ngram"

        # If a direct high-frequency conversational match exists, blend the top words to the front
        if direct_match:
            existing_words = {p["word"] for p in predictions}
            boosted = []
            for w in direct_match[:top_k]:
                boosted.append({"word": w, "confidence": 85.0 if len(boosted)==0 else 60.0, "probability": 0.85, "type": "collocation"})
            for p in predictions:
                if p["word"] not in [b["word"] for b in boosted]:
                    boosted.append(p)
            predictions = boosted[:top_k]
            tot_p = sum(p.get("confidence", 10) for p in predictions) or 1.0
            for p in predictions:
                p["confidence"] = round((p.get("confidence", 10) / tot_p) * 100, 1)

        latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
        
        # Whole-line / sentence level autocorrect audit
        line_correction = None
        if self.autocorrect and full_text.strip():
            line_correction = self.autocorrect.correct_sentence(full_text.strip())
            
        return {
            "mode": "next_word",
            "model_used": model_used,
            "active_token": "",
            "context": context_text,
            "latency_ms": latency_ms,
            "predictions": predictions,
            "line_correction": line_correction
        }
