"""
Smart Shortcut & Abbreviation Expansion Engine
Expands shorthand tokens into full phrases and provides domain-specific related word suggestions.
Examples:
  fn    -> function (related: call, return, parameter, definition, arguments)
  algo  -> algorithm (related: efficiency, complexity, optimization, model)
  ml    -> machine learning (related: model, dataset, training, algorithms)
  plz   -> please (related: let, help, send, check, confirm)
"""

from typing import Dict, List, Optional, Any

DEFAULT_SHORTCUTS: Dict[str, Dict[str, Any]] = {
    # Programming & Tech
    "fn": {
        "expansion": "function",
        "category": "programming",
        "related": ["call", "return", "parameter", "definition", "arguments", "execution"]
    },
    "func": {
        "expansion": "function",
        "category": "programming",
        "related": ["definition", "call", "return", "parameter", "scope"]
    },
    "algo": {
        "expansion": "algorithm",
        "category": "computer science",
        "related": ["complexity", "efficiency", "optimization", "implementation", "design"]
    },
    "ml": {
        "expansion": "machine learning",
        "category": "data science",
        "related": ["models", "algorithms", "datasets", "training", "evaluation"]
    },
    "ai": {
        "expansion": "artificial intelligence",
        "category": "data science",
        "related": ["systems", "models", "applications", "research", "capabilities"]
    },
    "nlp": {
        "expansion": "natural language processing",
        "category": "data science",
        "related": ["tokenization", "sentiment", "models", "corpus", "classification"]
    },
    "rnn": {
        "expansion": "recurrent neural network",
        "category": "deep learning",
        "related": ["sequence", "hidden", "layers", "gradients", "lstm"]
    },
    "lstm": {
        "expansion": "long short term memory",
        "category": "deep learning",
        "related": ["network", "cell", "gate", "vanishing", "gradient"]
    },
    "nn": {
        "expansion": "neural network",
        "category": "deep learning",
        "related": ["layers", "weights", "biases", "training", "activation"]
    },
    "cnn": {
        "expansion": "convolutional neural network",
        "category": "deep learning",
        "related": ["filters", "pooling", "vision", "images", "layers"]
    },
    "acc": {
        "expansion": "accuracy",
        "category": "evaluation",
        "related": ["precision", "recall", "score", "metric", "evaluation"]
    },
    "approx": {
        "expansion": "approximately",
        "category": "math",
        "related": ["equal", "value", "estimate", "calculation", "measure"]
    },
    "def": {
        "expansion": "definition",
        "category": "general",
        "related": ["concept", "meaning", "explanation", "term", "statement"]
    },
    "param": {
        "expansion": "parameter",
        "category": "programming",
        "related": ["values", "arguments", "tuning", "optimization", "configuration"]
    },
    "params": {
        "expansion": "parameters",
        "category": "programming",
        "related": ["tuning", "weights", "inputs", "hyperparameters", "values"]
    },
    "config": {
        "expansion": "configuration",
        "category": "devops",
        "related": ["settings", "options", "environment", "setup", "parameters"]
    },
    "temp": {
        "expansion": "temperature",
        "category": "ml/weather",
        "related": ["softmax", "sampling", "setting", "parameter", "weather"]
    },
    "eval": {
        "expansion": "evaluation",
        "category": "general",
        "related": ["metrics", "results", "benchmarks", "performance", "score"]
    },
    "opt": {
        "expansion": "optimization",
        "category": "math/ml",
        "related": ["algorithm", "gradient", "loss", "convergence", "weights"]
    },
    "calc": {
        "expansion": "calculation",
        "category": "math",
        "related": ["results", "formula", "computation", "matrix", "analysis"]
    },
    "info": {
        "expansion": "information",
        "category": "general",
        "related": ["details", "data", "summary", "knowledge", "updates"]
    },
    "doc": {
        "expansion": "document",
        "category": "business",
        "related": ["file", "attached", "content", "summary", "report"]
    },
    "docs": {
        "expansion": "documentation",
        "category": "programming",
        "related": ["code", "manual", "guide", "reference", "api"]
    },
    "spec": {
        "expansion": "specification",
        "category": "engineering",
        "related": ["requirements", "details", "design", "standards", "architecture"]
    },
    "db": {
        "expansion": "database",
        "category": "software",
        "related": ["query", "records", "storage", "connection", "tables"]
    },
    "ui": {
        "expansion": "user interface",
        "category": "design",
        "related": ["experience", "design", "components", "responsive", "layout"]
    },
    "api": {
        "expansion": "application programming interface",
        "category": "software",
        "related": ["endpoints", "request", "response", "rest", "service"]
    },
    "dev": {
        "expansion": "development",
        "category": "software",
        "related": ["process", "team", "environment", "tools", "pipeline"]
    },

    # Conversational & Messaging
    "plz": {
        "expansion": "please",
        "category": "courtesy",
        "related": ["let", "help", "send", "check", "confirm", "find"]
    },
    "pls": {
        "expansion": "please",
        "category": "courtesy",
        "related": ["let", "help", "send", "review", "reply", "note"]
    },
    "thx": {
        "expansion": "thank you",
        "category": "courtesy",
        "related": ["so", "very", "much", "for", "your", "help"]
    },
    "ty": {
        "expansion": "thank you",
        "category": "courtesy",
        "related": ["very", "much", "for", "your", "support", "time"]
    },
    "msg": {
        "expansion": "message",
        "category": "messaging",
        "related": ["sent", "received", "content", "notification", "details"]
    },
    "idk": {
        "expansion": "i do not know",
        "category": "conversation",
        "related": ["what", "how", "if", "why", "the", "exact"]
    },
    "imo": {
        "expansion": "in my opinion",
        "category": "conversation",
        "related": ["this", "that", "it", "we", "the", "approach"]
    },
    "imho": {
        "expansion": "in my humble opinion",
        "category": "conversation",
        "related": ["we", "this", "that", "the", "best"]
    },
    "asap": {
        "expansion": "as soon as possible",
        "category": "business",
        "related": ["please", "let", "we", "will", "send"]
    },
    "btw": {
        "expansion": "by the way",
        "category": "conversation",
        "related": ["did", "have", "are", "what", "we", "you"]
    },
    "omw": {
        "expansion": "on my way",
        "category": "messaging",
        "related": ["to", "now", "home", "there", "will"]
    },
    "brb": {
        "expansion": "be right back",
        "category": "messaging",
        "related": ["in", "a", "few", "minutes", "soon"]
    },
    "faq": {
        "expansion": "frequently asked questions",
        "category": "general",
        "related": ["section", "answers", "help", "guide", "details"]
    },
    "wpm": {
        "expansion": "words per minute",
        "category": "typing",
        "related": ["typing", "speed", "test", "metric", "rate"]
    }
}

class ShortcutEngine:
    """
    Handles shorthand shortcut detection, full text expansion,
    and contextual related word suggestions.
    """
    def __init__(self, custom_shortcuts: Optional[Dict[str, Dict[str, Any]]] = None):
        self.shortcuts: Dict[str, Dict[str, Any]] = dict(DEFAULT_SHORTCUTS)
        if custom_shortcuts:
            self.shortcuts.update(custom_shortcuts)

    def is_shortcut(self, token: str) -> bool:
        """Check if token is a registered shortcut."""
        return token.lower().strip() in self.shortcuts

    def get_expansion(self, token: str) -> Optional[str]:
        """Get full expanded text for a token."""
        entry = self.shortcuts.get(token.lower().strip())
        return entry["expansion"] if entry else None

    def get_related_words(self, token: str) -> List[str]:
        """Get domain-specific related words for a shortcut token."""
        entry = self.shortcuts.get(token.lower().strip())
        return entry.get("related", []) if entry else []

    def get_shortcut_suggestions(self, token: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Generate candidate objects for the shortcut.
        First candidate: the full expansion (e.g. 'function').
        Subsequent candidates: related contextual words (e.g. 'call', 'return', 'parameter').
        """
        clean_token = token.lower().strip()
        if clean_token not in self.shortcuts:
            return []

        entry = self.shortcuts[clean_token]
        expansion = entry["expansion"]
        related = entry.get("related", [])

        results = []
        # Primary candidate: Full Expansion
        results.append({
            "word": expansion,
            "confidence": 92.0,
            "type": "shortcut",
            "shortcut_from": clean_token,
            "category": entry.get("category", "general")
        })

        # Related contextual words
        remaining = top_k - 1
        for i, rel_word in enumerate(related[:remaining]):
            results.append({
                "word": rel_word,
                "confidence": round(max(15.0, 75.0 - (i * 12.0)), 1),
                "type": "related",
                "shortcut_from": clean_token,
                "category": entry.get("category", "general")
            })

        # Normalize confidences across returned suggestions
        tot = sum(r["confidence"] for r in results) or 1.0
        for r in results:
            r["confidence"] = round((r["confidence"] / tot) * 100, 1)

        return results

    def expand_sentence(self, sentence: str) -> Dict[str, Any]:
        """
        Scan a sentence and replace any shortcuts with their full expansion.
        """
        import re
        words = sentence.split()
        if not words:
            return {"original": sentence, "expanded": sentence, "has_expansions": False, "expansions": []}

        expanded_words = []
        expansions = []

        for w in words:
            clean_w = re.sub(r"[^a-zA-Z0-9]", "", w).lower()
            if clean_w in self.shortcuts:
                entry = self.shortcuts[clean_w]
                full_text = entry["expansion"]

                # Preserve Title Case if original was capitalized
                if w.istitle():
                    full_text = full_text.capitalize()
                elif w.isupper():
                    full_text = full_text.upper()

                # Re-attach trailing punctuation
                trailing_punct = "".join([c for c in w if not c.isalnum()])
                replacement = full_text + trailing_punct

                expanded_words.append(replacement)
                expansions.append({
                    "from": w,
                    "to": replacement,
                    "related": entry.get("related", [])
                })
            else:
                expanded_words.append(w)

        expanded_sentence = " ".join(expanded_words)
        return {
            "original": sentence,
            "expanded": expanded_sentence,
            "has_expansions": len(expansions) > 0,
            "expansions": expansions
        }

    def get_all_shortcuts(self) -> Dict[str, Dict[str, Any]]:
        """Return registry of all active shortcuts."""
        return self.shortcuts
