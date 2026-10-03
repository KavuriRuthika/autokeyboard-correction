"""
N-Gram Language Model Implementation
Supports Unigram, Bigram, Trigram, and 4-Gram modeling with
Jelinek-Mercer Linear Interpolation and Backoff smoothing.
"""

import os
import math
import pickle
from collections import defaultdict, Counter
from typing import List, Tuple, Dict, Any, Optional
from src.preprocessing import tokenize_sentence

class NGramLanguageModel:
    """
    Interpolated N-Gram Language Model with n up to 4.
    """
    def __init__(self, n: int = 4, lambdas: Optional[Tuple[float, float, float, float]] = None):
        self.n = n
        # Linear interpolation weights: (unigram, bigram, trigram, 4-gram)
        # Default: heavier weight on higher-order contexts when available
        self.lambdas = lambdas or (0.05, 0.20, 0.35, 0.40)
        
        self.total_tokens: int = 0
        self.vocab: set = set()
        
        # N-gram count dictionaries
        self.unigram_counts: Counter = Counter()
        self.bigram_counts: Counter = Counter()
        self.trigram_counts: Counter = Counter()
        self.fourgram_counts: Counter = Counter()
        
        # Fast prefix lookup tables: context -> Counter(next_word -> count)
        self.bigram_contexts: Dict[str, Counter] = defaultdict(Counter)
        self.trigram_contexts: Dict[Tuple[str, str], Counter] = defaultdict(Counter)
        self.fourgram_contexts: Dict[Tuple[str, str, str], Counter] = defaultdict(Counter)

    def train(self, sentences: List[str]):
        """Train n-gram frequencies on a list of text sentences."""
        print(f"Training {self.n}-Gram Language Model on {len(sentences)} sentences...")
        self.unigram_counts.clear()
        self.bigram_counts.clear()
        self.trigram_counts.clear()
        self.fourgram_counts.clear()
        self.bigram_contexts.clear()
        self.trigram_contexts.clear()
        self.fourgram_contexts.clear()
        self.vocab.clear()
        self.total_tokens = 0
        
        for sentence in sentences:
            tokens = tokenize_sentence(sentence)
            if not tokens:
                continue
                
            # Add boundary padding
            padded_tokens = ["<s>"] * (self.n - 1) + tokens + ["</s>"]
            
            for i in range(self.n - 1, len(padded_tokens)):
                w = padded_tokens[i]
                if w != "<s>":
                    self.unigram_counts[w] += 1
                    self.vocab.add(w)
                    self.total_tokens += 1
                    
                # Bigram
                w_prev1 = padded_tokens[i - 1]
                self.bigram_counts[(w_prev1, w)] += 1
                self.bigram_contexts[w_prev1][w] += 1
                
                # Trigram
                if self.n >= 3:
                    w_prev2 = padded_tokens[i - 2]
                    self.trigram_counts[(w_prev2, w_prev1, w)] += 1
                    self.trigram_contexts[(w_prev2, w_prev1)][w] += 1
                    
                # 4-gram
                if self.n >= 4:
                    w_prev3 = padded_tokens[i - 3]
                    self.fourgram_counts[(w_prev3, w_prev2, w_prev1, w)] += 1
                    self.fourgram_contexts[(w_prev3, w_prev2, w_prev1)][w] += 1
                    
        print(f"N-Gram trained! Vocab: {len(self.vocab)}, Tokens: {self.total_tokens}, Bigrams: {len(self.bigram_counts)}, Trigrams: {len(self.trigram_counts)}")

    def get_prob(self, word: str, context: List[str]) -> float:
        """
        Calculate interpolated probability P(word | context).
        Uses Jelinek-Mercer linear interpolation:
        P = lambda1*P_uni + lambda2*P_bi + lambda3*P_tri + lambda4*P_4gram
        """
        # 1. Unigram probability
        p_uni = (self.unigram_counts.get(word, 0) + 1.0) / (self.total_tokens + len(self.vocab) + 1.0)
        
        # 2. Bigram probability
        p_bi = p_uni
        if context:
            w_prev1 = context[-1]
            c_ctx1 = self.unigram_counts.get(w_prev1, 0)
            if c_ctx1 > 0:
                p_bi = self.bigram_counts.get((w_prev1, word), 0) / c_ctx1
            else:
                p_bi = p_uni

        # 3. Trigram probability
        p_tri = p_bi
        if len(context) >= 2:
            ctx2 = (context[-2], context[-1])
            c_ctx2 = self.bigram_counts.get(ctx2, 0)
            if c_ctx2 > 0:
                p_tri = self.trigram_counts.get((ctx2[0], ctx2[1], word), 0) / c_ctx2
            else:
                p_tri = p_bi
                
        # 4. 4-gram probability
        p_four = p_tri
        if len(context) >= 3:
            ctx3 = (context[-3], context[-2], context[-1])
            c_ctx3 = self.trigram_counts.get(ctx3, 0)
            if c_ctx3 > 0:
                p_four = self.fourgram_counts.get((ctx3[0], ctx3[1], ctx3[2], word), 0) / c_ctx3
            else:
                p_four = p_tri

        l1, l2, l3, l4 = self.lambdas
        return (l1 * p_uni) + (l2 * p_bi) + (l3 * p_tri) + (l4 * p_four)

    def predict_next_words(self, context_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Predict top-k next words given a preceding context string.
        """
        tokens = tokenize_sentence(context_text)
        if not tokens:
            # Fallback to top unigrams
            top_uni = [w for w, _ in self.unigram_counts.most_common(top_k) if w not in ("<s>", "</s>")]
            return [{"word": w, "probability": round(self.unigram_counts[w] / max(1, self.total_tokens), 4), "confidence": 100 // top_k} for w in top_uni]

        # Gather candidate pool from matching higher-order contexts
        candidates = set()
        
        # 4-gram context match
        if len(tokens) >= 3:
            ctx3 = (tokens[-3], tokens[-2], tokens[-1])
            if ctx3 in self.fourgram_contexts:
                candidates.update(self.fourgram_contexts[ctx3].keys())
                
        # Trigram context match
        if len(tokens) >= 2:
            ctx2 = (tokens[-2], tokens[-1])
            if ctx2 in self.trigram_contexts:
                candidates.update(self.trigram_contexts[ctx2].keys())
                
        # Bigram context match
        ctx1 = tokens[-1]
        if ctx1 in self.bigram_contexts:
            candidates.update(self.bigram_contexts[ctx1].keys())
            
        # If candidate pool is too small, back off to top frequent words
        if len(candidates) < top_k:
            for w, _ in self.unigram_counts.most_common(50):
                candidates.add(w)
                if len(candidates) >= 50:
                    break

        # Remove sentence boundary tokens
        candidates.discard("<s>")
        candidates.discard("</s>")
        
        # Calculate probability for each candidate
        scored = []
        for word in candidates:
            prob = self.get_prob(word, tokens)
            scored.append({"word": word, "probability": prob})
            
        # Sort descending
        scored.sort(key=lambda x: x["probability"], reverse=True)
        top_candidates = scored[:top_k]
        
        # Normalize into percentages
        total_p = sum(c["probability"] for c in top_candidates) or 1.0
        for c in top_candidates:
            c["confidence"] = round((c["probability"] / total_p) * 100, 1)
            c["probability"] = round(c["probability"], 4)
            
        return top_candidates

    def calculate_perplexity(self, test_sentences: List[str]) -> float:
        """
        Calculate Perplexity on a held-out test set:
        PPL = exp(-1/N * sum(log(P(w_i | context))))
        """
        log_prob_sum = 0.0
        n_tokens = 0
        
        for sentence in test_sentences:
            tokens = tokenize_sentence(sentence)
            if not tokens:
                continue
            for i in range(len(tokens)):
                word = tokens[i]
                context = tokens[max(0, i - (self.n - 1)):i]
                p = self.get_prob(word, context)
                log_prob_sum += math.log(max(p, 1e-12))
                n_tokens += 1
                
        if n_tokens == 0:
            return float('inf')
            
        avg_neg_log_prob = - (log_prob_sum / n_tokens)
        perplexity = math.exp(avg_neg_log_prob)
        return round(perplexity, 2)

    def save(self, filepath: str):
        """Serialize model to file."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        print(f"N-Gram Model saved to {filepath}")

    @classmethod
    def load(cls, filepath: str) -> "NGramLanguageModel":
        """Deserialize model from file."""
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        return model
