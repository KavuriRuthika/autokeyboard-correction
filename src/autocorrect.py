"""
Intelligent Autocorrect and Prefix Completion Engine
Features:
- QWERTY physical keyboard layout proximity penalty matrix
- Bayesian noisy channel model: P(target|typed) proportional to P(typed|target) * P(target)
- RapidFuzz high-speed Damerau-Levenshtein distance matching
- Prefix auto-completion for in-progress typing
"""

import os
import json
import math
from typing import List, Dict, Any, Optional
from rapidfuzz import distance, process
from src.shortcuts import ShortcutEngine

# QWERTY physical neighbor adjacency table
QWERTY_NEIGHBORS = {
    'q': ['w', 'a', 's', '1', '2'],
    'w': ['q', 'e', 'a', 's', 'd', '2', '3'],
    'e': ['w', 'r', 's', 'd', 'f', '3', '4'],
    'r': ['e', 't', 'd', 'f', 'g', '4', '5'],
    't': ['r', 'y', 'f', 'g', 'h', '5', '6'],
    'y': ['t', 'u', 'g', 'h', 'j', '6', '7'],
    'u': ['y', 'i', 'h', 'j', 'k', '7', '8'],
    'i': ['u', 'o', 'j', 'k', 'l', '8', '9'],
    'o': ['i', 'p', 'k', 'l', '9', '0'],
    'p': ['o', 'l', '0'],
    'a': ['q', 'w', 's', 'z'],
    's': ['a', 'w', 'e', 'd', 'x', 'z'],
    'd': ['s', 'e', 'r', 'f', 'c', 'x'],
    'f': ['d', 'r', 't', 'g', 'v', 'c'],
    'g': ['f', 't', 'y', 'h', 'b', 'v'],
    'h': ['g', 'y', 'u', 'j', 'n', 'b'],
    'j': ['h', 'u', 'i', 'k', 'm', 'n'],
    'k': ['j', 'i', 'o', 'l', 'm'],
    'l': ['k', 'o', 'p'],
    'z': ['a', 's', 'x'],
    'x': ['z', 's', 'd', 'c'],
    'c': ['x', 'd', 'f', 'v'],
    'v': ['c', 'f', 'g', 'b'],
    'b': ['v', 'g', 'h', 'n'],
    'n': ['b', 'h', 'j', 'm'],
    'm': ['n', 'j', 'k']
}

class AutocorrectEngine:
    """
    State-of-the-art Autocorrect & Completion Engine combining
    keyboard ergonomics, Bayesian priors, and edit distance.
    """
    def __init__(self, frequencies_path: Optional[str] = None):
        self.word_freq: Dict[str, int] = {}
        self.total_tokens: int = 1
        self.vocabulary: List[str] = []
        self.vocab_set: set = set()
        self.shortcut_engine = ShortcutEngine()
        
        if frequencies_path and os.path.exists(frequencies_path):
            self.load_frequencies(frequencies_path)
            
    def load_frequencies(self, filepath: str):
        """Load word frequency dictionary from JSON."""
        with open(filepath, 'r', encoding='utf-8') as f:
            self.word_freq = json.load(f)
            
        self.total_tokens = sum(self.word_freq.values())
        # Filter for valid alphabetic words
        self.vocabulary = [w for w in self.word_freq.keys() if w.isalpha() and len(w) > 1]
        # Sort vocabulary by frequency descending
        self.vocabulary.sort(key=lambda w: self.word_freq[w], reverse=True)
        self.vocab_set = set(self.vocabulary)
        print(f"Autocorrect initialized with {len(self.vocabulary)} vocabulary words.")

    def get_prior_prob(self, word: str) -> float:
        """Calculate Bayesian prior P(w) with Laplace smoothing."""
        count = self.word_freq.get(word, 0)
        return (count + 1.0) / (self.total_tokens + len(self.word_freq) + 1.0)

    def keyboard_distance_penalty(self, typed: str, candidate: str) -> float:
        """
        Calculates distance between typed string and candidate word.
        Uses Damerau-Levenshtein distance, but discounts:
        1. Single-character substitutions on physically adjacent QWERTY keys (e.g. smarr -> smart)
        2. Single-character adjacent transpositions (e.g. teh -> the, thsi -> this, woudl -> would)
        3. Double-letter omissions (e.g. helo -> hello, mesage -> message)
        """
        raw_dist = float(distance.DamerauLevenshtein.distance(typed, candidate))
        
        if raw_dist == 1.0:
            # Check adjacent key substitution
            if len(typed) == len(candidate):
                diff_indices = [i for i, (c1, c2) in enumerate(zip(typed, candidate)) if c1 != c2]
                if len(diff_indices) == 1:
                    i = diff_indices[0]
                    if candidate[i] in QWERTY_NEIGHBORS.get(typed[i], []):
                        return 0.45
                elif len(diff_indices) == 2:
                    # Adjacent character transposition (e.g. 'teh' -> 'the')
                    i1, i2 = diff_indices
                    if abs(i1 - i2) == 1 and typed[i1] == candidate[i2] and typed[i2] == candidate[i1]:
                        return 0.40
            
            # Check double-letter omission (e.g. 'helo' -> 'hello')
            if len(candidate) == len(typed) + 1:
                for i in range(len(candidate) - 1):
                    if candidate[i] == candidate[i + 1]:
                        # If candidate with one of the duplicates removed equals typed
                        reduced = candidate[:i] + candidate[i+1:]
                        if reduced == typed:
                            return 0.40
                            
        return raw_dist

    def get_prefix_completions(self, prefix: str, max_results: int = 5) -> List[str]:
        """Find words that start with the prefix, ranked by frequency."""
        prefix = prefix.lower()
        if not prefix or len(prefix) < 2:
            return []
            
        completions = [w for w in self.vocabulary if w.startswith(prefix) and w != prefix]
        return completions[:max_results]

    def correct(self, typed_word: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Provide top autocorrect suggestions and completions for a typed token.
        Returns list of dicts with word, confidence, score, and match type.
        """
        typed_clean = typed_word.lower().strip()
        if not typed_clean or not typed_clean.isalpha():
            return []
            
        # Check if typed token is an abbreviation or shortcut (e.g. fn -> function)
        if self.shortcut_engine.is_shortcut(typed_clean):
            return self.shortcut_engine.get_shortcut_suggestions(typed_clean, top_k=top_k)
            
        is_exact_valid = typed_clean in self.vocab_set
        candidates = {}
        
        # 1. Exact match candidate (if valid)
        if is_exact_valid:
            candidates[typed_clean] = {
                "word": typed_clean,
                "distance": 0.0,
                "type": "exact",
                "prior": self.get_prior_prob(typed_clean)
            }
            
        # 2. Prefix completions (user is actively typing a longer word)
        prefix_matches = self.get_prefix_completions(typed_clean, max_results=top_k * 3)
        for pm in prefix_matches:
            # Small penalty based on length delta so shorter natural words rank comfortably
            length_delta = len(pm) - len(typed_clean)
            candidates[pm] = {
                "word": pm,
                "distance": 0.25 * min(length_delta, 4),
                "type": "completion",
                "prior": self.get_prior_prob(pm)
            }
                
        # 3. Fuzzy matches for typos using RapidFuzz
        fuzzy_matches = process.extract(
            typed_clean,
            self.vocabulary,  # Full vocabulary search (sub-15ms with RapidFuzz C++ backend)
            scorer=distance.DamerauLevenshtein.normalized_similarity,
            limit=25
        )
        
        for cand_word, sim_score, _ in fuzzy_matches:
            if cand_word not in candidates:
                kb_dist = self.keyboard_distance_penalty(typed_clean, cand_word)
                if kb_dist <= 2.0:  # Allow up to 2 edits or 1 adjacent typo
                    candidates[cand_word] = {
                        "word": cand_word,
                        "distance": kb_dist,
                        "type": "correction",
                        "prior": self.get_prior_prob(cand_word)
                    }

        # 4. Score all candidates using Bayesian Noisy Channel:
        # Score = exp(-alpha * distance) * (prior ^ beta)
        alpha = 2.2   # Penalty for edit distance
        beta = 0.35   # Weight of word popularity
        
        scored_results = []
        for word, meta in candidates.items():
            dist = meta["distance"]
            prior = meta["prior"]
            likelihood = math.exp(-alpha * dist)
            prior_factor = math.pow(prior, beta)
            
            # Bonus if exact match
            bonus = 1.3 if meta["type"] == "exact" else 1.0
            final_score = likelihood * prior_factor * bonus
            
            scored_results.append({
                "word": word,
                "score": final_score,
                "distance": round(dist, 2),
                "type": meta["type"],
                "frequency": self.word_freq.get(word, 0)
            })
            
        # Sort by final score descending
        scored_results.sort(key=lambda x: x["score"], reverse=True)
        top_results = scored_results[:top_k]
        
        # Normalize scores to pseudo-probabilities
        total_score = sum(r["score"] for r in top_results) or 1.0
        for r in top_results:
            r["confidence"] = round((r["score"] / total_score) * 100, 1)
            
        return top_results

    def correct_sentence(self, sentence: str) -> Dict[str, Any]:
        """
        Whole-line / full-sentence autocorrect:
        Scans all words in the input line, corrects every typo,
        preserves punctuation, capitalization, and formatting.
        """
        import re
        if not sentence.strip():
            return {"original": sentence, "corrected": sentence, "has_changes": False, "changes": []}

        # Step 1: Expand shortcuts and abbreviations across the sentence (e.g. fn -> function, ml -> machine learning)
        shortcut_res = self.shortcut_engine.expand_sentence(sentence)
        working_sentence = shortcut_res["expanded"]
        changes = [
            {"from": exp["from"], "to": exp["to"], "confidence": 100.0, "type": "shortcut", "related": exp.get("related", [])}
            for exp in shortcut_res["expansions"]
        ]
            
        # Step 2: Correct typos on the words in the working sentence
        words = working_sentence.split()
        corrected_words = []
        
        for w in words:
            clean_w = re.sub(r"[^a-zA-Z]", "", w).lower()
            if not clean_w:
                corrected_words.append(w)
                continue
                
            # If word is already a valid known vocabulary word, keep it
            if clean_w in self.vocab_set:
                corrected_words.append(w)
                continue
                
            # Otherwise, find the best autocorrect fix
            fixes = self.correct(clean_w, top_k=1)
            if fixes and fixes[0]["type"] in ("correction", "shortcut"):
                best_fix = fixes[0]["word"]
                if w.istitle():
                    best_fix = best_fix.capitalize()
                elif w.isupper():
                    best_fix = best_fix.upper()
                    
                prefix_punct = ""
                suffix_punct = ""
                for char in w:
                    if not char.isalnum():
                        prefix_punct += char
                    else:
                        break
                for char in reversed(w):
                    if not char.isalnum():
                        suffix_punct = char + suffix_punct
                    else:
                        break
                        
                replacement = prefix_punct + best_fix + suffix_punct
                corrected_words.append(replacement)
                changes.append({"from": w, "to": replacement, "confidence": fixes[0]["confidence"], "type": fixes[0].get("type", "correction")})
            else:
                corrected_words.append(w)
                
        corrected_line = " ".join(corrected_words)
        return {
            "original": sentence,
            "corrected": corrected_line,
            "has_changes": len(changes) > 0,
            "changes": changes
        }
