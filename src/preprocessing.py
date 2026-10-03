"""
NLP Preprocessing and Tokenization Module
Handles text cleaning, tokenization, vocabulary management, and sequence generation
for both N-gram and LSTM models.
"""

import os
import re
import json
import numpy as np
from collections import Counter
from typing import List, Tuple, Dict, Optional

SPECIAL_TOKENS = {
    "<PAD>": 0,
    "<UNK>": 1,
    "<EOS>": 2
}

def clean_text(text: str) -> str:
    """Normalize text: lowercase, expand common contractions, remove invalid symbols."""
    text = text.lower().strip()
    
    # Common contractions expansion
    contractions = {
        r"won't": "will not",
        r"can't": "cannot",
        r"n't": " not",
        r"'re": " are",
        r"'s": " is",
        r"'d": " would",
        r"'ll": " will",
        r"'t": " not",
        r"'ve": " have",
        r"'m": " am"
    }
    for pattern, replacement in contractions.items():
        text = re.sub(pattern, replacement, text)
        
    # Replace non-alphabetical characters (except spaces and apostrophes)
    text = re.sub(r"[^a-z\s']", ' ', text)
    # Remove isolated apostrophes
    text = re.sub(r"\s+'|'\s+", ' ', text)
    # Normalize multiple whitespace
    text = re.sub(r"\s+", ' ', text).strip()
    return text

def tokenize_sentence(sentence: str) -> List[str]:
    """Tokenize a cleaned sentence into word tokens."""
    cleaned = clean_text(sentence)
    if not cleaned:
        return []
    return cleaned.split()

class TextTokenizer:
    """
    Vocabulary builder and sequence encoder/decoder.
    Maintains word-to-index and index-to-word mappings.
    """
    def __init__(self, max_vocab_size: int = 5000, min_freq: int = 1):
        self.max_vocab_size = max_vocab_size
        self.min_freq = min_freq
        self.word2idx: Dict[str, int] = dict(SPECIAL_TOKENS)
        self.idx2word: Dict[int, str] = {v: k for k, v in SPECIAL_TOKENS.items()}
        self.word_counts: Counter = Counter()
        self.vocab_size: int = len(SPECIAL_TOKENS)

    def fit_on_texts(self, texts: List[str]):
        """Build vocabulary from a list of text sentences."""
        self.word_counts = Counter()
        for text in texts:
            tokens = tokenize_sentence(text)
            self.word_counts.update(tokens)
            
        # Select most common words within limit and meeting min frequency
        most_common = [
            word for word, count in self.word_counts.most_common()
            if count >= self.min_freq and word not in SPECIAL_TOKENS
        ]
        
        # Limit to max_vocab_size minus special tokens
        allowed_count = self.max_vocab_size - len(SPECIAL_TOKENS)
        selected_words = most_common[:allowed_count]
        
        self.word2idx = dict(SPECIAL_TOKENS)
        for idx, word in enumerate(selected_words, start=len(SPECIAL_TOKENS)):
            self.word2idx[word] = idx
            
        self.idx2word = {v: k for k, v in self.word2idx.items()}
        self.vocab_size = len(self.word2idx)
        print(f"Tokenizer fitted: Vocab size = {self.vocab_size} (from {len(self.word_counts)} unique tokens)")

    def texts_to_sequences(self, texts: List[str]) -> List[List[int]]:
        """Convert list of sentences into integer index sequences."""
        unk_idx = self.word2idx.get("<UNK>", 1)
        sequences = []
        for text in texts:
            tokens = tokenize_sentence(text)
            seq = [self.word2idx.get(token, unk_idx) for token in tokens]
            if seq:
                sequences.append(seq)
        return sequences

    def sequence_to_text(self, sequence: List[int]) -> str:
        """Convert list of token indices back to readable string."""
        words = [self.idx2word.get(idx, "<UNK>") for idx in sequence]
        return " ".join([w for w in words if w not in ("<PAD>", "<EOS>")])

    def save(self, filepath: str):
        """Save tokenizer vocabulary and configuration to JSON file."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        data = {
            "max_vocab_size": self.max_vocab_size,
            "min_freq": self.min_freq,
            "vocab_size": self.vocab_size,
            "word2idx": self.word2idx,
            "word_counts": dict(self.word_counts.most_common(1000))
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        print(f"Tokenizer saved to {filepath}")

    @classmethod
    def load(cls, filepath: str) -> "TextTokenizer":
        """Load tokenizer vocabulary from JSON file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        tok = cls(max_vocab_size=data["max_vocab_size"], min_freq=data["min_freq"])
        tok.word2idx = data["word2idx"]
        tok.idx2word = {int(v): k for k, v in data["word2idx"].items()}
        tok.vocab_size = data["vocab_size"]
        return tok


def generate_lstm_data(sequences: List[List[int]], seq_length: int = 4) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate (X, y) training pairs using a sliding window.
    X: context window of shape (num_samples, seq_length)
    y: target next token index of shape (num_samples,)
    """
    X, y = [], []
    pad_idx = SPECIAL_TOKENS["<PAD>"]
    
    for seq in sequences:
        if len(seq) < 2:
            continue
        # Sliding sub-sequences
        for i in range(1, len(seq)):
            target = seq[i]
            context = seq[max(0, i - seq_length):i]
            
            # Left-pad context if shorter than seq_length
            if len(context) < seq_length:
                padded = [pad_idx] * (seq_length - len(context)) + context
            else:
                padded = context[-seq_length:]
                
            X.append(padded)
            y.append(target)
            
    return np.array(X, dtype=np.int32), np.array(y, dtype=np.int32)
