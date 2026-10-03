"""
Model Evaluation and Benchmarking Suite
Calculates:
- Perplexity (PPL) on unseen test sets
- Top-1, Top-3, and Top-5 Next-Word Accuracy
- Keystroke Savings Rate (KSR) Simulation
- Prediction Latency Benchmark
"""

import os
import time
import math
import numpy as np
from typing import Dict, Any, List
from src.preprocessing import tokenize_sentence, TextTokenizer, generate_lstm_data
from src.ngram_model import NGramLanguageModel
from src.lstm_model import LSTMNextWordModel

def evaluate_models(max_sentences: int = 150) -> Dict[str, Any]:
    """
    Run comprehensive evaluation on test dataset.
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    test_path = os.path.join(base_dir, "data", "processed", "test_corpus.txt")
    tok_path = os.path.join(base_dir, "models", "tokenizer.json")
    ngram_path = os.path.join(base_dir, "models", "ngram_model.pkl")
    lstm_path = os.path.join(base_dir, "models", "lstm_model.keras")

    if not os.path.exists(test_path):
        raise FileNotFoundError("Test corpus missing.")

    with open(test_path, 'r', encoding='utf-8') as f:
        test_sentences = [line.strip() for line in f if line.strip()][:max_sentences]

    tokenizer = TextTokenizer.load(tok_path)
    ngram = NGramLanguageModel.load(ngram_path)
    lstm = LSTMNextWordModel.load(lstm_path, tokenizer)

    results = {
        "dataset_info": {
            "test_sentences_evaluated": len(test_sentences),
            "vocab_size": tokenizer.vocab_size
        },
        "ngram": {
            "top1_acc": 0.0,
            "top3_acc": 0.0,
            "top5_acc": 0.0,
            "perplexity": 0.0,
            "avg_latency_ms": 0.0,
            "keystroke_savings_pct": 0.0
        },
        "lstm": {
            "top1_acc": 0.0,
            "top3_acc": 0.0,
            "top5_acc": 0.0,
            "perplexity": 0.0,
            "avg_latency_ms": 0.0,
            "keystroke_savings_pct": 0.0
        }
    }

    # Evaluate N-gram Perplexity
    results["ngram"]["perplexity"] = ngram.calculate_perplexity(test_sentences)

    # Evaluate LSTM loss and Perplexity
    test_seqs = tokenizer.texts_to_sequences(test_sentences)
    X_test, y_test = generate_lstm_data(test_seqs, seq_length=lstm.seq_length)
    if len(X_test) > 0 and lstm.model is not None:
        loss, acc = lstm.model.evaluate(X_test, y_test, verbose=0)
        results["lstm"]["perplexity"] = round(math.exp(min(loss, 15.0)), 2)

    # Evaluation loop for Top-K accuracy, latency, and keystroke savings
    ngram_top1, ngram_top3, ngram_top5 = 0, 0, 0
    lstm_top1, lstm_top3, lstm_top5 = 0, 0, 0
    total_eval_tokens = 0

    ngram_latencies = []
    lstm_latencies = []

    total_chars_in_text = 0
    ngram_chars_saved = 0
    lstm_chars_saved = 0

    for sentence in test_sentences:
        tokens = tokenize_sentence(sentence)
        if len(tokens) < 3:
            continue
            
        full_sent_str = " ".join(tokens)
        total_chars_in_text += len(full_sent_str)
        
        for i in range(1, len(tokens)):
            target_word = tokens[i]
            context = " ".join(tokens[:i])
            total_eval_tokens += 1
            
            # 1. N-Gram Prediction Benchmark
            t0 = time.perf_counter()
            ng_preds = [p["word"] for p in ngram.predict_next_words(context, top_k=5)]
            ngram_latencies.append((time.perf_counter() - t0) * 1000)
            
            if ng_preds and target_word == ng_preds[0]:
                ngram_top1 += 1
            if target_word in ng_preds[:3]:
                ngram_top3 += 1
                # Keystroke savings: if in top 3 before typing, saves full word length + 1 space minus 1 tab key
                ngram_chars_saved += max(0, len(target_word) - 1)
            if target_word in ng_preds[:5]:
                ngram_top5 += 1

            # 2. LSTM Prediction Benchmark
            t0 = time.perf_counter()
            ls_preds = [p["word"] for p in lstm.predict_next_words(context, top_k=5)]
            lstm_latencies.append((time.perf_counter() - t0) * 1000)

            if ls_preds and target_word == ls_preds[0]:
                lstm_top1 += 1
            if target_word in ls_preds[:3]:
                lstm_top3 += 1
                lstm_chars_saved += max(0, len(target_word) - 1)
            if target_word in ls_preds[:5]:
                lstm_top5 += 1

    if total_eval_tokens > 0:
        results["ngram"]["top1_acc"] = round((ngram_top1 / total_eval_tokens) * 100, 2)
        results["ngram"]["top3_acc"] = round((ngram_top3 / total_eval_tokens) * 100, 2)
        results["ngram"]["top5_acc"] = round((ngram_top5 / total_eval_tokens) * 100, 2)
        results["ngram"]["avg_latency_ms"] = round(float(np.mean(ngram_latencies)), 2)
        results["ngram"]["keystroke_savings_pct"] = round((ngram_chars_saved / max(1, total_chars_in_text)) * 100, 2)

        results["lstm"]["top1_acc"] = round((lstm_top1 / total_eval_tokens) * 100, 2)
        results["lstm"]["top3_acc"] = round((lstm_top3 / total_eval_tokens) * 100, 2)
        results["lstm"]["top5_acc"] = round((lstm_top5 / total_eval_tokens) * 100, 2)
        results["lstm"]["avg_latency_ms"] = round(float(np.mean(lstm_latencies)), 2)
        results["lstm"]["keystroke_savings_pct"] = round((lstm_chars_saved / max(1, total_chars_in_text)) * 100, 2)

    return results

if __name__ == "__main__":
    print("Running evaluation suite...")
    metrics = evaluate_models(max_sentences=100)
    print("\nEVALUATION RESULTS:")
    print("-" * 50)
    print(f"Metrics (N-Gram): Top-1: {metrics['ngram']['top1_acc']}%, Top-3: {metrics['ngram']['top3_acc']}%, PPL: {metrics['ngram']['perplexity']}, Latency: {metrics['ngram']['avg_latency_ms']}ms, KSR: {metrics['ngram']['keystroke_savings_pct']}%")
    print(f"Metrics (LSTM):   Top-1: {metrics['lstm']['top1_acc']}%, Top-3: {metrics['lstm']['top3_acc']}%, PPL: {metrics['lstm']['perplexity']}, Latency: {metrics['lstm']['avg_latency_ms']}ms, KSR: {metrics['lstm']['keystroke_savings_pct']}%")
