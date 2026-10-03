"""
Unified Model Training Pipeline
Trains both N-Gram Language Model and TensorFlow/Keras LSTM Model,
saves models, weights, and tokenizers to models/ directory.
"""

import os
import time
from src.preprocessing import TextTokenizer, generate_lstm_data
from src.ngram_model import NGramLanguageModel
from src.lstm_model import LSTMNextWordModel

def train_all_models():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    train_file = os.path.join(base_dir, "data", "processed", "train_corpus.txt")
    test_file = os.path.join(base_dir, "data", "processed", "test_corpus.txt")
    
    if not os.path.exists(train_file):
        raise FileNotFoundError(f"Training corpus not found at {train_file}. Run prepare_dataset.py first.")
        
    with open(train_file, 'r', encoding='utf-8') as f:
        train_sentences = [line.strip() for line in f if line.strip()]
        
    with open(test_file, 'r', encoding='utf-8') as f:
        test_sentences = [line.strip() for line in f if line.strip()]
        
    print(f"Loaded {len(train_sentences)} training sentences and {len(test_sentences)} test sentences.")
    
    # 1. Build and save Tokenizer
    print("\n" + "="*50)
    print("STEP 1: Fitting and saving Tokenizer...")
    print("="*50)
    tokenizer = TextTokenizer(max_vocab_size=3500, min_freq=2)
    tokenizer.fit_on_texts(train_sentences)
    tokenizer_path = os.path.join(base_dir, "models", "tokenizer.json")
    tokenizer.save(tokenizer_path)

    # 2. Train N-Gram Language Model
    print("\n" + "="*50)
    print("STEP 2: Training N-Gram Language Model (Trigram / 4-Gram with Interpolation)...")
    print("="*50)
    t0 = time.time()
    ngram = NGramLanguageModel(n=4)
    ngram.train(train_sentences)
    ngram_time = time.time() - t0
    ngram_path = os.path.join(base_dir, "models", "ngram_model.pkl")
    ngram.save(ngram_path)
    
    # Quick Perplexity check
    ngram_ppl = ngram.calculate_perplexity(test_sentences[:200])
    print(f"N-Gram Training complete in {ngram_time:.2f}s | Test Perplexity (sample): {ngram_ppl}")

    # 3. Train LSTM Neural Network
    print("\n" + "="*50)
    print("STEP 3: Training TensorFlow/Keras LSTM Next-Word Prediction Model...")
    print("="*50)
    seq_length = 4
    train_seqs = tokenizer.texts_to_sequences(train_sentences)
    test_seqs = tokenizer.texts_to_sequences(test_sentences)
    
    X_train, y_train = generate_lstm_data(train_seqs, seq_length=seq_length)
    X_val, y_val = generate_lstm_data(test_seqs, seq_length=seq_length)
    
    print(f"Generated {len(X_train)} training sequences and {len(X_val)} validation sequences (context window = {seq_length}).")
    
    t0 = time.time()
    lstm = LSTMNextWordModel(
        vocab_size=tokenizer.vocab_size,
        seq_length=seq_length,
        embedding_dim=64,
        lstm_units=128
    )
    lstm.tokenizer = tokenizer
    lstm.build_model()
    
    # Train for 8 epochs with batch_size 64
    lstm.train(X_train, y_train, X_val, y_val, epochs=8, batch_size=64)
    lstm_time = time.time() - t0
    
    lstm_path = os.path.join(base_dir, "models", "lstm_model.keras")
    lstm.save(lstm_path)
    print(f"LSTM Training complete in {lstm_time:.2f}s | Saved to {lstm_path}")
    
    # Quick sanity test predictions
    test_prompts = [
        "artificial intelligence is",
        "please let me know if",
        "machine learning models",
        "thank you very much for"
    ]
    
    print("\n" + "="*50)
    print("SAMPLE PREDICTIONS COMPARISON:")
    print("="*50)
    for p in test_prompts:
        ng_preds = [x['word'] for x in ngram.predict_next_words(p, top_k=3)]
        ls_preds = [x['word'] for x in lstm.predict_next_words(p, top_k=3)]
        print(f"Prompt: '{p}'")
        print(f"  -> N-Gram: {ng_preds}")
        print(f"  -> LSTM:   {ls_preds}")
        
    print("\nAll models trained and saved successfully!")

if __name__ == "__main__":
    train_all_models()
