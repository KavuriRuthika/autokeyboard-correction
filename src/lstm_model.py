"""
LSTM / Recurrent Neural Network Next-Word Prediction Model
Built with TensorFlow and Keras.
Features:
- Word Embedding Layer
- Stacked Recurrent LSTM Layers with Dropout
- Dense Projection & Softmax Probability Distribution
- Temperature-controlled inference and top-K candidate extraction
"""

import os
import json
import numpy as np
import tensorflow as tf
from typing import List, Dict, Any, Optional
from src.preprocessing import TextTokenizer, clean_text, tokenize_sentence

class LSTMNextWordModel:
    """
    TensorFlow/Keras LSTM Next-Word Prediction Model.
    """
    def __init__(self, vocab_size: int = 5000, seq_length: int = 4, embedding_dim: int = 64, lstm_units: int = 128):
        self.vocab_size = vocab_size
        self.seq_length = seq_length
        self.embedding_dim = embedding_dim
        self.lstm_units = lstm_units
        self.model: Optional[tf.keras.Model] = None
        self.tokenizer: Optional[TextTokenizer] = None

    def build_model(self):
        """Construct the neural network architecture."""
        inputs = tf.keras.Input(shape=(self.seq_length,), dtype=tf.int32, name="context_tokens")
        
        # Word embedding
        x = tf.keras.layers.Embedding(
            input_dim=self.vocab_size,
            output_dim=self.embedding_dim,
            name="word_embedding"
        )(inputs)
        
        # Stacked LSTM layers with dropout for regularization
        x = tf.keras.layers.LSTM(self.lstm_units, return_sequences=True, name="lstm_1")(x)
        x = tf.keras.layers.Dropout(0.2, name="dropout_1")(x)
        x = tf.keras.layers.LSTM(self.lstm_units, return_sequences=False, name="lstm_2")(x)
        x = tf.keras.layers.Dropout(0.2, name="dropout_2")(x)
        
        # Dense feature projection
        x = tf.keras.layers.Dense(self.lstm_units, activation="relu", name="dense_proj")(x)
        x = tf.keras.layers.Dropout(0.2, name="dropout_3")(x)
        
        # Softmax classification over vocabulary
        outputs = tf.keras.layers.Dense(self.vocab_size, activation="softmax", name="next_word_prob")(x)
        
        self.model = tf.keras.Model(inputs=inputs, outputs=outputs, name="NextWordLSTM")
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.003),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=["accuracy"]
        )
        print("LSTM Model compiled successfully:")
        self.model.summary()

    def train(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, epochs: int = 12, batch_size: int = 64):
        """Train the model with validation and early stopping."""
        if self.model is None:
            self.build_model()
            
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=3,
                restore_best_weights=True,
                verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=1,
                min_lr=1e-5,
                verbose=1
            )
        ]
        
        print(f"Beginning training on {len(X_train)} samples, validating on {len(X_val)} samples...")
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        return history

    def predict_next_words(self, context_text: str, top_k: int = 5, temperature: float = 1.0) -> List[Dict[str, Any]]:
        """
        Generate top-k next word predictions from raw context text.
        """
        if self.model is None or self.tokenizer is None:
            return []

        tokens = tokenize_sentence(context_text)
        unk_idx = self.tokenizer.word2idx.get("<UNK>", 1)
        pad_idx = self.tokenizer.word2idx.get("<PAD>", 0)
        
        token_ids = [self.tokenizer.word2idx.get(t, unk_idx) for t in tokens]
        
        # Prepare context window of length seq_length
        if len(token_ids) < self.seq_length:
            padded_seq = [pad_idx] * (self.seq_length - len(token_ids)) + token_ids
        else:
            padded_seq = token_ids[-self.seq_length:]
            
        input_array = np.array([padded_seq], dtype=np.int32)
        
        # Forward pass prediction
        predictions = self.model.predict(input_array, verbose=0)[0]
        
        # Suppress special tokens (<PAD>, <UNK>, <EOS>)
        for special in ("<PAD>", "<UNK>", "<EOS>"):
            idx = self.tokenizer.word2idx.get(special)
            if idx is not None and idx < len(predictions):
                predictions[idx] = 0.0

        # Apply temperature scaling if temperature != 1.0
        if temperature > 0.0 and temperature != 1.0:
            predictions = np.log(np.maximum(predictions, 1e-10)) / temperature
            exp_preds = np.exp(predictions - np.max(predictions))
            predictions = exp_preds / np.sum(exp_preds)
        else:
            predictions = predictions / (np.sum(predictions) + 1e-10)

        # Get top-k indices
        top_indices = np.argsort(predictions)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            word = self.tokenizer.idx2word.get(int(idx), "<UNK>")
            prob = float(predictions[idx])
            results.append({
                "word": word,
                "probability": round(prob, 4),
                "confidence": round(prob * 100, 1)
            })
            
        # Re-normalize displayed confidence across top-k
        total_top_p = sum(r["confidence"] for r in results) or 1.0
        for r in results:
            r["confidence"] = round((r["confidence"] / total_top_p) * 100, 1)
            
        return results

    def save(self, filepath: str):
        """Save model weights and metadata."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.model.save(filepath)
        
        # Save metadata config
        meta_path = filepath.rsplit('.', 1)[0] + "_meta.json"
        meta = {
            "vocab_size": self.vocab_size,
            "seq_length": self.seq_length,
            "embedding_dim": self.embedding_dim,
            "lstm_units": self.lstm_units
        }
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(meta, f, indent=2)
            
        print(f"LSTM Model saved to {filepath} and {meta_path}")

    @classmethod
    def load(cls, filepath: str, tokenizer: TextTokenizer) -> "LSTMNextWordModel":
        """Load trained model and associate tokenizer."""
        meta_path = filepath.rsplit('.', 1)[0] + "_meta.json"
        with open(meta_path, 'r', encoding='utf-8') as f:
            meta = json.load(f)
            
        instance = cls(
            vocab_size=meta["vocab_size"],
            seq_length=meta["seq_length"],
            embedding_dim=meta["embedding_dim"],
            lstm_units=meta["lstm_units"]
        )
        instance.model = tf.keras.models.load_model(filepath)
        instance.tokenizer = tokenizer
        print(f"LSTM Model loaded successfully from {filepath}")
        return instance
