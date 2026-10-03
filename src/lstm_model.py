"""
LSTM / Recurrent Neural Network Next-Word Prediction Model
Built with TensorFlow / Keras with a high-performance pure NumPy fallback
for seamless, lightweight deployment on serverless platforms (e.g., Vercel, Render).

Features:
- Word Embedding Layer
- Stacked Recurrent LSTM Layers with Dropout
- Dense Projection & Softmax Probability Distribution
- Temperature-controlled inference and top-K candidate extraction
- Dual runtime: TensorFlow/Keras or ultra-fast pure NumPy forward pass
"""

import os
import json
import numpy as np
from typing import List, Dict, Any, Optional
from src.preprocessing import TextTokenizer, clean_text, tokenize_sentence

try:
    import tensorflow as tf
    HAS_TF = True
except (ImportError, ModuleNotFoundError):
    tf = None
    HAS_TF = False


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid function."""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


def _run_lstm_cell(x_t: np.ndarray, h_prev: np.ndarray, c_prev: np.ndarray,
                   kernel: np.ndarray, recurrent_kernel: np.ndarray, bias: np.ndarray,
                   units: int):
    """
    Standard Keras-compatible LSTM cell forward step.
    Keras gate order in kernel/bias is: [input (i), forget (f), cell (c), output (o)].
    """
    z = np.dot(x_t, kernel) + np.dot(h_prev, recurrent_kernel) + bias
    i = _sigmoid(z[0:units])
    f = _sigmoid(z[units:2 * units])
    c_cand = np.tanh(z[2 * units:3 * units])
    o = _sigmoid(z[3 * units:4 * units])
    c = f * c_prev + i * c_cand
    h = o * np.tanh(c)
    return h, c


class LSTMNextWordModel:
    """
    LSTM Next-Word Prediction Model with dual TensorFlow & pure NumPy inference engines.
    """
    def __init__(self, vocab_size: int = 5000, seq_length: int = 4, embedding_dim: int = 64, lstm_units: int = 128):
        self.vocab_size = vocab_size
        self.seq_length = seq_length
        self.embedding_dim = embedding_dim
        self.lstm_units = lstm_units
        self.model: Optional[Any] = None
        self.tokenizer: Optional[TextTokenizer] = None
        self.numpy_weights: Optional[Dict[str, np.ndarray]] = None

    def build_model(self):
        """Construct the neural network architecture via TensorFlow/Keras."""
        if not HAS_TF:
            raise RuntimeError("TensorFlow is required to train or construct the model architecture.")

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
        if not HAS_TF:
            raise RuntimeError("TensorFlow is required for training.")

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

    def forward_numpy(self, token_ids: List[int]) -> np.ndarray:
        """
        Pure NumPy forward pass through Embedding -> LSTM1 -> LSTM2 -> Dense -> Softmax.
        Executes in ~2ms with zero TensorFlow dependency.
        """
        w = self.numpy_weights
        embeds = w["w_embed"][token_ids]  # shape: (seq_length, embedding_dim)
        units = self.lstm_units

        # Layer 1: LSTM with return_sequences=True
        h = np.zeros(units, dtype=np.float32)
        c = np.zeros(units, dtype=np.float32)
        seq_h = []
        for t in range(self.seq_length):
            h, c = _run_lstm_cell(embeds[t], h, c, w["w_l1_k"], w["w_l1_r"], w["b_l1"], units)
            seq_h.append(h)

        # Layer 2: LSTM with return_sequences=False
        h2 = np.zeros(units, dtype=np.float32)
        c2 = np.zeros(units, dtype=np.float32)
        for t in range(self.seq_length):
            h2, c2 = _run_lstm_cell(seq_h[t], h2, c2, w["w_l2_k"], w["w_l2_r"], w["b_l2"], units)

        # Dense projection with ReLU
        dense = np.maximum(0.0, np.dot(h2, w["w_dense_k"]) + w["b_dense"])

        # Output projection and Softmax
        logits = np.dot(dense, w["w_out_k"]) + w["b_out"]
        exp_logits = np.exp(logits - np.max(logits))
        return exp_logits / np.sum(exp_logits)

    def predict_next_words(self, context_text: str, top_k: int = 5, temperature: float = 1.0) -> List[Dict[str, Any]]:
        """
        Generate top-k next word predictions from raw context text.
        """
        if (self.numpy_weights is None and self.model is None) or self.tokenizer is None:
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
            
        # Fast inference: use pure NumPy if available, else Keras model
        if self.numpy_weights is not None:
            predictions = self.forward_numpy(padded_seq)
        elif self.model is not None:
            input_array = np.array([padded_seq], dtype=np.int32)
            predictions = self.model.predict(input_array, verbose=0)[0]
        else:
            return []
        
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
        """Save model weights in both Keras and compressed NumPy formats for serverless portability."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        if HAS_TF and self.model is not None:
            self.model.save(filepath)
            
            # Export compressed NumPy weights for zero-dependency inference
            npz_path = filepath.rsplit('.', 1)[0] + "_weights.npz"
            np.savez_compressed(
                npz_path,
                w_embed=self.model.get_layer("word_embedding").get_weights()[0],
                w_l1_k=self.model.get_layer("lstm_1").get_weights()[0],
                w_l1_r=self.model.get_layer("lstm_1").get_weights()[1],
                b_l1=self.model.get_layer("lstm_1").get_weights()[2],
                w_l2_k=self.model.get_layer("lstm_2").get_weights()[0],
                w_l2_r=self.model.get_layer("lstm_2").get_weights()[1],
                b_l2=self.model.get_layer("lstm_2").get_weights()[2],
                w_dense_k=self.model.get_layer("dense_proj").get_weights()[0],
                b_dense=self.model.get_layer("dense_proj").get_weights()[1],
                w_out_k=self.model.get_layer("next_word_prob").get_weights()[0],
                b_out=self.model.get_layer("next_word_prob").get_weights()[1]
            )
            print(f"Exported portable NumPy weights to {npz_path}")
        
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
        """Load trained model weights and associate tokenizer."""
        meta_path = filepath.rsplit('.', 1)[0] + "_meta.json"
        npz_path = filepath.rsplit('.', 1)[0] + "_weights.npz"
        
        vocab_size = 520
        seq_length = 4
        embedding_dim = 64
        lstm_units = 128
        
        if os.path.exists(meta_path):
            with open(meta_path, 'r', encoding='utf-8') as f:
                meta = json.load(f)
                vocab_size = meta.get("vocab_size", vocab_size)
                seq_length = meta.get("seq_length", seq_length)
                embedding_dim = meta.get("embedding_dim", embedding_dim)
                lstm_units = meta.get("lstm_units", lstm_units)
            
        instance = cls(
            vocab_size=vocab_size,
            seq_length=seq_length,
            embedding_dim=embedding_dim,
            lstm_units=lstm_units
        )
        instance.tokenizer = tokenizer

        # 1. Prefer loading portable NumPy weights (instant, zero TF dependency, serverless safe)
        if os.path.exists(npz_path):
            loaded_npz = np.load(npz_path)
            instance.numpy_weights = {k: loaded_npz[k] for k in loaded_npz.files}
            print(f"Loaded portable NumPy LSTM weights from {npz_path}")
            return instance

        # 2. Fallback to Keras model if TensorFlow is installed
        if HAS_TF and os.path.exists(filepath):
            instance.model = tf.keras.models.load_model(filepath)
            print(f"LSTM Model loaded successfully from {filepath}")
            return instance

        print("Warning: Neither lstm_weights.npz nor Keras model could be loaded.")
        return instance
