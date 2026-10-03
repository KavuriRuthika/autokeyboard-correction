"""
Flask Application for Smart Autocorrect Keyboard & Next-Word Prediction
Provides REST APIs for real-time predictions, autocorrect, evaluation, and interactive web UI.
"""

import os
import json
import time
from flask import Flask, request, jsonify, render_template
from src.predictor import SmartKeyboardPredictor
from src.evaluator import evaluate_models

app = Flask(__name__, static_folder="static", template_folder="templates")

# Initialize global predictor
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")

predictor = None
cached_metrics = None

def get_predictor():
    global predictor
    if predictor is None:
        predictor = SmartKeyboardPredictor(models_dir=MODELS_DIR, data_dir=DATA_DIR)
    return predictor

@app.route("/")
def index():
    """Render the AI Smart Keyboard interface."""
    return render_template("index.html")

@app.route("/api/status", methods=["GET"])
def get_status():
    """Return backend status and model readiness."""
    p = get_predictor()
    ngram_ready = p.ngram_model is not None
    lstm_ready = p.lstm_model is not None
    autocorrect_ready = p.autocorrect is not None
    vocab_size = p.tokenizer.vocab_size if p.tokenizer else (len(p.autocorrect.vocabulary) if p.autocorrect else 0)
    
    return jsonify({
        "status": "ready" if (ngram_ready or lstm_ready) else "training",
        "ngram_loaded": ngram_ready,
        "lstm_loaded": lstm_ready,
        "autocorrect_loaded": autocorrect_ready,
        "vocab_size": vocab_size,
        "default_engine": "lstm" if lstm_ready else "ngram"
    })

@app.route("/api/predict", methods=["POST"])
def predict():
    """
    Main prediction endpoint.
    Automatically differentiates between in-word autocorrect and post-space next-word prediction.
    """
    try:
        data = request.get_json(force=True) or {}
        text = data.get("text", "")
        engine = data.get("engine", "lstm")
        top_k = int(data.get("top_k", 5))
        temperature = float(data.get("temperature", 1.0))
        enable_autocorrect = bool(data.get("enable_autocorrect", True))
        
        p = get_predictor()
        # Fallback to ngram if lstm is not loaded yet
        if engine == "lstm" and p.lstm_model is None and p.ngram_model is not None:
            engine = "ngram"
            
        result = p.predict(
            full_text=text,
            engine=engine,
            top_k=top_k,
            temperature=temperature,
            enable_autocorrect=enable_autocorrect
        )
        return jsonify({"status": "success", **result})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/api/correct", methods=["POST"])
def correct_word():
    """Direct autocorrect endpoint for single word typos."""
    try:
        data = request.get_json(force=True) or {}
        word = data.get("word", "").strip()
        top_k = int(data.get("top_k", 5))
        
        p = get_predictor()
        if not p.autocorrect:
            return jsonify({"status": "error", "message": "Autocorrect engine not ready"}), 503
            
        t0 = time.perf_counter()
        suggestions = p.autocorrect.correct(word, top_k=top_k)
        latency = round((time.perf_counter() - t0) * 1000, 2)
        
        return jsonify({
            "status": "success",
            "word": word,
            "latency_ms": latency,
            "suggestions": suggestions
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/api/correct_sentence", methods=["POST"])
def correct_sentence_endpoint():
    """Whole-line / full-sentence autocorrect endpoint."""
    try:
        data = request.get_json(force=True) or {}
        text = data.get("text", "").strip()
        p = get_predictor()
        if not p.autocorrect:
            return jsonify({"status": "error", "message": "Autocorrect engine not ready"}), 503
            
        t0 = time.perf_counter()
        result = p.autocorrect.correct_sentence(text)
        latency = round((time.perf_counter() - t0) * 1000, 2)
        
        return jsonify({
            "status": "success",
            "latency_ms": latency,
            **result
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/api/shortcuts", methods=["GET"])
def get_shortcuts():
    """Return registry of shorthand abbreviations and domain-specific related words."""
    try:
        from src.shortcuts import DEFAULT_SHORTCUTS
        p = get_predictor()
        shortcuts_data = p.autocorrect.shortcut_engine.shortcuts if (p.autocorrect and hasattr(p.autocorrect, 'shortcut_engine')) else DEFAULT_SHORTCUTS
        return jsonify({"status": "success", "shortcuts": shortcuts_data})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/api/evaluate", methods=["GET"])
def get_evaluation():
    """Compute or return cached benchmarking metrics."""
    global cached_metrics
    try:
        # Check if re-evaluation is forced
        force = request.args.get("force", "false").lower() == "true"
        if cached_metrics is None or force:
            cached_metrics = evaluate_models(max_sentences=60)
            
        return jsonify({"status": "success", "metrics": cached_metrics})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == "__main__":
    print("Starting AI Smart Keyboard Server on http://127.0.0.1:5000 ...")
    app.run(host="127.0.0.1", port=5000, debug=False)
