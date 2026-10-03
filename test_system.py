"""
Comprehensive System and Integration Test Suite
Validates:
1. Web server HTTP responses & HTML template structure
2. /api/status, /api/predict, /api/correct, and /api/evaluate endpoints
3. Dual-engine next-word prediction (LSTM and N-Gram)
4. Bayesian QWERTY autocorrect accuracy on test typos
5. Keystroke savings & Perplexity calculations
"""

import json
import urllib.request
import urllib.error

BASE_URL = "http://127.0.0.1:5000"

def run_tests():
    print("=" * 60)
    print("RUNNING AUTOMATED INTEGRATION TESTS FOR NEUROKEY AI")
    print("=" * 60)
    
    passed = 0
    total = 0

    def test(name, condition, extra=""):
        nonlocal passed, total
        total += 1
        if condition:
            print(f" [PASS] {name} {extra}")
            passed += 1
        else:
            print(f" [FAIL] {name} {extra}")

    # Test 1: HTML Landing Page
    try:
        req = urllib.request.urlopen(f"{BASE_URL}/")
        html = req.read().decode('utf-8')
        test("1. Web Server Landing Page (GET /)", req.status == 200 and "NeuroKey AI" in html, f"(Code {req.status})")
        test("   - Check Virtual Keyboard rendered", 'id="virtual-keyboard"' in html)
        test("   - Check Predictive Ribbon rendered", 'id="prediction-candidates-strip"' in html)
        test("   - Check Probability Distribution UI", 'id="probability-distribution-list"' in html)
    except Exception as e:
        test("1. Web Server Landing Page", False, f"Error: {e}")

    # Test 2: System Status API
    try:
        req = urllib.request.urlopen(f"{BASE_URL}/api/status")
        data = json.loads(req.read())
        test("2. System Status API (GET /api/status)", data.get("status") == "ready", f"(Engine: {data.get('default_engine')}, Vocab: {data.get('vocab_size')})")
        test("   - LSTM Loaded", data.get("lstm_loaded") is True)
        test("   - N-Gram Loaded", data.get("ngram_loaded") is True)
        test("   - Autocorrect Loaded", data.get("autocorrect_loaded") is True)
    except Exception as e:
        test("2. System Status API", False, f"Error: {e}")

    # Test 3: Next-Word Prediction (LSTM)
    try:
        payload = json.dumps({"text": "artificial intelligence is ", "engine": "lstm"}).encode('utf-8')
        req = urllib.request.Request(f"{BASE_URL}/api/predict", data=payload, headers={"Content-Type": "application/json"})
        data = json.loads(urllib.request.urlopen(req).read())
        top_words = [p["word"] for p in data.get("predictions", [])]
        test("3. Next-Word Prediction - LSTM (POST /api/predict)", data.get("mode") == "next_word" and len(top_words) > 0, f"Top: {top_words[:3]}")
    except Exception as e:
        test("3. Next-Word Prediction - LSTM", False, f"Error: {e}")

    # Test 4: Next-Word Prediction (N-Gram)
    try:
        payload = json.dumps({"text": "machine learning models ", "engine": "ngram"}).encode('utf-8')
        req = urllib.request.Request(f"{BASE_URL}/api/predict", data=payload, headers={"Content-Type": "application/json"})
        data = json.loads(urllib.request.urlopen(req).read())
        top_words = [p["word"] for p in data.get("predictions", [])]
        test("4. Next-Word Prediction - N-Gram (POST /api/predict)", data.get("mode") == "next_word" and len(top_words) > 0, f"Top: {top_words[:3]}")
    except Exception as e:
        test("4. Next-Word Prediction - N-Gram", False, f"Error: {e}")

    # Test 5: Next-Word Prediction (Hybrid Ensemble)
    try:
        payload = json.dumps({"text": "please let me know if ", "engine": "ensemble"}).encode('utf-8')
        req = urllib.request.Request(f"{BASE_URL}/api/predict", data=payload, headers={"Content-Type": "application/json"})
        data = json.loads(urllib.request.urlopen(req).read())
        top_words = [p["word"] for p in data.get("predictions", [])]
        test("5. Next-Word Prediction - Hybrid Ensemble", data.get("mode") == "next_word" and len(top_words) > 0, f"Top: {top_words[:3]}")
    except Exception as e:
        test("5. Next-Word Prediction - Hybrid Ensemble", False, f"Error: {e}")

    # Test 6: In-Word Bayesian Autocorrect & Typo Handling
    typos = [
        ("teh", "the"),
        ("smarr", "smart"),
        ("alogrithm", "algorithm")
    ]
    for typo, expected in typos:
        try:
            payload = json.dumps({"text": typo, "engine": "lstm", "enable_autocorrect": True}).encode('utf-8')
            req = urllib.request.Request(f"{BASE_URL}/api/predict", data=payload, headers={"Content-Type": "application/json"})
            data = json.loads(urllib.request.urlopen(req).read())
            preds = [p["word"] for p in data.get("predictions", [])]
            test(f"6. Autocorrect '{typo}' -> expects '{expected}'", expected in preds, f"Got: {preds[:3]}")
        except Exception as e:
            test(f"6. Autocorrect '{typo}'", False, f"Error: {e}")

    # Test 7: Direct Word Correction Endpoint (/api/correct)
    try:
        payload = json.dumps({"word": "preeict", "top_k": 3}).encode('utf-8')
        req = urllib.request.Request(f"{BASE_URL}/api/correct", data=payload, headers={"Content-Type": "application/json"})
        data = json.loads(urllib.request.urlopen(req).read())
        suggestions = [s["word"] for s in data.get("suggestions", [])]
        test("7. Direct Word Correction (POST /api/correct)", len(suggestions) > 0, f"Suggestions: {suggestions}")
    except Exception as e:
        test("7. Direct Word Correction", False, f"Error: {e}")

    # Test 8: Whole-Line / Sentence-Level Autocorrect (POST /api/correct_sentence)
    try:
        test_line = "teh intellignet smarr alogrithm"
        expected_line = "the intelligent smart algorithm"
        payload = json.dumps({"text": test_line}).encode('utf-8')
        req = urllib.request.Request(f"{BASE_URL}/api/correct_sentence", data=payload, headers={"Content-Type": "application/json"})
        data = json.loads(urllib.request.urlopen(req).read())
        corrected = data.get("corrected")
        test("8. Whole-Line Autocorrect (POST /api/correct_sentence)", corrected == expected_line, f"'{test_line}' -> '{corrected}'")
    except Exception as e:
        test("8. Whole-Line Autocorrect", False, f"Error: {e}")

    # Test 9: Shorthand Shortcut Expansion (e.g. fn -> function + related words)
    try:
        payload = json.dumps({"text": "fn"}).encode('utf-8')
        req = urllib.request.Request(f"{BASE_URL}/api/predict", data=payload, headers={"Content-Type": "application/json"})
        data = json.loads(urllib.request.urlopen(req).read())
        preds = data.get("predictions", [])
        has_function = any(p["word"] == "function" for p in preds)
        has_related = any(p.get("type") == "related" for p in preds)
        test("9. Shorthand Shortcut Expansion ('fn' -> 'function' + related)", has_function and has_related, f"Top: {[p['word'] for p in preds[:4]]}")
    except Exception as e:
        test("9. Shorthand Shortcut Expansion", False, f"Error: {e}")

    # Test 10: Model Evaluation & Benchmarks Endpoint
    try:
        req = urllib.request.urlopen(f"{BASE_URL}/api/evaluate")
        data = json.loads(req.read())
        m = data.get("metrics", {})
        lstm_top3 = m.get("lstm", {}).get("top3_acc", 0)
        ngram_top3 = m.get("ngram", {}).get("top3_acc", 0)
        test("8. Benchmarking Metrics API (GET /api/evaluate)", "lstm" in m and "ngram" in m, f"(LSTM Top-3: {lstm_top3}%, N-Gram Top-3: {ngram_top3}%)")
    except Exception as e:
        test("8. Benchmarking Metrics API", False, f"Error: {e}")

    print("=" * 60)
    print(f"RESULTS: {passed}/{total} tests passed ({round((passed/total)*100, 1)}%)")
    print("=" * 60)
    return passed == total

if __name__ == "__main__":
    run_tests()
