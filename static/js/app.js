/**
 * NeuroKey AI Application Logic
 * Orchestrates real-time predictions, autocorrect, telemetry, and UI updates.
 */

document.addEventListener('DOMContentLoaded', () => {
  // Elements
  const textarea = document.getElementById('main-typing-input');
  const candidatesStrip = document.getElementById('prediction-candidates-strip');
  const ribbonBadge = document.getElementById('ribbon-mode-badge');
  const ribbonText = document.getElementById('ribbon-mode-text');
  const ghostTag = document.getElementById('ghost-prediction-tag');
  const ghostWord = document.getElementById('ghost-word-text');

  // Telemetry elements
  const statWords = document.getElementById('stat-words');
  const statChars = document.getElementById('stat-chars');
  const statWpm = document.getElementById('stat-wpm');
  const statSavedKeys = document.getElementById('stat-saved-keys');
  const headerLatency = document.getElementById('header-latency-stat');
  const headerSavings = document.getElementById('header-savings-stat');
  const headerEngine = document.getElementById('header-active-engine');
  const engineDot = document.getElementById('engine-status-dot');
  const engineTagLabel = document.getElementById('engine-tag-label');

  // Controls
  const toggleAutocorrect = document.getElementById('toggle-autocorrect');
  const sliderTemperature = document.getElementById('slider-temperature');
  const valTemperature = document.getElementById('val-temperature');
  const demoSelect = document.getElementById('demo-scenario-select');

  // Probability and Context lists
  const probList = document.getElementById('probability-distribution-list');
  const contextChipsList = document.getElementById('context-chips-list');

  // Ergonomics & KSR
  const ksrCircle = document.getElementById('ksr-progress-circle');
  const ksrPercentVal = document.getElementById('ksr-percent-val');
  const metricTyped = document.getElementById('metric-keys-typed');
  const metricSaved = document.getElementById('metric-keys-saved');
  const metricTotal = document.getElementById('metric-keys-total');

  // Actions
  const btnSpeak = document.getElementById('btn-speak-text');
  const btnCopy = document.getElementById('btn-copy-text');
  const btnClear = document.getElementById('btn-clear-text');
  const btnFixEntireLine = document.getElementById('btn-fix-entire-line');

  // Whole Line Autocorrect elements
  const lineBanner = document.getElementById('line-autocorrect-banner');
  const lineCorrectedText = document.getElementById('line-corrected-text');
  const btnApplyLineFix = document.getElementById('btn-apply-line-fix');

  // Benchmarks Modal
  const modalBenchmarks = document.getElementById('modal-benchmarks');
  const btnOpenBenchmarks = document.getElementById('btn-open-benchmarks');
  const btnCloseModal = document.getElementById('btn-close-benchmark-modal');
  const btnDismissModal = document.getElementById('btn-dismiss-benchmark');
  const btnRecomputeBenchmark = document.getElementById('btn-recompute-benchmark');

  // State
  let currentEngine = 'lstm';
  let activePredictions = [];
  let currentLineCorrection = '';
  let debounceTimer = null;
  let typingStartTime = null;
  let keysTypedCount = 0;
  let keysSavedCount = 0;
  let lastInferenceTime = 0;

  // Initialize Virtual Keyboard
  const virtualKeyboard = new window.VirtualKeyboard();

  // Initialize Engine Selection buttons
  const engineBtns = document.querySelectorAll('.model-toggle-btn');
  engineBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      engineBtns.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      currentEngine = btn.getAttribute('data-engine');
      updateEngineHeader();
      fetchPredictions(true);
    });
  });

  function updateEngineHeader() {
    if (currentEngine === 'lstm') {
      headerEngine.textContent = 'Deep LSTM';
      engineTagLabel.textContent = 'Neural Net';
    } else if (currentEngine === 'ngram') {
      headerEngine.textContent = 'N-Gram (4-Gram)';
      engineTagLabel.textContent = 'Katz/Interp';
    } else {
      headerEngine.textContent = 'Hybrid Ensemble';
      engineTagLabel.textContent = 'Blended';
    }
  }

  // Temperature Slider
  sliderTemperature.addEventListener('input', (e) => {
    valTemperature.textContent = parseFloat(e.target.value).toFixed(1);
    fetchPredictions(false);
  });

  // Autocorrect Toggle
  toggleAutocorrect.addEventListener('change', () => {
    fetchPredictions(true);
  });

  // Textarea Input Listener
  textarea.addEventListener('input', (e) => {
    if (!typingStartTime) typingStartTime = Date.now();
    keysTypedCount++;
    updateTelemetry();

    // Debounced prediction fetch (120ms)
    clearTimeout(debounceTimer);
    debounceTimer = setTimeout(() => {
      fetchPredictions();
    }, 120);
  });

  // Handle Space and Tab keys in Textarea for smart auto-correction & prediction acceptance
  textarea.addEventListener('keydown', (e) => {
    if (e.key === 'Tab') {
      e.preventDefault();
      acceptPrimaryPrediction();
      return;
    }

    // Ctrl+Enter or Cmd+Enter to auto-fix the entire line
    if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
      e.preventDefault();
      applyWholeLineCorrection();
      return;
    }

    if (e.key === ' ' || e.code === 'Space') {
      // Smart Spacebar Autocorrect & Shortcut Expansion
      const isCorrectionOrShortcut = ribbonText.textContent.startsWith('Autocorrecting') || ribbonText.textContent.startsWith('⚡ Expanding');
      if (isCorrectionOrShortcut && activePredictions.length > 0) {
        const top = activePredictions[0];
        // If it's a genuine correction or a shortcut expansion
        if ((top.type === 'correction' && top.confidence >= 35) || top.type === 'shortcut') {
          e.preventDefault();
          applyPrediction(top, true);
          virtualKeyboard.playKeyClick(true);
          return;
        }
      }
    }
  });

  // Custom event from Virtual Keyboard Tab
  window.addEventListener('keyboard-tab-press', () => {
    acceptPrimaryPrediction();
  });

  // Custom event from Virtual Keyboard Space
  window.addEventListener('keyboard-space-press', (e) => {
    const isCorrectionOrShortcut = ribbonText.textContent.startsWith('Autocorrecting') || ribbonText.textContent.startsWith('⚡ Expanding');
    if (isCorrectionOrShortcut && activePredictions.length > 0) {
      const top = activePredictions[0];
      if ((top.type === 'correction' && top.confidence >= 35) || top.type === 'shortcut') {
        applyPrediction(top, true);
        return;
      }
    }
    virtualKeyboard.insertText(' ');
  });

  // Whole Line Autocorrect application
  async function applyWholeLineCorrection() {
    let corrected = currentLineCorrection;
    if (!corrected) {
      // Fetch directly from server if not already cached
      try {
        const res = await fetch('/api/correct_sentence', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ text: textarea.value })
        });
        const data = await res.json();
        if (data.status === 'success' && data.has_changes) {
          corrected = data.corrected;
        }
      } catch (e) {}
    }

    if (!corrected) {
      showToast("No typos detected in this line!");
      return;
    }

    const oldText = textarea.value;
    textarea.value = corrected + ' ';
    const saved = Math.max(2, Math.abs(corrected.length - oldText.length) + 4);
    keysSavedCount += saved;

    currentLineCorrection = '';
    lineBanner.style.display = 'none';

    textarea.focus();
    textarea.selectionStart = textarea.selectionEnd = textarea.value.length;
    textarea.dispatchEvent(new Event('input', { bubbles: true }));

    showToast(`✨ Auto-corrected whole line: "${corrected}"`);
    virtualKeyboard.playKeyClick(true);
  }

  btnApplyLineFix.addEventListener('click', applyWholeLineCorrection);
  btnFixEntireLine.addEventListener('click', applyWholeLineCorrection);

  // Scenario Selector
  demoSelect.addEventListener('change', (e) => {
    const val = e.target.value;
    if (val) {
      textarea.value = val;
      textarea.focus();
      textarea.dispatchEvent(new Event('input', { bubbles: true }));
      showToast(`Loaded demo: "${val}"`);
    }
  });

  // Speak / Copy / Clear
  btnSpeak.addEventListener('click', () => {
    const text = textarea.value.trim();
    if (!text) {
      showToast("Canvas is empty to speak!");
      return;
    }
    if ('speechSynthesis' in window) {
      window.speechSynthesis.cancel();
      const utterance = new SpeechSynthesisUtterance(text);
      utterance.rate = 1.0;
      utterance.pitch = 1.0;
      window.speechSynthesis.speak(utterance);
      showToast("Speaking text aloud...");
    } else {
      showToast("Speech synthesis not supported in this browser.");
    }
  });

  btnCopy.addEventListener('click', () => {
    if (!textarea.value) {
      showToast("Nothing to copy!");
      return;
    }
    navigator.clipboard.writeText(textarea.value).then(() => {
      showToast("Copied text to clipboard!");
    });
  });

  btnClear.addEventListener('click', () => {
    textarea.value = '';
    keysTypedCount = 0;
    keysSavedCount = 0;
    typingStartTime = null;
    updateTelemetry();
    fetchPredictions(true);
    showToast("Canvas cleared.");
  });

  // ==========================================================================
  // PREDICTION FETCHING & RENDERING
  // ==========================================================================
  async function fetchPredictions(immediate = false) {
    const text = textarea.value;
    const enableAc = toggleAutocorrect.checked;
    const temp = parseFloat(sliderTemperature.value) || 1.0;

    try {
      const response = await fetch('/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          text: text,
          engine: currentEngine,
          top_k: 5,
          temperature: temp,
          enable_autocorrect: enableAc
        })
      });

      const data = await response.json();
      if (data.status === 'success') {
        activePredictions = data.predictions || [];
        lastInferenceTime = data.latency_ms || 0;
        headerLatency.textContent = `${lastInferenceTime} ms`;
        
        renderRibbon(data);
        renderProbabilities(data.predictions);
        renderContext(data.context);
        updateGhostPrediction(data);

        // Whole-line autocorrect banner update
        if (data.line_correction && data.line_correction.has_changes) {
          currentLineCorrection = data.line_correction.corrected;
          lineCorrectedText.textContent = data.line_correction.corrected;
          lineBanner.style.display = 'flex';
        } else {
          currentLineCorrection = '';
          lineBanner.style.display = 'none';
        }
      }
    } catch (err) {
      console.error("Prediction fetch failed:", err);
    }
  }

  function renderRibbon(data) {
    const isAutocorrect = data.mode === 'autocorrect' || data.mode === 'shortcut';
    const isShortcut = data.mode === 'shortcut';

    if (isShortcut) {
      ribbonText.textContent = `⚡ Expanding Shortcut: "${data.active_token}"`;
      ribbonBadge.style.color = 'var(--accent-violet)';
    } else if (isAutocorrect) {
      ribbonText.textContent = `Autocorrecting: "${data.active_token}"`;
      ribbonBadge.style.color = 'var(--accent-amber)';
    } else {
      ribbonText.textContent = `Anticipating Next Word (${data.model_used.toUpperCase()})`;
      ribbonBadge.style.color = 'var(--text-secondary)';
    }

    if (!data.predictions || data.predictions.length === 0) {
      candidatesStrip.innerHTML = '<span style="font-size:0.8rem;color:var(--text-muted);padding:0.4rem;">No suggestions available</span>';
      return;
    }

    candidatesStrip.innerHTML = '';
    data.predictions.forEach((item, index) => {
      const pill = document.createElement('button');
      pill.className = `prediction-pill ${index === 0 ? 'primary' : ''}`;
      pill.setAttribute('data-word', item.word);

      let badgeText = `${item.confidence}%`;
      let badgeClass = '';
      if (item.type === 'shortcut') {
        badgeText = '⚡ Expand';
        badgeClass = 'shortcut';
      } else if (item.type === 'related') {
        badgeText = '🔗 Related';
        badgeClass = 'related';
      } else if (item.type === 'correction') {
        badgeText = 'Fix';
      } else if (item.type === 'completion') {
        badgeText = 'Complete';
      }

      pill.innerHTML = `
        <span class="pill-word">${item.word}</span>
        <span class="pill-badge ${badgeClass}">${badgeText}</span>
        ${index === 0 ? '<span class="pill-hotkey">[Tab]</span>' : ''}
      `;

      pill.addEventListener('click', () => {
        applyPrediction(item, isAutocorrect);
      });

      candidatesStrip.appendChild(pill);
    });
  }

  function updateGhostPrediction(data) {
    if (data.predictions && data.predictions.length > 0) {
      const topWord = data.predictions[0].word;
      ghostWord.textContent = topWord;
      ghostTag.style.display = 'flex';
    } else {
      ghostTag.style.display = 'none';
    }
  }

  function acceptPrimaryPrediction() {
    if (activePredictions.length > 0) {
      const isCorrectionOrShortcut = ribbonText.textContent.startsWith('Autocorrecting') || ribbonText.textContent.startsWith('⚡ Expanding');
      applyPrediction(activePredictions[0], isCorrectionOrShortcut);
      virtualKeyboard.playKeyClick(true);
    }
  }

  function applyPrediction(predictionItem, isAutocorrect) {
    const word = typeof predictionItem === 'object' ? predictionItem.word : predictionItem;
    const itemType = typeof predictionItem === 'object' ? predictionItem.type : null;
    const shortcutFrom = typeof predictionItem === 'object' ? predictionItem.shortcut_from : null;

    const currentVal = textarea.value;
    let savedChars = 0;

    if (isAutocorrect) {
      // Replace the active last partial word
      const words = currentVal.trimEnd().split(/\s+/);
      const partialWord = words[words.length - 1] || '';

      // If user directly clicked a related word for an active shortcut (e.g. typed 'fn' and clicked 'call')
      if (itemType === 'related' && shortcutFrom) {
        const shortcutPred = activePredictions.find(p => p.type === 'shortcut');
        const expansion = shortcutPred ? shortcutPred.word : shortcutFrom;
        words[words.length - 1] = `${expansion} ${word}`;
        savedChars = Math.max(0, (expansion.length + word.length + 1) - partialWord.length);
      } else {
        words[words.length - 1] = word;
        savedChars = Math.max(0, word.length - partialWord.length);
      }
      textarea.value = words.join(' ') + ' ';
    } else {
      // Append next word with trailing space
      if (!currentVal.endsWith(' ') && currentVal.length > 0) {
        textarea.value += ' ' + word + ' ';
      } else {
        textarea.value += word + ' ';
      }
      // Saved keystrokes: full word length minus 1 (for Tab)
      savedChars = Math.max(0, word.length - 1);
    }

    keysSavedCount += savedChars;
    textarea.focus();
    textarea.selectionStart = textarea.selectionEnd = textarea.value.length;
    textarea.dispatchEvent(new Event('input', { bubbles: true }));

    showToast(`Inserted "${word}" (+${savedChars} keys saved)`);
  }

  // ==========================================================================
  // TELEMETRY & ERGONOMICS COMPUTATION
  // ==========================================================================
  function updateTelemetry() {
    const text = textarea.value;
    const words = text.trim() ? text.trim().split(/\s+/).length : 0;
    const chars = text.length;

    statWords.textContent = words;
    statChars.textContent = chars;
    statSavedKeys.textContent = keysSavedCount;

    // Calculate WPM
    if (typingStartTime && words > 0) {
      const minutesElapsed = (Date.now() - typingStartTime) / 60000;
      const wpm = minutesElapsed > 0.05 ? Math.round(words / minutesElapsed) : 0;
      statWpm.textContent = Math.min(wpm, 250);
    } else {
      statWpm.textContent = 0;
    }

    // Keystroke Savings Rate (KSR)
    const totalSimulated = keysTypedCount + keysSavedCount;
    const ksrPercent = totalSimulated > 0 ? Math.round((keysSavedCount / totalSimulated) * 100) : 0;

    ksrPercentVal.textContent = `${ksrPercent}%`;
    headerSavings.textContent = `${ksrPercent}%`;
    metricTyped.textContent = keysTypedCount;
    metricSaved.textContent = keysSavedCount;
    metricTotal.textContent = totalSimulated;

    // Update SVG stroke-dasharray (circumference is 100 for r=15.9155)
    ksrCircle.setAttribute('stroke-dasharray', `${ksrPercent}, 100`);
  }

  function renderProbabilities(predictions) {
    if (!predictions || predictions.length === 0) {
      probList.innerHTML = '<div class="empty-prob-state">No active probabilities to visualize.</div>';
      return;
    }

    probList.innerHTML = '';
    predictions.forEach(p => {
      const row = document.createElement('div');
      row.className = 'prob-row';
      row.innerHTML = `
        <div class="prob-labels">
          <span class="prob-word">${p.word}</span>
          <span class="prob-pct">${p.confidence}%</span>
        </div>
        <div class="prob-bar-track">
          <div class="prob-bar-fill" style="width: ${Math.max(4, p.confidence)}%;"></div>
        </div>
      `;
      probList.appendChild(row);
    });
  }

  function renderContext(contextStr) {
    if (!contextStr || !contextStr.trim()) {
      contextChipsList.innerHTML = '<span class="context-chip placeholder">&lt;Awaiting preceding tokens&gt;</span>';
      return;
    }

    const tokens = contextStr.trim().split(/\s+/).slice(-4);
    contextChipsList.innerHTML = '';
    tokens.forEach((t, i) => {
      const chip = document.createElement('span');
      chip.className = 'context-chip';
      chip.textContent = `${t} [t-${tokens.length - 1 - i}]`;
      contextChipsList.appendChild(chip);
    });
  }

  // ==========================================================================
  // BENCHMARK MODAL & EVALUATION
  // ==========================================================================
  btnOpenBenchmarks.addEventListener('click', openBenchmarkModal);
  btnCloseModal.addEventListener('click', () => modalBenchmarks.style.display = 'none');
  btnDismissModal.addEventListener('click', () => modalBenchmarks.style.display = 'none');
  modalBenchmarks.addEventListener('click', (e) => {
    if (e.target === modalBenchmarks) modalBenchmarks.style.display = 'none';
  });

  btnRecomputeBenchmark.addEventListener('click', () => {
    loadBenchmarks(true);
  });

  function openBenchmarkModal() {
    modalBenchmarks.style.display = 'flex';
    loadBenchmarks(false);
  }

  async function loadBenchmarks(force = false) {
    const benchLstmTop1 = document.getElementById('bench-lstm-top1');
    const benchLstmTop3 = document.getElementById('bench-lstm-top3');
    const benchLstmPpl = document.getElementById('bench-lstm-ppl');
    const benchLstmLat = document.getElementById('bench-lstm-lat');
    const benchLstmKsr = document.getElementById('bench-lstm-ksr');

    const benchNgramTop1 = document.getElementById('bench-ngram-top1');
    const benchNgramTop3 = document.getElementById('bench-ngram-top3');
    const benchNgramPpl = document.getElementById('bench-ngram-ppl');
    const benchNgramLat = document.getElementById('bench-ngram-lat');
    const benchNgramKsr = document.getElementById('bench-ngram-ksr');

    benchLstmTop1.textContent = 'Computing...';
    benchNgramTop1.textContent = 'Computing...';

    try {
      const res = await fetch(`/api/evaluate?force=${force}`);
      const data = await res.json();

      if (data.status === 'success' && data.metrics) {
        const m = data.metrics;

        benchLstmTop1.textContent = `${m.lstm.top1_acc}%`;
        benchLstmTop3.textContent = `${m.lstm.top3_acc}%`;
        benchLstmPpl.textContent = `${m.lstm.perplexity}`;
        benchLstmLat.textContent = `${m.lstm.avg_latency_ms} ms`;
        benchLstmKsr.textContent = `+${m.lstm.keystroke_savings_pct}%`;

        benchNgramTop1.textContent = `${m.ngram.top1_acc}%`;
        benchNgramTop3.textContent = `${m.ngram.top3_acc}%`;
        benchNgramPpl.textContent = `${m.ngram.perplexity}`;
        benchNgramLat.textContent = `${m.ngram.avg_latency_ms} ms`;
        benchNgramKsr.textContent = `+${m.ngram.keystroke_savings_pct}%`;

        showToast("Benchmark metrics loaded successfully!");
      }
    } catch (e) {
      console.error("Benchmark error:", e);
      showToast("Failed to load benchmarks.");
    }
  }

  // ==========================================================================
  // SHORTCUTS & TEXT EXPANSION MODAL
  // ==========================================================================
  const modalShortcuts = document.getElementById('modal-shortcuts');
  const btnOpenShortcuts = document.getElementById('btn-open-shortcuts');
  const btnCloseShortcutsModal = document.getElementById('btn-close-shortcuts-modal');
  const btnDismissShortcuts = document.getElementById('btn-dismiss-shortcuts');
  const shortcutsSearch = document.getElementById('shortcuts-search-input');
  const shortcutsGrid = document.getElementById('shortcuts-cards-grid');

  let shortcutsCache = {};

  if (btnOpenShortcuts) {
    btnOpenShortcuts.addEventListener('click', openShortcutsModal);
  }
  if (btnCloseShortcutsModal) {
    btnCloseShortcutsModal.addEventListener('click', () => modalShortcuts.style.display = 'none');
  }
  if (btnDismissShortcuts) {
    btnDismissShortcuts.addEventListener('click', () => modalShortcuts.style.display = 'none');
  }
  if (modalShortcuts) {
    modalShortcuts.addEventListener('click', (e) => {
      if (e.target === modalShortcuts) modalShortcuts.style.display = 'none';
    });
  }

  async function openShortcutsModal() {
    modalShortcuts.style.display = 'flex';
    if (Object.keys(shortcutsCache).length === 0) {
      try {
        const res = await fetch('/api/shortcuts');
        const data = await res.json();
        if (data.status === 'success') {
          shortcutsCache = data.shortcuts || {};
        }
      } catch (e) {}
    }
    renderShortcutsGrid('');
  }

  if (shortcutsSearch) {
    shortcutsSearch.addEventListener('input', (e) => {
      renderShortcutsGrid(e.target.value.toLowerCase().trim());
    });
  }

  function renderShortcutsGrid(filterText) {
    if (!shortcutsGrid) return;
    shortcutsGrid.innerHTML = '';
    const keys = Object.keys(shortcutsCache);
    const filtered = keys.filter(k => {
      const item = shortcutsCache[k];
      return k.includes(filterText) || 
             (item.expansion && item.expansion.toLowerCase().includes(filterText)) || 
             (item.category && item.category.toLowerCase().includes(filterText));
    });

    if (filtered.length === 0) {
      shortcutsGrid.innerHTML = '<div style="color:var(--text-muted);padding:1rem;">No matching shortcuts found.</div>';
      return;
    }

    filtered.forEach(k => {
      const item = shortcutsCache[k];
      const card = document.createElement('div');
      card.className = 'shortcut-card';
      const relatedTags = (item.related || []).map(r => `<span class="sc-rel-tag">${r}</span>`).join('');
      card.innerHTML = `
        <div class="sc-header-row">
          <span class="sc-code-badge">${k}</span>
          <span class="sc-cat-pill">${item.category || 'general'}</span>
        </div>
        <div class="sc-expansion">→ ${item.expansion}</div>
        <div class="sc-related-chips">${relatedTags}</div>
      `;
      card.addEventListener('click', () => {
        // Insert expansion directly to canvas
        const currentVal = textarea.value.trim();
        textarea.value = (currentVal ? currentVal + ' ' : '') + item.expansion + ' ';
        modalShortcuts.style.display = 'none';
        textarea.focus();
        textarea.dispatchEvent(new Event('input', { bubbles: true }));
        showToast(`✨ Expanded "${k}" → "${item.expansion}"`);
      });
      shortcutsGrid.appendChild(card);
    });
  }

  // Toast Notification Helper
  function showToast(msg) {
    const container = document.getElementById('toast-container');
    const toast = document.createElement('div');
    toast.className = 'toast-message';
    toast.textContent = msg;
    container.appendChild(toast);

    setTimeout(() => {
      toast.style.opacity = '0';
      toast.style.transform = 'translateY(10px)';
      toast.style.transition = 'all 0.3s ease';
      setTimeout(() => toast.remove(), 300);
    }, 2500);
  }

  // Initial trigger
  fetchPredictions(true);
});
