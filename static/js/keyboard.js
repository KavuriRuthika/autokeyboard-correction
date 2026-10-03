/**
 * Virtual & Physical Keyboard Controller
 * Handles tactile animations, audio synthesis, layout shifting, and hardware sync.
 */

class VirtualKeyboard {
  constructor() {
    this.capsLock = false;
    this.shift = false;
    this.soundEnabled = true;
    this.audioCtx = null;

    this.textarea = document.getElementById('main-typing-input');
    this.keyboardContainer = document.getElementById('virtual-keyboard');
    this.capsBtn = document.getElementById('btn-capslock');
    this.shiftLeftBtn = document.getElementById('btn-shift-left');
    this.shiftRightBtn = document.getElementById('btn-shift-right');
    this.soundToggle = document.getElementById('toggle-keyboard-sound');

    this.initAudio();
    this.initEventListeners();
  }

  initAudio() {
    try {
      const AudioContext = window.AudioContext || window.webkitAudioContext;
      if (AudioContext) {
        this.audioCtx = new AudioContext();
      }
    } catch (e) {
      console.warn("AudioContext not supported:", e);
    }
  }

  playKeyClick(isSpecial = false) {
    if (!this.soundEnabled || !this.audioCtx) return;
    if (this.audioCtx.state === 'suspended') {
      this.audioCtx.resume();
    }

    try {
      const osc = this.audioCtx.createOscillator();
      const gain = this.audioCtx.createGain();

      osc.type = isSpecial ? 'triangle' : 'sine';
      osc.frequency.setValueAtTime(isSpecial ? 240 : 480 + (Math.random() * 80 - 40), this.audioCtx.currentTime);

      gain.gain.setValueAtTime(0.04, this.audioCtx.currentTime);
      gain.gain.exponentialRampToValueAtTime(0.0001, this.audioCtx.currentTime + 0.05);

      osc.connect(gain);
      gain.connect(this.audioCtx.destination);

      osc.start();
      osc.stop(this.audioCtx.currentTime + 0.05);
    } catch (e) {}
  }

  initEventListeners() {
    // Sound Toggle
    if (this.soundToggle) {
      this.soundEnabled = this.soundToggle.checked;
      this.soundToggle.addEventListener('change', (e) => {
        this.soundEnabled = e.target.checked;
      });
    }

    // Virtual Keyboard Click delegation
    this.keyboardContainer.addEventListener('click', (e) => {
      const btn = e.target.closest('.key-btn');
      if (!btn) return;
      this.handleKeyClick(btn);
    });

    // Hardware Keyboard listeners for bi-directional mirroring
    window.addEventListener('keydown', (e) => this.handleHardwareKeyDown(e));
    window.addEventListener('keyup', (e) => this.handleHardwareKeyUp(e));
  }

  handleKeyClick(btn) {
    const key = btn.getAttribute('data-key');
    const isSpecial = btn.classList.contains('key-backspace') || 
                      btn.classList.contains('key-enter') || 
                      btn.classList.contains('key-space') || 
                      btn.classList.contains('key-tab');

    this.playKeyClick(isSpecial);

    if (key === 'CapsLock') {
      this.toggleCapsLock();
      return;
    }

    if (key === 'Shift') {
      this.toggleShift();
      return;
    }

    if (key === 'Backspace') {
      this.deleteChar();
      return;
    }

    if (key === 'Enter') {
      this.insertText('\n');
      return;
    }

    if (key === 'Tab') {
      // Trigger tab acceptance through custom event
      window.dispatchEvent(new CustomEvent('keyboard-tab-press'));
      return;
    }

    if (key === ' ') {
      // Trigger space with autocorrect consideration
      window.dispatchEvent(new CustomEvent('keyboard-space-press'));
      return;
    }

    // Regular character
    let charToInsert = key;
    if (charToInsert.length === 1 && charToInsert.match(/[a-z]/i)) {
      charToInsert = (this.capsLock !== this.shift) ? charToInsert.toUpperCase() : charToInsert.toLowerCase();
    }

    this.insertText(charToInsert);

    // If shift was on, turn it off after a character
    if (this.shift) {
      this.toggleShift(false);
    }
  }

  insertText(text) {
    if (!this.textarea) return;
    this.textarea.focus();

    const start = this.textarea.selectionStart;
    const end = this.textarea.selectionEnd;
    const currentVal = this.textarea.value;

    this.textarea.value = currentVal.substring(0, start) + text + currentVal.substring(end);
    this.textarea.selectionStart = this.textarea.selectionEnd = start + text.length;

    // Trigger input event to update predictions and stats
    this.textarea.dispatchEvent(new Event('input', { bubbles: true }));
  }

  deleteChar() {
    if (!this.textarea) return;
    this.textarea.focus();

    const start = this.textarea.selectionStart;
    const end = this.textarea.selectionEnd;
    const currentVal = this.textarea.value;

    if (start === end) {
      if (start > 0) {
        this.textarea.value = currentVal.substring(0, start - 1) + currentVal.substring(end);
        this.textarea.selectionStart = this.textarea.selectionEnd = start - 1;
      }
    } else {
      this.textarea.value = currentVal.substring(0, start) + currentVal.substring(end);
      this.textarea.selectionStart = this.textarea.selectionEnd = start;
    }

    this.textarea.dispatchEvent(new Event('input', { bubbles: true }));
  }

  toggleCapsLock(forced = null) {
    this.capsLock = forced !== null ? forced : !this.capsLock;
    if (this.capsBtn) {
      this.capsBtn.classList.toggle('active', this.capsLock);
    }
    this.updateKeyLabels();
  }

  toggleShift(forced = null) {
    this.shift = forced !== null ? forced : !this.shift;
    const active = this.shift;
    if (this.shiftLeftBtn) this.shiftLeftBtn.classList.toggle('active', active);
    if (this.shiftRightBtn) this.shiftRightBtn.classList.toggle('active', active);
    this.updateKeyLabels();
  }

  updateKeyLabels() {
    const isUpper = (this.capsLock !== this.shift);
    const keyButtons = this.keyboardContainer.querySelectorAll('.key-btn');
    keyButtons.forEach(btn => {
      const key = btn.getAttribute('data-key');
      if (key && key.length === 1 && key.match(/[a-z]/i)) {
        btn.textContent = isUpper ? key.toUpperCase() : key.toLowerCase();
      }
    });
  }

  handleHardwareKeyDown(e) {
    // If user is typing in select or non-main input, skip
    if (e.target && e.target.tagName === 'SELECT') return;

    // Visual ripple on corresponding virtual key
    const btn = this.findKeyElement(e);
    if (btn) {
      btn.classList.add('pressed', 'hardware-ripple');
      const isSpecial = e.code === 'Backspace' || e.code === 'Enter' || e.code === 'Space' || e.code === 'Tab';
      this.playKeyClick(isSpecial);
    }

    if (e.key === 'CapsLock') {
      this.toggleCapsLock(e.getModifierState('CapsLock'));
    }
    if (e.key === 'Shift') {
      this.toggleShift(true);
    }
  }

  handleHardwareKeyUp(e) {
    const btn = this.findKeyElement(e);
    if (btn) {
      btn.classList.remove('pressed');
      setTimeout(() => btn.classList.remove('hardware-ripple'), 300);
    }

    if (e.key === 'Shift') {
      this.toggleShift(false);
    }
  }

  findKeyElement(e) {
    // Match by code first (e.g. KeyQ, Space, Digit1)
    let btn = this.keyboardContainer.querySelector(`.key-btn[data-code="${e.code}"]`);
    if (!btn) {
      btn = this.keyboardContainer.querySelector(`.key-btn[data-key="${e.key.toLowerCase()}"]`);
    }
    return btn;
  }
}

window.VirtualKeyboard = VirtualKeyboard;
