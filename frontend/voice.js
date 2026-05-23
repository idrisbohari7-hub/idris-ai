/* ================================================================
   VOICE ASSISTANT — ChatGPT-style voice with barge-in
   Continuous listening in voice mode; interrupt TTS + LLM stream
   when the user speaks while the assistant is talking.
   ================================================================ */

class VoiceAssistant {
  constructor(deps) {
    this.deps = deps;
    this.enabled = false;
    this.state = "idle"; // idle | listening | processing | speaking
    this._vadStream = null;
    this._vadContext = null;
    this._vadAnalyser = null;
    this._vadRaf = null;
    this._interruptCooldown = false;
    this._pendingTranscript = "";
    this._speechRestartTimer = null;
  }

  get isActive() {
    return (
      this.enabled &&
      (this.state === "listening" ||
        this.state === "processing" ||
        this.state === "speaking")
    );
  }

  setState(next) {
    this.state = next;
    const el = this.deps.voiceStatusEl;
    if (!el) return;
    const labels = {
      idle: "",
      listening: "Listening…",
      processing: "Thinking…",
      speaking: "Speaking — interrupt anytime",
    };
    el.textContent = this.enabled ? labels[next] || "" : "";
    el.classList.toggle("active", this.enabled && next !== "idle");
  }

  async enable() {
    if (this.enabled) return;
    this.enabled = true;
    const { ttsPlayer, voiceModeBtn, recognition } = this.deps;
    if (ttsPlayer && !ttsPlayer.enabled) {
      ttsPlayer.enabled = true;
      if (this.deps.ttsBtn) this.deps.ttsBtn.classList.add("tts-active");
    }
    if (voiceModeBtn) voiceModeBtn.classList.add("voice-mode-active");
    if (ttsPlayer) {
      ttsPlayer.unlock();
    }
    if (!recognition) {
      this.setState("idle");
      if (this.deps.voiceStatusEl)
        this.deps.voiceStatusEl.textContent = "Speech not supported in this browser";
      return;
    }
    recognition.continuous = true;
    recognition.interimResults = true;
    this.setState("listening");
    this._startRecognition();
    this._startVadMonitor();
  }

  disable() {
    this.enabled = false;
    this._stopVadMonitor();
    this._clearSpeechRestart();
    if (this.deps.voiceModeBtn)
      this.deps.voiceModeBtn.classList.remove("voice-mode-active");
    this.setState("idle");
    this._stopRecognition();
    if (this.deps.onInterrupt) this.deps.onInterrupt(false);
  }

  toggle() {
    if (this.enabled) this.disable();
    else this.enable();
  }

  /** Called when assistant starts streaming a reply */
  onProcessingStart() {
    if (!this.enabled) return;
    this.setState("processing");
    this._stopRecognition();
  }

  /** Called when TTS starts playing */
  onSpeakingStart() {
    if (!this.enabled) return;
    this.setState("speaking");
    this._startVadMonitor();
    this._startRecognitionForInterrupt();
  }

  /** Called when response fully done (stream + TTS queue drained) */
  onTurnComplete() {
    if (!this.enabled) return;
    this._stopVadMonitor();
    this.setState("listening");
    this._startRecognition();
  }

  /** User or VAD detected speech during assistant output */
  handleInterrupt(source) {
    if (!this.enabled || this._interruptCooldown) return;
    const { isStreaming, ttsPlayer, onInterrupt } = this.deps;
    const speaking = ttsPlayer && (ttsPlayer.playing || ttsPlayer.queue.length > 0);
    if (!isStreaming && !speaking && this.state !== "processing") return;

    this._interruptCooldown = true;
    setTimeout(() => {
      this._interruptCooldown = false;
    }, 600);

    if (onInterrupt) onInterrupt(true);
    if (ttsPlayer) ttsPlayer.stop();
    this.setState("listening");
    this._startRecognition();
  }

  handleSpeechResult(event) {
    if (!this.enabled) return;
    const result = event.results[event.results.length - 1];
    const text = result[0].transcript.trim();
    if (!text) return;

    const { isStreaming, ttsPlayer, onSend, onInterrupt } = this.deps;
    const speaking = ttsPlayer && (ttsPlayer.playing || ttsPlayer.queue.length > 0);

    if (isStreaming || speaking || this.state === "processing") {
      if (!result.isFinal) {
        this.handleInterrupt("speech-interim");
        return;
      }
      if (onInterrupt) onInterrupt(true);
      if (ttsPlayer) ttsPlayer.stop();
      this._pendingTranscript = text;
      this._stopRecognition();
      setTimeout(() => {
        if (this._pendingTranscript && onSend) {
          onSend(this._pendingTranscript);
          this._pendingTranscript = "";
        }
      }, 150);
      return;
    }

    if (this.deps.messageInput) {
      this.deps.messageInput.value = text;
      if (this.deps.autoResizeInput) this.deps.autoResizeInput();
    }

    if (result.isFinal) {
      this._stopRecognition();
      if (onSend) onSend(text);
    }
  }

  _startRecognition() {
    const { recognition, isStreaming } = this.deps;
    if (!recognition || !this.enabled || isStreaming) return;
    this._clearSpeechRestart();
    try {
      recognition.stop();
    } catch (_) {}
    this._speechRestartTimer = setTimeout(() => {
      if (!this.enabled) return;
      try {
        recognition.start();
      } catch (e) {
        if (e.name === "InvalidStateError") {
          /* already running */
        }
      }
    }, 120);
  }

  _startRecognitionForInterrupt() {
    const { recognition } = this.deps;
    if (!recognition || !this.enabled) return;
    try {
      recognition.start();
    } catch (_) {}
  }

  _stopRecognition() {
    const { recognition } = this.deps;
    this._clearSpeechRestart();
    if (!recognition) return;
    try {
      recognition.stop();
    } catch (_) {}
  }

  _clearSpeechRestart() {
    if (this._speechRestartTimer) {
      clearTimeout(this._speechRestartTimer);
      this._speechRestartTimer = null;
    }
  }

  async _startVadMonitor() {
    if (this._vadRaf) return;
    const { isStreaming, ttsPlayer } = this.deps;
    const speaking = ttsPlayer && (ttsPlayer.playing || ttsPlayer.queue.length > 0);
    if (!isStreaming && !speaking) return;

    try {
      if (!this._vadStream) {
        this._vadStream = await navigator.mediaDevices.getUserMedia({
          audio: {
            echoCancellation: true,
            noiseSuppression: true,
            autoGainControl: true,
          },
        });
        this._vadContext = new (window.AudioContext || window.webkitAudioContext)();
        const source = this._vadContext.createMediaStreamSource(this._vadStream);
        this._vadAnalyser = this._vadContext.createAnalyser();
        this._vadAnalyser.fftSize = 512;
        this._vadAnalyser.smoothingTimeConstant = 0.4;
        source.connect(this._vadAnalyser);
      }
    } catch (e) {
      console.warn("VAD mic unavailable:", e);
      return;
    }

    const data = new Uint8Array(this._vadAnalyser.frequencyBinCount);
    let loudFrames = 0;
    const threshold = 42;
    const requiredFrames = 8;

    const tick = () => {
      if (!this.enabled || !this._vadAnalyser) {
        this._stopVadMonitor();
        return;
      }
      const { isStreaming: streaming, ttsPlayer: tts } = this.deps;
      const stillSpeaking =
        tts && (tts.playing || tts.queue.length > 0);
      if (!streaming && !stillSpeaking) {
        this._stopVadMonitor();
        return;
      }

      this._vadAnalyser.getByteFrequencyData(data);
      let sum = 0;
      for (let i = 0; i < data.length; i++) sum += data[i];
      const avg = sum / data.length;

      if (avg > threshold) {
        loudFrames++;
        if (loudFrames >= requiredFrames) {
          this.handleInterrupt("vad");
          loudFrames = 0;
        }
      } else {
        loudFrames = Math.max(0, loudFrames - 1);
      }
      this._vadRaf = requestAnimationFrame(tick);
    };
    this._vadRaf = requestAnimationFrame(tick);
  }

  _stopVadMonitor() {
    if (this._vadRaf) {
      cancelAnimationFrame(this._vadRaf);
      this._vadRaf = null;
    }
  }

  destroy() {
    this.disable();
    if (this._vadStream) {
      this._vadStream.getTracks().forEach((t) => t.stop());
      this._vadStream = null;
    }
    if (this._vadContext) {
      this._vadContext.close().catch(() => {});
      this._vadContext = null;
    }
    this._vadAnalyser = null;
  }
}

window.VoiceAssistant = VoiceAssistant;
