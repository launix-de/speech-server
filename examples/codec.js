(function (root, factory) {
  if (typeof module === "object" && module.exports) {
    module.exports = factory();
  } else {
    root.AudioCodec = factory();
  }
})(typeof self !== "undefined" ? self : this, function () {
  const FRAME_SAMPLES = 1024;
  const SAMPLE_RATE = 48_000;
  const FFT_SIZE = 1024;
  const HEADER_SIZE = 12;
  const VERSION = 2;

  // --- Bit-weight functions (ISO 226 psychoacoustic weighting) ---

  function lowBits(freq) {
    if (freq < 50) return 5;
    if (freq < 125) return 12;
    if (freq < 250) return 11;
    if (freq < 500) return 10;
    if (freq < 1000) return 9;
    if (freq < 3000) return 8;
    if (freq < 7000) return 7;
    if (freq < 9000) return 6;
    if (freq < 13000) return 5;
    return 4;
  }

  function mediumBits(freq) {
    if (freq < 50) return 7;
    if (freq < 125) return 14;
    if (freq < 250) return 13;
    if (freq < 500) return 12;
    if (freq < 1000) return 11;
    if (freq < 3000) return 10;
    if (freq < 7000) return 9;
    if (freq < 9000) return 8;
    if (freq < 13000) return 7;
    return 6;
  }

  function highBits(freq) {
    if (freq < 50) return 9;
    if (freq < 125) return 16;
    if (freq < 250) return 15;
    if (freq < 500) return 14;
    if (freq < 1000) return 13;
    if (freq < 3000) return 12;
    if (freq < 7000) return 11;
    if (freq < 9000) return 10;
    if (freq < 13000) return 9;
    return 8;
  }

  function fullBits() {
    return 16;
  }

  // --- Profile definitions ---

  function buildWeights(binCount, bitFn) {
    const w = new Uint8Array(binCount);
    for (let i = 0; i < binCount; i++) {
      w[i] = bitFn((i * SAMPLE_RATE) / FFT_SIZE);
    }
    return w;
  }

  const PROFILES = {};
  [
    { name: "low",    binCount: 160, profileId: 0, bitFn: lowBits },
    { name: "medium", binCount: 256, profileId: 1, bitFn: mediumBits },
    { name: "high",   binCount: 384, profileId: 2, bitFn: highBits },
    { name: "full",   binCount: 512, profileId: 3, bitFn: fullBits },
  ].forEach(function (def) {
    const weights = buildWeights(def.binCount, def.bitFn);
    let totalBits = 0;
    for (let i = 0; i < weights.length; i++) totalBits += weights[i] * 2;
    PROFILES[def.name] = {
      name: def.name,
      binCount: def.binCount,
      profileId: def.profileId,
      weights: weights,
      totalBits: totalBits,
      payloadBytes: Math.ceil(totalBits / 8),
    };
  });

  // Reverse lookup: profileId -> profile
  const PROFILES_BY_ID = {};
  Object.keys(PROFILES).forEach(function (k) {
    PROFILES_BY_ID[PROFILES[k].profileId] = PROFILES[k];
  });

  // --- FFT (Cooley-Tukey, radix-2) ---

  const bitRev = new Uint16Array(FFT_SIZE);
  (function buildBitRev() {
    const bits = Math.log2(FFT_SIZE) | 0;
    for (let i = 0; i < FFT_SIZE; i++) {
      let j = 0;
      for (let k = 0; k < bits; k++) {
        if (i & (1 << k)) j |= 1 << (bits - 1 - k);
      }
      bitRev[i] = j;
    }
  })();

  function fft(real, imag, invert) {
    const n = real.length;
    for (let i = 0; i < n; i++) {
      const j = bitRev[i];
      if (i < j) {
        let t = real[i]; real[i] = real[j]; real[j] = t;
        t = imag[i]; imag[i] = imag[j]; imag[j] = t;
      }
    }
    for (let len = 2; len <= n; len <<= 1) {
      const ang = (2 * Math.PI) / len * (invert ? -1 : 1);
      const wLenR = Math.cos(ang), wLenI = Math.sin(ang);
      for (let i = 0; i < n; i += len) {
        let wR = 1, wI = 0;
        for (let j = 0; j < len / 2; j++) {
          const uR = real[i + j], uI = imag[i + j];
          const k = i + j + len / 2;
          const vR = real[k] * wR - imag[k] * wI;
          const vI = real[k] * wI + imag[k] * wR;
          real[i + j] = uR + vR; imag[i + j] = uI + vI;
          real[k] = uR - vR; imag[k] = uI - vI;
          const nR = wR * wLenR - wI * wLenI;
          wI = wR * wLenI + wI * wLenR;
          wR = nR;
        }
      }
    }
    if (invert) {
      for (let i = 0; i < n; i++) { real[i] /= n; imag[i] /= n; }
    }
  }

  // --- Bit I/O ---

  function writeBits(buffer, baseOffset, bitIndex, value, bits) {
    for (let i = bits - 1; i >= 0; i--) {
      const bit = (value >> i) & 1;
      const byteOff = baseOffset + (bitIndex >> 3);
      buffer[byteOff] |= bit << (7 - (bitIndex & 7));
      bitIndex++;
    }
    return bitIndex;
  }

  function readBits(buffer, baseOffset, bitIndex, bits) {
    let value = 0;
    for (let i = 0; i < bits; i++) {
      const byteOff = baseOffset + ((bitIndex + i) >> 3);
      const bit = (buffer[byteOff] >> (7 - ((bitIndex + i) & 7))) & 1;
      value = (value << 1) | bit;
    }
    return value;
  }

  // --- Encode / Decode ---

  let frameCounter = 0;

  function encodeFrame(frameSamples, profile) {
    profile = profile || "low";
    const prof = PROFILES[profile];
    if (!prof) throw new Error("Unknown profile: " + profile);
    if (frameSamples.length !== FRAME_SAMPLES) {
      throw new Error("encodeFrame expects " + FRAME_SAMPLES + " samples");
    }

    const real = new Float32Array(FFT_SIZE);
    const imag = new Float32Array(FFT_SIZE);
    real.set(frameSamples);
    fft(real, imag, false);

    let maxAbs = 0;
    for (let i = 0; i < prof.binCount; i++) {
      const a = Math.max(Math.abs(real[i]), Math.abs(imag[i]));
      if (a > maxAbs) maxAbs = a;
    }
    if (maxAbs < 1e-9) maxAbs = 1e-9;

    const buffer = new Uint8Array(HEADER_SIZE + prof.payloadBytes);
    const view = new DataView(buffer.buffer, buffer.byteOffset, buffer.byteLength);

    view.setUint8(0, VERSION);
    view.setUint8(1, prof.binCount & 0xFF);
    view.setUint8(2, prof.profileId);
    view.setUint8(3, 0);
    view.setFloat32(4, maxAbs, true);
    view.setUint32(8, frameCounter++ >>> 0, true);

    let bitIndex = 0;
    for (let i = 0; i < prof.binCount; i++) {
      const bits = prof.weights[i];
      const maxQuant = (1 << bits) - 1;
      const r = Math.max(-maxAbs, Math.min(maxAbs, real[i]));
      const im = Math.max(-maxAbs, Math.min(maxAbs, imag[i]));
      const rq = Math.round(((r / maxAbs) + 1) * 0.5 * maxQuant);
      const iq = Math.round(((im / maxAbs) + 1) * 0.5 * maxQuant);
      bitIndex = writeBits(buffer, HEADER_SIZE, bitIndex, Math.max(0, Math.min(maxQuant, rq)), bits);
      bitIndex = writeBits(buffer, HEADER_SIZE, bitIndex, Math.max(0, Math.min(maxQuant, iq)), bits);
    }

    return buffer;
  }

  function decodeFrame(encoded) {
    if (!(encoded instanceof Uint8Array)) {
      encoded = new Uint8Array(encoded);
    }
    if (encoded.length < HEADER_SIZE) {
      throw new Error("Encoded frame too small");
    }
    const view = new DataView(encoded.buffer, encoded.byteOffset, encoded.byteLength);
    const version = view.getUint8(0);
    if (version !== VERSION) {
      throw new Error("Unsupported codec version " + version);
    }
    // Header byte 1 only holds binCount & 0xFF which wraps at 256,
    // so always prefer the profile lookup via profileId (byte 2).
    const profileId = view.getUint8(2);
    const scale = view.getFloat32(4, true);

    const prof = PROFILES_BY_ID[profileId] || PROFILES_BY_ID[0];
    const count = prof.binCount;

    const resultReal = new Float32Array(FFT_SIZE);
    const resultImag = new Float32Array(FFT_SIZE);

    let bitIndex = 0;
    for (let i = 0; i < count; i++) {
      const bits = prof.weights[i];
      const maxQuant = (1 << bits) - 1;
      const rq = readBits(encoded, HEADER_SIZE, bitIndex, bits);
      bitIndex += bits;
      const iq = readBits(encoded, HEADER_SIZE, bitIndex, bits);
      bitIndex += bits;

      const r = ((rq / maxQuant) * 2 - 1) * scale;
      const im = ((iq / maxQuant) * 2 - 1) * scale;

      resultReal[i] = r;
      resultImag[i] = im;
      if (i !== 0) {
        const mirror = FFT_SIZE - i;
        resultReal[mirror] = r;
        resultImag[mirror] = -im;
      }
    }

    fft(resultReal, resultImag, true);
    return resultReal.subarray(0, FRAME_SAMPLES);
  }

  function frameSizeBytes(profile) {
    profile = profile || "low";
    const prof = PROFILES[profile];
    if (!prof) throw new Error("Unknown profile: " + profile);
    return HEADER_SIZE + prof.payloadBytes;
  }

  // --- Plug-and-play mic capture & playback ---

  var FRAME_DURATION = FRAME_SAMPLES / SAMPLE_RATE;

  /**
   * Open microphone, encode frames with the given profile, and send
   * as binary messages on the WebSocket.
   *
   * Returns a handle with:
   *   .close()           — stop capture and release mic
   *   .muted (get/set)   — mute toggle (frames are not sent while muted)
   *   .rms               — current mic RMS level (0..1), updated per frame
   *   .speaking           — true if RMS > threshold
   *   .txFrames          — total frames sent
   *   .txBytes           — total bytes sent
   *   .onrms(rms)        — callback fired each frame with the current RMS
   *
   * @param {WebSocket} ws
   * @param {string} [profile='low']
   * @returns {Promise<object>} resolves when mic is ready
   */
  function openMic(ws, profile) {
    profile = profile || 'low';
    var handle = {
      muted: false,
      rms: 0,
      speaking: false,
      txFrames: 0,
      txBytes: 0,
      onrms: null,
      close: function () { cleanup(); },
    };
    var mediaStream = null;
    var audioCtx = null;
    var scriptNode = null;
    var analyser = null;
    var levelRaf = null;
    var closed = false;

    function updateLevel() {
      if (!analyser || closed) return;
      var buf = new Uint8Array(analyser.fftSize);
      analyser.getByteTimeDomainData(buf);
      var sum = 0;
      for (var i = 0; i < buf.length; i++) {
        var v = (buf[i] - 128) / 128;
        sum += v * v;
      }
      handle.rms = Math.sqrt(sum / buf.length);
      handle.speaking = handle.rms > 0.02;
      if (handle.onrms) handle.onrms(handle.rms);
      if (!closed) levelRaf = requestAnimationFrame(updateLevel);
    }

    function cleanup() {
      closed = true;
      if (levelRaf) cancelAnimationFrame(levelRaf);
      if (scriptNode) { scriptNode.disconnect(); scriptNode = null; }
      if (analyser) { analyser.disconnect(); analyser = null; }
      if (mediaStream) { mediaStream.getTracks().forEach(function (t) { t.stop(); }); mediaStream = null; }
      if (audioCtx) { audioCtx.close(); audioCtx = null; }
    }

    var ready = navigator.mediaDevices.getUserMedia({
      audio: { channelCount: 1, echoCancellation: false, noiseSuppression: false }
    }).then(function (stream) {
      if (closed) { stream.getTracks().forEach(function (t) { t.stop(); }); return handle; }
      mediaStream = stream;
      audioCtx = new AudioContext({ sampleRate: SAMPLE_RATE });
      var source = audioCtx.createMediaStreamSource(stream);

      analyser = audioCtx.createAnalyser();
      analyser.fftSize = 512;
      source.connect(analyser);

      scriptNode = audioCtx.createScriptProcessor(FRAME_SAMPLES, 1, 1);
      scriptNode.onaudioprocess = function (e) {
        if (closed || !ws || ws.readyState !== WebSocket.OPEN || handle.muted) return;
        var float32 = e.inputBuffer.getChannelData(0);
        var encoded = encodeFrame(float32, profile);
        ws.send(encoded);
        handle.txFrames++;
        handle.txBytes += encoded.byteLength;
      };
      source.connect(scriptNode);
      scriptNode.connect(audioCtx.destination);
      updateLevel();
      return handle;
    });

    handle._ready = ready;
    return ready;
  }

  /**
   * Open a playback channel: receive binary WebSocket messages,
   * decode as codec frames, and schedule for playback.
   *
   * Returns a handle with:
   *   .close()           — stop playback, remove listener
   *   .rxFrames          — total frames received
   *   .rxBytes           — total bytes received
   *   .onframe(samples)  — callback fired with decoded Float32Array per frame
   *
   * @param {WebSocket} ws
   * @returns {object}
   */
  function openSpeaker(ws) {
    var audioCtx = new AudioContext({ sampleRate: SAMPLE_RATE });
    var nextPlayTime = 0;
    var closed = false;

    var handle = {
      rxFrames: 0,
      rxBytes: 0,
      onframe: null,
      close: function () {
        closed = true;
        ws.removeEventListener('message', onMessage);
        audioCtx.close();
      },
    };

    function onMessage(evt) {
      if (closed) return;
      if (typeof evt.data === 'string') return;
      handle.rxFrames++;
      handle.rxBytes += evt.data.byteLength;
      try {
        var samples = decodeFrame(new Uint8Array(evt.data));
        // Schedule playback
        var buffer = audioCtx.createBuffer(1, samples.length, SAMPLE_RATE);
        buffer.getChannelData(0).set(samples);
        var src = audioCtx.createBufferSource();
        src.buffer = buffer;
        src.connect(audioCtx.destination);
        var now = audioCtx.currentTime;
        if (nextPlayTime < now) nextPlayTime = now;
        src.start(nextPlayTime);
        nextPlayTime += FRAME_DURATION;
        if (handle.onframe) handle.onframe(samples);
      } catch (e) {
        // skip bad frames
      }
    }

    ws.addEventListener('message', onMessage);
    return handle;
  }

  /**
   * Compute RMS of a Float32Array (e.g. a decoded frame).
   * @param {Float32Array} samples
   * @returns {number} RMS in [0, 1]
   */
  function computeRMS(samples) {
    var sum = 0;
    for (var i = 0; i < samples.length; i++) {
      sum += samples[i] * samples[i];
    }
    return Math.sqrt(sum / samples.length);
  }

  /**
   * Compute peak level of a Float32Array.
   * @param {Float32Array} samples
   * @returns {number} peak absolute value in [0, 1]
   */
  function computePeak(samples) {
    var peak = 0;
    for (var i = 0; i < samples.length; i++) {
      var a = Math.abs(samples[i]);
      if (a > peak) peak = a;
    }
    return peak;
  }

  /**
   * Convert RMS or peak level to decibels (dBFS).
   * @param {number} level — linear level (0..1)
   * @returns {number} dBFS (0 dB = full scale, -Infinity for silence)
   */
  function levelToDb(level) {
    if (level <= 0) return -Infinity;
    return 20 * Math.log10(level);
  }

  /**
   * Join a conference: open WebSocket, start mic + speaker in one call.
   * @param {string} wsUrl — full WebSocket URL (wss://host/ws/phone/nonce)
   * @param {object} [opts] — optional settings
   * @param {string} [opts.profile='low'] — codec profile
   * @param {function} [opts.onstate] — called with state string ('connecting','ready','connected','disconnected','error')
   * @param {function} [opts.onrms] — called with mic RMS level
   * @param {function} [opts.onmessage] — called with parsed JSON control messages from server
   * @returns {Promise<{close:function, ws:WebSocket, mic:object, speaker:object}>}
   */
  function joinConference(wsUrl, opts) {
    opts = opts || {};
    var profile = opts.profile || 'medium';
    var onstate = opts.onstate || function () {};
    var onrms = opts.onrms || null;
    var onmessage = opts.onmessage || null;

    onstate('connecting');

    var ws = new WebSocket(wsUrl);
    ws.binaryType = 'arraybuffer';

    return new Promise(function (resolve, reject) {
      var mic = null, speaker = null, closed = false;

      var handle = {
        ws: ws,
        mic: null,
        speaker: null,
        close: function () {
          if (closed) return;
          closed = true;
          if (handle.mic) { handle.mic.close(); handle.mic = null; }
          if (handle.speaker) { handle.speaker.close(); handle.speaker = null; }
          if (ws.readyState <= 1) {
            try { ws.send(JSON.stringify({hangup: true})); } catch(e) {}
            ws.close();
          }
          onstate('disconnected');
        },
      };

      ws.onopen = function () {
        onstate('ready');
        // 1. Tell server we want to join (server builds pipeline + codec session)
        ws.send(JSON.stringify({accept: true}));
        // 2. Send codec handshake (server's CodecSocketSession expects this)
        ws.send(JSON.stringify({type: 'hello', profiles: [profile]}));
      };

      ws.onmessage = function (evt) {
        if (typeof evt.data !== 'string') return; // binary = audio, handled by speaker
        try {
          var msg = JSON.parse(evt.data);
          if (msg.hangup) { handle.close(); return; }

          // 2. Server's codec session sends hello response → start mic + speaker
          if (msg.type === 'hello' && !handle.mic) {
            handle.speaker = openSpeaker(ws);
            openMic(ws, msg.profile || profile).then(function (m) {
              if (closed) { m.close(); return; }
              handle.mic = m;
              if (onrms) m.onrms = onrms;
              onstate('connected');
              resolve(handle);
            }).catch(function (e) {
              onstate('error');
              reject(e);
            });
            return;
          }

          if (onmessage) onmessage(msg);
        } catch(e) {}
      };

      ws.onerror = function () {
        onstate('error');
        if (!handle.mic) reject(new Error('WebSocket error'));
      };

      ws.onclose = function () {
        if (!closed) handle.close();
      };
    });
  }

  return {
    FRAME_SAMPLES,
    SAMPLE_RATE,
    FFT_SIZE,
    HEADER_SIZE,
    VERSION,
    PROFILES,
    PROFILES_BY_ID,
    encodeFrame,
    decodeFrame,
    frameSizeBytes,
    openMic,
    openSpeaker,
    computeRMS,
    computePeak,
    levelToDb,
    joinConference,
  };
});
