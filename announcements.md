# speech-server Launch Announcements

---

## Hacker News (Show HN)

**Title:** Show HN: Open-source self-hosted telephony platform with composable audio DSL

**Body:**

I've been building an open-source telephony platform that replaces cloud services like Twilio for self-hosted deployments. It handles SIP conferencing, real-time TTS/STT, voice conversion, and call routing -- all through a composable DSL:

```
sip:leg1{"completed":"/cb"} -> tee:tap -> call:conf1 -> sip:leg1
tee:tap -> stt:de -> webhook:https://crm.example.com/transcript
play:hold{"url":"music.mp3","loop":true} -> call:conf1
```

One API endpoint, one DSL syntax for everything: bridging SIP legs, playing hold music, streaming TTS into a conference, attaching STT sidechains, or rendering offline audio.

**What it does:**

- Multi-tenant SIP conferencing with PCMU, PCMA, G722, Opus codec negotiation
- N-to-M conference mixer with per-participant mix-minus (echo cancellation)
- Real-time Whisper STT with webhook delivery
- Piper TTS with streaming synthesis and voice conversion (FreeVC)
- Browser participants via custom FFT codec over WebSocket
- Webhook-driven call control (designed for CRM integration)
- Account isolation, PBX pinning, subscriber management

**What it doesn't do:** No hosted service, no per-minute billing. You run it on your own hardware. A single node handles dozens of concurrent calls with STT+TTS.

The conference mixer runs at 48kHz internally. SIP codecs are negotiated per-leg and sample rate conversion is automatic -- an Opus webclient and a PCMU desk phone in the same conference just works.

Stack: Python, Flask, pyVoIP, faster-whisper, Piper ONNX, PyAV (G722/Opus). ~430 tests including end-to-end RTP audio quality checks with cross-correlation similarity measurement.

GPLv3 licensed. We use it in production for a German CRM system handling inbound/outbound calls with hold, transfer, STT transcription, and browser-based participation.

GitHub: https://github.com/launix-de/speech-server

---

## r/selfhosted

**Title:** I built an open-source, self-hosted alternative to Twilio for voice calls -- with real-time transcription and text-to-speech

**Body:**

Hey r/selfhosted,

I got tired of per-minute charges and sending customer audio to cloud APIs, so I built **speech-server** -- a self-hosted telephony platform that handles everything Twilio Voice does, but runs on your own box.

**What it does:**

- **SIP conferencing** -- connect desk phones, softphones, or browser clients in the same call
- **Real-time transcription** -- Whisper STT runs locally, transcripts delivered via webhook
- **Text-to-speech** -- Piper voices, streaming synthesis, voice conversion
- **Hold music, transfers, multi-party** -- the full call center experience
- **Browser phone** -- participants join from Chrome via WebSocket (no plugin needed)
- **Webhook-driven** -- your app controls call flow via REST API + callbacks

**What it replaces:**

| | Twilio | speech-server |
|---|---|---|
| Cost | $0.02/min STT + $0.0085/min TTS | Your compute cost |
| Privacy | Audio goes to cloud | Everything stays on your server |
| Latency | Cloud round-trip | Local, sub-100ms |
| Customization | Limited API | Full DSL + stage pipeline |

**How it works:**

Everything is a pipeline described in a simple DSL:

```
# Bridge a SIP call with live transcription
sip:leg1 -> tee:tap -> call:conference1 -> sip:leg1
tee:tap -> stt:de -> webhook:https://mycrm.local/transcript

# Play hold music
play:hold{"url":"hold.mp3","loop":true} -> call:conference1

# TTS announcement
tts:de{"text":"Please hold, connecting you now."} -> call:conference1
```

Runs with pm2, works behind Apache/nginx reverse proxy. We run it in production for a German ERP system with real phone calls daily.

**Tech:** Python, GPLv3 licensed, ~430 tests, supports PCMU/PCMA/G722/Opus codecs.

GitHub: https://github.com/launix-de/speech-server

Happy to answer questions!

---

## r/Python

**Title:** speech-server: composable audio pipeline library with automatic format conversion -- TTS, STT, SIP conferencing, voice conversion

**Body:**

I've been working on a Python library for real-time audio processing that uses a pipe-based architecture similar to UNIX pipes. Stages snap together with `.pipe()` and format conversion (sample rate, encoding) is inserted automatically:

```python
source = TTSProducer(voice, syn, text="Hello!")
pipeline = (
    source
    .pipe(VCConverter("target_voice.wav"))    # voice conversion
    .pipe(PitchAdjuster(pitch_override_st=2.0))  # pitch shift
    .pipe(FileRecorder("output.wav", sample_rate=24000))
)
pipeline.run()
```

Or as a DSL string:

```python
# CLI
echo "Hello" | speech-pipeline run "cli:text | tts:en_US-amy-medium | cli:raw" > out.raw
```

The interesting part architecturally: every `Stage` declares `input_format` and `output_format` (sample rate + encoding). When you call `a.pipe(b)`, it inspects both formats and auto-inserts `SampleRateConverter` or `EncodingConverter` stages as needed. So you can pipe a 48kHz Opus SIP leg into a 16kHz Whisper STT without thinking about resampling.

**What's in the box:**

- **TTS:** Piper ONNX (streaming, multi-voice)
- **STT:** faster-whisper (real-time, chunked)
- **Voice Conversion:** FreeVC
- **SIP telephony:** Full conferencing platform with mix-minus, codec negotiation (Opus/G722/PCMU/PCMA), webhook-driven call control
- **Browser audio:** Custom FFT codec over WebSocket
- **REST API:** Create pipelines via DSL, render TTS as WAV, stream text into live conferences

The pipeline DSL supports both `|` and `->` separators with inline JSON params:

```
sip:leg1{"completed":"/callback"} -> tee:tap -> call:conference1 -> sip:leg1
tee:tap -> stt:de -> webhook:https://example.com/transcript
text_input | tts:de_DE-thorsten-medium | vc:target_voice | conference:call1
```

One parser, one API endpoint for everything.

~430 tests including end-to-end audio quality checks -- we pump real audio (MP3) through RTP sessions and measure output similarity via cross-correlation. Every codec (PCMU, PCMA, G722, Opus) is tested individually and in cross-codec conferences.

GPLv3 licensed, production-tested. GitHub: https://github.com/launix-de/speech-server

---

## r/VOIP

**Title:** Open-source SIP conferencing server with built-in TTS, STT, and browser participants -- looking for feedback

**Body:**

I've been building an open-source SIP conferencing platform and would love feedback from this community.

**The pitch:** A self-hosted server that handles SIP call routing, multi-party conferencing with mix-minus, and integrates TTS/STT natively. Everything is controlled via REST API -- designed for CRM/PBX integration, not as a standalone PBX.

**SIP specifics:**

- Built-in SIP stack (UDP signaling) + pyVoIP for inbound/outbound
- Codec negotiation: Opus (48kHz), G722 (16kHz), PCMU, PCMA
- Conference mixer at 48kHz with automatic sample rate conversion per leg
- Mix-minus per participant (A doesn't hear A's own audio)
- DTMF detection
- SIP REGISTER as trunk client (tested with VoIPXS/gnTel)
- SIP Registrar for local device auth (WebRTC-style browser clients)

**Call control via DSL:**

```
# Bridge inbound leg into conference (bidirectional)
sip:leg-abc{"completed":"/webhook/done"} -> call:call-xyz -> sip:leg-abc

# Hold: detach leg, play music directly to it
play:hold{"url":"hold.mp3","loop":true,"volume":50} -> sip:leg-abc

# STT sidechain (tee splits audio without affecting the main path)
sip:leg-abc -> tee:tap -> call:call-xyz -> sip:leg-abc
tee:tap -> stt:de -> webhook:https://crm.example.com/transcript
```

**What it's NOT:** Not a PBX replacement. No IVR menus, no voicemail, no call queues. It's the audio engine that a PBX or CRM talks to via REST API. Think of it as "Twilio Voice but self-hosted."

**Current limitations:**

- No oRTP/oSIP -- custom lightweight SIP stack, good enough for trunk registration and basic call signaling but not a full SIP proxy
- No oLEGACY -- no T.38 fax, no SRTP (DTLS-SRTP in progress), no ICE/STUN
- Single-node only (no cluster mode yet, though the architecture supports it via codec bridges)

We use it in production with a German CRM system. Handles inbound/outbound calls, hold/transfer, live transcription, and browser-based participation.

GPLv3 licensed. GitHub: https://github.com/launix-de/speech-server

What are the biggest gaps you see for your use cases?

---

## r/opensource

**Title:** speech-server -- open-source self-hosted voice platform (TTS, STT, SIP conferencing, voice conversion) -- GPLv3 licensed

**Body:**

Releasing **speech-server**, an open-source toolkit for building voice applications. It combines text-to-speech, speech-to-text, SIP telephony, and audio processing into a single self-hosted platform.

**Why we built it:** We needed voice capabilities for a CRM system but didn't want to depend on Twilio (cost, privacy, vendor lock-in). Asterisk was too complex for our use case. So we built a modern Python-based alternative with a simple DSL for describing audio pipelines.

**Core capabilities:**

- Multi-voice TTS (Piper ONNX models)
- Real-time STT (faster-whisper)
- Voice conversion (FreeVC) and pitch shifting
- SIP conferencing with multi-codec support (Opus, G722, G.711)
- Browser participation via WebSocket
- REST API for programmatic call control
- Webhook-driven CRM integration

**Example -- describe a complete call flow in 3 lines:**

```
sip:inbound_leg -> tee:recorder -> call:conference -> sip:inbound_leg
tee:recorder -> stt:de -> webhook:https://myapp.com/transcript
play:hold_music{"url":"music.mp3","loop":true} -> call:conference
```

**Tech stack:** Python 3.10+, Flask, pyVoIP, faster-whisper, Piper, PyAV. GPLv3 licensed.

430+ tests including end-to-end audio quality verification through real RTP sessions.

GitHub: https://github.com/launix-de/speech-server

Contributions welcome -- especially around WebRTC support, additional codec implementations, and IVR building blocks.
