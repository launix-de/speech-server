# speech-pipeline

A composable, real-time speech processing toolkit for Python. Build text-to-speech, speech-to-text, voice conversion, and telephony pipelines from simple building blocks -- as a library, HTTP server, or CLI tool.

Stages snap together like UNIX pipes. Format conversion (sample rate, encoding) is automatic. Streaming is the default: audio plays as it is synthesized, transcriptions arrive as words are spoken.

```
echo "Hallo Welt" | speech-pipeline run "cli:text | tts:de_DE-thorsten-medium | cli:raw" > out.raw
```

## Key Features

**Text-to-Speech** -- Multi-voice TTS via [Piper](https://github.com/rhasspy/piper) ONNX models with automatic voice discovery, streaming synthesis, and configurable speed/pitch/noise parameters.

**Speech-to-Text** -- Real-time transcription via [faster-whisper](https://github.com/SYSTRAN/faster-whisper) with pause-based chunking, multi-language support, and NDJSON output.

**Voice Conversion** -- Transform speaker identity in real time using FreeVC (Coqui TTS). Swap any voice to sound like a target recording.

**Pitch Adjustment** -- Formant-preserving pitch shifting via ffmpeg rubberband. Auto-estimated from source/target F0 or set explicitly in semitones.

**Unified Pipeline DSL** -- One DSL for everything: `sip:leg1 -> call:conf1 -> sip:leg1` for telephony, `text_input | tts:voice | conference:call1` for streaming TTS, `tts:de{"text":"Hallo"} | pitch:2.0` for offline rendering. All through a single REST API.

**Telephony Platform** -- Multi-tenant conferencing with PBX management, SIP legs (PCMU, PCMA, G722, Opus), browser participants, hold music, DTMF, live STT, and webhook-driven call control.

**Fourier Codec** -- Custom FFT-based audio codec with four quality profiles (low/medium/high/full) for compressed real-time audio over WebSockets.

**HTTP & WebSocket Server** -- Ready-to-deploy server with REST and WebSocket endpoints for TTS, STT, streaming pipelines, and codec-compressed audio. CORS enabled, pm2-friendly.

**CLI Tool** -- Run pipelines from the command line, list voices, or start the server. Pipe stdin/stdout like any UNIX tool.

## Installation

```bash
pip install -e ".[tts,stt,server]"         # core + TTS + STT + HTTP server
pip install -e ".[all]"                     # everything including VC and SIP
pip install -e /path/to/piper              # Piper TTS from source
```

System dependencies:

```bash
sudo apt install espeak-ng ffmpeg
```

### Quick Start

```bash
# Start the server with API enabled
speech-pipeline serve --admin-token SECRET --voices-path voices-piper

# List available voices
speech-pipeline voices --voices-path voices-piper

# Synthesize from the command line
echo "Hallo Welt" | speech-pipeline run "cli:text | tts:de_DE-thorsten-medium | cli:raw" > out.raw
```

## Library Usage

Use `speech_pipeline` as a Python library to build custom audio pipelines.

### TTS: Text to Audio File

```python
from speech_pipeline import TTSProducer, FileRecorder, SampleRateConverter
from speech_pipeline.registry import TTSRegistry

registry = TTSRegistry("voices-piper")
voice = registry.ensure_loaded("de_DE-thorsten-medium")
syn = registry.create_synthesis_config(voice, {})

source = TTSProducer(voice, syn, text="Hallo Welt!", sentence_silence=0.2)
recorder = FileRecorder("output.mp3", sample_rate=voice.config.sample_rate)
source.pipe(recorder)
recorder.run()
```

### Streaming TTS: Lines to PCM

```python
from speech_pipeline import StreamingTTSProducer
from speech_pipeline.registry import TTSRegistry

registry = TTSRegistry("voices-piper")
voice = registry.ensure_loaded("de_DE-thorsten-medium")
syn = registry.create_synthesis_config(voice, {})

lines = ["First sentence.", "Second sentence.", "Third sentence."]
source = StreamingTTSProducer(lines, voice, syn)

for pcm_chunk in source.stream_pcm24k():
    # pcm_chunk is bytes (s16le mono at voice.config.sample_rate)
    process_audio(pcm_chunk)
```

### STT: Audio File to Text

```python
from speech_pipeline import AudioReader, SampleRateConverter
from speech_pipeline.WhisperSTT import WhisperTranscriber

source = AudioReader("interview.wav")
stt = WhisperTranscriber("small", chunk_seconds=3.0, language="de")
source.pipe(SampleRateConverter(24000, 16000)).pipe(stt)

for ndjson_chunk in stt.stream_pcm24k():
    print(ndjson_chunk.decode())
    # {"text": "hallo welt", "start": 0.0, "end": 1.5}
```

### Composing Stages with `.pipe()`

Stages chain with `.pipe()`. Format conversion (sample rate, encoding) is inserted automatically when needed:

```python
from speech_pipeline import (
    TTSProducer, VCConverter, PitchAdjuster, FileRecorder
)

source = TTSProducer(voice, syn, text="Hello!")
pipeline = (
    source
    .pipe(VCConverter("target_voice.wav"))
    .pipe(PitchAdjuster("target_voice.wav", pitch_override_st=2.0))
    .pipe(FileRecorder("output.wav", sample_rate=24000))
)
pipeline.run()
```

## Architecture

The library uses a pipeline of composable stages that process audio as a stream of PCM chunks. Each stage extends `Stage` (`speech_pipeline/base.py`), implements `stream_pcm24k() -> Iterator[bytes]`, and connects via `.pipe()`:

```
Source --> Processor --> Processor --> Sink
```

Format conversion between stages is automatic: `.pipe()` inserts `SampleRateConverter` and `EncodingConverter` stages when the output format of one stage does not match the input format of the next.

### Telephony Boundaries

The telephony server is intentionally a runtime executor, not the business
orchestration brain.

- The server owns runtime primitives: PBXs, SIP dialogs, calls, legs, webclient
  sessions, mixers, and the pipeline DSL.
- The CRM owns business logic: when to create calls, which legs/pipes to attach,
  when to answer, which webhooks to deploy, and how SIP users map to CRM users.
- The startup callback is a generic admin-side provisioner hook. In production
  LDS uses it to provision PBXs/accounts; once an account is provisioned, the
  server pings that CRM's heartbeat so the CRM can register its webhooks.
- SIP identities intentionally encode CRM ownership (`user@crm-domain` or the
  mangled legacy variants) so the server can route auth and device-dial events
  to exactly one CRM tenant without broadcasting across tenants.
- Incoming calls are also routed to exactly one CRM: first by explicit DID,
  otherwise by the wildcard subscriber for that PBX.

This means the speech server may perform SIP-mechanical steps such as `183
Session Progress`, deferred `200 OK`, RTP allocation, and `BYE`/`CANCEL`, but
the decision which audio flows where is made by the CRM through the REST/DSL
API.

### All Stages

#### Sources (produce PCM, no upstream)

| Stage | Module | Description |
|-------|--------|-------------|
| `TTSProducer` | `speech_pipeline.TTSProducer` | Fixed text to PCM via Piper ONNX. Streams sentence by sentence. |
| `StreamingTTSProducer` | `speech_pipeline.StreamingTTSProducer` | Text iterable to PCM. Synthesizes each line as it arrives. |
| `AudioReader` | `speech_pipeline.AudioReader` | Reads audio from file/URL via ffmpeg. Bearer auth for remote files. |
| `PCMInputReader` | `speech_pipeline.PCMInputReader` | Reads raw PCM bytes from a stream (HTTP body, microphone). |
| `WebSocketReader` | `speech_pipeline.WebSocketReader` | Binary/text from flask-sock WebSocket. |
| `SIPSource` | `speech_pipeline.SIPSource` | RTP audio from a SIP call (pyVoIP or RTPSession). Detects A-law/µ-law. |
| `CLIReader` | `speech_pipeline.CLIReader` | Text lines from stdin. |
| `QueueSource` | `speech_pipeline.QueueSource` | PCM from a `queue.Queue`. Bridge for AudioTee/AudioMixer. |
| `ConferenceMixer` | `speech_pipeline.ConferenceMixer` | N-to-M mixer with atomic per-sink mix-minus (echo cancellation). Real-time paced. |
| `CodecSocketSource` | `speech_pipeline.CodecSocketSource` | Decoded PCM from Fourier codec WebSocket. |

#### Processors (transform PCM, have upstream)

| Stage | Module | Description |
|-------|--------|-------------|
| `VCConverter` | `speech_pipeline.VCConverter` | Voice conversion via FreeVC. Passthrough if unavailable. |
| `PitchAdjuster` | `speech_pipeline.PitchAdjuster` | Pitch shifting via ffmpeg rubberband (formant-preserving). |
| `SampleRateConverter` | `speech_pipeline.SampleRateConverter` | Resampling via audioop (zero-latency). No-op when rates match. |
| `EncodingConverter` | `speech_pipeline.EncodingConverter` | s16le <-> u8. Auto-inserted by `pipe()`. |
| `AudioTee` | `speech_pipeline.AudioTee` | Pass-through with side-chain sinks via queues. Hot-pluggable. |
| `ConferenceLeg` | `speech_pipeline.ConferenceLeg` | Bidirectional conference participant with mix-minus output. |
| `GainStage` | `speech_pipeline.GainStage` | Runtime-adjustable volume. |
| `DelayLine` | `speech_pipeline.DelayLine` | Runtime-adjustable audio delay. |

#### Sinks (consume PCM, produce output)

| Stage | Module | Description |
|-------|--------|-------------|
| `WhisperTranscriber` | `speech_pipeline.WhisperSTT` | PCM to NDJSON transcription via faster-whisper. |
| `SIPSink` | `speech_pipeline.SIPSink` | PCM as RTP packets into a SIP call. Detects A-law/µ-law/G722/Opus. |
| `FileRecorder` | `speech_pipeline.FileRecorder` | Records PCM to file (MP3/WAV/OGG) via ffmpeg. |
| `WebhookSink` | `speech_pipeline.WebhookSink` | POST NDJSON lines to an HTTP endpoint (e.g. STT transcription). |
| `QueueSink` | `speech_pipeline.QueueSink` | Drains a stage pipeline into a `queue.Queue`. |
| `ResponseWriter` | `speech_pipeline.ResponseWriter` | Streams PCM as WAV HTTP response. |
| `WebSocketWriter` | `speech_pipeline.WebSocketWriter` | PCM as binary WebSocket messages. |
| `CLIWriter` | `speech_pipeline.CLIWriter` | NDJSON, text, or raw binary to stdout. |
| `CodecSocketSink` | `speech_pipeline.CodecSocketSink` | Encodes PCM to Fourier codec frames. |

## CLI Reference

```bash
# Run a pipeline from a DSL string
speech-pipeline run "cli:text | tts:de_DE-thorsten-medium | cli:raw"

# Start the HTTP/WebSocket server with API
speech-pipeline serve --admin-token SECRET --voices-path voices-piper

# Start with SIP stack enabled
speech-pipeline serve --admin-token SECRET --sip-port 5061

# List available voices
speech-pipeline voices --voices-path voices-piper
```

## Pipeline DSL

All pipeline operations use a single DSL. Separators `->` and `|` are interchangeable. Each element has the form `type`, `type:id`, or `type:id{json_params}`.

### Element Reference

| Type | ID | JSON Params | Description |
|------|-----|-------------|-------------|
| `sip` | leg_id | `{completed, dtmf}` | SIP leg source/sink (bidirectional in `sip:L -> call:C -> sip:L`) |
| `call` | call_id | -- | Conference participant (ConferenceLeg with mix-minus) |
| `conference` | call_id | -- | Alias for `call` |
| `play` | stage_name | `{url, loop, volume, completed}` | Audio playback from URL into conference or SIP leg |
| `tts` | voice | `{text}` | TTS: fixed text with `{"text":"..."}`, or streaming from upstream `text_input` |
| `stt` | language | `{model, chunk}` | Speech-to-text via Whisper |
| `tee` | tee_id | -- | Audio splitter for sidechains (`tee:tap -> stt:de -> webhook:URL`) |
| `webhook` | url | `{bearer}` | POST NDJSON to HTTP endpoint (terminal) |
| `text_input` | -- | -- | Queue-backed text source for API-driven streaming TTS |
| `gain` | factor | -- | Volume adjustment (1.0 = unity) |
| `delay` | ms | -- | Audio delay line |
| `pitch` | semitones | -- | Pitch shift (formant-preserving) |
| `vc` | voice_ref | -- | Voice conversion via FreeVC |
| `save` | name | `{rate, format}` | Record to managed download URL (returns via webhook) |
| `answer` | leg_id | -- | Send SIP 200 OK on inbound leg (single-element action) |
| `originate` | number | `{ringing, answered, completed, ...}` | Dial outbound (2-elem pipe: `originate:N{cb} -> call:C`) |
| `webclient` | user | `{callback, base_url}` | Create webclient session + iframe URL (single action) |
| `ws` | mode | -- | WebSocket source/sink (`ws:pcm`, `ws:text`, `ws:ndjson`) |
| `cli` | mode | -- | CLI stdin/stdout (`cli:text`, `cli:raw`, `cli:ndjson`) |
| `codec` | session_id | `{profile}` | Fourier codec WebSocket source/sink |
| `mix` | mixer_name | `{rate}` | Named mixer source |
| `mixminus` | queue_name | -- | Subtract own audio from full mix |

### Example DSL Pipelines

**Telephony (via REST API):**
```
# Bidirectional SIP bridge with STT sidechain
sip:leg1{"completed":"/cb/done"} -> tee:leg1_tap -> call:call-xxx -> sip:leg1
tee:leg1_tap -> stt:de -> webhook:https://crm.example.com/stt

# Hold music (looping, 50% volume)
play:hold{"url":"https://cdn.example.com/hold.mp3","loop":true,"volume":50} -> call:call-xxx

# Hold music directly to a held SIP leg
play:hold_b{"url":"hold.mp3","loop":true,"volume":50} -> sip:leg2

# TTS announcement
tts:de{"text":"Bitte warten Sie."} -> call:call-xxx

# Streaming TTS into conference (text fed via API)
text_input | tts:de_DE-thorsten-medium | conference:call-xxx

# Streaming TTS with voice conversion
text_input | tts:de_DE-thorsten-medium | vc:de_DE-thorsten-high | conference:call-xxx

# Originate outbound leg (async fire-and-forget with callbacks)
originate:+4917099999{"answered":"/cb/ans","completed":"/cb/end"} -> call:call-xxx

# Answer an inbound leg (single-element action)
answer:leg-abc

# Create webclient session (single-element action)
webclient:user42{"callback":"/cb/wc","base_url":"https://srv.example.com"}

# Record to managed download URL (webhook fires with URL when done)
sip:leg1 -> tee:rec -> call:call-xxx -> sip:leg1
tee:rec -> save:recording_xyz{"format":"wav"}
```

**Offline rendering (returns WAV):**
```
tts:de_DE-thorsten-medium{"text":"Hallo Welt"}
tts:de{"text":"Test"} | pitch:3.0
tts:de{"text":"Voice test"} | vc:target_voice | pitch:2.0
```

**WebSocket / CLI:**
```
ws:pcm | stt:de | ws:ndjson
ws:text | tts:de_DE-thorsten-medium | ws:pcm
cli:text | tts:de_DE-thorsten-medium | cli:raw
codec:session1 | conference:call-xxx | codec:session1
```

## REST API

### Core Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/healthz` | Liveness check |
| `GET` | `/voices` | Available voices with metadata |
| `GET/POST` | `/tts/say` | Synthesize speech (query params, returns WAV). Rate limit: 1/IP |
| `POST` | `/tts/stream` | Streaming TTS (text body in, WAV out). Rate limit: 1/IP |
| `WS` | `/ws/stt` | WebSocket STT. Rate limit: 1 concurrent/IP |
| `WS` | `/ws/tts` | WebSocket TTS. Rate limit: 1 concurrent/IP |
| `WS` | `/ws/pipe` | Generic DSL pipeline over WebSocket (requires auth) |
| `WS` | `/ws/socket/<id>` | Fourier codec bidirectional audio (unguessable session ID) |

### Pipeline API

All `/api/` endpoints require `Authorization: Bearer <token>`. Both admin tokens and account tokens are accepted. Account tokens can only access their own calls and subscribers.

#### Pipeline API — the single audio-routing endpoint

All call-control and audio routing goes through `/api/pipelines`.
HTTP method selects the operation:

```
POST   /api/pipelines              Create/execute DSL: {"dsl": "...", "render"?: bool}
GET    /api/pipelines              List pipelines (no args) OR lookup (?dsl=item)
DELETE /api/pipelines              Kill a stage: {"dsl": "stage_id"}
GET    /api/pipelines/<pid>        Pipeline detail (stages, edges, formats)
DELETE /api/pipelines/<pid>        Stop and remove a running pipeline
POST   /api/pipelines/<pid>/input  Feed text into text_input: {"text": "...", "eof": true}
GET    /api/saves/<filename>       Download a save: output (from save: DSL element)
```

**Creating a telephony pipeline:**
```bash
curl -X POST -H "Authorization: Bearer TOKEN" \
  -d '{"dsl": "sip:leg1 -> call:call-xxx -> sip:leg1"}' \
  http://localhost:5000/api/pipelines
```

**Rendering TTS as WAV (synchronous, returns audio/wav):**
```bash
curl -X POST -H "Authorization: Bearer TOKEN" \
  -d '{"dsl": "tts:de{\"text\":\"Hallo Welt\"}", "render": true}' \
  -o output.wav \
  http://localhost:5000/api/pipelines
```

**Look up a live object by DSL:**
```bash
curl -H "Authorization: Bearer TOKEN" \
  "http://localhost:5000/api/pipelines?dsl=call:call-xxx"   # call + participants
  "http://localhost:5000/api/pipelines?dsl=sip:leg-abc"     # leg details
  "http://localhost:5000/api/pipelines?dsl=play:hold"       # stage status
```

**Kill a stage:**
```bash
curl -X DELETE -H "Authorization: Bearer TOKEN" \
  -d '{"dsl": "play:hold_music"}' \
  http://localhost:5000/api/pipelines
```

**Streaming TTS into a conference:**
```bash
# 1. Create pipeline
curl -X POST -H "Authorization: Bearer TOKEN" \
  -d '{"dsl": "text_input | tts:de_DE-thorsten-medium | conference:call-xxx"}' \
  http://localhost:5000/api/pipelines
# Returns {"id": "abc123", ...}

# 2. Feed text
curl -X POST -H "Authorization: Bearer TOKEN" \
  -d '{"text": "Hallo, willkommen in der Konferenz."}' \
  http://localhost:5000/api/pipelines/abc123/input

# 3. End stream
curl -X POST -H "Authorization: Bearer TOKEN" \
  -d '{"eof": true}' \
  http://localhost:5000/api/pipelines/abc123/input
```

#### Stage Manipulation

```
GET    /api/pipelines/<pid>/stages          List stages
GET    /api/pipelines/<pid>/stages/<sid>    Stage detail (type, config, formats)
PATCH  /api/pipelines/<pid>/stages/<sid>    Hot-update config (gain, delay_ms)
DELETE /api/pipelines/<pid>/stages/<sid>    Remove processor, reconnect neighbors
POST   /api/pipelines/<pid>/stages/<sid>/replace  Replace with new DSL element
```

### Telephony API

#### PBX Management (admin-only)

```
PUT    /api/pbx/<pbx_id>       Register/update PBX (sip_proxy, credentials)
GET    /api/pbx                List all PBX connections
DELETE /api/pbx/<pbx_id>       Remove PBX
```

#### Account Management (admin-only)

```
PUT    /api/accounts/<id>      Register account (token, PBX pin, features, call limits)
GET    /api/accounts           List accounts
GET    /api/accounts/<id>      Account details
DELETE /api/accounts/<id>      Delete account and its subscribers
```

#### Subscriber Management (account-scoped)

```
PUT    /api/subscribe/<id>     Register subscriber (base_url, bearer_token, DIDs)
GET    /api/subscribers        List subscribers
GET    /api/subscribers/<id>   Subscriber details
DELETE /api/subscribers/<id>   Unsubscribe
```

#### Call Management (account-scoped)

Calls are conference containers. All audio routing and leg operations
go through `/api/pipelines`. Idle calls (no sources/sinks for 30s)
auto-cancel to prevent memory leaks.

```
POST   /api/calls              Create call/conference
GET    /api/calls              List calls
GET    /api/calls/<id>         Call details (participants, status)
GET    /api/calls/<id>/participants  List active participants
DELETE /api/calls/<id>         End call (hangs up all legs, cleans up)
```

Leg operations use the pipeline DSL:

| Old endpoint | New DSL |
|---|---|
| `POST /api/legs/originate` | `POST /api/pipelines {"dsl": "originate:NUM{cb} -> call:C"}` |
| `POST /api/legs/{id}/answer` | `POST /api/pipelines {"dsl": "answer:LEG"}` |
| `POST /api/legs/{id}/bridge` | `POST /api/pipelines {"dsl": "sip:LEG -> call:C -> sip:LEG"}` |
| `GET /api/legs/{id}` | `GET /api/pipelines?dsl=sip:LEG` |
| `DELETE /api/calls/{id}/stages/{sid}` | `DELETE /api/pipelines {"dsl": "STAGE_ID"}` |
| `POST /api/calls/{id}/commands (webclient)` | `POST /api/pipelines {"dsl": "webclient:USER{cb,base_url,call_id}"}` |
| `GET /api/calls/{id}` | `GET /api/pipelines?dsl=call:ID` (participants included) |
| `GET /api/calls/{id}/participants` | `GET /api/pipelines?dsl=call:ID` (nested `participants`) |

#### Nonce Management (for webclient auth)

```
POST   /api/nonce              Create nonce (1h TTL)
GET    /api/nonces             List nonces
DELETE /api/nonce/<nonce>      Revoke nonce
```

### CRM Integration Pattern

The typical call flow for a CRM subscriber — everything through `/api/pipelines`:

```
1. Inbound call → speech-server webhook → CRM creates DB record
2. CRM: POST /api/calls → create conference
3. CRM: POST /api/pipelines → bridge inbound SIP leg + wait music + STT sidechain
   DSL: "sip:LEG{"completed":"/cb"} -> tee:LEG_tap -> call:CALL -> sip:LEG"
   DSL: "tee:LEG_tap -> stt:de -> webhook:https://crm/stt"
   DSL: "play:CALL_wait{"url":"hold.mp3","loop":true} -> call:CALL"
4. CRM: POST /api/pipelines {"dsl": "originate:+4917...{cb} -> call:CALL"}
   → async SIP INVITE; CRM receives "answered" callback
5. On answered: CRM: POST /api/pipelines → bridge outbound leg
6. CRM: DELETE /api/pipelines {"dsl": "play:CALL_wait"} → stop wait music
7. Hold: DELETE /api/pipelines {"dsl": "bridge:LEG"} +
        POST /api/pipelines with "play:hold{...} -> sip:LEG"
8. Unhold: DELETE hold music stage + POST /api/pipelines → rebridge
9. CRM: DELETE /api/calls/CALL → teardown (or auto-cleanup after 30s idle)
```

### RTP Codecs

The SIP stack negotiates codecs automatically. Supported:

| Codec | PT | Sample Rate | Bandwidth |
|-------|----|-------------|-----------|
| Opus | 111 | 48 kHz | Variable (32 kbps default) |
| G.722 | 9 | 16 kHz (8 kHz RTP clock) | 64 kbps |
| PCMU (G.711 µ-law) | 0 | 8 kHz | 64 kbps |
| PCMA (G.711 A-law) | 8 | 8 kHz | 64 kbps |

Preference order: Opus > G.722 > PCMU > PCMA. The conference mixer runs at 48 kHz internally; sample rate conversion is automatic.

## Fourier Codec

Custom FFT-based audio codec for compressed real-time audio over WebSockets.

| Profile | Bins | Freq range | ~Bytes/frame | Use case |
|---------|------|------------|--------------|----------|
| `low` | 160 | 0-7.5 kHz | ~157 | Telephone, low bandwidth |
| `medium` | 256 | 0-12 kHz | ~410 | Good speech quality |
| `high` | 384 | 0-18 kHz | ~920 | Near-CD quality |
| `full` | 512 | 0-24 kHz | ~2060 | Lossless (within FFT) |

## Voice Models

Voice models (`.onnx` files) are not included. Place them in `voices-piper/` or specify `--voices-path`.

```bash
mkdir -p voices-piper && cd voices-piper

# German - Thorsten (medium)
wget https://huggingface.co/rhasspy/piper-voices/resolve/main/de/de_DE/thorsten/medium/de_DE-thorsten-medium.onnx
wget https://huggingface.co/rhasspy/piper-voices/resolve/main/de/de_DE/thorsten/medium/de_DE-thorsten-medium.onnx.json
```

Browse all voices: https://github.com/rhasspy/piper/blob/master/VOICES.md

## Browser Demos

| Demo | File | Description |
|------|------|-------------|
| STT | `examples/stt.html` | Microphone -> WebSocket STT -> transcript display |
| STS | `examples/sts.html` | Microphone -> STT -> TTS -> speaker (robot voice) |
| Codec | `examples/codec-demo.html` | Mic -> Fourier codec -> WS -> server -> decode -> playback |
| AI Assistant | `examples/ai.py` | Multi-user voice AI with LLM, streaming TTS, and conference |

## Apache Proxy

```apache
ProxyPass /tts/ws/pipe ws://localhost:5000/ws/pipe
ProxyPass /tts/ws/stt ws://localhost:5000/ws/stt
ProxyPass /tts/ws/tts ws://localhost:5000/ws/tts
ProxyPass /tts http://localhost:5000
```

## Requirements

- Python 3.10+
- espeak-ng (`sudo apt install espeak-ng`)
- ffmpeg with rubberband support (`sudo apt install ffmpeg`)
- [Piper](https://github.com/rhasspy/piper) Python bindings (install from source)
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (optional, for STT)
- [pyVoIP](https://pypi.org/project/pyVoIP/) (optional, for SIP)

## License

GPLv3
