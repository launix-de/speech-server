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

**Pipeline DSL** -- Describe complex audio flows as simple text: `ws:pcm | resample:48000:16000 | stt:de | ws:ndjson`. The builder wires up all the stages, converters, and adapters automatically.

**Telephony Platform** -- Multi-tenant conferencing with PBX management, SIP legs, browser participants, hold music, DTMF, live STT, and webhook-driven call control. Full REST API for accounts, subscribers, and call commands.

**SIP Bridge** -- Dial into Asterisk conferences as a bot with full-duplex STT/TTS. Transcribe what others say, speak generated text back.

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
# Start the server
speech-pipeline serve --voices-path voices-piper

# List available voices
speech-pipeline voices --voices-path voices-piper

# Synthesize from the command line
echo "Hallo Welt" | speech-pipeline run "cli:text | tts:de_DE-thorsten-medium | cli:raw" > out.raw

# Or use the server script (creates venv, installs deps, starts server)
bash run_server.sh
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

### Running a Pipeline from DSL

```python
from speech_pipeline.PipelineBuilder import PipelineBuilder
from speech_pipeline.registry import TTSRegistry
import argparse

registry = TTSRegistry("voices-piper")
args = argparse.Namespace(whisper_model="small", cuda=False)
builder = PipelineBuilder(ws=None, registry=registry, args=args)

run = builder.build("cli:text | tts:de_DE-thorsten-medium | cli:raw")
run.run()
```

## Architecture

The library uses a pipeline of composable stages that process audio as a stream of PCM chunks. Each stage extends `Stage` (`speech_pipeline/base.py`), implements `stream_pcm24k() -> Iterator[bytes]`, and connects via `.pipe()`:

```
Source --> Processor --> Processor --> Sink
```

Format conversion between stages is automatic: `.pipe()` inserts `SampleRateConverter` and `EncodingConverter` stages when the output format of one stage does not match the input format of the next.

### Example Pipelines

```
POST /              TTS:            TTSProducer --> VCConverter --> PitchAdjuster --> ResponseWriter
POST /tts/stream    Streaming TTS:  text_lines(request.stream) --> StreamingTTSProducer --> ResponseWriter
POST /inputstream   STT:            PCMInputReader --> [SampleRateConverter] --> WhisperTranscriber --> NDJSON
WS   /ws/pipe       Generic:        DSL-defined (e.g. ws:pcm | resample:48000:16000 | stt:de | ws:ndjson)
CLI                 TTS:            cli:text | tts:de_DE-thorsten-medium | cli:raw
```

### All Stages

#### Sources (produce PCM, no upstream)

| Stage | Module | Description |
|-------|--------|-------------|
| `TTSProducer` | `speech_pipeline.TTSProducer` | Fixed text to PCM via Piper ONNX. Streams sentence by sentence. |
| `StreamingTTSProducer` | `speech_pipeline.StreamingTTSProducer` | Text iterable to PCM. Synthesizes each line as it arrives. |
| `AudioReader` | `speech_pipeline.AudioReader` | Reads audio from file/URL via ffmpeg. Bearer auth for remote files. |
| `PCMInputReader` | `speech_pipeline.PCMInputReader` | Reads raw PCM bytes from a stream (HTTP body, microphone). |
| `WebSocketReader` | `speech_pipeline.WebSocketReader` | Binary/text from flask-sock WebSocket. |
| `SIPSource` | `speech_pipeline.SIPSource` | RTP audio from a SIP call via pyVoIP. |
| `CLIReader` | `speech_pipeline.CLIReader` | Text lines from stdin. |
| `QueueSource` | `speech_pipeline.QueueSource` | PCM from a `queue.Queue`. Bridge for AudioTee/AudioMixer. |
| `AudioMixer` | `speech_pipeline.AudioMixer` | Mixes N input queues. Hot-pluggable. |
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
| `MixMinus` | `speech_pipeline.MixMinus` | Subtract own audio from full mix (manual mix-minus). |
| `GainStage` | `speech_pipeline.GainStage` | Runtime-adjustable volume. |
| `DelayLine` | `speech_pipeline.DelayLine` | Runtime-adjustable audio delay. |

#### Sinks (consume PCM, produce output)

| Stage | Module | Description |
|-------|--------|-------------|
| `ResponseWriter` | `speech_pipeline.ResponseWriter` | Streams PCM as WAV HTTP response. |
| `RawResponseWriter` | `speech_pipeline.RawResponseWriter` | Raw file passthrough. |
| `WhisperTranscriber` | `speech_pipeline.WhisperSTT` | PCM to NDJSON transcription via faster-whisper. |
| `WebSocketWriter` | `speech_pipeline.WebSocketWriter` | PCM as binary WebSocket messages. |
| `SIPSink` | `speech_pipeline.SIPSink` | PCM as RTP packets into a SIP call. |
| `CLIWriter` | `speech_pipeline.CLIWriter` | NDJSON, text, or raw binary to stdout. |
| `FileRecorder` | `speech_pipeline.FileRecorder` | Records PCM to file (MP3/WAV/OGG) via ffmpeg. |
| `CodecSocketSink` | `speech_pipeline.CodecSocketSink` | Encodes PCM to Fourier codec frames. |
| `QueueSink` | `speech_pipeline.QueueSink` | Drains a stage pipeline into a `queue.Queue`. |
| `WebhookSink` | `speech_pipeline.WebhookSink` | POST NDJSON lines to an HTTP endpoint (e.g. STT transcription). |

#### Utilities

| Component | Module | Description |
|-----------|--------|-------------|
| `NdjsonToText` | `speech_pipeline.NdjsonToText` | Iterator adapter: extracts `.text` from NDJSON bytes for STT->TTS transitions. |
| `PipelineBuilder` | `speech_pipeline.PipelineBuilder` | DSL parser and stage wiring. |
| `TTSRegistry` | `speech_pipeline.registry` | Voice discovery, caching and lazy loading. |
| `FileFetcher` | `speech_pipeline.FileFetcher` | Downloads HTTP(S) URLs or local files. Bearer auth. |
| `FreeVCService` | `speech_pipeline.vc_service` | Singleton FreeVC model manager. |
| `SIPSession` | `speech_pipeline.SIPSession` | pyVoIP lifecycle manager. |
| `CodecSocketSession` | `speech_pipeline.CodecSocketSession` | WebSocket session for Fourier codec. |
| `fourier_codec` | `speech_pipeline.fourier_codec` | FFT-based codec with multi-profile support. |
| `telephony` | `speech_pipeline.telephony` | Multi-tenant telephony platform (PBX, accounts, calls, commands). |

## CLI Reference

```bash
# Run a pipeline from a DSL string
speech-pipeline run "cli:text | tts:de_DE-thorsten-medium | cli:raw"

# Start the HTTP/WebSocket server
speech-pipeline serve --host 0.0.0.0 --port 5000 --voices-path voices-piper

# Start with pipeline control API enabled
speech-pipeline serve --admin-token SECRET --voices-path voices-piper

# Start the SIP conference bridge
speech-pipeline sip-bridge -- --voice de_DE-thorsten-medium --lang de

# List available voices
speech-pipeline voices --voices-path voices-piper
```

## Pipeline DSL

Syntax: `element | element | ... | element`

Each element: `type:param1:param2`

| Type | Params | Stage |
|------|--------|-------|
| `cli:text` | -- | CLIReader (first) / CLIWriter text (last) |
| `cli:raw` | -- | CLIWriter binary (last) |
| `cli:ndjson` | -- | CLIWriter NDJSON (last) |
| `ws:pcm` | -- | WebSocketReader / WebSocketWriter |
| `ws:text` | -- | WebSocketReader.text_lines() / ws.send() |
| `ws:ndjson` | -- | ws.send(NDJSON line) |
| `resample` | FROM:TO | SampleRateConverter |
| `stt` | LANG or LANG:CHUNK:MODEL | WhisperTranscriber |
| `tts` | VOICE | StreamingTTSProducer |
| `sip` | TARGET | SIPSource / SIPSink |
| `vc` | VOICE2 | VCConverter |
| `pitch` | ST | PitchAdjuster |
| `record` | FILE or FILE:RATE | AudioTee + FileRecorder sidechain |
| `tee` | NAME | AudioTee feeding named mixer |
| `mix` | NAME or NAME:RATE | AudioMixer source |
| `gain` | FACTOR | GainStage (1.0 = unity) |
| `delay` | MS | DelayLine |
| `codec` | ID or ID:PROFILE | CodecSocketSource / CodecSocketSink |
| `conference` | CALL_ID | ConferenceLeg (bidirectional mix-minus participant) |

### Example DSL Pipelines

```
STT:      ws:pcm | resample:48000:16000 | stt:de | ws:ndjson
TTS:      ws:text | tts:de_DE-thorsten-medium | ws:pcm
STS:      ws:pcm | resample:48000:16000 | stt:de | tts:de_DE-thorsten-medium | ws:pcm
CLI-TTS:  cli:text | tts:de_DE-thorsten-medium | cli:raw
SIP-TX:   ws:text | tts:de_DE-thorsten-medium | resample:22050:8000 | sip:100@pbx
SIP-RX:   sip:100@pbx | resample:8000:16000 | stt:de | ws:ndjson
```

## HTTP API

### `GET /healthz`
Liveness check. Returns `200 OK`.

### `GET /voices`
Returns JSON map of available voices with metadata.

### `POST /`
Synthesize speech. Parameters: `text`, `voice`, `voice2` (VC target), `sound` (audio source), `lang`, `speaker`/`speaker_id`, `length_scale`, `noise_scale`, `noise_w_scale`, `sentence_silence`, `pitch_st`, `pitch_factor`, `pitch_disable`.

### `POST /inputstream`
Streaming STT. Send raw PCM via request body, receive NDJSON transcription.

```bash
arecord -f S16_LE -r 16000 -c 1 -t raw -q - | \
  curl -sN -T - -H "Content-Type: application/octet-stream" \
  http://localhost:5000/inputstream
```

### `POST /tts/stream`
Streaming TTS. Send text lines via request body, receive streaming WAV audio.

```bash
echo "Hallo Welt." | curl -T - -H 'Content-Type: text/plain' \
  -o out.wav 'http://localhost:5000/tts/stream?voice=de_DE-thorsten-medium'
```

### `WS /ws/stt`
WebSocket STT. Binary PCM in, NDJSON text out.

### `WS /ws/tts`
WebSocket TTS. Text in, binary PCM out.

### `WS /ws/pipe`
Generic pipeline endpoint. Send JSON config (`{"pipe": "..."}` or `{"pipes": [...]}`), data flows according to the DSL.

### `WS /ws/socket/<id>`
Fourier codec bidirectional audio socket with profile handshake.

## Pipeline Control API

All `/api/*` endpoints require `--admin-token` to be set on the server and `Authorization: Bearer <token>` on every request.

```bash
# Start server with admin API enabled
speech-pipeline serve --admin-token SECRET
```

### `GET /api/pipelines`
List all running pipelines (id, DSL, state, stage count).

### `POST /api/pipelines`
Create a pipeline from DSL. Body: `{"dsl": "cli:text | tts:voice | cli:raw"}`.

### `GET /api/pipelines/<pid>`
Pipeline detail with full stage list, audio formats, and edge graph.

### `DELETE /api/pipelines/<pid>`
Cancel and remove a pipeline.

### `GET /api/pipelines/<pid>/stages`
List stages in a pipeline (id, type, config, cancelled).

### `GET /api/pipelines/<pid>/stages/<sid>`
Single stage detail including input/output audio format.

### `PATCH /api/pipelines/<pid>/stages/<sid>`
Hot-update stage config without restart. Supported: `GainStage` (gain), `DelayLine` (delay_ms).

```bash
curl -X PATCH -H "Authorization: Bearer SECRET" \
  -d '{"gain": 0.5}' http://localhost:5000/api/pipelines/abc123/stages/def456
```

### `DELETE /api/pipelines/<pid>/stages/<sid>`
Remove a processor stage and reconnect its neighbors.

### `POST /api/pipelines/<pid>/stages/<sid>/replace`
Replace a stage with a new one built from a DSL element. Body: `{"element": "gain:2.0"}`.

## Fourier Codec

Custom FFT-based audio codec for compressed real-time audio over WebSockets.

| Profile | Bins | Freq range | ~Bytes/frame | Use case |
|---------|------|------------|--------------|----------|
| `low` | 160 | 0-7.5 kHz | ~157 | Telephone, low bandwidth |
| `medium` | 256 | 0-12 kHz | ~410 | Good speech quality |
| `high` | 384 | 0-18 kHz | ~920 | Near-CD quality |
| `full` | 512 | 0-24 kHz | ~2060 | Lossless (within FFT) |

## SIP Bridge

Dial into Asterisk conferences as a bot with full-duplex STT/TTS.

```bash
speech-pipeline sip-bridge -- --voice de_DE-thorsten-medium --lang de
```

```
RX: SIPSource --> SampleRateConverter(8k->16k) --> WhisperTranscriber --> CLIWriter
TX: CLIReader --> StreamingTTSProducer --> SampleRateConverter(native->8k) --> SIPSink
```

SIP stages require `pyVoIP` (`pip install pyVoIP`).

## Telephony Platform

Multi-tenant conferencing platform with PBX integration, subscriber webhooks, and browser-based participation. All telephony API endpoints live under `/api/` and require `--admin-token` to be set.

### Architecture

```
Subscriber App ──webhook──► speech-pipeline ──SIP──► Asterisk PBX
                                │                        │
                                ├── ConferenceMixer ◄────┘
                                │       │
                                ├── TTS/STT/Play (as conference sources/sinks)
                                │
                                └── WebClient (browser via Fourier codec)
```

Each call owns a `ConferenceMixer` that handles N-to-M audio mixing with per-sink mix-minus (echo cancellation). Participants (SIP legs, TTS, STT, browser clients) connect as sources and sinks.

### Telephony API

#### PBX Management (admin-only)

```
PUT    /api/pbx/<pbx_id>       Register/update PBX (sip_proxy, ari_url, credentials)
GET    /api/pbx                List all PBX connections
DELETE /api/pbx/<pbx_id>       Remove PBX and stop its SIP listener
```

#### Account Management (admin-only)

```
PUT    /api/accounts/<id>      Register account (token, PBX pin, features, call limits)
GET    /api/accounts            List accounts
GET    /api/accounts/<id>      Account details
DELETE /api/accounts/<id>      Delete account and its subscribers
```

#### Subscriber Management (account-scoped)

```
PUT    /api/subscribe/<id>     Register subscriber (base_url, bearer_token, DIDs, events)
GET    /api/subscribers         List subscribers
GET    /api/subscribers/<id>   Subscriber details (includes liveliness status)
DELETE /api/subscribers/<id>   Unsubscribe
```

#### Call Management (account-scoped)

```
POST   /api/calls              Create call/conference
GET    /api/calls              List calls
GET    /api/calls/<id>         Call details (participants, status)
POST   /api/calls/<id>/commands  Execute commands (async)
GET    /api/calls/<id>/participants  List participants
DELETE /api/calls/<id>         End call
```

#### Leg Management (account-scoped)

```
GET    /api/legs               List SIP legs
GET    /api/legs/<id>          Leg details
DELETE /api/legs/<id>          Hang up leg
POST   /api/legs/<id>/bridge   Bridge leg into conference
POST   /api/legs/originate     Originate outbound call into existing conference
```

#### Nonce Management (for webclient iframe auth)

```
POST   /api/nonce              Create nonce (1h TTL)
GET    /api/nonces             List nonces
DELETE /api/nonce/<nonce>      Revoke nonce
```

### Call Commands

Commands are sent via `POST /api/calls/<id>/commands` and execute asynchronously. Subscribers receive results via webhook callbacks.

| Command | Description | Key Params |
|---------|-------------|------------|
| `originate` | Create outbound SIP leg, bridge into conference | `to`, `callbacks` |
| `add_leg` | Bridge existing inbound leg into conference | `leg_id`, `callbacks` |
| `tts` | Speak text into conference (fire-and-forget) | `text`, `voice`, `callback` |
| `play` | Play audio file into conference (supports loop & volume) | `url`, `loop`, `volume`, `callback` |
| `stop_play` | Stop looping playback instantly | `participant_id` |
| `hangup` | Hang up a leg or entire call | `leg_id` (optional) |
| `transfer` | Move participant between conferences | `participant_id`, `target_call_id` |
| `dtmf` | Send DTMF tones inband | `digits`, `duration_ms` |
| `stt_start` | Start live STT, POST transcripts to webhook | `language`, `model`, `callback` |
| `stt_stop` | Stop STT | -- |
| `webclient` | Create browser participant slot with iframe URL | `user`, `callback`, `base_url` |

### WebClient

Browser-based conference participation via Fourier codec WebSocket. The webclient command returns an iframe URL with a nonce-authenticated phone UI. Internally, the browser connects as:

```
codec:SESSION | conference:CALL | codec:SESSION
```

### Subscriber Webhooks

Events (incoming call, leg answered/completed, etc.) are delivered as HTTP POST to `subscriber.base_url`. The subscriber can respond with `{"commands": [...]}` to chain further operations.

## Cluster Architecture

For high-throughput deployments, speech-pipeline supports horizontal scaling across multiple nodes.

### Overview

```
Clients ──► Entry Nodes (pipeline orchestration)
                │
                ├──► GPU Worker A (TTS/STT stages)
                ├──► GPU Worker B (TTS/STT stages)
                └──► GPU Worker C (voice conversion)
```

**Entry nodes** accept WebSocket connections and instantiate the full pipeline. When an entry node is at capacity, it discovers free worker nodes in the cluster and offloads compute-intensive stages (TTS, STT, voice conversion) to them. The offloaded stages communicate back through WebSocket channels with Fourier codec compression -- the same codec transport already used for client connections.

**Client-side load balancing**: DNS returns shuffled IPs for entry nodes. Each client connects to a random entry node, distributing connections without a central load balancer.

### How It Works

1. Client connects to an entry node via `WS /ws/pipe` with a DSL config
2. Entry node builds the full pipeline locally
3. If the entry node is under heavy load, it uses the pipeline control API (`/api/`) to:
   - Locate a worker node with free capacity
   - Deploy the heavy stages (e.g. `stt`, `tts`, `vc`) on the worker via `POST /api/pipelines`
   - Replace the local stages with `codec` bridge stages that route audio to/from the worker
4. From the client's perspective, nothing changes -- the pipeline still streams as before

### Stage Offloading

Any processor stage can be offloaded by replacing it with a codec bridge pair:

```
Before:  ws:pcm | stt:de | ws:ndjson        (all local)
After:   ws:pcm | codec:worker1 | ws:ndjson  (STT runs on worker1)
```

The Fourier codec adds minimal latency. At `low` profile, the FFT computation is ~6 microseconds per frame; the bottleneck is Python bit-packing at ~700 microseconds, supporting ~30 concurrent streams per CPU core. Use `medium` or `high` profiles for better audio quality at higher bandwidth.

### Performance Notes

| Profile | Concurrent streams/core | Bandwidth/stream | Quality |
|---------|------------------------|-------------------|---------|
| `low` | ~30 | ~12 KB/s | Telephone |
| `medium` | ~15 | ~32 KB/s | Good speech |
| `high` | ~8 | ~72 KB/s | Near-CD |

GPU-bound stages (TTS, STT, VC) scale with GPU count. CPU-bound codec transport scales linearly with cores.

## Voice Models

Voice models (`.onnx` files) are not included. Place them in `voices-piper/` or specify `--voices-path`.

```bash
mkdir -p voices-piper && cd voices-piper

# German - Thorsten (medium)
wget https://huggingface.co/rhasspy/piper-voices/resolve/main/de/de_DE/thorsten/medium/de_DE-thorsten-medium.onnx
wget https://huggingface.co/rhasspy/piper-voices/resolve/main/de/de_DE/thorsten/medium/de_DE-thorsten-medium.onnx.json

# English - Amy (medium)
wget https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/amy/medium/en_US-amy-medium.onnx
wget https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/amy/medium/en_US-amy-medium.onnx.json
```

Browse all voices: https://github.com/rhasspy/piper/blob/master/VOICES.md

## Browser Demos

| Demo | File | Description |
|------|------|-------------|
| STT | `examples/stt.html` | Microphone -> WebSocket STT -> transcript display |
| STS | `examples/sts.html` | Microphone -> STT -> TTS -> speaker (robot voice) |
| Codec | `examples/codec-demo.html` | Mic -> Fourier codec -> WS -> server -> decode -> playback |
| Conference | `examples/webconference.py` | Multi-user conference with STT, TTS, hold music, webclient |

Open via `https://server/tts/examples/stt.html?api=https://server/tts`

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

MIT
