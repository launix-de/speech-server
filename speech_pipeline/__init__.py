from .base import Stage, AudioFormat
from .EncodingConverter import EncodingConverter
from .AudioReader import AudioReader
from .TTSProducer import TTSProducer
from .VCConverter import VCConverter
from .PitchAdjuster import PitchAdjuster
from .ResponseWriter import ResponseWriter
from .FileFetcher import FileFetcher
from .RawResponseWriter import RawResponseWriter
from .PCMInputReader import PCMInputReader
from .StreamingTTSProducer import StreamingTTSProducer
from .SampleRateConverter import SampleRateConverter
from .WhisperSTT import WhisperTranscriber
from .WebSocketReader import WebSocketReader
from .WebSocketWriter import WebSocketWriter
from .NdjsonToText import NdjsonToText
from .WebhookSink import WebhookSink
from .CLIReader import CLIReader
from .CLIWriter import CLIWriter
from .PipelineBuilder import PipelineBuilder
from .AudioSocketSession import AudioSocketSession
from .AudioSocketSource import AudioSocketSource
from .AudioSocketSink import AudioSocketSink
from .QueueSource import QueueSource
from .FileRecorder import FileRecorder
from .AudioTee import AudioTee
from .AudioMixer import AudioMixer
from .GainStage import GainStage
from .DelayLine import DelayLine
from .CodecSocketSession import CodecSocketSession, get_session as get_codec_session
from .CodecSocketSource import CodecSocketSource
from .CodecSocketSink import CodecSocketSink
from . import fourier_codec

# Optional: SIP stages (require pyVoIP)
try:
    from .SIPSession import SIPSession
    from .SIPSource import SIPSource
    from .SIPSink import SIPSink
except Exception:
    pass

# Optional convenience re-exports for TTS registry
try:
    from .registry import TTSRegistry, load_voice_info, VoiceInfo  # type: ignore
except Exception:
    # Keep package importable even if registry deps are missing at runtime
    pass

