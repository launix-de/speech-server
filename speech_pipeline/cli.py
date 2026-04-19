"""CLI entry point for speech-pipeline.

Usage:
    speech-pipeline run  "cli:text -> tts{\"voice\":\"de_DE-thorsten-medium\"} -> cli:raw"
    speech-pipeline serve [--host HOST] [--port PORT] [--voices-path DIR]
    speech-pipeline sip-bridge [--extension EXT] [--voice VOICE] [--lang LANG]
    speech-pipeline voices [--voices-path DIR]
"""
from __future__ import annotations

import argparse
import sys


def cmd_run(args: argparse.Namespace) -> None:
    """Run a pipeline from the DSL string."""
    from .PipelineBuilder import PipelineBuilder
    from .registry import TTSRegistry

    registry = TTSRegistry(args.voices_path, use_cuda=args.cuda)
    builder = PipelineBuilder(ws=None, registry=registry, args=args)
    run = builder.build(args.pipeline)
    try:
        run.run()
    except KeyboardInterrupt:
        run.cancel()


def cmd_serve(args: argparse.Namespace) -> None:
    """Start the HTTP/WebSocket server (delegates to piper_multi_server)."""
    # Import the server module from the repo root (not from the package)
    import importlib.util
    from pathlib import Path

    server_path = Path(__file__).resolve().parents[1] / "piper_multi_server.py"
    if not server_path.exists():
        sys.stderr.write(f"Server script not found: {server_path}\n")
        sys.exit(1)
    spec = importlib.util.spec_from_file_location("piper_multi_server", server_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    # Build an args namespace matching what create_app expects
    server_args = argparse.Namespace(
        host=args.host,
        port=args.port,
        model=None,
        voices_path=args.voices_path,
        scan_dir=None,
        cuda=args.cuda,
        sentence_silence=0.0,
        soundpath=getattr(args, "soundpath", "../voices/%s.wav"),
        bearer=getattr(args, "bearer", ""),
        whisper_model=getattr(args, "whisper_model", "small"),
        media_folder=getattr(args, "media_folder", None),
        admin_token=getattr(args, "admin_token", ""),
        debug=args.debug,
        voice_ttl_seconds=7200,
        voice_cache_max=64,
    )
    import logging
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)
    app = mod.create_app(server_args)
    app.run(host=args.host, port=args.port, threaded=True)


def cmd_sip_bridge(args: argparse.Namespace) -> None:
    """Start the SIP conference bridge (delegates to sip_bridge.py)."""
    import importlib.util
    from pathlib import Path

    bridge_path = Path(__file__).resolve().parents[1] / "sip_bridge.py"
    if not bridge_path.exists():
        sys.stderr.write(f"SIP bridge script not found: {bridge_path}\n")
        sys.exit(1)
    spec = importlib.util.spec_from_file_location("sip_bridge", bridge_path)
    mod = importlib.util.module_from_spec(spec)
    # Replace sys.argv so the bridge's own argparse picks up our flags
    saved_argv = sys.argv
    sys.argv = ["sip-bridge"] + args.extra
    try:
        spec.loader.exec_module(mod)
        mod.main()
    finally:
        sys.argv = saved_argv


def cmd_voices(args: argparse.Namespace) -> None:
    """List available voices."""
    from .registry import TTSRegistry

    registry = TTSRegistry(args.voices_path, use_cuda=False)
    if not registry.index:
        sys.stderr.write(f"No voices found in {args.voices_path}\n")
        sys.exit(1)
    for model_id in sorted(registry.index.keys()):
        path = registry.index[model_id]
        sys.stdout.write(f"{model_id}  {path}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="speech-pipeline",
        description="Composable stage-based speech pipeline",
    )
    parser.add_argument("--debug", action="store_true")
    sub = parser.add_subparsers(dest="command")

    # --- run ---
    p_run = sub.add_parser("run", help="Run a pipeline from DSL string")
    p_run.add_argument("pipeline", help='Pipeline DSL, e.g. "cli:text -> tts{\\"voice\\":\\"de_DE-thorsten-medium\\"} -> cli:raw"')
    p_run.add_argument("--voices-path", default="voices-piper")
    p_run.add_argument("--cuda", action="store_true")
    p_run.add_argument("--whisper-model", default="small")
    p_run.add_argument("--media-folder", default=None)

    # --- serve ---
    p_serve = sub.add_parser("serve", help="Start the HTTP/WS server")
    p_serve.add_argument("--host", default="0.0.0.0")
    p_serve.add_argument("--port", type=int, default=5000)
    p_serve.add_argument("--voices-path", default="voices-piper")
    p_serve.add_argument("--cuda", action="store_true")
    p_serve.add_argument("--whisper-model", default="small")
    p_serve.add_argument("--media-folder", default=None)
    p_serve.add_argument("--soundpath", default="../voices/%s.wav",
                         help="Deprecated legacy voice-ref template. Prefer JSON params with URLs or --media-folder filenames.")
    p_serve.add_argument("--bearer", default="")
    p_serve.add_argument("--admin-token", default="", help="Bearer token to enable /api/ pipeline control")

    # --- sip-bridge ---
    p_sip = sub.add_parser("sip-bridge", help="SIP conference bridge with STT/TTS")
    p_sip.add_argument("extra", nargs="*", help="Extra args forwarded to sip_bridge.py")

    # --- voices ---
    p_voices = sub.add_parser("voices", help="List available voices")
    p_voices.add_argument("--voices-path", default="voices-piper")

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)

    if args.command == "run":
        cmd_run(args)
    elif args.command == "serve":
        cmd_serve(args)
    elif args.command == "sip-bridge":
        cmd_sip_bridge(args)
    elif args.command == "voices":
        cmd_voices(args)


if __name__ == "__main__":
    main()
