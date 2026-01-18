#!/usr/bin/env python3
import argparse
import asyncio
import contextlib
import logging
import os
import signal
from functools import partial
from pathlib import Path

from wyoming.info import Attribution, Info, TtsProgram, TtsVoice
from wyoming.server import AsyncServer, AsyncTcpServer

from . import SERVICE_NAME, __version__
from .audio import SUPPORTED_LANGUAGES
from .config import Settings
from .download import BASE_URL, check_model_exists, download_model
from .engine import XTTSEngine, set_seed
from .handler import XTTSEventHandler

_LOGGER = logging.getLogger(__name__)


def parse_args(defaults: Settings) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Wyoming XTTS TTS server")
    parser.add_argument("--uri", default=defaults.uri)
    parser.add_argument("--assets", type=Path, default=defaults.assets)
    parser.add_argument("--deepspeed", action=argparse.BooleanOptionalAction, default=defaults.deepspeed)
    parser.add_argument("--no-download-model", action="store_true", default=defaults.no_download_model)
    parser.add_argument(
        "--zeroconf",
        default=defaults.zeroconf,
        help="Zeroconf service name (enables discovery)",
    )
    parser.add_argument(
        "--log-level",
        default=defaults.log_level,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    parser.add_argument(
        "--language-fallback",
        default=defaults.language_fallback,
        help="Fallback language when detection fails",
    )
    parser.add_argument(
        "--language-no-detect",
        action="store_true",
        default=defaults.language_no_detect,
        help="Disable language auto-detection",
    )
    parser.add_argument("--temperature", type=float, default=defaults.temperature)
    parser.add_argument("--speed", type=float, default=defaults.speed)
    parser.add_argument("--top-k", type=int, default=defaults.top_k)
    parser.add_argument("--top-p", type=float, default=defaults.top_p)
    parser.add_argument("--repetition-penalty", type=float, default=defaults.repetition_penalty)
    parser.add_argument("--stream-chunk-size", type=int, default=defaults.stream_chunk_size)
    parser.add_argument("--min-segment-chars", type=int, default=defaults.min_segment_chars)
    parser.add_argument(
        "--seed",
        type=int,
        default=defaults.seed,
        help="Fixed seed for reproducible synthesis (Omit for random)",
    )
    parser.add_argument("--version", action="version", version=__version__)
    return parser.parse_args()


def scan_voices(voices_path: Path) -> list[TtsVoice]:
    languages = list(SUPPORTED_LANGUAGES)
    return [
        TtsVoice(
            name=wav.stem,
            description=f"Voice: {wav.stem}",
            version=None,
            attribution=Attribution(name="", url=""),
            installed=True,
            languages=languages,
        )
        for wav in sorted(voices_path.glob("*.wav"))
    ]


def build_info(voices: list[TtsVoice]) -> Info:
    return Info(
        tts=[
            TtsProgram(
                name=SERVICE_NAME,
                description="XTTS v2 text-to-speech with voice cloning",
                attribution=Attribution(name="Coqui", url="https://github.com/coqui-ai/TTS"),
                installed=True,
                voices=voices,
                version=__version__,
                supports_synthesize_streaming=True,
            )
        ]
    )


async def main() -> None:
    settings = Settings()
    args = parse_args(settings)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )

    if args.seed is not None:
        set_seed(args.seed)

    models_path = args.assets / "models"
    voices_path = args.assets / "voices"
    cache_path = args.assets / "cache"

    cache_path.mkdir(parents=True, exist_ok=True)
    os.environ["TORCH_EXTENSIONS_DIR"] = str(cache_path / "torch_extensions")

    _LOGGER.info("Starting %s service", SERVICE_NAME)
    _LOGGER.info("Assets path: %s", args.assets)

    if not voices_path.exists():
        voices_path.mkdir(parents=True, exist_ok=True)
        _LOGGER.warning("No voices directory found, created: %s", voices_path)

    if not check_model_exists(models_path):
        if args.no_download_model:
            _LOGGER.error("XTTS model not found in %s", models_path)
            raise SystemExit(1)
        _LOGGER.info("No XTTS model found. Downloading [%s]", BASE_URL)
        download_model(models_path)

    voices = scan_voices(voices_path)
    if not voices:
        _LOGGER.warning("No voice samples found in %s", voices_path)

    wyoming_info = build_info(voices)

    engine = XTTSEngine(
        model_path=models_path,
        use_deepspeed=args.deepspeed,
        temperature=args.temperature,
        speed=args.speed,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        stream_chunk_size=args.stream_chunk_size,
        seed=args.seed,
    )
    await engine.load()
    _LOGGER.info(
        "Synthesis settings: temp=%.2f, speed=%.2f, top_k=%d, top_p=%.2f, rep_penalty=%.2f, chunk_size=%d",
        args.temperature,
        args.speed,
        args.top_k,
        args.top_p,
        args.repetition_penalty,
        args.stream_chunk_size,
    )

    server = AsyncServer.from_uri(args.uri)
    zeroconf = None

    if args.zeroconf:
        if not isinstance(server, AsyncTcpServer):
            raise ValueError("Zeroconf requires tcp:// URI")

        from wyoming.zeroconf import HomeAssistantZeroconf

        zeroconf_host = None if server.host in ("0.0.0.0", "::") else server.host
        zeroconf = HomeAssistantZeroconf(
            name=args.zeroconf,
            port=server.port,
            host=zeroconf_host,
        )
        await zeroconf.register_server()
        _LOGGER.info(
            "Zeroconf discovery enabled: %s -> %s:%s",
            args.zeroconf,
            zeroconf.host,
            server.port,
        )

    loop = asyncio.get_running_loop()
    shutdown_event = asyncio.Event()

    def handle_signal() -> None:
        _LOGGER.info("Shutdown signal received")
        shutdown_event.set()

    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, handle_signal)

    _LOGGER.info("Server ready on %s", args.uri)

    try:
        server_task = asyncio.create_task(
            server.run(
                partial(
                    XTTSEventHandler,
                    wyoming_info,
                    engine,
                    voices_path,
                    args.language_fallback,
                    args.language_no_detect,
                    args.min_segment_chars,
                )
            )
        )
        shutdown_task = asyncio.create_task(shutdown_event.wait())

        _, pending = await asyncio.wait(
            [server_task, shutdown_task],
            return_when=asyncio.FIRST_COMPLETED,
        )

        for task in pending:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

    finally:
        _LOGGER.info("Shutdown complete")


def run() -> None:
    asyncio.run(main())


if __name__ == "__main__":
    run()
