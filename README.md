# wyoming-xtts

A Wyoming protocol server for XTTS v2 text-to-speech, built for Home Assistant.

## Why

I wanted to use XTTS with Home Assistant but could not find a proper solution. Spent a day with various tools and bridges, but nothing really fit what I needed. So I decided I just wrote my own.

It simply does XTTS over the Wyoming protocol. No web interface, no API bridges, no configuration files. You put your voice samples in a folder and it works.

## Features

- Wyoming protocol native, follows wyoming-piper reference (Zeroconf discovery included)
- Bidirectional streaming support (text streams in from LLM, audio streams out)
- DeepSpeed (faster inference, trades VRAM for speed)

With bidirectional streaming and DeepSpeed you should expect a good, snappy performance. DeepSpeed halved the response time on my 1080. 

## Quick Start

```bash

mkdir -p /path/to/your/assets/voices

docker run -d \
  --gpus all \
  -p 10200:10200 \
  --name wyoming-xtts \
  -v /path/to/your/assets:/data \
  lmo3/wyoming-xtts
```

Then add to Home Assistant:

1) Settings -> Devices & services -> Add integration -> Wyoming Protocol -> Enter IP and Port (Default 10200) / Or, use the auto detected wyoming-xtts node if HA received the Zeroconf advertisement. 
2) Settings -> Voice assistants -> [Add or Select existing Assistant] -> Text-to-speech -> wyoming-xtts
3) Configure voice, language (Currently HA doesn't send the selected language to any wyoming-tts server, so auto detect will be used until this is fixed.)
4) ???
5) Profit

## Assets

Mount a folder or volume to `/data`. The server handles the rest:

```
/data/
├── models/    # XTTS model files (~2GB, auto-downloaded if missing)
├── voices/    # Your voice samples (WAV files, 6-30 seconds each)
└── cache/     # Torch compilation cache
```

Voice files are picked up by filename. Put `sarah.wav` in the voices folder, select "sarah" in Home Assistant.

Voice samples should be WAV files, mono, 22050 Hz, 16-bit PCM. XTTS resamples other formats internally but this avoids unnecessary conversion. Aim for 6-30 seconds of clear speech without background noise.

When DeepSpeed is enabled, it compiles a few libraries on first start. These go into the cache folder, so you don't have to compile them again after redeploying the docker container.

## Configuration

No config files. Environment variables only.

| Variable | Default | Description |
|----------|---------|-------------|
| `XTTS_URI` | `tcp://0.0.0.0:10200` | Server address |
| `XTTS_ASSETS` | (local)`./assets`, (docker)`/data` | Assets directory |
| `XTTS_ZEROCONF` | `wyoming-xtts` | Zeroconf service name (set empty to disable) |
| `XTTS_DEEPSPEED` | `false` | Enable DeepSpeed (faster, uses more VRAM) |
| `XTTS_LANGUAGE_FALLBACK` | `en` | Fallback when HA doesn't send language and detection fails |
| `XTTS_LANGUAGE_NO_DETECT` | `false` | Disable language auto-detection, always use fallback |
| `XTTS_LOG_LEVEL` | `INFO` | Log level (DEBUG, INFO, WARNING, ERROR) |
| `XTTS_NO_DOWNLOAD_MODEL` | `false` | Disable XTTS model auto-download |

### Synthesis Parameters

Note: All XTTS defaults are taken from the Xtts library config defaults.

| Variable | Default | Description |
|----------|---------|-------------|
| `XTTS_TEMPERATURE` | `0.85` | Sampling temperature (higher = more creative, less stable) |
| `XTTS_SPEED` | `1.0` | Speech speed multiplier |
| `XTTS_TOP_K` | `50` | Top-k sampling (fewer = faster, less diverse) |
| `XTTS_TOP_P` | `0.85` | Nucleus sampling threshold |
| `XTTS_REPETITION_PENALTY` | `2.0` | Repetition penalty |
| `XTTS_STREAM_CHUNK_SIZE` | `20` | Tokens per audio chunk (lower = faster first audio, may stutter) |
| `XTTS_MIN_SEGMENT_CHARS` | `20` | Minimum characters before synthesizing (prevents short segment hallucinations) |
| `XTTS_SEED` | `42` | Fixed seed for reproducible synthesis, set `XTTS_SEED=""` for a random seed and random synthesis |

Or use CLI arguments (`--deepspeed`, `--fallback-language de`, `--top-k 30`, etc.).

### Why is there a seed?

XTTS uses random sampling and sometimes doesn't stop when it should. A 2 second sentence can become 10 seconds of gibberish. This is a [known issue](https://github.com/coqui-ai/TTS/discussions/4146) with no real fix.

A fixed seed makes output deterministic. Same text sounds the same every time. This doesn't prevent hallucinations, but if seed 42 works for your voice samples, it will keep working. If you get hallucinations, try a different seed value. Set `XTTS_SEED=""` for random behavior, but expect inconsistent results.

## Supported Languages

en, es, fr, de, it, pt, pl, tr, ru, nl, cs, ar, zh-cn, hu, ko, ja, hi

Language is meant to be sent by Home Assistant based on your voice assistant config. Currently Home Assistant does not do this. Hence, this server auto-detects from text for now. If that fails, it uses `XTTS_LANGUAGE_FALLBACK` (default: `en`). Set `XTTS_LANGUAGE_NO_DETECT=true` to skip detection entirely.

## Requirements

- NVIDIA GPU (Pascal/GTX 10xx or newer)
- Docker with nvidia-container-toolkit
- ~2GB disk space for the model

On purpose this service uses PyTorch cu126 which still includes sm_60 support. 

Newer PyTorch builds dropped this, so Pascal cards (GTX 1080 etc.) would not work. 
This build should support everything from GTX 10xx series upwards. 


## License

MIT
