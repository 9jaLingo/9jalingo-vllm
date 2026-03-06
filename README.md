# 9jaLingo TTS-vLLM (9javox) — Legacy Model

[![License](https://img.shields.io/badge/License-Proprietary-red.svg)](LICENSE)

> **Proprietary Software** — Copyright (c) 2025 9jaLingo. All rights reserved.  
> This is a closed-source commercial product. Unauthorized copying, distribution, or reverse-engineering is strictly prohibited. See [LICENSE](LICENSE).

A high-performance Text-to-Speech (TTS) inference server for **Nigerian languages** powered by vLLM, with **speaker embeddings** and **voice cloning** support.

This is 9jaLingo's **legacy TTS model** — the foundation that proved production-grade Nigerian language TTS is achievable. Future models will build on the insights gained here.

**Model:** 9javox · **Supported Languages:** Hausa · Igbo · Yoruba · Pidgin · English (Nigerian Accent — coming soon)

Built on the [`naijalingo-tts-2`](https://pypi.org/project/naijalingo-tts-2/) module and the `9jaLingo/9javox-9jalingo-finetuned-full-v1` model.

## Features

- **Ultra-Fast Inference**: vLLM's optimized engine for high-throughput generation
- **9jaLingo Speech API**: Production-ready `/v1/audio/speech` endpoint
- **Speaker Embeddings**: 240+ pre-computed voices (128-dim) organized by language — true voice control
- **Voice Cloning**: Clone any voice from a short reference audio (3–30 seconds)
- **Multi-Language Support**: Hausa, Igbo, Yoruba, Nigerian Pidgin (English Nigerian Accent coming soon)
- **Extended Generation**: Up to 40 seconds of continuous high-quality audio per generation
- **Real-Time Streaming**: Server-Sent Events (SSE) with both vLLM and speaker paths
- **Long-Form Generation**: Automatic text chunking for texts beyond 40 seconds
- **Flexible Sampling**: Temperature, top-p, and repetition penalty at generation time
- **Flexible Output Formats**: WAV, PCM, FLAC, ALAC, MP3, AAC, OGG, or streaming SSE
- **Dual Engine Architecture**: vLLM fast path + Direct model path for speaker features

## Architecture

```
                     ┌─────────────────────────────────────────┐
                     │   FastAPI Server (9javox API)       │
                     └───────────────────┴─────────────────────┘
                              │
              ┌───────────────┴───────────────┐
              │                               │
    ┌─────────┴─────────┐      ┌───────┴────────┐
    │  vLLM Fast Path   │      │ Direct Model Path│
    │  (no speaker emb) │      │ (speaker + clone)│
    └────────────────────┘      └────────────────┘
              │                               │
    AsyncLLMEngine              NaijaLingoTTS + SpeakerEmbedder
              │                               │
              └───────────────┬───────────────┘
                              │
              NeMo NanoCodec Decoder (22050 Hz)
                              │
              Output: WAV / FLAC / MP3 / AAC / OGG / ALAC / PCM / SSE
```

The system uses:
- **TTS Model**: `9jaLingo/9javox-9jalingo-finetuned-full-v1` (LFM2-based, 9javox)
- **Audio Codec**: `nvidia/nemo-nano-codec-22khz-0.6kbps-12.5fps`
- **Speaker Embedder**: `nineninesix/speaker-emb-tbr` (WavLM-based, 128-dim)
- **Inference**: vLLM AsyncEngine (fast) + Direct HuggingFace (speaker features)
- **Sample Rate**: 22050 Hz, 16-bit, mono

## Installation

### Prerequisites
- Linux (or WSL on Windows)
- Python 3.10 – 3.12
- NVIDIA GPU with CUDA 12.8+ (for GPU mode) — CPU mode also supported for testing
- 12GB+ VRAM recommended (GPU mode)
- **FFmpeg** (required for MP3, AAC, OGG, and ALAC output formats):
  ```bash
  sudo apt install ffmpeg
  ```

### Install Dependencies

1. Install `uv`:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
uv --version
```

2. Create and activate virtual environment:
```bash
cd 9jalingo-vllm
uv venv --python 3.12
source .venv/bin/activate
```

#### GPU Mode (Production — requires NVIDIA GPU)

3. Install the project and its dependencies:
```bash
uv pip install -e . --torch-backend=auto
```

4. Install vLLM (GPU):
```bash
uv pip install vllm --torch-backend=auto
```

Here is the [vLLM GPU documentation](https://docs.vllm.ai/en/stable/getting_started/installation/gpu.html) for custom installation.

#### CPU Mode (Testing — no GPU required)

3. Install the project and its dependencies with CPU-only PyTorch:
```bash
CFLAGS="-g -O3 -Wall -fPIC" UV_HTTP_TIMEOUT=300 \
  uv pip install -e . --torch-backend cpu
```

4. Install vLLM CPU (from the [official CPU wheels](https://docs.vllm.ai/en/stable/getting_started/installation/cpu/)):
```bash
UV_HTTP_TIMEOUT=300 \
  uv pip install vllm \
    --extra-index-url https://wheels.vllm.ai/nightly/cpu \
    --index-strategy first-index \
    --torch-backend cpu
```

5. (Recommended) Install TCMalloc for better CPU performance:
```bash
sudo apt-get install -y --no-install-recommends libtcmalloc-minimal4
```

> **Note**: CPU and GPU modes install dependencies in stages using `uv pip install` (not `uv run`) to avoid resolution conflicts between `nemo-toolkit` and `vllm`. Use `python server.py` to start the server after installation.

**Known Issues**

- vLLM does not support Windows natively. Use WSL with a compatible Linux distribution.

- There may be a dependency conflict between `nemo-toolkit[tts]` (requires specific `transformers`) and vLLM. Installing in stages (project first, then vLLM) avoids this.

- `nemo-toolkit[tts]` requires `ffmpeg`. Install it with `apt install ffmpeg` if not already available.

- On some systems, building C extensions (e.g., `cdifflib`) may fail with uv's standalone Python. Use `CFLAGS="-g -O3 -Wall -fPIC"` to override problematic compiler flags.

## Quick Start

### Start the Server

```bash
python server.py
```

The server starts on `http://localhost:8000` and downloads required models on first run.

### Check Server Health

```bash
curl http://localhost:8000/health
```

### Generate Pidgin Speech

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "text": "How far, my guy? Na so life be sometimes.",
    "language": "pcm",
    "response_format": "wav"
  }' \
  --output speech.wav
```

### Generate Hausa Speech

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Sannu da zuwa, yaya kake? Ina fatan ka samu lafiya lau lau.",
    "language": "ha",
    "response_format": "wav"
  }' \
  --output hausa_speech.wav
```

### Generate Yoruba Speech (Streaming)

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Bawo ni o se wa? Mo fe ki a ba ara wa soro.",
    "language": "yo",
    "stream_format": "sse"
  }'
```

### Generate Igbo Speech

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Kedu ka i mere? Anyi nwere obi uto na anyi na-ekwuri okwu taa.",
    "language": "ig",
    "response_format": "wav"
  }' \
  --output igbo_speech.wav
```

### Generate MP3 Output

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "text": "How far, my people? Today na beautiful day.",
    "language": "pcm",
    "response_format": "mp3"
  }' \
  --output speech.mp3
```

### Generate FLAC (Lossless) Output

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Sannu da zuwa. Yaya aiki?",
    "language": "ha",
    "response_format": "flac"
  }' \
  --output speech.flac
```

### Generate with a Speaker

```bash
# List available Hausa speakers
curl http://localhost:8000/v1/speakers/ha

# Generate with a specific speaker (language auto-detected from speaker ID)
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Sannu da zuwa, yaya kake?",
    "speaker": "abdullahi_ha",
    "response_format": "wav"
  }' \
  --output abdullahi_speech.wav
```

### Voice Cloning

```bash9
# Step 1: Extract speaker embedding from reference audio
curl -X POST http://localhost:8000/v1/voice/clone \
  -F "file=@reference_voice.wav"

# Step 2: Use the returned embedding in speech generation
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "text": "How far, my people?",
    "language": "pcm",
    "speaker_embedding": [0.12, -0.34, ...],
    "response_format": "wav"
  }' \
  --output cloned_speech.wav

# Or do both in one step:
curl -X POST http://localhost:8000/v1/voice/clone/generate \
  -F "file=@reference_voice.wav" \
  -F "text=How far, my people?" \
  -F "language=pcm" \
  -F "response_format=wav" \
  --output cloned_speech.wav
```

## API Reference

### POST `/v1/audio/speech`

9jaLingo Speech API endpoint for text-to-speech generation with optional speaker embedding.

**Routing**: If `speaker` or `speaker_embedding` is provided, the request uses the Direct model path (supports voice control). Otherwise, it uses the vLLM fast path.

#### Request Body

```json
{
  "text": "Text to convert to speech",
  "model": "9javox",
  "language": "pcm",
  "response_format": "wav",
  "stream_format": null,
  "speaker": null,
  "speaker_embedding": null,
  "temperature": 0.6,
  "top_p": 0.85,
  "repetition_penalty": 1.3,
  "max_chunk_duration": 20.0,
  "silence_duration": 0.2
}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `text` | string | required | Text to convert to speech (also accepts `input` for compatibility) |
| `model` | string | `"9javox"` | Model name (`9javox`) |
| `language` | string | `"pcm"` | Language tag (`ha`, `ig`, `yo`, `pcm`) or `default` |
| `response_format` | string | `"wav"` | Output format: `wav`, `pcm`, `flac`, `alac`, `mp3`, `aac`, `ogg` |
| `stream_format` | string | `null` | `"sse"` for streaming, `null` for complete file |
| `speaker` | string | `null` | Speaker ID (e.g., `"abdullahi_ha"`) — auto-detects language |
| `speaker_embedding` | float[] | `null` | Raw 128-dim speaker embedding vector |
| `temperature` | float | `0.6` | Sampling temperature (0.3–1.5) |
| `top_p` | float | `0.85` | Nucleus sampling threshold |
| `repetition_penalty` | float | `1.3` | Repetition penalty (1.0–1.5) |
| `max_chunk_duration` | float | `20.0` | Max duration per chunk in long-form mode |
| `silence_duration` | float | `0.2` | Silence between chunks |

#### Available Language Tags

| Language Tag | Language | Example |
|-------------|----------|---------|
| `ha` | Hausa | "Sannu da zuwa, yaya kake?" |
| `ig` | Igbo | "Kedu ka i mere?" |
| `yo` | Yoruba | "Bawo ni o se wa?" |
| `pcm` | Nigerian Pidgin | "How far, my guy?" |
| `default` | No language prefix | English or auto-detect |

#### Response Formats

**Non-Streaming (response_format)**:

| Format | Type | MIME Type | Requires FFmpeg |
|--------|------|-----------|------------------|
| `wav` | Uncompressed | `audio/wav` | No |
| `pcm` | Uncompressed | `audio/pcm` | No |
| `flac` | Lossless | `audio/flac` | No (uses soundfile) |
| `alac` | Lossless | `audio/mp4` | Yes |
| `mp3` | Lossy | `audio/mpeg` | Yes |
| `aac` | Lossy | `audio/aac` | Yes |
| `ogg` | Lossy | `audio/ogg` | Yes |

**Streaming (stream_format)**:
- `sse` — Server-Sent Events with base64-encoded audio chunks
- `audio` — Raw audio streaming

#### Streaming Event Format (SSE)

```
data: {"type": "speech.audio.delta", "audio": "<base64_pcm_chunk>"}
data: {"type": "speech.audio.delta", "audio": "<base64_pcm_chunk>"}
data: {"type": "speech.audio.done", "usage": {"input_tokens": 25, "output_tokens": 487, "total_tokens": 512}}
```

### GET `/health`

Returns server and model status.

```json
{
  "status": "healthy",
  "tts_initialized": true,
  "speakers_loaded": 288,
  "direct_model_loaded": false,
  "voice_cloner_loaded": false,
  "supported_languages": {"ha": "Hausa", "ig": "Igbo", "yo": "Yoruba", "pcm": "Pidgin"}
}
```

### GET `/v1/speakers`

List all available speakers organized by language.

```json
{
  "total_speakers": 288,
  "languages": {
    "ha": {"name": "Hausa", "speaker_count": 33},
    "ig": {"name": "Igbo", "speaker_count": 75},
    "pcm": {"name": "Pidgin", "speaker_count": 86},
    "yo": {"name": "Yoruba", "speaker_count": 59}
  },
  "speakers": {
    "ha": [
      {"id": "abdullahi_ha", "display_name": "Abdullahi (ha)", "language": "ha"},
      {"id": "abubakar_ha", "display_name": "Abubakar (ha)", "language": "ha"}
    ]
  }
}
```

### GET `/v1/speakers/{language}`

List speakers for a specific language (`ha`, `ig`, `yo`, `pcm`).

```bash
curl http://localhost:8000/v1/speakers/yo
```

### POST `/v1/voice/clone`

Upload a reference audio file (3–30 seconds recommended) to extract a 128-dim speaker embedding.

**Request**: `multipart/form-data` with `file` field

**Response**:
```json
{
  "embedding": [0.12, -0.34, 0.56, ...],
  "dim": 128,
  "source_file": "reference_voice.wav",
  "usage": "Pass this embedding in the 'speaker_embedding' field of /v1/audio/speech"
}
```

### POST `/v1/voice/clone/generate`

Clone a voice from reference audio and generate speech in one step.

**Request**: `multipart/form-data`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `file` | file | required | Reference audio file |
| `text` | string | required | Text to speak |
| `language` | string | `"pcm"` | Language tag |
| `response_format` | string | `"wav"` | Output format |
| `temperature` | float | `0.6` | Sampling temperature |
| `top_p` | float | `0.85` | Nucleus sampling |
| `repetition_penalty` | float | `1.3` | Repetition penalty |

## Long-Form Generation

The model supports up to **40 seconds** of continuous audio per generation. For texts estimated beyond that (`LONG_FORM_THRESHOLD_SECONDS = 40.0` in `config.py`), the system automatically:

1. Splits text into sentence-based chunks ~20 seconds each
2. Generates each chunk with language tag and speaker embedding consistency
3. Concatenates audio segments with configurable silence (default: 0.2s)
4. Returns seamless combined audio

**Control long-form behavior**:
```json
{
  "text": "Very long text in Hausa...",
  "language": "ha",
  "speaker": "abdullahi_ha",
  "max_chunk_duration": 20.0,
  "silence_duration": 0.2
}
```

## Configuration

Key configuration parameters in [config.py](config.py):

```python
# Audio Settings
SAMPLE_RATE = 22050                    # Hz
CODEBOOK_SIZE = 4032                   # Codes per codebook
CHUNK_SIZE = 25                        # Frames per streaming chunk
LOOKBACK_FRAMES = 15                   # Context frames for decoding

# Generation Parameters
TEMPERATURE = 0.6                      # Sampling temperature
TOP_P = 0.85                           # Nucleus sampling threshold
REPETITION_PENALTY = 1.3               # Prevent repetition
MAX_TOKENS = 1500                      # ~30 seconds max audio

# Long-Form Settings
LONG_FORM_THRESHOLD_SECONDS = 40.0     # Auto-enable threshold
LONG_FORM_CHUNK_DURATION = 12.0        # Target chunk duration
LONG_FORM_SILENCE_DURATION = 0.2       # Inter-chunk silence

# Model
MODEL_NAME = "9jaLingo/9javox-9jalingo-finetuned-full-v1"
CODEC_MODEL_NAME = "nvidia/nemo-nano-codec-22khz-0.6kbps-12.5fps"

# Supported Languages
SUPPORTED_LANGUAGES = {
    "ha": "Hausa",
    "ig": "Igbo",
    "yo": "Yoruba",
    "pcm": "Pidgin",
}
```

## Performance

### Real-Time Factor (RTF)

Test generation speed across all supported languages:
```bash
uv run python test_rtf.py
```

Expected performance:
- **RTF Target**: < 0.5 (faster than real-time)
- **GPU Memory**: ~16GB, depends on `gpu_memory_utilization` parameter in `VLLMTTSGenerator`

### Optimization Tips

1. **GPU Memory**: Adjust `gpu_memory_utilization` in [server.py](server.py):
   ```python
   gpu_memory_utilization=0.9  # Reduce if OOM occurs
   ```

2. **Multi-GPU**: Enable tensor parallelism:
   ```python
   tensor_parallel_size=2  # For 2 GPUs
   ```

3. **Batch Processing**: Increase `max_num_seqs` for concurrent requests:
   ```python
   max_num_seqs=4  # Process 4 requests simultaneously
   ```

## Project Structure

```
9jalingo-vllm/
├── server.py               # FastAPI application and main entry point
├── config.py               # Configuration and constants
├── test_rtf.py             # Performance testing utility
├── requirements.txt        # Python dependencies
├── audio/                  # Audio processing modules
│   ├── __init__.py
│   ├── encoder.py          # Multi-format audio encoder (WAV/FLAC/MP3/AAC/OGG/ALAC)
│   ├── player.py           # NeMo audio codec decoder
│   └── streaming.py        # Streaming audio writer with sliding window
└── generation/             # TTS generation modules
    ├── __init__.py
    ├── vllm_generator.py   # vLLM engine wrapper and generation
    └── chunking.py         # Text chunking for long-form
```

## How It Works

### 1. Token Generation Pipeline

```
Input Text
    |
[Add language tag prefix + special tokens]
    |
vLLM AsyncEngine (streaming token generation)
    |
Token Stream: Text + START_OF_SPEECH + Audio Tokens + END_OF_SPEECH
    |
Filter audio tokens (groups of 4 for codec)
```

### 2. Audio Decoding

```
Audio Tokens (groups of 4 per frame)
    |
Buffer tokens in streaming writer
    |
Sliding window decoder (with lookback context)
    |
NVIDIA NeMo NanoCodec (4 codebooks → PCM)
    |
16-bit PCM audio @ 22050 Hz
```

### 3. Special Token Architecture

The model uses special tokens to structure generation:
- `START_OF_HUMAN`, `END_OF_HUMAN` — Wrap input text
- `START_OF_AI`, `END_OF_AI` — Mark model's response boundaries
- `START_OF_SPEECH`, `END_OF_SPEECH` — Delimit audio token sequences
- Audio tokens map to 4 codebook indices per 80ms frame

### 4. Language Selection

Language selection is achieved by prepending language tags to prompts:
```
Input: "Sannu da zuwa"
Language: "ha"
Prompt: "ha: Sannu da zuwa"
```

This guides the model to generate speech in the requested Nigerian language.

## Advanced Usage

### Adjusting Generation Quality

Modify generation parameters in [config.py](config.py):
```python
TEMPERATURE = 0.6         # Lower = more deterministic (0.3-1.5)
TOP_P = 0.85              # Nucleus sampling threshold
REPETITION_PENALTY = 1.3  # Prevent repetition (1.0-1.5)
```

### PCM Output with Custom Processing

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Na wetin dey happen for Lagos today.",
    "language": "pcm",
    "response_format": "pcm"
  }' \
  --output speech.pcm

# Headers will include:
# X-Sample-Rate: 22050
# X-Channels: 1
# X-Bit-Depth: 16
```

### Compressed Output Formats

For lossy formats (MP3, AAC, OGG) and lossless ALAC, FFmpeg must be installed on the system:

```bash
# MP3 — best compatibility, lossy
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"text": "Bawo ni o se wa?", "language": "yo", "response_format": "mp3"}' \
  --output speech.mp3

# AAC — modern lossy, good quality/size ratio
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"text": "Kedu ka i mere?", "language": "ig", "response_format": "aac"}' \
  --output speech.aac

# OGG Vorbis — open-source lossy
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"text": "Ina kwana?", "language": "ha", "response_format": "ogg"}' \
  --output speech.ogg

# FLAC — lossless compression (no FFmpeg needed)
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"text": "How body?", "language": "pcm", "response_format": "flac"}' \
  --output speech.flac

# ALAC — Apple lossless (requires FFmpeg)
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"text": "Sannu da zuwa.", "language": "ha", "response_format": "alac"}' \
  --output speech.m4a
```

## Troubleshooting

### Out of Memory (OOM)

Reduce GPU memory utilization in [server.py](server.py):
```python
gpu_memory_utilization=0.7  # Lower from 0.9
```

Or reduce max model length:
```python
max_model_len=1024  # 50 tokens ≈ 1 second of audio
```

### Slow Generation

1. Check RTF with `python test_rtf.py`
2. Ensure CUDA is properly installed: `torch.cuda.is_available()`
3. Verify GPU utilization: `nvidia-smi`
4. Consider enabling CUDA graphs (already default)

### FFmpeg Not Found

If you get an error when requesting MP3, AAC, OGG, or ALAC formats:
```bash
sudo apt install ffmpeg
ffmpeg -version  # verify installation
```
FLAC and WAV/PCM formats work without FFmpeg.

### Audio Quality Issues

1. Ensure sample rate matches (22050 Hz)
2. For long-form, adjust chunk duration:
   ```json
   {"max_chunk_duration": 10.0}
   ```
3. Increase lookback frames for smoother transitions in [config.py](config.py):
   ```python
   LOOKBACK_FRAMES = 20  # More context
   ```

### Model Download Issues

Models are automatically downloaded from HuggingFace on first run. If downloads fail:
```bash
# Pre-download models
python -c "
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('9jaLingo/9javox-9jalingo-finetuned-full-v1')
# Model will be downloaded by vLLM on first use
"
```

### vLLM Model Architecture Support

If vLLM does not yet support the model's architecture (LFM2), you may see an error at startup. Check the [vLLM supported models list](https://docs.vllm.ai/en/latest/models/supported_models.html) for compatibility. You can also try:
```bash
uv pip install vllm --upgrade
```

## Production Deployment

### Security Considerations

**Update CORS settings in** [server.py](server.py):
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://yourdomain.com"],  # Restrict origins
    allow_credentials=True,
    allow_methods=["POST", "GET"],
    allow_headers=["Content-Type"],
)
```

### Recommendations

1. **Add Authentication**: Implement API keys or OAuth
2. **Rate Limiting**: Prevent abuse with request limits
3. **Monitoring**: Track token usage via the `usage` field in responses
4. **Timeouts**: Adjust request timeouts for long-form generation
5. **Load Balancing**: Deploy multiple instances with GPU-aware routing

### Docker Deployment

Create a `Dockerfile`:
```dockerfile
FROM nvidia/cuda:12.8.0-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    curl \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.cargo/bin:$PATH"

COPY . .

RUN uv pip install "naijalingo-tts-2==0.1.1" && \
    uv pip install fastapi "uvicorn[standard]" && \
    uv pip install vllm --torch-backend=auto && \
    uv pip install "transformers==4.56.0"

EXPOSE 8000

CMD ["uv", "run", "python", "server.py"]
```

Build and run:
```bash
docker build -t 9jalingo-vllm-tts .
docker run --gpus all -p 8000:8000 9jalingo-vllm-tts
```

## Limitations

1. **Speaker Embeddings**: The vLLM version uses language tags for voice selection. Speaker-specific voice control (via 128-dim embeddings) is available in the standard `naijalingo-tts-2` module but not in the vLLM path due to engine constraints. Voice cloning requires the non-vLLM pipeline.
2. **Max Audio Length**: ~30 seconds per single generation (max_tokens=1500). Use long-form mode for longer texts.
3. **Codec Artifacts**: 0.6 kbps compression may introduce minor artifacts (quality/speed tradeoff).
4. **GPU Inference**: Designed for GPU inference; not tested on CPU or TPU.
5. **Single Request Processing**: Optimized for one request at a time (increase `max_num_seqs` for concurrent processing).

## Related Projects

- [`naijalingo-tts-2`](https://pypi.org/project/naijalingo-tts-2/) — Core TTS module with full speaker embedding and voice cloning support
- [`9jalingoTTS-2`](../9jalingoTTS-2/) — Gradio-based demo with speaker selection UI

## License

Proprietary — Copyright (c) 2025 9jaLingo. All rights reserved.

This software is closed-source and may not be copied, modified, distributed, or reverse-engineered without explicit written authorization from 9jaLingo. See [LICENSE](LICENSE) for full terms.

For licensing inquiries, contact: legal@9jalingo.com

## References

- [vLLM Documentation](https://docs.vllm.ai/)
- [naijalingo-tts-2 on PyPI](https://pypi.org/project/naijalingo-tts-2/)

---

**9javox** is 9jaLingo's legacy TTS model — the first production-grade speech system for Nigerian languages.
