"""FastAPI server for 9jaLingo TTS with streaming, speaker embeddings, and voice cloning"""

import io
import tempfile
import os
from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, Response
from pydantic import AliasChoices, BaseModel, ConfigDict, Field
from typing import Optional, Literal, List
import numpy as np
from scipy.io.wavfile import write as wav_write
import base64
import json
import torch

from audio import NaijaLingoAudioPlayer, StreamingAudioWriter, encode_audio, SUPPORTED_FORMATS
from generation.vllm_generator import VLLMTTSGenerator
from generation.direct_generator import DirectTTSGenerator
from speakers import SpeakerManager
from config import (
    CHUNK_SIZE, LOOKBACK_FRAMES, TEMPERATURE, TOP_P, MAX_TOKENS,
    LONG_FORM_THRESHOLD_SECONDS, LONG_FORM_SILENCE_DURATION,
    LONG_FORM_CHUNK_DURATION, SUPPORTED_LANGUAGES,
    SPEAKERS_DIR, SPEAKER_EMB_DIM, GENERATION_MANIFEST_PATH,
)

from nemo.utils.nemo_logging import Logger

nemo_logger = Logger()
nemo_logger.remove_stream_handlers()


app = FastAPI(title="9jaLingo TTS API", version="2.0.0",
             description="Text-to-Speech for Nigerian Languages with Speaker Embeddings & Voice Cloning")

# Add CORS middleware to allow client connections
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances (initialized on startup)
generator = None        # vLLM engine — fast path (no speaker embeddings)
player = None           # NeMo audio codec decoder
speaker_manager = None  # Speaker .pt file manager
direct_generator = None # Direct model — speaker embedding & voice cloning path


class TTSRequest(BaseModel):
    text: str
    language_tag: Optional[str] = None
    temperature: Optional[float] = TEMPERATURE
    max_tokens: Optional[int] = MAX_TOKENS
    top_p: Optional[float] = TOP_P
    chunk_size: Optional[int] = CHUNK_SIZE
    lookback_frames: Optional[int] = LOOKBACK_FRAMES


class OpenAISpeechRequest(BaseModel):
    """OpenAI-compatible speech request model with speaker embedding support"""
    text: str = Field(..., validation_alias=AliasChoices("text", "input"), description="Text to convert to speech")
    model: Literal["tts-1", "tts-1-hd", "9javox"] = Field(
        default="9javox",
        description="TTS model to use (9javox is the 9jaLingo model)"
    )
    voice: str = Field(
        default="pcm",
        description="Language tag (ha, ig, yo, pcm) or 'default' for no language prefix"
    )
    response_format: Literal["wav", "pcm", "flac", "alac", "mp3", "aac", "ogg"] = Field(
        default="wav",
        description="Audio output format. Uncompressed: wav, pcm. Lossless: flac, alac. Lossy: mp3, aac, ogg."
    )
    stream_format: Optional[Literal["sse", "audio"]] = Field(
        default=None,
        description="Use 'sse' for Server-Sent Events streaming"
    )
    # Speaker embedding support
    speaker: Optional[str] = Field(
        default=None,
        description="Speaker ID (e.g., 'abdullahi_ha') — auto-detects language from speaker. Uses direct model path."
    )
    speaker_embedding: Optional[List[float]] = Field(
        default=None,
        description="Raw 128-dim speaker embedding vector. Uses direct model path."
    )
    # Generation parameters
    temperature: Optional[float] = Field(
        default=TEMPERATURE,
        description="Sampling temperature (0.3-1.5, default 1.0)"
    )
    top_p: Optional[float] = Field(
        default=TOP_P,
        description="Nucleus sampling threshold (default 0.95)"
    )
    repetition_penalty: Optional[float] = Field(
        default=1.1,
        description="Repetition penalty (1.0-1.5, default 1.1)"
    )
    # Long-form generation parameters
    enable_long_form: Optional[bool] = Field(
        default=True,
        description="Auto-detect and use long-form generation for texts >40s"
    )
    max_chunk_duration: Optional[float] = Field(
        default=20.0,
        description="Max duration per chunk in long-form mode (seconds)"
    )
    silence_duration: Optional[float] = Field(
        default=0.2,
        description="Silence between chunks in long-form mode (seconds)"
    )


@app.on_event("startup")
async def startup_event():
    """Initialize models on startup"""
    global generator, player, speaker_manager, direct_generator
    print("🚀 Initializing 9jaLingo vLLM TTS models...")

    # 1. vLLM engine — fast path (language-tag-based generation, no speaker embeddings)
    generator = VLLMTTSGenerator(
        tensor_parallel_size=1,        # Increase for multi-GPU
        gpu_memory_utilization=0.5,    # Adjust based on available VRAM
        max_model_len=1024             # Maximum sequence length
    )
    await generator.initialize_engine()

    player = NaijaLingoAudioPlayer(generator.tokenizer)

    # 2. Speaker manager — load pre-computed speaker embeddings
    speaker_manager = SpeakerManager(
        speakers_dir=SPEAKERS_DIR,
        manifest_path=GENERATION_MANIFEST_PATH,
    )

    # 3. Direct generator — lazy-loaded only when speaker/cloning requests arrive
    direct_generator = DirectTTSGenerator()

    print("✅ 9jaLingo vLLM TTS models initialized successfully!")
    print(f"   Supported languages: {', '.join(f'{v} ({k})' for k, v in SUPPORTED_LANGUAGES.items())}")
    print(f"   Speakers loaded: {speaker_manager.total_speakers} across {speaker_manager.available_languages}")
    print(f"   Speaker/cloning path: lazy-loaded on first request")


@app.get("/health")
async def health_check():
    """Check if server is ready"""
    return {
        "status": "healthy",
        "tts_initialized": generator is not None and player is not None,
        "speakers_loaded": speaker_manager.total_speakers if speaker_manager else 0,
        "direct_model_loaded": direct_generator._model_loaded if direct_generator else False,
        "voice_cloner_loaded": direct_generator._embedder_loaded if direct_generator else False,
        "supported_languages": SUPPORTED_LANGUAGES,
    }


@app.post("/v1/audio/speech",
          responses={200: {"content": {"audio/wav": {}, "audio/mpeg": {}, "audio/flac": {}, "audio/ogg": {}, "audio/aac": {}}}},
          response_class=Response)
async def openai_speech(request: OpenAISpeechRequest):
    """9jaLingo speech generation endpoint

    Dual-engine routing:
    - If speaker or speaker_embedding is provided → Direct NaijaLingoTTS model (supports speaker embeddings)
    - Otherwise → vLLM AsyncEngine (fast path, language-tag only)

    Supports both streaming (SSE) and non-streaming modes.
    The 'voice' parameter selects a language tag (ha, ig, yo, pcm) or 'default'.
    """
    if not generator or not player:
        raise HTTPException(status_code=503, detail="TTS models not initialized")

    # ── Resolve speaker embedding (triggers direct model path if present) ──
    speaker_emb = None
    use_direct_path = False

    if request.speaker:
        # Load pre-computed speaker embedding from .pt file
        if not speaker_manager or not speaker_manager.has_speaker(request.speaker):
            raise HTTPException(
                status_code=404,
                detail=f"Speaker '{request.speaker}' not found. Use GET /v1/speakers to list available speakers."
            )
        speaker_emb = speaker_manager.get_embedding(request.speaker)
        if speaker_emb is None:
            raise HTTPException(status_code=500, detail=f"Failed to load embedding for speaker '{request.speaker}'")
        use_direct_path = True
        # Auto-detect language from speaker ID if voice is default
        speaker_lang = speaker_manager.get_speaker_language(request.speaker)
        if speaker_lang:
            language_tag = speaker_lang
        else:
            language_tag = request.voice if request.voice in SUPPORTED_LANGUAGES else None
        print(f"[Server] Speaker path: {request.speaker} → language={language_tag}")

    elif request.speaker_embedding:
        # Parse raw embedding vector
        if len(request.speaker_embedding) != SPEAKER_EMB_DIM:
            raise HTTPException(
                status_code=400,
                detail=f"speaker_embedding must be {SPEAKER_EMB_DIM}-dim, got {len(request.speaker_embedding)}"
            )
        speaker_emb = torch.tensor([request.speaker_embedding], dtype=torch.float32)
        use_direct_path = True
        language_tag = request.voice if request.voice in SUPPORTED_LANGUAGES else None
        print(f"[Server] Custom embedding path → language={language_tag}")

    else:
        # Standard vLLM path — no speaker embedding
        language_tag = request.voice if request.voice in SUPPORTED_LANGUAGES else None

    # Build prompt text with optional language tag prefix
    prompt_text = VLLMTTSGenerator.build_prompt(request.text, language_tag)

    # ── Direct model path (speaker embeddings) ──
    if use_direct_path:
        return await _generate_direct(request, language_tag, speaker_emb)

    # ── vLLM fast path (no speaker embedding) ──
    # Streaming mode (SSE)
    if request.stream_format == "sse":
        async def sse_generator():
            """Generate Server-Sent Events with audio chunks"""
            import asyncio
            import queue as thread_queue
            from generation.chunking import estimate_duration, split_into_sentences

            chunk_queue = thread_queue.Queue()

            # Estimate duration to determine if we need long-form generation
            estimated_duration = estimate_duration(request.text)
            lang_for_generation = language_tag
            use_long_form = estimated_duration > LONG_FORM_THRESHOLD_SECONDS

            # Track token counts for usage reporting
            input_token_count = 0
            output_token_count = 0

            if use_long_form:
                # Long-form streaming: stream each sentence chunk as it's generated
                print(f"[Server] Using long-form SSE streaming (estimated {estimated_duration:.1f}s)")

                async def generate_async_long_form():
                    nonlocal input_token_count, output_token_count
                    try:
                        chunks = split_into_sentences(
                            request.text,
                            max_duration_seconds=request.max_chunk_duration or LONG_FORM_CHUNK_DURATION
                        )
                        total_chunks = len(chunks)

                        for i, text_chunk in enumerate(chunks):
                            # Custom list wrapper that pushes chunks to queue
                            class ChunkList(list):
                                def append(self, chunk):
                                    super().append(chunk)
                                    chunk_queue.put(("chunk", chunk))

                            audio_writer = StreamingAudioWriter(
                                player,
                                output_file=None,
                                chunk_size=CHUNK_SIZE,
                                lookback_frames=LOOKBACK_FRAMES
                            )
                            audio_writer.audio_chunks = ChunkList()
                            audio_writer.start()

                            # Generate with language tag prefix
                            chunk_prompt = VLLMTTSGenerator.build_prompt(text_chunk, lang_for_generation)
                            result = await generator._generate_async(
                                chunk_prompt,
                                audio_writer,
                                max_tokens=MAX_TOKENS
                            )
                            audio_writer.finalize()

                            # Track tokens
                            input_token_count += len(generator.prepare_input(chunk_prompt))
                            output_token_count += len(result.get('all_token_ids', []))

                            # Add silence between chunks (except after last chunk)
                            if i < total_chunks - 1:
                                silence_samples = int(
                                    (request.silence_duration or LONG_FORM_SILENCE_DURATION) * 22050
                                )
                                silence = np.zeros(silence_samples, dtype=np.float32)
                                chunk_queue.put(("chunk", silence))

                        chunk_queue.put(("done", {"input": input_token_count, "output": output_token_count}))
                    except Exception as e:
                        print(f"Generation error: {e}")
                        import traceback
                        traceback.print_exc()
                        chunk_queue.put(("error", str(e)))

                gen_task = asyncio.create_task(generate_async_long_form())
            else:
                # Standard streaming for short texts
                print(f"[Server] Using standard SSE streaming (estimated {estimated_duration:.1f}s)")

                class ChunkList(list):
                    def append(self, chunk):
                        super().append(chunk)
                        chunk_queue.put(("chunk", chunk))

                audio_writer = StreamingAudioWriter(
                    player,
                    output_file=None,
                    chunk_size=CHUNK_SIZE,
                    lookback_frames=LOOKBACK_FRAMES
                )
                audio_writer.audio_chunks = ChunkList()

                async def generate_async():
                    nonlocal input_token_count, output_token_count
                    try:
                        audio_writer.start()
                        result = await generator._generate_async(
                            prompt_text,
                            audio_writer,
                            max_tokens=MAX_TOKENS
                        )
                        audio_writer.finalize()

                        input_token_count = len(generator.prepare_input(prompt_text))
                        output_token_count = len(result.get('all_token_ids', []))

                        chunk_queue.put(("done", {"input": input_token_count, "output": output_token_count}))
                    except Exception as e:
                        print(f"Generation error: {e}")
                        import traceback
                        traceback.print_exc()
                        chunk_queue.put(("error", str(e)))

                gen_task = asyncio.create_task(generate_async())

            # Stream chunks as they arrive
            try:
                while True:
                    msg_type, data = await asyncio.get_event_loop().run_in_executor(
                        None, lambda: chunk_queue.get(timeout=30)
                    )

                    if msg_type == "chunk":
                        # Convert numpy array to int16 PCM
                        pcm_data = (data * 32767).astype(np.int16)

                        # Encode as base64
                        audio_base64 = base64.b64encode(pcm_data.tobytes()).decode('utf-8')

                        # Send SSE event: speech.audio.delta
                        event_data = {
                            "type": "speech.audio.delta",
                            "audio": audio_base64
                        }
                        yield f"data: {json.dumps(event_data)}\n\n"

                    elif msg_type == "done":
                        token_counts = data
                        event_data = {
                            "type": "speech.audio.done",
                            "usage": {
                                "input_tokens": token_counts["input"],
                                "output_tokens": token_counts["output"],
                                "total_tokens": token_counts["input"] + token_counts["output"]
                            }
                        }
                        yield f"data: {json.dumps(event_data)}\n\n"
                        break

                    elif msg_type == "error":
                        error_data = {
                            "type": "error",
                            "error": data
                        }
                        yield f"data: {json.dumps(error_data)}\n\n"
                        break

            finally:
                await gen_task

        return StreamingResponse(
            sse_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"
            }
        )

    # Non-streaming mode (complete audio file)
    else:
        try:
            from generation.chunking import estimate_duration
            estimated_duration = estimate_duration(request.text)

            lang_for_generation = language_tag
            use_long_form = estimated_duration > LONG_FORM_THRESHOLD_SECONDS

            if use_long_form:
                print(f"[Server] Using long-form generation (estimated {estimated_duration:.1f}s)")
                result = await generator.generate_long_form_async(
                    text=request.text,
                    language_tag=lang_for_generation,
                    player=player,
                    max_chunk_duration=request.max_chunk_duration or LONG_FORM_CHUNK_DURATION,
                    silence_duration=request.silence_duration or LONG_FORM_SILENCE_DURATION,
                    max_tokens=MAX_TOKENS
                )
                full_audio = result['audio']
            else:
                print(f"[Server] Using standard generation (estimated {estimated_duration:.1f}s)")
                audio_writer = StreamingAudioWriter(
                    player,
                    output_file=None,
                    chunk_size=CHUNK_SIZE,
                    lookback_frames=LOOKBACK_FRAMES
                )
                audio_writer.start()

                result = await generator._generate_async(
                    prompt_text,
                    audio_writer,
                    max_tokens=MAX_TOKENS
                )

                audio_writer.finalize()

                if not audio_writer.audio_chunks:
                    raise HTTPException(status_code=500, detail="No audio generated")

                full_audio = np.concatenate(audio_writer.audio_chunks)

            # Encode audio into requested format
            fmt = request.response_format
            encoded_bytes, mime_type = encode_audio(full_audio, fmt)

            # Build response headers
            headers = {"Content-Type": mime_type}
            if fmt == "pcm":
                headers.update({
                    "X-Sample-Rate": "22050",
                    "X-Channels": "1",
                    "X-Bit-Depth": "16",
                })

            return Response(
                content=encoded_bytes,
                media_type=mime_type,
                headers=headers,
            )

        except Exception as e:
            print(e)
            raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def root():
    """Root endpoint with API info"""
    return {
        "name": "9jaLingo TTS API (9javox)",
        "version": "2.0.0",
        "model": "9javox",
        "description": "Text-to-Speech for Nigerian Languages — Hausa, Igbo, Yoruba, Pidgin",
        "features": [
            "vLLM fast inference (language-tag path)",
            "Speaker embeddings (128-dim, pre-computed .pt files)",
            "Voice cloning (upload reference audio)",
            "SSE streaming",
            "Multi-format output (WAV, FLAC, MP3, AAC, OGG, ALAC, PCM)",
            "Up to 40s continuous generation",
            "Long-form auto-chunking",
        ],
        "endpoints": {
            "/v1/audio/speech": "POST - Speech generation (supports speaker & cloning)",
            "/v1/speakers": "GET - List all speakers by language",
            "/v1/speakers/{language}": "GET - List speakers for a language",
            "/v1/voice/clone": "POST - Extract speaker embedding from audio",
            "/v1/voice/clone/generate": "POST - Clone voice + generate speech",
            "/health": "GET - Health check",
        },
        "supported_languages": SUPPORTED_LANGUAGES,
        "speakers_loaded": speaker_manager.total_speakers if speaker_manager else 0,
        "supported_formats": {
            k: {"mime": v["mime"], "category": v["category"]}
            for k, v in SUPPORTED_FORMATS.items()
        },
    }


# ═══════════════════════════════════════════════════════════════════════
# Direct model path — speaker embedding & voice cloning generation
# ═══════════════════════════════════════════════════════════════════════

async def _generate_direct(request: OpenAISpeechRequest, language_tag, speaker_emb):
    """Generate speech using the direct NaijaLingoTTS model with speaker embedding.

    Used when speaker or speaker_embedding is provided in the request.
    Supports both streaming (SSE) and non-streaming modes.
    """
    from generation.chunking import estimate_duration

    # SSE streaming with speaker: generate full audio, then stream in chunks
    if request.stream_format == "sse":
        async def sse_direct_generator():
            try:
                estimated_duration = estimate_duration(request.text)
                print(f"[Server/Direct SSE] Estimated: {estimated_duration:.1f}s, with speaker embedding")

                if estimated_duration > LONG_FORM_THRESHOLD_SECONDS:
                    full_audio = direct_generator.generate_long_form(
                        text=request.text,
                        language_tag=language_tag,
                        speaker_emb=speaker_emb,
                        max_chunk_duration=request.max_chunk_duration or LONG_FORM_CHUNK_DURATION,
                        silence_duration=request.silence_duration or LONG_FORM_SILENCE_DURATION,
                        temperature=request.temperature or TEMPERATURE,
                        top_p=request.top_p or TOP_P,
                        repetition_penalty=request.repetition_penalty or 1.1,
                    )
                else:
                    full_audio, _ = direct_generator.generate(
                        text=request.text,
                        language_tag=language_tag,
                        speaker_emb=speaker_emb,
                        temperature=request.temperature or TEMPERATURE,
                        top_p=request.top_p or TOP_P,
                        repetition_penalty=request.repetition_penalty or 1.1,
                    )

                # Stream the generated audio in PCM chunks via SSE
                CHUNK_SAMPLES = int(0.5 * 22050)  # 0.5s chunks
                for i in range(0, len(full_audio), CHUNK_SAMPLES):
                    chunk = full_audio[i:i + CHUNK_SAMPLES]
                    pcm_data = (chunk * 32767).astype(np.int16)
                    audio_base64 = base64.b64encode(pcm_data.tobytes()).decode('utf-8')
                    event_data = {"type": "speech.audio.delta", "audio": audio_base64}
                    yield f"data: {json.dumps(event_data)}\n\n"

                # Done event
                event_data = {
                    "type": "speech.audio.done",
                    "engine": "direct",
                    "usage": {"audio_duration": len(full_audio) / 22050},
                }
                yield f"data: {json.dumps(event_data)}\n\n"

            except Exception as e:
                print(f"[Server/Direct SSE] Error: {e}")
                import traceback
                traceback.print_exc()
                yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"

        return StreamingResponse(
            sse_direct_generator(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
        )

    # Non-streaming with speaker
    else:
        try:
            estimated_duration = estimate_duration(request.text)
            print(f"[Server/Direct] Estimated: {estimated_duration:.1f}s, with speaker embedding")

            if estimated_duration > LONG_FORM_THRESHOLD_SECONDS:
                full_audio = direct_generator.generate_long_form(
                    text=request.text,
                    language_tag=language_tag,
                    speaker_emb=speaker_emb,
                    max_chunk_duration=request.max_chunk_duration or LONG_FORM_CHUNK_DURATION,
                    silence_duration=request.silence_duration or LONG_FORM_SILENCE_DURATION,
                    temperature=request.temperature or TEMPERATURE,
                    top_p=request.top_p or TOP_P,
                    repetition_penalty=request.repetition_penalty or 1.1,
                )
            else:
                full_audio, _ = direct_generator.generate(
                    text=request.text,
                    language_tag=language_tag,
                    speaker_emb=speaker_emb,
                    temperature=request.temperature or TEMPERATURE,
                    top_p=request.top_p or TOP_P,
                    repetition_penalty=request.repetition_penalty or 1.1,
                )

            # Encode audio into requested format
            fmt = request.response_format
            encoded_bytes, mime_type = encode_audio(full_audio, fmt)

            headers = {"Content-Type": mime_type, "X-Engine": "direct"}
            if fmt == "pcm":
                headers.update({"X-Sample-Rate": "22050", "X-Channels": "1", "X-Bit-Depth": "16"})

            return Response(content=encoded_bytes, media_type=mime_type, headers=headers)

        except Exception as e:
            print(f"[Server/Direct] Error: {e}")
            import traceback
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=str(e))


# ═══════════════════════════════════════════════════════════════════════
# Speaker endpoints
# ═══════════════════════════════════════════════════════════════════════

@app.get("/v1/speakers")
async def list_speakers(
    gender: Optional[str] = Query(None, description="Filter by gender: male or female"),
    domain: Optional[str] = Query(None, description="Filter by domain name"),
):
    """List all available speakers organized by language.

    Returns speakers grouped by language with counts, gender breakdown,
    and available domains. Supports optional gender and domain filtering.
    """
    if not speaker_manager:
        raise HTTPException(status_code=503, detail="Speaker manager not initialized")

    languages = speaker_manager.get_languages()
    speakers_by_lang = {}
    for lang in languages:
        speakers_by_lang[lang] = speaker_manager.list_speakers(
            language=lang, gender=gender, domain=domain
        )

    total_filtered = sum(len(s) for s in speakers_by_lang.values())

    return {
        "total_speakers": speaker_manager.total_speakers,
        "filtered_count": total_filtered,
        "languages": languages,
        "domains": speaker_manager.get_domains(),
        "speakers": speakers_by_lang,
    }


@app.get("/v1/speakers/{language}")
async def list_speakers_by_language(
    language: str,
    gender: Optional[str] = Query(None, description="Filter by gender: male or female"),
    domain: Optional[str] = Query(None, description="Filter by domain name"),
):
    """List speakers for a specific language.

    Args:
        language: Language code (ha, ig, yo, pcm)
        gender: Optional gender filter
        domain: Optional domain filter
    """
    if not speaker_manager:
        raise HTTPException(status_code=503, detail="Speaker manager not initialized")

    if language not in SUPPORTED_LANGUAGES:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown language '{language}'. Available: {list(SUPPORTED_LANGUAGES.keys())}"
        )

    speakers = speaker_manager.list_speakers(
        language=language, gender=gender, domain=domain
    )
    lang_name = SUPPORTED_LANGUAGES.get(language, language)

    return {
        "language": language,
        "language_name": lang_name,
        "count": len(speakers),
        "speakers": speakers,
    }


# ═══════════════════════════════════════════════════════════════════════
# Voice cloning endpoints
# ═══════════════════════════════════════════════════════════════════════

@app.post("/v1/voice/clone")
async def clone_voice(file: UploadFile = File(..., description="Reference audio file (WAV, MP3, FLAC, etc.)")):
    """Extract a 128-dim speaker embedding from a reference audio file.

    Upload a short audio clip (3-30 seconds recommended) and receive a speaker
    embedding that can be used in /v1/audio/speech via the speaker_embedding field.

    Returns:
        JSON with embedding vector (128 floats) and metadata
    """
    if not direct_generator:
        raise HTTPException(status_code=503, detail="Voice cloning not available")

    # Save uploaded file to temp location
    suffix = os.path.splitext(file.filename)[1] if file.filename else ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        embedding = direct_generator.clone_voice(tmp_path)
        embedding_list = embedding.squeeze().tolist()

        return {
            "embedding": embedding_list,
            "dim": len(embedding_list),
            "source_file": file.filename,
            "usage": "Pass this embedding in the 'speaker_embedding' field of /v1/audio/speech",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Voice cloning failed: {str(e)}")
    finally:
        os.unlink(tmp_path)


@app.post("/v1/voice/clone/generate",
          responses={200: {"content": {"audio/wav": {}, "audio/mpeg": {}, "audio/flac": {}, "audio/ogg": {}, "audio/aac": {}}}},
          response_class=Response)
async def clone_and_generate(
    file: UploadFile = File(..., description="Reference audio file for voice cloning"),
    text: str = Form(..., description="Text to convert to speech"),
    voice: str = Form(default="pcm", description="Language tag (ha, ig, yo, pcm)"),
    response_format: str = Form(default="wav", description="Output format (wav, mp3, flac, etc.)"),
    temperature: float = Form(default=1.0, description="Sampling temperature"),
    top_p: float = Form(default=0.95, description="Nucleus sampling threshold"),
    repetition_penalty: float = Form(default=1.1, description="Repetition penalty"),
):
    """Clone a voice from reference audio and generate speech in one step.

    Upload a reference audio + text, and receive generated speech in the cloned voice.
    This combines /v1/voice/clone and /v1/audio/speech into a single request.
    """
    if not direct_generator:
        raise HTTPException(status_code=503, detail="Voice cloning not available")

    # Save uploaded file
    suffix = os.path.splitext(file.filename)[1] if file.filename else ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        language_tag = voice if voice in SUPPORTED_LANGUAGES else None

        audio, speaker_emb = direct_generator.clone_and_generate(
            text=text,
            audio_data=tmp_path,
            language_tag=language_tag,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        )

        # Encode audio
        fmt = response_format if response_format in SUPPORTED_FORMATS else "wav"
        encoded_bytes, mime_type = encode_audio(audio, fmt)

        headers = {
            "Content-Type": mime_type,
            "X-Engine": "direct",
            "X-Speaker-Embedding-Dim": str(SPEAKER_EMB_DIM),
        }
        if fmt == "pcm":
            headers.update({"X-Sample-Rate": "22050", "X-Channels": "1", "X-Bit-Depth": "16"})

        return Response(content=encoded_bytes, media_type=mime_type, headers=headers)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Clone + generate failed: {str(e)}")
    finally:
        os.unlink(tmp_path)


if __name__ == "__main__":
    import uvicorn
    print("🇳🇬 Starting 9jaLingo TTS Server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
