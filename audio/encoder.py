"""Audio format encoding utilities for 9jaLingo TTS.

Converts raw float32 PCM audio (22050 Hz, mono) into various output formats.

Supported formats:
    Uncompressed:
        - wav   — 16-bit PCM WAV (maximum fidelity, largest files)
        - pcm   — Raw 16-bit signed little-endian PCM bytes

    Lossless:
        - flac  — Free Lossless Audio Codec (smaller than WAV, bit-perfect)
        - alac  — Apple Lossless Audio Codec (Apple ecosystem compatible)

    Lossy:
        - mp3   — MPEG-1 Audio Layer III (universal playback, ~192 kbps)
        - aac   — Advanced Audio Coding (better quality/size than MP3)
        - ogg   — Ogg Vorbis (open-source, great for streaming)

Requires: ffmpeg (system binary) for lossy and ALAC encoding.
          soundfile + libsndfile for FLAC encoding (fallback to ffmpeg).
"""

import io
import subprocess
import shutil
import numpy as np
from scipy.io.wavfile import write as wav_write
from typing import Literal

from config import SAMPLE_RATE


# All supported response formats
AudioFormat = Literal["wav", "pcm", "flac", "alac", "mp3", "aac", "ogg"]

SUPPORTED_FORMATS = {
    "wav":  {"mime": "audio/wav",           "ext": ".wav",  "category": "uncompressed"},
    "pcm":  {"mime": "application/octet-stream", "ext": ".pcm", "category": "uncompressed"},
    "flac": {"mime": "audio/flac",          "ext": ".flac", "category": "lossless"},
    "alac": {"mime": "audio/mp4",           "ext": ".m4a",  "category": "lossless"},
    "mp3":  {"mime": "audio/mpeg",          "ext": ".mp3",  "category": "lossy"},
    "aac":  {"mime": "audio/aac",           "ext": ".m4a",  "category": "lossy"},
    "ogg":  {"mime": "audio/ogg",           "ext": ".ogg",  "category": "lossy"},
}


def _check_ffmpeg() -> bool:
    """Check if ffmpeg is available on the system."""
    return shutil.which("ffmpeg") is not None


def _audio_to_wav_bytes(audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> bytes:
    """Convert float32 audio array to WAV bytes (16-bit PCM)."""
    # Clip to [-1, 1] and convert to int16
    audio_clipped = np.clip(audio, -1.0, 1.0)
    audio_int16 = (audio_clipped * 32767).astype(np.int16)
    buf = io.BytesIO()
    wav_write(buf, sample_rate, audio_int16)
    buf.seek(0)
    return buf.read()


def _ffmpeg_convert(wav_bytes: bytes, output_format: str, codec: str,
                    extra_args: list = None) -> bytes:
    """Convert WAV bytes to target format using ffmpeg subprocess.

    Args:
        wav_bytes: Input WAV audio bytes
        output_format: ffmpeg output format name (e.g. 'mp3', 'ogg', 'flac')
        codec: ffmpeg codec name (e.g. 'libmp3lame', 'libvorbis', 'aac')
        extra_args: Additional ffmpeg arguments

    Returns:
        Encoded audio bytes

    Raises:
        RuntimeError: If ffmpeg is not installed or conversion fails
    """
    if not _check_ffmpeg():
        raise RuntimeError(
            "ffmpeg is required for audio format conversion. "
            "Install it with: apt install ffmpeg (Linux) or brew install ffmpeg (macOS)"
        )

    cmd = [
        "ffmpeg",
        "-hide_banner", "-loglevel", "error",
        "-i", "pipe:0",           # Read WAV from stdin
        "-vn",                    # No video
        "-acodec", codec,
    ]
    if extra_args:
        cmd.extend(extra_args)
    cmd.extend([
        "-f", output_format,
        "pipe:1",                 # Write to stdout
    ])

    proc = subprocess.run(
        cmd,
        input=wav_bytes,
        capture_output=True,
        timeout=60,
    )

    if proc.returncode != 0:
        error_msg = proc.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"ffmpeg conversion to {output_format} failed: {error_msg}")

    return proc.stdout


def encode_audio(audio: np.ndarray, fmt: AudioFormat,
                 sample_rate: int = SAMPLE_RATE) -> tuple[bytes, str]:
    """Encode a float32 audio array into the requested format.

    Args:
        audio: Float32 audio samples in range [-1, 1], shape (num_samples,)
        fmt: Target audio format
        sample_rate: Audio sample rate (default: 22050)

    Returns:
        Tuple of (encoded_bytes, mime_type)

    Raises:
        ValueError: If format is not supported
        RuntimeError: If ffmpeg is required but not installed
    """
    if fmt not in SUPPORTED_FORMATS:
        raise ValueError(
            f"Unsupported format '{fmt}'. "
            f"Supported: {', '.join(SUPPORTED_FORMATS.keys())}"
        )

    mime = SUPPORTED_FORMATS[fmt]["mime"]

    # ── Uncompressed ────────────────────────────────────────────

    if fmt == "pcm":
        audio_clipped = np.clip(audio, -1.0, 1.0)
        pcm_data = (audio_clipped * 32767).astype(np.int16)
        return pcm_data.tobytes(), mime

    if fmt == "wav":
        return _audio_to_wav_bytes(audio, sample_rate), mime

    # ── Lossless ────────────────────────────────────────────────

    if fmt == "flac":
        # Try soundfile first (faster, no subprocess), fallback to ffmpeg
        try:
            import soundfile as sf
            buf = io.BytesIO()
            audio_clipped = np.clip(audio, -1.0, 1.0)
            sf.write(buf, audio_clipped, sample_rate, format="FLAC", subtype="PCM_16")
            buf.seek(0)
            return buf.read(), mime
        except ImportError:
            wav_bytes = _audio_to_wav_bytes(audio, sample_rate)
            return _ffmpeg_convert(wav_bytes, "flac", "flac"), mime

    if fmt == "alac":
        wav_bytes = _audio_to_wav_bytes(audio, sample_rate)
        return _ffmpeg_convert(
            wav_bytes, "ipod", "alac",
            extra_args=["-movflags", "+faststart"]
        ), mime

    # ── Lossy ───────────────────────────────────────────────────

    if fmt == "mp3":
        wav_bytes = _audio_to_wav_bytes(audio, sample_rate)
        return _ffmpeg_convert(
            wav_bytes, "mp3", "libmp3lame",
            extra_args=["-b:a", "192k"]
        ), mime

    if fmt == "aac":
        wav_bytes = _audio_to_wav_bytes(audio, sample_rate)
        return _ffmpeg_convert(
            wav_bytes, "adts", "aac",
            extra_args=["-b:a", "192k"]
        ), mime

    if fmt == "ogg":
        wav_bytes = _audio_to_wav_bytes(audio, sample_rate)
        return _ffmpeg_convert(
            wav_bytes, "ogg", "libvorbis",
            extra_args=["-q:a", "6"]
        ), mime

    # Should never reach here
    raise ValueError(f"Unhandled format: {fmt}")
