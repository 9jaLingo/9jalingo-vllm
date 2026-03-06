"""Direct NaijaLingoTTS inference with speaker embedding and voice cloning support.

This module provides the direct (non-vLLM) inference path for features that
require passing speaker embeddings through the model's forward method.
vLLM's AsyncLLMEngine cannot pass custom kwargs like speaker_emb, so we
use the standard NaijaLingoTTS model for:

- Speaker-specific voice generation (pre-computed .pt embeddings)
- Voice cloning (real-time embedding extraction from reference audio)
- Any generation requiring speaker_emb parameter

Both the model and the embedder are LAZY-LOADED to avoid consuming GPU memory
unless speaker features are actually used.
"""

import torch
import numpy as np
from typing import Optional, Tuple, Union

from config import MODEL_NAME, SAMPLE_RATE, MAX_TOKENS


class DirectTTSGenerator:
    """Direct HuggingFace model inference with speaker embedding support.

    Wraps NaijaLingoTTS and SpeakerEmbedder with lazy loading — these heavy
    models are only loaded when a speaker/cloning request first arrives.
    """

    def __init__(self):
        self._model = None
        self._embedder = None
        self._model_loaded = False
        self._embedder_loaded = False

    def ensure_model(self):
        """Lazy-load the NaijaLingoTTS model on first speaker request."""
        if not self._model_loaded:
            import os
            from naijalingo_tts_2 import NaijaLingoTTS
            from huggingface_hub import snapshot_download

            # Resolve to local cache path so NaijaLingoTTS never hits the network.
            # The model was already downloaded by prepare_model / vLLM engine init.
            hf_token = os.environ.get("HF_TOKEN")
            local_path = snapshot_download(
                MODEL_NAME,
                token=hf_token,
                local_files_only=True,
            )
            print(f"🔄 Loading direct NaijaLingoTTS model from cache: {local_path}")

            self._model = NaijaLingoTTS(
                local_path,
                max_new_tokens=MAX_TOKENS,
                suppress_logs=True,
                show_info=False,
            )
            self._model_loaded = True
            print("✅ Direct NaijaLingoTTS model ready for speaker-aware generation!")

    def ensure_embedder(self):
        """Lazy-load the SpeakerEmbedder for voice cloning."""
        if not self._embedder_loaded:
            from naijalingo_tts_2 import SpeakerEmbedder

            print("🔄 Loading SpeakerEmbedder for voice cloning...")
            self._embedder = SpeakerEmbedder()
            self._embedder_loaded = True
            print("✅ SpeakerEmbedder ready!")

    @property
    def model(self):
        self.ensure_model()
        return self._model

    @property
    def embedder(self):
        self.ensure_embedder()
        return self._embedder

    # ── Generation ───────────────────────────────────────────────

    def generate(
        self,
        text: str,
        language_tag: Optional[str] = None,
        speaker_emb: Optional[torch.Tensor] = None,
        temperature: float = 1.0,
        top_p: float = 0.95,
        repetition_penalty: float = 1.1,
    ) -> Tuple[np.ndarray, str]:
        """Generate speech with optional speaker embedding.

        Args:
            text: Input text
            language_tag: Language tag (ha, ig, yo, pcm)
            speaker_emb: Speaker embedding tensor [1, 128] or path to .pt file
            temperature: Sampling temperature
            top_p: Nucleus sampling threshold
            repetition_penalty: Repetition penalty

        Returns:
            Tuple of (audio_numpy_array, text)
        """
        self.ensure_model()

        # Handle string path to .pt file
        if isinstance(speaker_emb, str):
            speaker_emb = torch.load(speaker_emb, map_location="cpu", weights_only=True)
            if speaker_emb.dim() == 1:
                speaker_emb = speaker_emb.unsqueeze(0)

        audio, out_text = self._model(
            text,
            language_tag=language_tag,
            speaker_emb=speaker_emb,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        )

        return audio, out_text

    def generate_long_form(
        self,
        text: str,
        language_tag: Optional[str] = None,
        speaker_emb: Optional[torch.Tensor] = None,
        max_chunk_duration: float = 20.0,
        silence_duration: float = 0.2,
        temperature: float = 1.0,
        top_p: float = 0.95,
        repetition_penalty: float = 1.1,
    ) -> np.ndarray:
        """Generate long-form speech with speaker embedding.

        Automatically chunks texts estimated to exceed 40 seconds.

        Args:
            text: Input text (any length)
            language_tag: Language tag for consistency across chunks
            speaker_emb: Speaker embedding tensor [1, 128]
            max_chunk_duration: Target duration per chunk in seconds
            silence_duration: Silence between chunks in seconds
            temperature: Sampling temperature
            top_p: Nucleus sampling threshold
            repetition_penalty: Repetition penalty

        Returns:
            Concatenated audio as numpy array
        """
        from generation.chunking import split_into_sentences, estimate_duration

        self.ensure_model()

        estimated_duration = estimate_duration(text)
        print(f"[Direct Long-form] Estimated: {estimated_duration:.1f}s, Chunk target: {max_chunk_duration}s")

        if estimated_duration <= 40.0:
            audio, _ = self.generate(
                text, language_tag, speaker_emb, temperature, top_p, repetition_penalty
            )
            return audio

        # Chunk and generate
        chunks = split_into_sentences(text, max_duration_seconds=max_chunk_duration)
        audio_segments = []

        for i, chunk in enumerate(chunks):
            print(f"[Direct Long-form] Chunk {i+1}/{len(chunks)}: '{chunk[:60]}...'")
            audio, _ = self.generate(
                chunk, language_tag, speaker_emb, temperature, top_p, repetition_penalty
            )
            audio_segments.append(audio)

        # Concatenate with silence
        if len(audio_segments) == 1:
            return audio_segments[0]

        silence = np.zeros(int(silence_duration * SAMPLE_RATE), dtype=np.float32)
        result = audio_segments[0]
        for seg in audio_segments[1:]:
            result = np.concatenate([result, silence, seg])

        return result

    # ── Voice Cloning ────────────────────────────────────────────

    def clone_voice(
        self,
        audio_data: Union[np.ndarray, torch.Tensor, str],
        sample_rate: int = 16000,
    ) -> torch.Tensor:
        """Extract speaker embedding from audio for voice cloning.

        Args:
            audio_data: Audio as numpy array, torch tensor, or file path (str)
            sample_rate: Sample rate of input audio (ignored if file path)

        Returns:
            Speaker embedding tensor [1, 128] (L2-normalized)
        """
        self.ensure_embedder()

        if isinstance(audio_data, str):
            return self._embedder.embed_audio_file(audio_data)
        else:
            return self._embedder.embed_audio(audio_data, sample_rate=sample_rate)

    def clone_and_generate(
        self,
        text: str,
        audio_data: Union[np.ndarray, torch.Tensor, str],
        sample_rate: int = 16000,
        language_tag: Optional[str] = None,
        temperature: float = 1.0,
        top_p: float = 0.95,
        repetition_penalty: float = 1.1,
    ) -> Tuple[np.ndarray, torch.Tensor]:
        """Clone voice from audio and generate speech in one step.

        Args:
            text: Text to convert to speech
            audio_data: Reference audio for voice cloning
            sample_rate: Sample rate of reference audio
            language_tag: Language tag (ha, ig, yo, pcm)
            temperature: Sampling temperature
            top_p: Nucleus sampling threshold
            repetition_penalty: Repetition penalty

        Returns:
            Tuple of (generated_audio_array, speaker_embedding_tensor)
        """
        speaker_emb = self.clone_voice(audio_data, sample_rate)
        audio, _ = self.generate(
            text, language_tag, speaker_emb, temperature, top_p, repetition_penalty
        )
        return audio, speaker_emb
