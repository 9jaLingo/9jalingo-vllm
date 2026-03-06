"""vLLM-based text-to-speech generation logic with async streaming for 9jaLingo TTS"""

import asyncio
import os
import time
import torch
import numpy as np

# On GPU: LFM2's custom ops cause vLLM V1's Triton kernel compilation to fail.
# Force V0 engine BEFORE importing vllm (reads env at import time).
# On CPU: leave default — the CPU nightly already works correctly.
if torch.cuda.is_available():
    os.environ.setdefault("VLLM_USE_V1", "0")

from vllm import AsyncEngineArgs, AsyncLLMEngine, SamplingParams
from transformers import AutoTokenizer

# ─── CPU-only patches (skipped on GPU where triton/CUDA kernels work natively) ──
if not torch.cuda.is_available():
    # Patch 1: dispatch_cpu_unquantized_gemm crashes on 3D conv weights.
    # LFM2 has conv layers with 3D weights but the CPU GEMM dispatcher assumes 2D.
    import vllm.model_executor.layers.utils as _vllm_layer_utils

    _orig_dispatch = _vllm_layer_utils.dispatch_cpu_unquantized_gemm

    def _patched_dispatch(layer: torch.nn.Module, remove_weight: bool) -> None:
        if layer.weight.ndim != 2:
            layer.cpu_linear = torch.nn.functional.linear
            return
        return _orig_dispatch(layer, remove_weight)

    _vllm_layer_utils.dispatch_cpu_unquantized_gemm = _patched_dispatch

    # Patch 2: causal_conv1d ops use triton which doesn't work on CPU.
    # Replace with pure PyTorch implementations for LFM2 short convolution layers.
    import torch.nn.functional as F
    import vllm.model_executor.layers.mamba.ops.causal_conv1d as _causal_conv1d_mod

    def _cpu_causal_conv1d_fn(
        x, weight, bias=None, activation=None, conv_states=None,
        has_initial_state=None, cache_indices=None, query_start_loc=None,
        metadata=None, pad_slot_id=-1, **kwargs,
    ):
        """Pure PyTorch replacement for triton-based causal_conv1d_fn (prefill)."""
        original_dtype = x.dtype
        if conv_states is not None:
            x = x.to(conv_states.dtype)

        dim, total_tokens = x.shape
        width = weight.size(1)
        out = torch.empty_like(x)

        if query_start_loc is not None:
            batch = query_start_loc.size(0) - 1
        else:
            batch = 1
            query_start_loc = torch.tensor([0, total_tokens], device=x.device)

        w = weight.unsqueeze(1)  # (dim, 1, width) for depthwise conv1d

        for i in range(batch):
            start = query_start_loc[i].item()
            end = query_start_loc[i + 1].item()
            seq = x[:, start:end]  # (dim, seq_len)

            cache_idx = cache_indices[i].item() if cache_indices is not None else i

            # Prepend initial state or zero-pad for causal convolution
            if (conv_states is not None and has_initial_state is not None
                    and has_initial_state[i]):
                init = conv_states[cache_idx][:, -(width - 1):]
                padded = torch.cat([init, seq], dim=-1)
            else:
                padded = F.pad(seq, (width - 1, 0))

            # Depthwise causal conv1d (cross-correlation matches triton kernel)
            result = F.conv1d(
                padded.unsqueeze(0), w, bias=bias, groups=dim,
            ).squeeze(0)  # (dim, seq_len)
            out[:, start:end] = result

            # Update conv state with the last (width-1) input values
            if conv_states is not None:
                state_len = conv_states.size(2)
                combined = padded  # already has state prepended
                conv_states[cache_idx, :, :state_len] = combined[:, -state_len:]

        if activation in ("silu", "swish"):
            out = torch.nn.functional.silu(out)

        return out.to(original_dtype)

    def _cpu_causal_conv1d_update(
        x, conv_state, weight, bias=None, activation=None,
        conv_state_indices=None, pad_slot_id=-1, **kwargs,
    ):
        """Pure PyTorch replacement for triton-based causal_conv1d_update (decode)."""
        original_dtype = x.dtype
        x = x.to(conv_state.dtype)

        unsqueeze = x.dim() == 2
        if unsqueeze:
            x = x.unsqueeze(-1)  # (batch, dim, 1)

        batch, dim, seqlen = x.shape
        width = weight.size(1)
        out = torch.empty_like(x)

        for i in range(batch):
            idx = conv_state_indices[i].item() if conv_state_indices is not None else i
            if idx == pad_slot_id:
                continue
            state = conv_state[idx]  # (dim, state_len) where state_len = width - 1

            for s in range(seqlen):
                # Concat state (width-1 elements) + new token (1) = width elements
                window = torch.cat([state, x[i, :, s:s + 1]], dim=-1)  # (dim, width)
                out[i, :, s] = (window * weight).sum(dim=-1)
                if bias is not None:
                    out[i, :, s] += bias
                # Update state: drop oldest, append new token
                state = torch.cat([state[:, 1:], x[i, :, s:s + 1]], dim=-1)

            conv_state[idx] = state

        if activation in ("silu", "swish"):
            out = torch.nn.functional.silu(out)

        if unsqueeze:
            out = out.squeeze(-1)

        return out.to(original_dtype)

    _causal_conv1d_mod.causal_conv1d_fn = _cpu_causal_conv1d_fn
    _causal_conv1d_mod.causal_conv1d_update = _cpu_causal_conv1d_update
    # Also patch the imports in short_conv.py which imported these at module level
    import vllm.model_executor.layers.mamba.short_conv as _short_conv_mod
    _short_conv_mod.causal_conv1d_fn = _cpu_causal_conv1d_fn
    _short_conv_mod.causal_conv1d_update = _cpu_causal_conv1d_update

    print("✅ CPU-only patches applied (GEMM dispatch, causal_conv1d)")
else:
    print("🚀 GPU detected — skipping CPU patches (triton/CUDA kernels used)")


# ─── Patch 3: Frame-level position encoding (needed on BOTH CPU and GPU) ───────
# The model was trained with frame-level positions: every 4 audio tokens in a frame
# share the SAME position ID (compressing the audio position space by 4x).
# vLLM assigns sequential positions (0, 1, 2, ...) which is completely wrong for
# audio tokens and produces gibberish. This patch intercepts positions before they
# reach the compiled model graph and transforms them to frame-level positions.
from config import AUDIO_TOKENS_START

_FRAME_POS_TOKENS_PER_FRAME = 4
_FRAME_POS_AUDIO_STEP = 1.0
_FRAME_POS_AUDIO_START = AUDIO_TOKENS_START  # 64410

# Per-sequence state for tracking positions during decode
_frame_pos_state = None  # {'text_count': int, 'audio_count': int}

import vllm.model_executor.models.lfm2 as _lfm2_mod
_OrigLfm2ForCausalLM_forward = _lfm2_mod.Lfm2ForCausalLM.forward


def _frame_level_lfm2_forward(self, input_ids, positions, intermediate_tensors=None,
                               inputs_embeds=None, **kwargs):
    """Wrapper that converts vLLM's sequential positions to frame-level positions."""
    global _frame_pos_state

    if input_ids is not None:
        n_tokens = input_ids.numel()

        if n_tokens > 1:
            # Prefill (or chunked prefill): have multiple tokens
            flat_ids = input_ids.flatten()
            is_audio = flat_ids >= _FRAME_POS_AUDIO_START
            n_text = int((~is_audio).sum().item())
            n_audio = int(is_audio.sum().item())

            # Detect new sequence (positions starting from 0)
            if positions.min().item() == 0:
                _frame_pos_state = {'text_count': 0, 'audio_count': 0}

            if _frame_pos_state is not None:
                if n_audio > 0:
                    # Compute frame-level positions for mixed text/audio prefill
                    # (rare for TTS since prefill is all text, but handle it)
                    new_positions = torch.empty_like(positions)
                    tc = _frame_pos_state['text_count']
                    ac = _frame_pos_state['audio_count']
                    for j in range(n_tokens):
                        tok = flat_ids[j].item()
                        frame_pos = tc + int(ac // _FRAME_POS_TOKENS_PER_FRAME * _FRAME_POS_AUDIO_STEP)
                        new_positions.view(-1)[j] = frame_pos
                        if tok >= _FRAME_POS_AUDIO_START:
                            ac += 1
                        else:
                            tc += 1
                    positions = new_positions
                    _frame_pos_state['text_count'] = tc
                    _frame_pos_state['audio_count'] = ac
                else:
                    # All text tokens — positions stay sequential, just update counts
                    _frame_pos_state['text_count'] += n_text
                    _frame_pos_state['audio_count'] += n_audio
        else:
            # Decode: single token — use tracked state
            if _frame_pos_state is not None:
                token_id = input_ids.flatten()[0].item()
                tc = _frame_pos_state['text_count']
                ac = _frame_pos_state['audio_count']

                frame_pos = tc + int(ac // _FRAME_POS_TOKENS_PER_FRAME * _FRAME_POS_AUDIO_STEP)
                positions = positions.new_full(positions.shape, frame_pos)

                if token_id >= _FRAME_POS_AUDIO_START:
                    _frame_pos_state['audio_count'] += 1
                else:
                    _frame_pos_state['text_count'] += 1

    return _OrigLfm2ForCausalLM_forward(self, input_ids, positions, intermediate_tensors,
                                         inputs_embeds, **kwargs)


_lfm2_mod.Lfm2ForCausalLM.forward = _frame_level_lfm2_forward

from config import (
    MODEL_NAME, START_OF_HUMAN, END_OF_TEXT, END_OF_HUMAN, END_OF_AI,
    TEMPERATURE, TOP_P, REPETITION_PENALTY, MAX_TOKENS, SAMPLE_RATE,
    SUPPORTED_LANGUAGES,
)
from prepare_model import prepare as prepare_vllm_model


class VLLMTTSGenerator:
    def __init__(self, tensor_parallel_size=1, gpu_memory_utilization=0.9, max_model_len=2048):
        """Initialize vLLM-based TTS generator with async streaming support

        Args:
            tensor_parallel_size: Number of GPUs to use for tensor parallelism
            gpu_memory_utilization: Fraction of GPU memory to use (0.0 to 1.0)
            max_model_len: Maximum sequence length
        """
        # Prepare vLLM-compatible model (strips learnable RoPE + speaker projection)
        vllm_model_path = prepare_vllm_model()
        print(f"Loading vLLM AsyncLLMEngine model: {vllm_model_path}")

        # Auto-detect dtype and eager mode based on GPU compute capability
        infer_dtype = "bfloat16"
        use_eager = False  # CPU default — was working, don't change
        if torch.cuda.is_available():
            cc = torch.cuda.get_device_capability()
            if cc[0] < 8:                # T4, V100 (cc 7.x) — no bf16 support
                infer_dtype = "float16"
            use_eager = True              # GPU + V0 engine — use eager mode
            print(f"GPU compute capability {cc[0]}.{cc[1]} — using {infer_dtype}")

        # Configure engine arguments — uses local model with standard Lfm2ForCausalLM
        engine_args = AsyncEngineArgs(
            model=vllm_model_path,
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            enforce_eager=use_eager,
            max_num_seqs=1,
            dtype=infer_dtype,
        )

        # Create async engine
        self.engine = None  # Will be initialized in async context
        self.engine_args = engine_args

        self.tokenizer = AutoTokenizer.from_pretrained(vllm_model_path)

        # Pre-configure sampling parameters
        self.sampling_params = SamplingParams(
            temperature=TEMPERATURE,
            top_p=TOP_P,
            max_tokens=MAX_TOKENS,
            repetition_penalty=REPETITION_PENALTY,
            stop_token_ids=[END_OF_AI],
        )

    async def initialize_engine(self):
        """Initialize the async engine — call this during startup to avoid lazy loading"""
        if self.engine is None:
            print("Initializing vLLM AsyncLLMEngine...")
            self.engine = AsyncLLMEngine.from_engine_args(self.engine_args)
            print("vLLM AsyncLLMEngine initialized and ready!")

    def prepare_input(self, prompt):
        """Build custom input_ids with special tokens

        Token format: [START_OF_HUMAN] + tokenized_text + [END_OF_TEXT, END_OF_HUMAN]
        """
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids

        # Add special tokens
        start_token = torch.tensor([[START_OF_HUMAN]], dtype=torch.int64)
        end_tokens = torch.tensor([[END_OF_TEXT, END_OF_HUMAN]], dtype=torch.int64)
        modified_input_ids = torch.cat([start_token, input_ids, end_tokens], dim=1)

        # Convert to list for vLLM
        return modified_input_ids[0].tolist()

    @staticmethod
    def build_prompt(text, language_tag=None):
        """Build prompt text with optional language tag prefix

        Args:
            text: Raw input text
            language_tag: Language tag (ha, ig, yo, pcm) or None for no prefix

        Returns:
            Prompt string, e.g. "ha: Sannu da zuwa" or just "Hello world"
        """
        if language_tag and language_tag in SUPPORTED_LANGUAGES:
            return f"{language_tag}: {text}"
        return text

    async def _generate_async(self, prompt, audio_writer, max_tokens=MAX_TOKENS):
        """Async generator that streams tokens as they are generated

        Args:
            prompt: Text prompt (with language tag already prepended if needed)
            audio_writer: StreamingAudioWriter instance to receive tokens
            max_tokens: Maximum number of tokens to generate

        Returns:
            Dictionary with generation metrics and results
        """
        # Initialize engine if needed
        if self.engine is None:
            self.engine = AsyncLLMEngine.from_engine_args(self.engine_args)

        # Prepare input_ids with special tokens
        input_ids = self.prepare_input(prompt)

        point_1 = time.time()

        # Override max_tokens if different from default
        if max_tokens != MAX_TOKENS:
            sampling_params = SamplingParams(
                temperature=TEMPERATURE,
                top_p=TOP_P,
                max_tokens=max_tokens,
                repetition_penalty=REPETITION_PENALTY,
                stop_token_ids=[END_OF_AI],
            )
        else:
            sampling_params = self.sampling_params

        # Generate unique request ID
        request_id = f"tts-{id(prompt)}-{time.time()}"

        # Stream tokens as they are generated
        all_token_ids = []
        audio_token_count = 0
        inside_speech = False

        # Add request to engine with token prompt
        results_generator = self.engine.generate(
            {"prompt_token_ids": input_ids},
            sampling_params,
            request_id=request_id
        )

        async for request_output in results_generator:
            # Get newly generated tokens
            new_token_ids = request_output.outputs[0].token_ids

            # Find which tokens are new since last iteration
            num_new_tokens = len(new_token_ids) - len(all_token_ids)
            if num_new_tokens > 0:
                new_tokens = new_token_ids[-num_new_tokens:]
                all_token_ids.extend(new_tokens)

                # Stream each new token to audio_writer and count audio tokens
                for token_id in new_tokens:
                    audio_writer.add_token(token_id)

                    # Track audio tokens efficiently during streaming
                    if token_id == audio_writer.player.start_of_speech:
                        inside_speech = True
                    elif token_id == audio_writer.player.end_of_speech:
                        inside_speech = False
                    elif inside_speech:
                        audio_token_count += 1

        point_2 = time.time()
        generation_time = point_2 - point_1

        # Calculate Real Time Factor (RTF)
        # Audio codec runs at 12.5 fps, audio tokens come in groups of 4 per frame
        FRAMES_PER_SECOND = 12.5
        TOKENS_PER_FRAME = 4

        num_frames = audio_token_count // TOKENS_PER_FRAME
        audio_duration = num_frames / FRAMES_PER_SECOND
        rtf = generation_time / audio_duration if audio_duration > 0 else 0

        # Calculate token counts
        prompt_tokens = len(input_ids)
        generated_tokens = len(all_token_ids)
        total_tokens = prompt_tokens + generated_tokens

        print(f"\n[vLLM] Generation complete. Prompt tokens: {prompt_tokens}, Generated tokens: {generated_tokens}, Total: {total_tokens}")
        print(f"       Audio tokens: {audio_token_count}, Frames: {num_frames}, Audio duration: {audio_duration:.2f}s")
        print(f"       Generation time: {generation_time:.2f}s, RTF: {rtf:.3f}")

        return {
            'all_token_ids': all_token_ids,
            'generation_time': generation_time,
            'audio_duration': audio_duration,
            'rtf': rtf,
            'point_1': point_1,
            'point_2': point_2
        }

    def generate(self, prompt, audio_writer, max_tokens=MAX_TOKENS):
        """Generate speech tokens from text prompt with streaming

        This is a synchronous wrapper around the async streaming implementation.

        Args:
            prompt: Text prompt to convert to speech
            audio_writer: StreamingAudioWriter instance to receive tokens
            max_tokens: Maximum number of tokens to generate

        Returns:
            Dictionary with generation metrics and results
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self._generate_async(prompt, audio_writer, max_tokens))
        else:
            import concurrent.futures
            import threading

            result = None
            exception = None

            def run_in_new_loop():
                nonlocal result, exception
                try:
                    new_loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(new_loop)
                    result = new_loop.run_until_complete(
                        self._generate_async(prompt, audio_writer, max_tokens)
                    )
                    new_loop.close()
                except Exception as e:
                    exception = e

            thread = threading.Thread(target=run_in_new_loop)
            thread.start()
            thread.join()

            if exception:
                raise exception

            return result

    async def generate_long_form_async(self, text, language_tag, player, max_chunk_duration=12.0,
                                       silence_duration=0.2, max_tokens=MAX_TOKENS):
        """Generate speech for long text by splitting into chunks

        This method handles texts longer than the model's training distribution (5-15s)
        by splitting into sentence-based chunks and generating each with the same
        language tag for consistency.

        Args:
            text: Input text (can be any length)
            language_tag: Language tag for consistency (e.g., 'ha', 'yo', 'ig', 'pcm')
            player: NaijaLingoAudioPlayer instance for decoding audio
            max_chunk_duration: Target duration per chunk in seconds (default 12s)
            silence_duration: Duration of silence between chunks in seconds (default 0.2s)
            max_tokens: Maximum tokens per generation

        Returns:
            Dictionary with:
                - audio: Concatenated audio as numpy array
                - chunks_info: List of info dicts for each chunk
                - total_duration: Total audio duration in seconds
                - total_generation_time: Total time spent generating
        """
        from generation.chunking import split_into_sentences, estimate_duration
        from audio.streaming import StreamingAudioWriter

        # Estimate if text needs chunking
        estimated_duration = estimate_duration(text)
        print(f"\n[Long-form] Estimated duration: {estimated_duration:.1f}s for text length: {len(text)} chars")

        # Split into chunks
        chunks = split_into_sentences(text, max_duration_seconds=max_chunk_duration)
        print(f"[Long-form] Split into {len(chunks)} chunks")

        if len(chunks) == 1:
            print("[Long-form] Single chunk — using standard generation")

        # Generate each chunk with language tag prefix for consistency
        audio_segments = []
        chunks_info = []
        total_generation_time = 0

        for i, chunk in enumerate(chunks):
            print(f"\n[Long-form] Generating chunk {i+1}/{len(chunks)}: '{chunk[:50]}...'")

            # Add language tag prefix for consistency
            prompt = self.build_prompt(chunk, language_tag)

            # Create audio writer for this chunk
            audio_writer = StreamingAudioWriter(
                player,
                output_file=None,
                chunk_size=25,
                lookback_frames=15
            )
            audio_writer.start()

            # Generate this chunk
            result = await self._generate_async(prompt, audio_writer, max_tokens=max_tokens)

            # Finalize and get audio
            audio = audio_writer.finalize()

            if audio is not None and len(audio) > 0:
                audio_segments.append(audio)
                chunks_info.append({
                    'chunk_index': i,
                    'text': chunk,
                    'duration': result['audio_duration'],
                    'generation_time': result['generation_time'],
                    'rtf': result['rtf']
                })
                total_generation_time += result['generation_time']
            else:
                print(f"[Long-form] Warning: No audio generated for chunk {i+1}")

        # Concatenate audio segments with silence
        if len(audio_segments) == 0:
            raise ValueError("No audio was generated")

        if len(audio_segments) == 1:
            final_audio = audio_segments[0]
        else:
            final_audio = self._concatenate_with_silence(
                audio_segments,
                silence_duration=silence_duration
            )

        total_duration = len(final_audio) / SAMPLE_RATE

        print(f"\n[Long-form] Complete!")
        print(f"  Total chunks: {len(chunks)}")
        print(f"  Total duration: {total_duration:.2f}s")
        print(f"  Total generation time: {total_generation_time:.2f}s")
        print(f"  Overall RTF: {total_generation_time / total_duration:.3f}")

        return {
            'audio': final_audio,
            'chunks_info': chunks_info,
            'total_duration': total_duration,
            'total_generation_time': total_generation_time,
            'num_chunks': len(chunks)
        }

    def _concatenate_with_silence(self, audio_segments, silence_duration=0.2):
        """Concatenate audio segments with short silence between them

        Args:
            audio_segments: List of numpy audio arrays
            silence_duration: Duration of silence in seconds

        Returns:
            Concatenated audio as numpy array
        """
        if len(audio_segments) == 1:
            return audio_segments[0]

        silence_samples = int(silence_duration * SAMPLE_RATE)
        silence = np.zeros(silence_samples, dtype=audio_segments[0].dtype)

        result = audio_segments[0]
        for next_segment in audio_segments[1:]:
            result = np.concatenate([result, silence, next_segment])

        return result
