"""Quick test to verify RTF output for 9jaLingo TTS"""

import asyncio
from generation.vllm_generator import VLLMTTSGenerator
from audio import NaijaLingoAudioPlayer, StreamingAudioWriter
from config import CHUNK_SIZE, LOOKBACK_FRAMES

async def main():
    print("Initializing 9jaLingo vLLM generator...")
    generator = VLLMTTSGenerator(
        tensor_parallel_size=1,
        gpu_memory_utilization=0.9,
        max_model_len=2048
    )

    # Initialize engine
    await generator.initialize_engine()

    player = NaijaLingoAudioPlayer(generator.tokenizer)

    # Test prompts for each supported language
    test_prompts = [
        ("pcm", "How far, my guy? Na so life be sometimes, you go dey struggle today but tomorrow go shine."),
        ("ha", "Sannu da zuwa, yaya kake? Ina fatan ka samu lafiya lau lau."),
        ("yo", "Bawo ni o se wa? Mo fe ki a ba ara wa soro nipa ohun ti a le se."),
        ("ig", "Kedu ka i mere? Anyi nwere obi uto na anyi na-ekwuri okwu taa."),
    ]

    for lang_tag, text in test_prompts:
        prompt = generator.build_prompt(text, lang_tag)
        print(f"\n{'='*60}")
        print(f"Language: {lang_tag} | Prompt: {prompt[:60]}...")

        audio_writer = StreamingAudioWriter(
            player,
            output_file=None,
            chunk_size=CHUNK_SIZE,
            lookback_frames=LOOKBACK_FRAMES
        )
        audio_writer.start()

        # Generate
        result = await generator._generate_async(prompt, audio_writer)
        audio_writer.finalize()

        # Print results
        print(f"  Tokens: {len(result['all_token_ids'])}")
        print(f"  Audio duration: {result['audio_duration']:.2f}s")
        print(f"  Generation time: {result['generation_time']:.2f}s")
        print(f"  RTF: {result['rtf']:.3f}")

if __name__ == "__main__":
    asyncio.run(main())
