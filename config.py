"""Configuration and constants for 9jaLingo TTS vLLM"""

# Tokenizer configuration (matches naijalingo-tts-2 module)
TOKENIZER_LENGTH = 64400

# Special tokens
START_OF_TEXT = 1
END_OF_TEXT = 2
START_OF_SPEECH = TOKENIZER_LENGTH + 1
END_OF_SPEECH = TOKENIZER_LENGTH + 2
START_OF_HUMAN = TOKENIZER_LENGTH + 3
END_OF_HUMAN = TOKENIZER_LENGTH + 4
START_OF_AI = TOKENIZER_LENGTH + 5
END_OF_AI = TOKENIZER_LENGTH + 6
PAD_TOKEN = TOKENIZER_LENGTH + 7
AUDIO_TOKENS_START = TOKENIZER_LENGTH + 10

# Audio configuration
CODEBOOK_SIZE = 4032
SAMPLE_RATE = 22050

# Streaming configuration
CHUNK_SIZE = 25  # Number of new frames to output per iteration
LOOKBACK_FRAMES = 15  # Number of frames to include from previous context

# Generation configuration
TEMPERATURE = 1.0
TOP_P = 0.95
REPETITION_PENALTY = 1.1
REPETITION_CONTEXT_SIZE = 20
MAX_TOKENS = 3000  # ~40 seconds max continuous audio (40s * 12.5fps * 4 tokens/frame ≈ 2000 audio tokens + prompt)

# Long-form generation configuration
LONG_FORM_THRESHOLD_SECONDS = 40.0  # Model supports up to 40s continuous; auto-chunk beyond that
LONG_FORM_CHUNK_DURATION = 12.0     # Target duration per chunk (within 5-15s training distribution for quality)
LONG_FORM_SILENCE_DURATION = 0.2    # Silence between chunks in seconds

# Speaker embedding configuration
SPEAKER_EMB_DIM = 128      # Dimension of speaker embeddings (128-dim L2-normalized)
SPEAKERS_DIR = "./speakers"  # Directory containing .pt speaker embedding files
GENERATION_MANIFEST_PATH = None  # Optional: path to generation_manifest.json for speaker metadata

# Supported languages
SUPPORTED_LANGUAGES = {
    "ha": "Hausa",
    "ig": "Igbo",
    "yo": "Yoruba",
    "pcm": "Pidgin",
    # "en_NG": "English (Nigerian Accent)",  # Coming soon
}

# Model paths
MODEL_NAME = "9jaLingo/9javox-9jalingo-finetuned-full-v1"
CODEC_MODEL_NAME = "nvidia/nemo-nano-codec-22khz-0.6kbps-12.5fps"
