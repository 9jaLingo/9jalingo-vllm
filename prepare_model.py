"""
Prepare model for vLLM inference.

The fine-tuned model uses FlashCompatibleLfm2ForCausalLM (custom architecture)
with learnable RoPE and speaker embedding projection. vLLM's built-in Lfm2ForCausalLM
does not support these extra modules, so we create a vLLM-compatible copy:

1. Download the model from HuggingFace
2. Remove incompatible weight keys (learnable_rope_layers, speaker_emb_projection)
3. Update config.json (architectures → Lfm2ForCausalLM, use_learnable_rope → false)
4. Save to a local directory

The vLLM fast path doesn't use speaker embeddings anyway (those go through the
Direct model path), and the learnable RoPE alphas are small refinements that
don't significantly impact generation quality at inference.
"""

import json
import os
import shutil

from huggingface_hub import snapshot_download
from safetensors.torch import load_file, save_file


# Source model on HuggingFace
HF_MODEL_ID = "9jaLingo/9javox-9jalingo-finetuned-full-v1"

# Local output directory for vLLM-compatible model
VLLM_MODEL_DIR = os.path.join(os.path.dirname(__file__), "model_vllm")

# Weight key prefixes to strip (not supported by vLLM's Lfm2ForCausalLM)
STRIP_PREFIXES = (
    "model.learnable_rope_layers.",
    "model.speaker_emb_projection.",
)


def prepare():
    if os.path.isdir(VLLM_MODEL_DIR) and os.path.exists(
        os.path.join(VLLM_MODEL_DIR, "model.safetensors")
    ):
        print(f"vLLM-compatible model already exists at: {VLLM_MODEL_DIR}")
        return VLLM_MODEL_DIR

    print(f"Downloading {HF_MODEL_ID} from HuggingFace...")
    cache_dir = snapshot_download(
        HF_MODEL_ID,
        allow_patterns=["*.safetensors", "*.json", "*.jinja", "*.txt", "*.model"],
    )
    print(f"Downloaded to cache: {cache_dir}")

    # Create output directory
    os.makedirs(VLLM_MODEL_DIR, exist_ok=True)

    # Copy all non-safetensors files (tokenizer, config, etc.)
    for fname in os.listdir(cache_dir):
        src = os.path.join(cache_dir, fname)
        dst = os.path.join(VLLM_MODEL_DIR, fname)
        if os.path.isfile(src) and not fname.endswith(".safetensors"):
            shutil.copy2(src, dst)

    # Load safetensors, strip incompatible keys, re-save
    safetensors_files = [f for f in os.listdir(cache_dir) if f.endswith(".safetensors")]

    all_tensors = {}
    for sf in safetensors_files:
        all_tensors.update(load_file(os.path.join(cache_dir, sf)))

    original_count = len(all_tensors)
    stripped_keys = []

    for key in list(all_tensors.keys()):
        if any(key.startswith(prefix) for prefix in STRIP_PREFIXES):
            stripped_keys.append(key)
            del all_tensors[key]

    print(f"Stripped {len(stripped_keys)} incompatible keys from {original_count} total:")
    for key in stripped_keys:
        print(f"  - {key}")

    # Save cleaned weights
    save_file(all_tensors, os.path.join(VLLM_MODEL_DIR, "model.safetensors"))
    print(f"Saved {len(all_tensors)} weights to {VLLM_MODEL_DIR}/model.safetensors")

    # Update config.json
    config_path = os.path.join(VLLM_MODEL_DIR, "config.json")
    with open(config_path) as f:
        config = json.load(f)

    config["architectures"] = ["Lfm2ForCausalLM"]
    config["use_learnable_rope"] = False

    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Updated config.json: architectures → Lfm2ForCausalLM, use_learnable_rope → false")

    print(f"\nvLLM-compatible model ready at: {VLLM_MODEL_DIR}")
    return VLLM_MODEL_DIR


if __name__ == "__main__":
    prepare()
