"""Speaker management for 9jaLingo TTS — loads and organizes speaker embeddings by language.

Speaker .pt files follow the naming convention: {name}_{lang_code}.pt
e.g., abdullahi_ha.pt, adaeze_ig.pt, blessing_pcm.pt

speaker_map.json stores rich metadata per speaker:
{
  "abdullahi_ha": {
    "id": "abdullahi_ha",
    "display_name": "Abdullahi (ha)",
    "name": "Abdullahi",
    "language": "ha",
    "language_name": "Hausa",
    "path": "./speakers/abdullahi_ha.pt",
    "gender": "male",
    "domains": ["Everyday Conversation"]
  }
}

Supports:
- Loading from rich speaker_map.json (id → metadata + .pt path)
- Auto-discovery from directory of .pt files (fallback)
- Per-language speaker listing and filtering
- Filtering by gender and domain
- Lazy embedding loading with caching
"""

import os
import json
import torch
from typing import Dict, List, Optional
from pathlib import Path

from config import SUPPORTED_LANGUAGES, SPEAKER_EMB_DIM


class SpeakerManager:
    """Manages pre-computed speaker embeddings (.pt files) organized by language.

    Speaker embeddings are 128-dim L2-normalized vectors extracted from reference
    audio via a WavLM-based speaker embedder. Each .pt file contains a single
    torch.Tensor of shape [1, 128] or [128].
    """

    def __init__(
        self,
        speakers_dir: str = "./speakers",
        speaker_map_path: Optional[str] = None,
        manifest_path: Optional[str] = None,
    ):
        """Initialize speaker manager.

        Args:
            speakers_dir: Directory containing .pt speaker embedding files
            speaker_map_path: Path to speaker_map.json (default: speakers_dir/speaker_map.json)
            manifest_path: Ignored (metadata now lives in speaker_map.json directly)
        """
        self.speakers_dir = Path(speakers_dir)
        self.speaker_map_path = speaker_map_path or str(self.speakers_dir / "speaker_map.json")

        # speaker_base_name -> {display_name, path, language, gender (optional)}
        self._speakers: Dict[str, dict] = {}
        # language_code -> [speaker_base_names]
        self._by_language: Dict[str, List[str]] = {}
        # Cache loaded embedding tensors
        self._embedding_cache: Dict[str, torch.Tensor] = {}

        self._load_speakers()


    # ── Loading ──────────────────────────────────────────────────

    def _load_speakers(self):
        """Load speakers from speaker_map.json or auto-discover from directory."""
        if os.path.exists(self.speaker_map_path):
            self._load_from_map()
        elif self.speakers_dir.exists():
            self._discover_from_directory()
        else:
            print(f"⚠️  No speakers directory found at {self.speakers_dir}")
            return

        # Build per-language index
        for name, info in self._speakers.items():
            lang = info["language"]
            self._by_language.setdefault(lang, []).append(name)

        # Sort each language's speakers
        for lang in self._by_language:
            self._by_language[lang].sort()

        total = len(self._speakers)
        langs = {lang: len(names) for lang, names in sorted(self._by_language.items())}
        print(f"📢 Loaded {total} speakers: {langs}")

    def _load_from_map(self):
        """Load from rich speaker_map.json keyed by speaker_id.

        Format: {"abdullahi_ha": {id, display_name, name, language, language_name, path, gender?, domains?}}
        Also supports legacy format: {"Display Name (lang)": "./speakers/name.pt"}
        """
        with open(self.speaker_map_path) as f:
            raw_map = json.load(f)

        for key, value in raw_map.items():
            # Rich format: value is a dict with metadata
            if isinstance(value, dict):
                base_name = value.get("id", key)
                pt_path = value.get("path", "")

                # Resolve path relative to speakers_dir
                if pt_path and not os.path.isabs(pt_path):
                    resolved_path = str(self.speakers_dir / Path(pt_path).name)
                else:
                    resolved_path = pt_path

                self._speakers[base_name] = {
                    "display_name": value.get("display_name", key),
                    "name": value.get("name", base_name.rsplit("_", 1)[0].capitalize()),
                    "path": resolved_path,
                    "language": value.get("language", "unknown"),
                    "language_name": value.get("language_name", ""),
                }
                if "gender" in value:
                    self._speakers[base_name]["gender"] = value["gender"]
                if "domains" in value:
                    self._speakers[base_name]["domains"] = value["domains"]

            # Legacy format: value is a string path
            elif isinstance(value, str):
                lang = self._extract_lang_from_display(key)
                base_name = Path(value).stem
                resolved_path = str(self.speakers_dir / Path(value).name) if not os.path.isabs(value) else value
                self._speakers[base_name] = {
                    "display_name": key,
                    "name": base_name.rsplit("_", 1)[0].capitalize(),
                    "path": resolved_path,
                    "language": lang,
                    "language_name": SUPPORTED_LANGUAGES.get(lang, lang),
                }

    def _discover_from_directory(self):
        """Auto-discover speakers from .pt files in the directory."""
        for pt_file in sorted(self.speakers_dir.glob("*.pt")):
            base_name = pt_file.stem  # e.g., "abdullahi_ha"
            parts = base_name.rsplit("_", 1)
            if len(parts) != 2:
                continue

            name_part, lang = parts
            if lang not in SUPPORTED_LANGUAGES:
                continue

            display_name = f"{name_part.capitalize()} ({lang})"
            self._speakers[base_name] = {
                "display_name": display_name,
                "name": name_part.capitalize(),
                "path": str(pt_file),
                "language": lang,
                "language_name": SUPPORTED_LANGUAGES.get(lang, lang),
            }

    @staticmethod
    def _extract_lang_from_display(display_name: str) -> str:
        """Extract language code from 'Name (lang)' format."""
        try:
            return display_name.rsplit("(", 1)[1].rstrip(")")
        except (IndexError, AttributeError):
            return "unknown"

    # ── Queries ──────────────────────────────────────────────────

    def list_speakers(
        self,
        language: Optional[str] = None,
        gender: Optional[str] = None,
        domain: Optional[str] = None,
    ) -> List[dict]:
        """List speakers with full metadata, optionally filtered.

        Args:
            language: Language code to filter by (ha, ig, yo, pcm), or None for all
            gender: Filter by 'male' or 'female', or None for all
            domain: Filter by domain name, or None for all

        Returns:
            List of speaker info dicts with all available metadata
        """
        if language and language in self._by_language:
            names = self._by_language[language]
        else:
            names = sorted(self._speakers.keys())

        result = []
        for name in names:
            info = self._speakers[name]

            # Apply gender filter
            if gender and info.get("gender") != gender:
                continue

            # Apply domain filter
            if domain and domain not in info.get("domains", []):
                continue

            entry = {
                "id": name,
                "display_name": info["display_name"],
                "name": info.get("name", name.rsplit("_", 1)[0].capitalize()),
                "language": info["language"],
                "language_name": info.get("language_name", ""),
            }
            if "gender" in info:
                entry["gender"] = info["gender"]
            if "domains" in info:
                entry["domains"] = info["domains"]
            result.append(entry)
        return result

    def get_speaker_info(self, speaker_id: str) -> Optional[dict]:
        """Get full metadata for a single speaker."""
        resolved = self._resolve_speaker(speaker_id)
        if resolved is None:
            return None
        info = self._speakers[resolved]
        return {
            "id": resolved,
            "display_name": info["display_name"],
            "name": info.get("name", resolved.rsplit("_", 1)[0].capitalize()),
            "language": info["language"],
            "language_name": info.get("language_name", ""),
            "gender": info.get("gender"),
            "domains": info.get("domains", []),
        }

    def get_languages(self) -> Dict[str, dict]:
        """Get available languages with speaker counts and gender breakdown."""
        result = {}
        for lang in sorted(self._by_language.keys()):
            lang_name = SUPPORTED_LANGUAGES.get(lang, lang)
            speakers = self._by_language[lang]
            males = sum(1 for s in speakers if self._speakers[s].get("gender") == "male")
            females = sum(1 for s in speakers if self._speakers[s].get("gender") == "female")
            result[lang] = {
                "name": lang_name,
                "speaker_count": len(speakers),
                "male": males,
                "female": females,
            }
        return result

    def get_domains(self) -> List[str]:
        """Get all unique domain names across speakers."""
        domains = set()
        for info in self._speakers.values():
            for d in info.get("domains", []):
                domains.add(d)
        return sorted(domains)

    def get_embedding(self, speaker_id: str) -> Optional[torch.Tensor]:
        """Load and return speaker embedding tensor [1, 128].

        Args:
            speaker_id: Base speaker name (e.g., 'abdullahi_ha') or display name

        Returns:
            Speaker embedding tensor [1, SPEAKER_EMB_DIM] or None if not found
        """
        resolved = self._resolve_speaker(speaker_id)
        if resolved is None:
            return None

        # Check cache
        if resolved in self._embedding_cache:
            return self._embedding_cache[resolved]

        # Load from disk
        info = self._speakers[resolved]
        path = info["path"]

        if not os.path.exists(path):
            print(f"⚠️  Speaker file not found: {path}")
            return None

        embedding = torch.load(path, map_location="cpu", weights_only=True)

        # Ensure correct shape [1, SPEAKER_EMB_DIM]
        if isinstance(embedding, torch.Tensor):
            if embedding.dim() == 1:
                embedding = embedding.unsqueeze(0)
        else:
            print(f"⚠️  Unexpected type in {path}: {type(embedding)}")
            return None

        self._embedding_cache[resolved] = embedding
        return embedding

    def get_speaker_language(self, speaker_id: str) -> Optional[str]:
        """Get the language code for a speaker."""
        resolved = self._resolve_speaker(speaker_id)
        if resolved is None:
            return None
        return self._speakers[resolved]["language"]

    def _resolve_speaker(self, speaker_id: str) -> Optional[str]:
        """Resolve speaker_id to base name, accepting base name or display name."""
        # Direct match
        if speaker_id in self._speakers:
            return speaker_id

        # Try matching by display name (case-insensitive)
        lower = speaker_id.lower()
        for base_name, info in self._speakers.items():
            if info["display_name"].lower() == lower:
                return base_name

        # Try case-insensitive base name
        for base_name in self._speakers:
            if base_name.lower() == lower:
                return base_name

        return None

    def has_speaker(self, speaker_id: str) -> bool:
        """Check if a speaker exists."""
        return self._resolve_speaker(speaker_id) is not None

    @property
    def total_speakers(self) -> int:
        return len(self._speakers)

    @property
    def available_languages(self) -> List[str]:
        return sorted(self._by_language.keys())
