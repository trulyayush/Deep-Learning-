"""Base metadata schemas and adapter contract for ECHO-MIND dataset ingestion.

Guarantees:
- Unavailable metadata fields remain `None` (never inferred or fabricated).
- Content hashes use SHA-256 for verifiable provenance and unlearning audits.
- Official dataset splits are preserved without identity/dialogue leakage.
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> Optional[str]:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


@dataclass(frozen=True)
class DeepfakeMetadataRecord:
    sample_id: str
    dataset_id: str
    original_video_id: Optional[str]
    video_path: Optional[str]
    audio_path: Optional[str]
    visual_label: Optional[str]
    audio_label: Optional[str]
    audiovisual_label: Optional[str]
    manipulation_type: Optional[str]
    start_timestamp: Optional[float]
    end_timestamp: Optional[float]
    official_split: Optional[str]


@dataclass(frozen=True)
class EmotionMetadataRecord:
    sample_id: str
    dataset_id: str
    dialogue_id: Optional[str]
    utterance_id: Optional[str]
    actor_id: Optional[str]
    speaker_id: Optional[str]
    video_path: Optional[str]
    audio_path: Optional[str]
    transcript: Optional[str]
    emotion_label: Optional[str]
    sentiment_label: Optional[str]
    start_timestamp: Optional[float]
    end_timestamp: Optional[float]
    official_split: Optional[str]


@dataclass(frozen=True)
class ContinualLearningMetadataRecord:
    sample_id: str
    dataset_id: str
    task_id: Optional[str]
    class_id: Optional[str]
    source_id: Optional[str]
    session_id: Optional[str]
    file_path: Optional[str]
    official_split: Optional[str]
    incremental_order: Optional[int]


@dataclass(frozen=True)
class UnlearningMetadataRecord:
    sample_id: str
    dataset_id: str
    original_sample_id: Optional[str]
    file_path: Optional[str]
    content_hash: Optional[str]
    training_split: Optional[str]
    forget_set_status: Optional[str]
    retain_set_status: Optional[str]
    model_version_reference: Optional[str]
    replay_memory_reference: Optional[str]
    derived_feature_reference: Optional[str]


@dataclass
class DatasetInspectionSummary:
    dataset_id: str
    storage_dir: str
    available: bool
    total_records: int
    present_files: int
    missing_files: List[str] = field(default_factory=list)
    corrupt_files: List[str] = field(default_factory=list)
    split_counts: Dict[str, int] = field(default_factory=dict)
    label_distribution: Dict[str, int] = field(default_factory=dict)
    leakage_warnings: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class DatasetAdapter(ABC):
    """Contract implemented by each dataset-specific adapter."""

    dataset_id: str
    domain: str

    def __init__(self, dataset_dir: str | Path) -> None:
        self.dataset_dir = Path(dataset_dir)

    @abstractmethod
    def parse_records(self) -> List[Any]:
        """Parse official annotations into strictly typed domain metadata records."""

    @abstractmethod
    def validate(self) -> DatasetInspectionSummary:
        """Verify file existence, integrity, label counts, and split separation."""

    @staticmethod
    def count_values(values: List[Optional[str]]) -> Dict[str, int]:
        counts = Counter(v for v in values if v is not None)
        return dict(sorted(counts.items()))
