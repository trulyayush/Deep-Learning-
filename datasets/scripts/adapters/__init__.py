"""ECHO-MIND dataset-specific metadata adapters."""

from .base import (
    ContinualLearningMetadataRecord,
    DatasetAdapter,
    DatasetInspectionSummary,
    DeepfakeMetadataRecord,
    EmotionMetadataRecord,
    UnlearningMetadataRecord,
)
from .continual_adapters import CIFAR100Adapter, CIFAR10Adapter, CORe50Adapter
from .deepfake_adapters import (
    ASVspoof2021Adapter,
    AVDeepfake1MAdapter,
    DFDCAdapter,
    FaceForensicsAdapter,
    FakeAVCelebAdapter,
)
from .emotion_adapters import CREMADAdapter, MELDAdapter, RAVDESSAdapter
from .unlearning_adapters import MUBenchAdapter, TOFUAdapter

__all__ = [
    "ASVspoof2021Adapter",
    "AVDeepfake1MAdapter",
    "CIFAR100Adapter",
    "CIFAR10Adapter",
    "CORe50Adapter",
    "CREMADAdapter",
    "ContinualLearningMetadataRecord",
    "DFDCAdapter",
    "DatasetAdapter",
    "DatasetInspectionSummary",
    "DeepfakeMetadataRecord",
    "EmotionMetadataRecord",
    "FaceForensicsAdapter",
    "FakeAVCelebAdapter",
    "MELDAdapter",
    "MUBenchAdapter",
    "RAVDESSAdapter",
    "TOFUAdapter",
    "UnlearningMetadataRecord",
]
