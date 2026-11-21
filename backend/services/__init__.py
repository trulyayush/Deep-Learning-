"""Modular media processing and multimodal inference services for ECHO-MIND."""

from .audio_processing import AudioProcessingResult, AudioProcessor, AudioSegmentFeature, SpectralBandEnergy
from .inference import (
    AffectiveFusionAdapter,
    AudioForensicsAdapter,
    AuthenticityAssessment,
    CrossModalAVAdapter,
    EmotionPipelineResult,
    ForensicsPipelineResult,
    MultimodalInferenceService,
    VisualForensicsAdapter,
)
from .video_processing import ExtractedFrameArtifact, FaceRegion, VideoProcessingResult, VideoProcessor

__all__ = [
    "AffectiveFusionAdapter",
    "AudioForensicsAdapter",
    "AudioProcessingResult",
    "AudioProcessor",
    "AudioSegmentFeature",
    "AuthenticityAssessment",
    "CrossModalAVAdapter",
    "EmotionPipelineResult",
    "ExtractedFrameArtifact",
    "FaceRegion",
    "ForensicsPipelineResult",
    "MultimodalInferenceService",
    "SpectralBandEnergy",
    "VideoProcessingResult",
    "VideoProcessor",
    "VisualForensicsAdapter",
]
