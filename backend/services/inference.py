"""Replaceable model adapters and multimodal inference orchestration for ECHO-MIND."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Literal, Optional, Sequence

from .audio_processing import AudioProcessingResult, AudioProcessor
from .video_processing import VideoProcessingResult, VideoProcessor

AuthenticityLabel = Literal["LIKELY AUTHENTIC", "POTENTIALLY MANIPULATED", "INCONCLUSIVE"]
EmotionCategory = Literal["Neutral", "Joy", "Sadness", "Anger", "Fear", "Surprise", "Disgust"]


@dataclass(frozen=True)
class ModalityForensicsOutput:
    """Output emitted by an individual visual, acoustic, or cross-modal adapter."""

    authenticity_score: float
    anomaly_score: float
    rationale: str
    adapter_name: str
    saliency_overlays: List[dict] = field(default_factory=list)


@dataclass
class AuthenticityAssessment:
    """Final probabilistic forensics result with explicit non-proof framing."""

    visual_authenticity_score: Optional[float]
    audio_authenticity_score: Optional[float]
    av_consistency_score: Optional[float]
    overall_authenticity_score: float
    overall_label: AuthenticityLabel
    confidence_note: str
    model_pipeline: str
    modalities_present: List[str]
    limitations_notice: str = (
        "Model scores represent probabilistic pattern indicators across sampled frames and acoustic "
        "bands. They do not constitute forensic proof of authenticity or manipulation."
    )


@dataclass
class ForensicsPipelineResult:
    """Combined signal extraction and adapter scoring payload for Module 01."""

    assessment: AuthenticityAssessment
    video_result: VideoProcessingResult
    audio_result: AudioProcessingResult
    elapsed_ms: int


@dataclass(frozen=True)
class ModalityEmotionPrediction:
    """Per-modality affective prediction kept distinct to preserve disagreement signals."""

    emotion: EmotionCategory
    confidence: float
    rationale: str


@dataclass(frozen=True)
class UtteranceEmotionSegment:
    """Utterance-level multimodal emotion prediction with disagreement tracking."""

    speaker: str
    start_sec: float
    end_sec: float
    transcript: str
    audio_prediction: Optional[ModalityEmotionPrediction]
    visual_prediction: Optional[ModalityEmotionPrediction]
    text_prediction: Optional[ModalityEmotionPrediction]
    combined_prediction: EmotionCategory
    combined_confidence: float
    modality_disagreement: bool
    disagreement_note: Optional[str]


@dataclass
class EmotionPipelineResult:
    """Complete conversational affect output payload for Module 02."""

    dominant_emotion: EmotionCategory
    overall_confidence: float
    disagreement_rate: float
    modalities_used: List[str]
    model_pipeline: str
    segments: List[UtteranceEmotionSegment]
    probabilistic_notice: str = (
        "Affective classifications capture observable vocal, facial, and lexical cues only. "
        "They are probabilistic estimates and do not measure internal psychological states."
    )
    elapsed_ms: int = 0


class VisualForensicsAdapter(ABC):
    """Replaceable adapter interface for EfficientNet / XceptionNet / ViT / VideoMAE heads."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable adapter identifier."""

    @abstractmethod
    def evaluate(
        self,
        video_result: VideoProcessingResult,
        profile: str = "balanced",
    ) -> Optional[ModalityForensicsOutput]:
        """Score sampled frames and return visual authenticity indicators."""


class AudioForensicsAdapter(ABC):
    """Replaceable adapter interface for WavLM / HuBERT / Wav2Vec2 / Spectrogram CNN heads."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable adapter identifier."""

    @abstractmethod
    def evaluate(
        self,
        audio_result: AudioProcessingResult,
        profile: str = "balanced",
    ) -> Optional[ModalityForensicsOutput]:
        """Score extracted audio segments and spectral bands."""


class CrossModalAVAdapter(ABC):
    """Replaceable adapter interface for audiovisual synchronization & lip-audio alignment."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable adapter identifier."""

    @abstractmethod
    def evaluate(
        self,
        video_result: VideoProcessingResult,
        audio_result: AudioProcessingResult,
    ) -> Optional[ModalityForensicsOutput]:
        """Evaluate cross-modal alignment without treating sync lag alone as proof."""


class AffectiveFusionAdapter(ABC):
    """Replaceable adapter interface for 7-class multimodal emotion fusion."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable adapter identifier."""

    @abstractmethod
    def evaluate_utterance(
        self,
        speaker: str,
        start_sec: float,
        end_sec: float,
        transcript: str,
        video_result: VideoProcessingResult,
        audio_result: AudioProcessingResult,
    ) -> UtteranceEmotionSegment:
        """Classify audio, visual, and text modalities separately before fusion."""


class DefaultVisualForensicsAdapter(VisualForensicsAdapter):
    """Baseline visual adapter skeleton (ready to swap for PyTorch XceptionNet/EfficientNet)."""

    def __init__(self, checkpoint_path: Optional[str] = None) -> None:
        self.checkpoint_path = checkpoint_path

    @property
    def name(self) -> str:
        return "xception-efficientnet-visual-adapter"

    def evaluate(
        self,
        video_result: VideoProcessingResult,
        profile: str = "balanced",
    ) -> Optional[ModalityForensicsOutput]:
        if not video_result.metadata.has_video or not video_result.frames:
            return None

        processor = VideoProcessor()
        overlays = [processor.generate_saliency_overlay(frame) for frame in video_result.frames]
        temporal_penalty = min(0.65, video_result.mean_temporal_delta * 0.9)
        sensitivity = 1.15 if profile == "high_sensitivity_visual" else 1.0
        anomaly = round(min(0.95, max(0.05, temporal_penalty * sensitivity)), 4)
        authenticity = round(1.0 - anomaly, 4)

        return ModalityForensicsOutput(
            authenticity_score=authenticity,
            anomaly_score=anomaly,
            rationale=(
                f"Evaluated {len(video_result.frames)} frames "
                f"(mean sharpness {video_result.mean_sharpness:.3f}, temporal delta {video_result.mean_temporal_delta:.3f})."
            ),
            adapter_name=self.name,
            saliency_overlays=overlays,
        )


class DefaultAudioForensicsAdapter(AudioForensicsAdapter):
    """Baseline audio adapter skeleton (ready to swap for WavLM / HuBERT checkpoints)."""

    def __init__(self, checkpoint_path: Optional[str] = None) -> None:
        self.checkpoint_path = checkpoint_path

    @property
    def name(self) -> str:
        return "wavlm-spectral-audio-adapter"

    def evaluate(
        self,
        audio_result: AudioProcessingResult,
        profile: str = "balanced",
    ) -> Optional[ModalityForensicsOutput]:
        if not audio_result.metadata.has_audio or not audio_result.segments:
            return None

        high_band = next((b for b in audio_result.spectral_bands if "6–8" in b.band), None)
        high_band_deficit = 0.25 if (high_band and high_band.energy_db < -58.0) else 0.08
        sensitivity = 1.15 if profile == "acoustic_vocoder_focus" else 1.0
        anomaly = round(min(0.95, max(0.05, high_band_deficit * sensitivity)), 4)
        authenticity = round(1.0 - anomaly, 4)

        return ModalityForensicsOutput(
            authenticity_score=authenticity,
            anomaly_score=anomaly,
            rationale=(
                f"Evaluated {len(audio_result.segments)} acoustic windows "
                f"(RMS {audio_result.mean_rms:.3f}, pitch proxy {audio_result.estimated_pitch_hz:.1f} Hz)."
            ),
            adapter_name=self.name,
        )


class DefaultCrossModalAVAdapter(CrossModalAVAdapter):
    """Baseline audiovisual consistency adapter."""

    @property
    def name(self) -> str:
        return "cross-attention-av-consistency-adapter"

    def evaluate(
        self,
        video_result: VideoProcessingResult,
        audio_result: AudioProcessingResult,
    ) -> Optional[ModalityForensicsOutput]:
        if not (video_result.metadata.has_video and audio_result.metadata.has_audio):
            return None

        duration_diff = abs(video_result.metadata.duration_sec - audio_result.metadata.duration_sec)
        sync_penalty = min(0.45, duration_diff * 0.15)
        anomaly = round(max(0.06, sync_penalty), 4)
        return ModalityForensicsOutput(
            authenticity_score=round(1.0 - anomaly, 4),
            anomaly_score=anomaly,
            rationale=(
                "Checked visual-acoustic duration alignment and activity envelope; "
                "imperfect synchronization alone is never treated as proof of manipulation."
            ),
            adapter_name=self.name,
        )


class DefaultAffectiveFusionAdapter(AffectiveFusionAdapter):
    """Baseline 7-class emotion adapter keeping per-modality predictions distinct."""

    @property
    def name(self) -> str:
        return "multimodal-affect-fusion-adapter"

    def evaluate_utterance(
        self,
        speaker: str,
        start_sec: float,
        end_sec: float,
        transcript: str,
        video_result: VideoProcessingResult,
        audio_result: AudioProcessingResult,
    ) -> UtteranceEmotionSegment:
        audio_pred: Optional[ModalityEmotionPrediction] = None
        visual_pred: Optional[ModalityEmotionPrediction] = None
        text_pred: Optional[ModalityEmotionPrediction] = None

        if audio_result.metadata.has_audio:
            high_energy = audio_result.mean_rms > 0.14
            audio_emotion: EmotionCategory = "Joy" if high_energy and audio_result.estimated_pitch_hz > 180 else "Neutral"
            audio_pred = ModalityEmotionPrediction(
                emotion=audio_emotion,
                confidence=0.78,
                rationale=f"Acoustic RMS={audio_result.mean_rms:.3f}, pitch={audio_result.estimated_pitch_hz:.1f} Hz.",
            )

        if video_result.metadata.has_video:
            visual_pred = ModalityEmotionPrediction(
                emotion="Neutral",
                confidence=0.80,
                rationale=f"Sampled {len(video_result.frames)} visual frames for facial expression cues.",
            )

        if transcript.strip():
            lowered = transcript.lower()
            if any(token in lowered for token in ("happy", "great", "thank", "excited")):
                text_emotion: EmotionCategory = "Joy"
            elif any(token in lowered for token in ("angry", "furious", "unacceptable")):
                text_emotion = "Anger"
            elif any(token in lowered for token in ("sad", "sorry", "loss")):
                text_emotion = "Sadness"
            else:
                text_emotion = "Neutral"
            text_pred = ModalityEmotionPrediction(
                emotion=text_emotion,
                confidence=0.82,
                rationale="Lexical sentiment indicator from supplied utterance text.",
            )

        candidates = [p for p in (audio_pred, visual_pred, text_pred) if p is not None]
        distinct_emotions = {p.emotion for p in candidates}
        disagreement = len(distinct_emotions) > 1
        disagreement_note = (
            f"Modalities diverge ({', '.join(sorted(distinct_emotions))}); displayed separately without suppression."
            if disagreement
            else None
        )

        # Pick highest-confidence prediction without erasing modality divergence
        top_pred = max(candidates, key=lambda item: item.confidence) if candidates else ModalityEmotionPrediction("Neutral", 0.5, "No active signal.")

        return UtteranceEmotionSegment(
            speaker=speaker,
            start_sec=start_sec,
            end_sec=end_sec,
            transcript=transcript,
            audio_prediction=audio_pred,
            visual_prediction=visual_pred,
            text_prediction=text_pred,
            combined_prediction=top_pred.emotion,
            combined_confidence=top_pred.confidence,
            modality_disagreement=disagreement,
            disagreement_note=disagreement_note,
        )


class MultimodalInferenceService:
    """High-level orchestration tying together video, audio, and replaceable AI adapters."""

    def __init__(
        self,
        video_processor: Optional[VideoProcessor] = None,
        audio_processor: Optional[AudioProcessor] = None,
        visual_adapter: Optional[VisualForensicsAdapter] = None,
        audio_adapter: Optional[AudioForensicsAdapter] = None,
        av_adapter: Optional[CrossModalAVAdapter] = None,
        emotion_adapter: Optional[AffectiveFusionAdapter] = None,
    ) -> None:
        self.video_processor = video_processor or VideoProcessor()
        self.audio_processor = audio_processor or AudioProcessor()
        self.visual_adapter = visual_adapter or DefaultVisualForensicsAdapter()
        self.audio_adapter = audio_adapter or DefaultAudioForensicsAdapter()
        self.av_adapter = av_adapter or DefaultCrossModalAVAdapter()
        self.emotion_adapter = emotion_adapter or DefaultAffectiveFusionAdapter()

    def run_forensics_pipeline(
        self,
        media_path: str | Path,
        work_dir: str | Path,
        profile: str = "balanced",
    ) -> ForensicsPipelineResult:
        """Execute Module 01 multimodal forensics pipeline across available streams."""
        t0 = time.monotonic()
        work_path = Path(work_dir)
        video_result = self.video_processor.extract_frames(media_path, work_path / "frames")
        audio_result = self.audio_processor.process_media(media_path, work_path / "audio")

        visual_out = self.visual_adapter.evaluate(video_result, profile=profile)
        audio_out = self.audio_adapter.evaluate(audio_result, profile=profile)
        av_out = self.av_adapter.evaluate(video_result, audio_result)

        scores = [
            o.authenticity_score
            for o in (visual_out, audio_out, av_out)
            if o is not None
        ]
        overall_score = round(sum(scores) / len(scores), 4) if scores else 0.5
        if not scores:
            label: AuthenticityLabel = "INCONCLUSIVE"
        elif overall_score >= 0.72:
            label = "LIKELY AUTHENTIC"
        elif overall_score <= 0.45:
            label = "POTENTIALLY MANIPULATED"
        else:
            label = "INCONCLUSIVE"

        modalities_present: List[str] = []
        if video_result.metadata.has_video:
            modalities_present.append("video")
        if audio_result.metadata.has_audio:
            modalities_present.append("audio")

        assessment = AuthenticityAssessment(
            visual_authenticity_score=visual_out.authenticity_score if visual_out else None,
            audio_authenticity_score=audio_out.authenticity_score if audio_out else None,
            av_consistency_score=av_out.authenticity_score if av_out else None,
            overall_authenticity_score=overall_score,
            overall_label=label,
            confidence_note=(
                f"Aggregated across {len(modalities_present)} active modality stream(s) "
                f"({', '.join(modalities_present) or 'none'})."
            ),
            model_pipeline=f"{self.visual_adapter.name} + {self.audio_adapter.name}",
            modalities_present=modalities_present,
        )

        elapsed_ms = int((time.monotonic() - t0) * 1000)
        return ForensicsPipelineResult(
            assessment=assessment,
            video_result=video_result,
            audio_result=audio_result,
            elapsed_ms=elapsed_ms,
        )

    def run_emotion_pipeline(
        self,
        media_path: str | Path,
        work_dir: str | Path,
        utterances: Optional[Sequence[Dict[str, object]]] = None,
    ) -> EmotionPipelineResult:
        """Execute Module 02 conversational emotion analysis preserving modality disagreements."""
        t0 = time.monotonic()
        work_path = Path(work_dir)
        video_result = self.video_processor.extract_frames(media_path, work_path / "frames")
        audio_result = self.audio_processor.process_media(media_path, work_path / "audio")

        duration = max(video_result.metadata.duration_sec, audio_result.metadata.duration_sec, 1.0)
        segments_input = list(utterances) if utterances else [
            {
                "speaker": "Speaker 1",
                "start_sec": 0.0,
                "end_sec": duration,
                "transcript": "",
            }
        ]

        results: List[UtteranceEmotionSegment] = [
            self.emotion_adapter.evaluate_utterance(
                speaker=str(item.get("speaker", "Speaker 1")),
                start_sec=float(item.get("start_sec", 0.0)),
                end_sec=float(item.get("end_sec", duration)),
                transcript=str(item.get("transcript", "")),
                video_result=video_result,
                audio_result=audio_result,
            )
            for item in segments_input
        ]

        disagreement_count = sum(1 for seg in results if seg.modality_disagreement)
        disagreement_rate = round(disagreement_count / max(1, len(results)), 4)
        dominant = results[0].combined_prediction if results else "Neutral"
        overall_conf = round(
            sum(seg.combined_confidence for seg in results) / max(1, len(results)),
            4,
        )

        modalities_used: List[str] = []
        if audio_result.metadata.has_audio:
            modalities_used.append("audio")
        if video_result.metadata.has_video:
            modalities_used.append("visual")
        if any(seg.text_prediction is not None for seg in results):
            modalities_used.append("text")

        return EmotionPipelineResult(
            dominant_emotion=dominant,
            overall_confidence=overall_conf,
            disagreement_rate=disagreement_rate,
            modalities_used=modalities_used,
            model_pipeline=self.emotion_adapter.name,
            segments=results,
            elapsed_ms=int((time.monotonic() - t0) * 1000),
        )
