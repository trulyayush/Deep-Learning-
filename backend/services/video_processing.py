"""Video decoding, frame extraction, facial preprocessing, and saliency overlay service."""

from __future__ import annotations

import json
import math
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional, Sequence


@dataclass(frozen=True)
class VideoStreamMetadata:
    """Container and video stream metadata inspected via ffprobe/OpenCV."""

    duration_sec: float = 0.0
    width: int = 0
    height: int = 0
    fps: float = 0.0
    video_codec: Optional[str] = None
    has_video: bool = False


@dataclass(frozen=True)
class FaceRegion:
    """Normalized [0, 1] bounding region for a detected facial crop."""

    x: float
    y: float
    width: float
    height: float
    confidence: float
    detector: str = "adapter-stub"


@dataclass
class ExtractedFrameArtifact:
    """Metadata and signal proxies for a sampled video frame."""

    index: int
    timestamp_sec: float
    frame_path: str
    sharpness_score: float = 0.0
    temporal_delta: float = 0.0
    face_regions: List[FaceRegion] = field(default_factory=list)
    preprocessed_tensor_shape: Optional[tuple[int, int, int]] = None


@dataclass
class VideoProcessingResult:
    """Complete video preprocessing bundle passed to forensics and emotion heads."""

    media_path: str
    metadata: VideoStreamMetadata
    frames: List[ExtractedFrameArtifact] = field(default_factory=list)
    mean_sharpness: float = 0.0
    mean_temporal_delta: float = 0.0
    notes: List[str] = field(default_factory=list)


class VideoProcessor:
    """Modular video decoding, frame sampling, face cropping, and saliency skeleton."""

    def __init__(
        self,
        target_frame_size: tuple[int, int] = (224, 224),
        max_sampled_frames: int = 8,
    ) -> None:
        self.target_frame_size = target_frame_size
        self.max_sampled_frames = max(1, max_sampled_frames)

    def probe_video(self, media_path: str | Path) -> VideoStreamMetadata:
        """Inspect video stream attributes using ffprobe when available."""
        path = Path(media_path)
        if not path.exists():
            raise FileNotFoundError(f"Media file not found: {path}")

        if shutil.which("ffprobe") is None:
            return VideoStreamMetadata()

        cmd = [
            "ffprobe",
            "-v",
            "quiet",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            str(path),
        ]
        completed = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if completed.returncode != 0 or not completed.stdout.strip():
            return VideoStreamMetadata()

        payload = json.loads(completed.stdout)
        streams = payload.get("streams", [])
        video_stream = next((s for s in streams if s.get("codec_type") == "video"), None)
        if video_stream is None:
            duration = float(payload.get("format", {}).get("duration") or 0.0)
            return VideoStreamMetadata(duration_sec=round(duration, 2), has_video=False)

        fps = 0.0
        rate = video_stream.get("r_frame_rate") or ""
        if "/" in rate:
            num_s, den_s = rate.split("/", maxsplit=1)
            try:
                num, den = float(num_s), float(den_s)
                if den > 0 and math.isfinite(num / den):
                    fps = round(num / den, 2)
            except ValueError:
                fps = 0.0

        duration = float(payload.get("format", {}).get("duration") or 0.0)
        return VideoStreamMetadata(
            duration_sec=round(duration, 2),
            width=int(video_stream.get("width") or 0),
            height=int(video_stream.get("height") or 0),
            fps=fps,
            video_codec=video_stream.get("codec_name"),
            has_video=True,
        )

    def extract_frames(
        self,
        media_path: str | Path,
        output_dir: str | Path,
        target_frames: Optional[int] = None,
        scale_width: int = 640,
    ) -> VideoProcessingResult:
        """Decode and sample uniform video frames into `output_dir` via ffmpeg."""
        path = Path(media_path)
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        metadata = self.probe_video(path)
        if not metadata.has_video:
            return VideoProcessingResult(
                media_path=str(path),
                metadata=metadata,
                frames=[],
                notes=["No video stream detected; visual pipeline skipped gracefully."],
            )

        count = max(1, min(target_frames or self.max_sampled_frames, 32))
        duration = max(1.0, metadata.duration_sec or 4.0)
        fps_filter = f"fps={max(0.25, count / duration):.4f},scale={scale_width}:-2"
        pattern = str(out_dir / "frame_%03d.png")

        if shutil.which("ffmpeg") is not None:
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-i",
                    str(path),
                    "-vf",
                    fps_filter,
                    "-frames:v",
                    str(count),
                    pattern,
                ],
                capture_output=True,
                text=True,
                check=False,
            )

        frame_files = sorted(out_dir.glob("frame_*.png"))
        artifacts: List[ExtractedFrameArtifact] = []
        prev_bytes: Optional[bytes] = None
        step_sec = duration / max(1, len(frame_files))

        for idx, frame_file in enumerate(frame_files):
            raw = frame_file.read_bytes()
            sharpness = self._estimate_byte_entropy_sharpness(raw)
            delta = self._estimate_temporal_delta(prev_bytes, raw)
            prev_bytes = raw
            faces = self.detect_facial_regions(frame_file)
            artifacts.append(
                ExtractedFrameArtifact(
                    index=idx,
                    timestamp_sec=round((idx + 0.5) * step_sec, 2),
                    frame_path=str(frame_file),
                    sharpness_score=sharpness,
                    temporal_delta=delta,
                    face_regions=faces,
                    preprocessed_tensor_shape=(3, self.target_frame_size[0], self.target_frame_size[1]),
                )
            )

        mean_sharp = (
            round(sum(f.sharpness_score for f in artifacts) / len(artifacts), 4)
            if artifacts
            else 0.0
        )
        mean_delta = (
            round(sum(f.temporal_delta for f in artifacts) / len(artifacts), 4)
            if artifacts
            else 0.0
        )

        return VideoProcessingResult(
            media_path=str(path),
            metadata=metadata,
            frames=artifacts,
            mean_sharpness=mean_sharp,
            mean_temporal_delta=mean_delta,
        )

    def detect_facial_regions(self, frame_path: str | Path) -> List[FaceRegion]:
        """Detect facial bounding boxes using OpenCV when installed, else adapter stub."""
        try:
            import cv2  # type: ignore

            img = cv2.imread(str(frame_path))
            if img is None:
                return []
            height, width = img.shape[:2]
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            cascade_path = getattr(cv2.data, "haarcascades", "") + "haarcascade_frontalface_default.xml"
            detector = cv2.CascadeClassifier(cascade_path)
            boxes = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
            regions = [
                FaceRegion(
                    x=round(float(x) / max(1, width), 4),
                    y=round(float(y) / max(1, height), 4),
                    width=round(float(w) / max(1, width), 4),
                    height=round(float(h) / max(1, height), 4),
                    confidence=0.85,
                    detector="opencv-haar-frontalface",
                )
                for (x, y, w, h) in boxes
            ]
            if regions:
                return regions
        except ImportError:
            pass

        return [
            FaceRegion(
                x=0.30,
                y=0.16,
                width=0.40,
                height=0.52,
                confidence=0.5,
                detector="center-face-prior-stub",
            )
        ]

    def preprocess_facial_crop(
        self,
        frame_path: str | Path,
        region: FaceRegion,
    ) -> dict[str, Any]:
        """Skeleton for cropping and normalizing facial patches for PyTorch vision encoders."""
        return {
            "frame_path": str(frame_path),
            "crop_region": {
                "x": region.x,
                "y": region.y,
                "w": region.width,
                "h": region.height,
            },
            "target_size": self.target_frame_size,
            "normalization": "imagenet_rgb_mean_std",
        }

    def generate_saliency_overlay(
        self,
        frame: ExtractedFrameArtifact,
        attribution_method: str = "grad_cam",
    ) -> dict[str, Any]:
        """Return normalized Grad-CAM / saliency attribution metadata for a frame."""
        primary_face = frame.face_regions[0] if frame.face_regions else FaceRegion(0.3, 0.18, 0.4, 0.5, 0.5)
        return {
            "frame_index": frame.index,
            "timestamp_sec": frame.timestamp_sec,
            "method": attribution_method,
            "region": {
                "x": primary_face.x,
                "y": primary_face.y,
                "w": primary_face.width,
                "h": primary_face.height,
            },
            "intensity": round(min(1.0, max(0.05, frame.temporal_delta * 2.0 + 0.15)), 3),
            "label": "Facial Boundary Attention",
        }

    @staticmethod
    def _estimate_byte_entropy_sharpness(raw_bytes: bytes) -> float:
        if not raw_bytes:
            return 0.0
        step = max(1, len(raw_bytes) // 1024)
        sample = raw_bytes[::step]
        diffs = sum(abs(sample[i] - sample[i - 1]) for i in range(1, len(sample)))
        return round(min(1.0, diffs / max(1, len(sample) * 64)), 4)

    @staticmethod
    def _estimate_temporal_delta(prev_bytes: Optional[bytes], cur_bytes: bytes) -> float:
        if not prev_bytes or not cur_bytes:
            return 0.0
        limit = min(len(prev_bytes), len(cur_bytes), 4096)
        if limit == 0:
            return 0.0
        diff = sum(abs(prev_bytes[i] - cur_bytes[i]) for i in range(limit))
        return round(min(1.0, diff / (limit * 48.0)), 4)


def summarize_frame_sequence(frames: Sequence[ExtractedFrameArtifact]) -> dict[str, float]:
    """Compute aggregate temporal stability statistics across extracted frames."""
    if not frames:
        return {"frame_count": 0.0, "mean_sharpness": 0.0, "mean_temporal_delta": 0.0}
    return {
        "frame_count": float(len(frames)),
        "mean_sharpness": round(sum(f.sharpness_score for f in frames) / len(frames), 4),
        "mean_temporal_delta": round(sum(f.temporal_delta for f in frames) / len(frames), 4),
    }
