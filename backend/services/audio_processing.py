"""Audio extraction, segmentation, acoustic feature, and spectrogram service."""

from __future__ import annotations

import json
import math
import shutil
import struct
import subprocess
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class AudioStreamMetadata:
    """Audio stream attributes inspected via ffprobe or WAV header."""

    duration_sec: float = 0.0
    sample_rate: Optional[int] = None
    channels: int = 1
    audio_codec: Optional[str] = None
    has_audio: bool = False


@dataclass(frozen=True)
class SpectralBandEnergy:
    """Acoustic energy measurement across a speech-forensics frequency band."""

    band: str
    energy_db: float
    baseline_db: float


@dataclass(frozen=True)
class AudioSegmentFeature:
    """Time-windowed acoustic statistics for deepfake and prosody analysis."""

    index: int
    start_sec: float
    end_sec: float
    rms_energy: float
    peak_amplitude: float
    zero_crossing_rate: float
    estimated_pitch_hz: float
    spectral_centroid_hz: float


@dataclass
class AudioProcessingResult:
    """Complete extracted audio artifact bundle for forensics and emotion heads."""

    media_path: str
    wav_path: Optional[str]
    metadata: AudioStreamMetadata
    mean_rms: float = 0.0
    peak_amplitude: float = 0.0
    zero_crossing_rate: float = 0.0
    estimated_pitch_hz: float = 0.0
    silence_ratio: float = 1.0
    spectral_bands: List[SpectralBandEnergy] = field(default_factory=list)
    segments: List[AudioSegmentFeature] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)


class AudioProcessor:
    """Modular audio extraction, segmentation, spectrogram, and acoustic feature pipeline."""

    def __init__(
        self,
        target_sample_rate: int = 16000,
        segment_window_sec: float = 2.0,
        segment_hop_sec: float = 1.5,
    ) -> None:
        self.target_sample_rate = target_sample_rate
        self.segment_window_sec = max(0.25, segment_window_sec)
        self.segment_hop_sec = max(0.25, segment_hop_sec)

    def probe_audio(self, media_path: str | Path) -> AudioStreamMetadata:
        """Inspect audio stream parameters using ffprobe when available."""
        path = Path(media_path)
        if not path.exists():
            raise FileNotFoundError(f"Media file not found: {path}")

        if path.suffix.lower() == ".wav" and shutil.which("ffprobe") is None:
            with wave.open(str(path), "rb") as wf:
                frames = wf.getnframes()
                sr = wf.getframerate()
                return AudioStreamMetadata(
                    duration_sec=round(frames / max(1, sr), 2),
                    sample_rate=sr,
                    channels=wf.getnchannels(),
                    audio_codec="pcm_s16le",
                    has_audio=True,
                )

        if shutil.which("ffprobe") is None:
            return AudioStreamMetadata()

        completed = subprocess.run(
            [
                "ffprobe",
                "-v",
                "quiet",
                "-print_format",
                "json",
                "-show_format",
                "-show_streams",
                str(path),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0 or not completed.stdout.strip():
            return AudioStreamMetadata()

        payload = json.loads(completed.stdout)
        streams = payload.get("streams", [])
        audio_stream = next((s for s in streams if s.get("codec_type") == "audio"), None)
        duration = float(payload.get("format", {}).get("duration") or 0.0)
        if audio_stream is None:
            return AudioStreamMetadata(duration_sec=round(duration, 2), has_audio=False)

        sr_raw = audio_stream.get("sample_rate")
        return AudioStreamMetadata(
            duration_sec=round(duration, 2),
            sample_rate=int(sr_raw) if sr_raw else None,
            channels=int(audio_stream.get("channels") or 1),
            audio_codec=audio_stream.get("codec_name"),
            has_audio=True,
        )

    def extract_mono_wav(
        self,
        media_path: str | Path,
        output_wav_path: str | Path,
    ) -> Optional[Path]:
        """Extract/transcode audio stream to 16 kHz mono 16-bit PCM WAV via ffmpeg."""
        src = Path(media_path)
        dst = Path(output_wav_path)
        dst.parent.mkdir(parents=True, exist_ok=True)

        metadata = self.probe_audio(src)
        if not metadata.has_audio:
            return None

        if shutil.which("ffmpeg") is None:
            if src.suffix.lower() == ".wav":
                shutil.copy2(src, dst)
                return dst
            return None

        completed = subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(src),
                "-vn",
                "-ac",
                "1",
                "-ar",
                str(self.target_sample_rate),
                "-c:a",
                "pcm_s16le",
                str(dst),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        return dst if completed.returncode == 0 and dst.exists() else None

    def load_pcm_samples(self, wav_path: str | Path) -> Tuple[List[float], int]:
        """Load normalized [-1.0, 1.0] mono audio samples from a 16-bit PCM WAV file."""
        with wave.open(str(wav_path), "rb") as wf:
            sample_rate = wf.getframerate()
            channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            raw = wf.readframes(wf.getnframes())

        if sampwidth != 2 or not raw:
            return [], sample_rate

        total_ints = len(raw) // 2
        ints = struct.unpack(f"<{total_ints}h", raw[: total_ints * 2])
        if channels > 1:
            samples = [
                sum(ints[i + c] for c in range(channels)) / (channels * 32768.0)
                for i in range(0, len(ints) - channels + 1, channels)
            ]
        else:
            samples = [v / 32768.0 for v in ints]
        return samples, sample_rate

    def segment_audio(
        self,
        samples: Sequence[float],
        sample_rate: int,
    ) -> List[AudioSegmentFeature]:
        """Slice audio into overlapping windows and extract RMS, ZCR, and pitch proxies."""
        if not samples or sample_rate <= 0:
            return []

        win_len = max(256, int(self.segment_window_sec * sample_rate))
        hop_len = max(128, int(self.segment_hop_sec * sample_rate))
        segments: List[AudioSegmentFeature] = []

        start = 0
        idx = 0
        while start < len(samples):
            chunk = samples[start : start + win_len]
            if len(chunk) < min(256, len(samples)):
                break

            rms = math.sqrt(sum(x * x for x in chunk) / len(chunk))
            peak = max(abs(x) for x in chunk)
            zcr = sum(
                1
                for i in range(1, len(chunk))
                if (chunk[i - 1] >= 0.0 and chunk[i] < 0.0) or (chunk[i - 1] < 0.0 and chunk[i] >= 0.0)
            ) / max(1, len(chunk))
            pitch = self._estimate_pitch_autocorr(chunk, sample_rate)
            centroid = round(min(sample_rate / 2.0, 350.0 + zcr * sample_rate * 0.45), 1)

            segments.append(
                AudioSegmentFeature(
                    index=idx,
                    start_sec=round(start / sample_rate, 2),
                    end_sec=round((start + len(chunk)) / sample_rate, 2),
                    rms_energy=round(rms, 4),
                    peak_amplitude=round(peak, 4),
                    zero_crossing_rate=round(zcr, 4),
                    estimated_pitch_hz=pitch,
                    spectral_centroid_hz=centroid,
                )
            )
            idx += 1
            if start + win_len >= len(samples):
                break
            start += hop_len

        return segments

    def compute_spectral_bands(
        self,
        samples: Sequence[float],
        sample_rate: int,
    ) -> List[SpectralBandEnergy]:
        """Compute 5-band speech-forensics dB energy via Goertzel DFT (or librosa when used)."""
        band_defs = [
            ("0–300 Hz", (120, 220), -34.0),
            ("300–1 kHz", (440, 720, 920), -22.0),
            ("1–3 kHz", (1300, 2000, 2700), -26.0),
            ("3–6 kHz", (3600, 4800), -35.0),
            ("6–8 kHz", (6400, 7400), -46.0),
        ]
        if len(samples) < 64 or sample_rate <= 0:
            return [SpectralBandEnergy(b, -60.0, base) for b, _, base in band_defs]

        dft_n = min(512, len(samples))
        offset = max(0, (len(samples) - dft_n) // 2)
        windowed = [
            samples[offset + n] * (0.54 - 0.46 * math.cos((2.0 * math.pi * n) / max(1, dft_n - 1)))
            for n in range(dft_n)
        ]

        results: List[SpectralBandEnergy] = []
        for band_name, freqs, baseline_db in band_defs:
            total_power = 0.0
            valid_freqs = [f for f in freqs if f < sample_rate / 2.0] or [freqs[0]]
            for freq in valid_freqs:
                k = (freq * dft_n) / float(sample_rate)
                omega = (2.0 * math.pi * k) / float(dft_n)
                coeff = 2.0 * math.cos(omega)
                q1 = 0.0
                q2 = 0.0
                for sample_val in windowed:
                    q0 = coeff * q1 - q2 + sample_val
                    q2 = q1
                    q1 = q0
                power = q1 * q1 + q2 * q2 - q1 * q2 * coeff
                total_power += max(power / float(dft_n * dft_n), 1e-10)
            avg_power = total_power / len(valid_freqs)
            energy_db = round(max(-72.0, min(0.0, 10.0 * math.log10(avg_power))), 1)
            results.append(SpectralBandEnergy(band=band_name, energy_db=energy_db, baseline_db=baseline_db))

        return results

    def process_media(
        self,
        media_path: str | Path,
        work_dir: str | Path,
    ) -> AudioProcessingResult:
        """Run end-to-end audio extraction, segmentation, and spectral feature computation."""
        metadata = self.probe_audio(media_path)
        if not metadata.has_audio:
            return AudioProcessingResult(
                media_path=str(media_path),
                wav_path=None,
                metadata=metadata,
                notes=["No audio stream detected; acoustic pipeline skipped gracefully."],
            )

        wav_out = Path(work_dir) / f"{Path(media_path).stem}_16k_mono.wav"
        extracted = self.extract_mono_wav(media_path, wav_out)
        if extracted is None:
            return AudioProcessingResult(
                media_path=str(media_path),
                wav_path=None,
                metadata=metadata,
                notes=["Unable to transcode audio stream to 16kHz PCM WAV."],
            )

        samples, sample_rate = self.load_pcm_samples(extracted)
        if not samples:
            return AudioProcessingResult(
                media_path=str(media_path),
                wav_path=str(extracted),
                metadata=metadata,
                notes=["Extracted WAV contained zero PCM frames."],
            )

        mean_rms = round(math.sqrt(sum(x * x for x in samples) / len(samples)), 4)
        peak = round(max(abs(x) for x in samples), 4)
        zcr = round(
            sum(
                1
                for i in range(1, len(samples))
                if (samples[i - 1] >= 0.0 and samples[i] < 0.0) or (samples[i - 1] < 0.0 and samples[i] >= 0.0)
            )
            / len(samples),
            4,
        )
        silence_ratio = round(sum(1 for x in samples if abs(x) < 0.015) / len(samples), 4)
        pitch = self._estimate_pitch_autocorr(samples[: min(len(samples), 2048)], sample_rate)
        bands = self.compute_spectral_bands(samples, sample_rate)
        segments = self.segment_audio(samples, sample_rate)

        return AudioProcessingResult(
            media_path=str(media_path),
            wav_path=str(extracted),
            metadata=metadata,
            mean_rms=mean_rms,
            peak_amplitude=peak,
            zero_crossing_rate=zcr,
            estimated_pitch_hz=pitch,
            silence_ratio=silence_ratio,
            spectral_bands=bands,
            segments=segments,
        )

    @staticmethod
    def _estimate_pitch_autocorr(samples: Sequence[float], sample_rate: int) -> float:
        n = min(1024, len(samples))
        if n < 64 or sample_rate <= 0:
            return 0.0
        min_lag = max(1, sample_rate // 400)
        max_lag = min(sample_rate // 70, n // 2)
        if max_lag <= min_lag:
            return 0.0

        best_lag = 0
        best_corr = 0.0
        window = samples[:n]
        for lag in range(min_lag, max_lag + 1):
            corr = sum(window[i] * window[i + lag] for i in range(n - lag))
            if corr > best_corr:
                best_corr = corr
                best_lag = lag
        return round(sample_rate / best_lag, 1) if best_lag > 0 and best_corr > 0.001 else 0.0
