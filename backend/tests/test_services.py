"""Unit tests for ECHO-MIND Python service layer and database schema definitions."""

import ast
import math
import struct
import tempfile
import unittest
import wave
from pathlib import Path

from backend.services.audio_processing import AudioProcessor
from backend.services.inference import MultimodalInferenceService
from backend.services.video_processing import VideoProcessor


def _write_test_wav(path: Path, freq_hz: float = 220.0, duration_sec: float = 2.0, sample_rate: int = 16000) -> None:
    total = int(duration_sec * sample_rate)
    samples = [
        int(max(-32767, min(32767, math.sin(2.0 * math.pi * freq_hz * (i / sample_rate)) * 0.4 * 32767)))
        for i in range(total)
    ]
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(struct.pack(f"<{len(samples)}h", *samples))


class BackendServicesTestCase(unittest.TestCase):
    def test_audio_only_forensics_handles_missing_video_gracefully(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            wav_file = tmp_path / "probe_audio.wav"
            _write_test_wav(wav_file, freq_hz=180.0, duration_sec=2.5)

            service = MultimodalInferenceService()
            result = service.run_forensics_pipeline(wav_file, tmp_path / "work")

            self.assertFalse(result.video_result.metadata.has_video)
            self.assertTrue(result.audio_result.metadata.has_audio)
            self.assertIsNone(result.assessment.visual_authenticity_score)
            self.assertIsNone(result.assessment.av_consistency_score)
            self.assertIsNotNone(result.assessment.audio_authenticity_score)
            self.assertEqual(result.assessment.modalities_present, ["audio"])
            self.assertGreaterEqual(len(result.audio_result.spectral_bands), 5)
            self.assertGreater(len(result.audio_result.segments), 0)

    def test_emotion_pipeline_preserves_modality_disagreement(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            wav_file = tmp_path / "emotion_probe.wav"
            _write_test_wav(wav_file, freq_hz=140.0, duration_sec=2.0)

            service = MultimodalInferenceService()
            result = service.run_emotion_pipeline(
                wav_file,
                tmp_path / "work",
                utterances=[
                    {
                        "speaker": "Speaker A",
                        "start_sec": 0.0,
                        "end_sec": 2.0,
                        "transcript": "I am furious about how unacceptable this situation has become.",
                    }
                ],
            )

            self.assertEqual(len(result.segments), 1)
            seg = result.segments[0]
            self.assertEqual(seg.audio_prediction.emotion, "Neutral")
            self.assertEqual(seg.text_prediction.emotion, "Anger")
            self.assertTrue(seg.modality_disagreement)
            self.assertIsNotNone(seg.disagreement_note)
            self.assertEqual(result.disagreement_rate, 1.0)

    def test_video_processor_probes_and_extracts_frames_from_real_sample_if_present(self) -> None:
        sample_mp4 = Path("data/samples/broadcast_news_briefing.mp4")
        if not sample_mp4.exists():
            self.skipTest("Sample MP4 not generated in workspace")

        with tempfile.TemporaryDirectory() as tmp:
            vp = VideoProcessor(max_sampled_frames=3)
            res = vp.extract_frames(sample_mp4, Path(tmp) / "frames", target_frames=3)
            self.assertTrue(res.metadata.has_video)
            self.assertEqual(len(res.frames), 3)
            overlay = vp.generate_saliency_overlay(res.frames[0])
            self.assertEqual(overlay["method"], "grad_cam")

    def test_database_schemas_define_required_entities_and_valid_syntax(self) -> None:
        schemas_path = Path("backend/database/schemas.py")
        source = schemas_path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        class_names = {node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)}
        expected = {
            "UserORM",
            "MediaUploadORM",
            "AnalysisJobORM",
            "AnalysisResultORM",
            "EmotionSegmentORM",
            "DatasetORM",
            "DatasetSampleORM",
            "ModelRegistryORM",
            "ModelVersionORM",
            "TrainingJobORM",
            "ContinualLearningHistoryORM",
            "ReplayMemoryRecordORM",
            "UnlearningRequestORM",
            "UnlearningAuditORM",
            "ReportORM",
            "ActivityLogORM",
            "SystemSettingORM",
            "UserRead",
            "MediaUploadRead",
            "AnalysisJobRead",
            "AnalysisResultRead",
            "ModelRegistryRead",
        }
        self.assertTrue(expected.issubset(class_names))


if __name__ == "__main__":
    unittest.main()
