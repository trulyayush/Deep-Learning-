"""Unit tests for ECHO-MIND dataset download manager, adapters, validation, and manifests."""

import csv
import http.server
import io
import os
import struct
import tarfile
import tempfile
import threading
import unittest
import wave
from pathlib import Path

from datasets.scripts.adapters.emotion_adapters import RAVDESSAdapter
from datasets.scripts.build_manifests import _write_csv, CONTINUAL_FIELDS
from datasets.scripts.adapters.base import ContinualLearningMetadataRecord
from datasets.scripts.download_manager import DatasetDownloadManager


class DatasetInfrastructureTestCase(unittest.TestCase):
    def test_access_gate_never_bypasses_restricted_datasets(self) -> None:
        mgr = DatasetDownloadManager("datasets")
        for gated_id in ("av_deepfake1m", "fakeavceleb", "faceforensics", "dfdc", "crema_d"):
            res = mgr.acquire_dataset(gated_id, confirm_large=True)
            self.assertEqual(res["status"], "ACCESS REQUIRED")
            self.assertIn("official_download_url", res)

    def test_large_archive_requires_explicit_confirmation(self) -> None:
        mgr = DatasetDownloadManager("datasets")
        res = mgr.acquire_dataset("meld", confirm_large=False, include_optional_large=True)
        self.assertEqual(res["status"], "READY TO DOWNLOAD")
        self.assertIn("--confirm-large", res["message"])

    def test_path_traversal_blocked_during_extraction(self) -> None:
        mgr = DatasetDownloadManager("datasets")
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            bad_tar = tmp_path / "evil.tar.gz"
            with tarfile.open(bad_tar, "w:gz") as tf:
                info = tarfile.TarInfo(name="../../escape.txt")
                payload = b"evil"
                info.size = len(payload)
                tf.addfile(info, io.BytesIO(payload))
            with self.assertRaises(ValueError):
                mgr.extract_archive(bad_tar, tmp_path / "out")

    def test_resumable_download_and_duplicate_skip(self) -> None:
        mgr = DatasetDownloadManager("datasets")
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            serve_dir = tmp_path / "srv"
            serve_dir.mkdir()
            payload = b"ECHO-MIND-RESUMABLE-TEST-" * 1000
            src_file = serve_dir / "sample.bin"
            src_file.write_bytes(payload)

            class RangeHandler(http.server.SimpleHTTPRequestHandler):
                def __init__(self, *args, **kwargs):
                    super().__init__(*args, directory=str(serve_dir), **kwargs)

                def send_head(self):
                    range_hdr = self.headers.get("Range")
                    if range_hdr and range_hdr.startswith("bytes="):
                        start = int(range_hdr.split("=")[1].split("-")[0])
                        data = src_file.read_bytes()[start:]
                        self.send_response(206)
                        self.send_header("Content-Length", str(len(data)))
                        self.send_header("Content-Range", f"bytes {start}-{len(payload)-1}/{len(payload)}")
                        self.end_headers()
                        return io.BytesIO(data)
                    return super().send_head()

                def log_message(self, format, *args):
                    return

            server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), RangeHandler)
            port = server.server_address[1]
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                dest_file = tmp_path / "dl" / "sample.bin"
                dest_file.parent.mkdir(parents=True)
                part_file = tmp_path / "dl" / "sample.bin.part"
                part_file.write_bytes(payload[:500])

                url = f"http://127.0.0.1:{port}/sample.bin"
                res1 = mgr.resumable_download(url, dest_file, expected_size=len(payload))
                self.assertEqual(res1["status"], "downloaded")
                self.assertEqual(dest_file.read_bytes(), payload)

                res2 = mgr.resumable_download(url, dest_file, expected_size=len(payload))
                self.assertEqual(res2["status"], "duplicate_verified_skip")
            finally:
                server.shutdown()
                server.server_close()

    def test_ravdess_parser_and_wav_validation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            wav_path = tmp_path / "Actor_01" / "03-01-05-02-01-01-01.wav"
            wav_path.parent.mkdir(parents=True)
            with wave.open(str(wav_path), "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(16000)
                wf.writeframes(struct.pack("<100h", *([0] * 100)))

            adapter = RAVDESSAdapter(tmp_path)
            summary = adapter.validate()
            self.assertTrue(summary.available)
            self.assertEqual(summary.total_records, 1)
            self.assertEqual(summary.corrupt_files, [])
            self.assertEqual(summary.leakage_warnings, [])
            self.assertIn("angry", summary.label_distribution)

    def test_manifest_keeps_unavailable_fields_empty_not_inferred(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_csv = Path(tmp) / "continual.csv"
            rec = ContinualLearningMetadataRecord(
                sample_id="s1",
                dataset_id="cifar10",
                task_id=None,
                class_id="cat",
                source_id=None,
                session_id=None,
                file_path="/tmp/b1",
                official_split="train",
                incremental_order=None,
            )
            _write_csv(out_csv, CONTINUAL_FIELDS, [rec])
            with out_csv.open("r", encoding="utf-8") as fh:
                rows = list(csv.DictReader(fh))
            self.assertEqual(rows[0]["task_id"], "")
            self.assertEqual(rows[0]["incremental_order"], "")


if __name__ == "__main__":
    unittest.main()
