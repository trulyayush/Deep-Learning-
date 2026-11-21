"""Deepfake forensics dataset adapters (Module 01)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

from .base import DatasetAdapter, DatasetInspectionSummary, DeepfakeMetadataRecord


class AVDeepfake1MAdapter(DatasetAdapter):
    dataset_id = "av_deepfake1m"
    domain = "deepfake"

    def parse_records(self) -> List[DeepfakeMetadataRecord]:
        meta_files = list(self.dataset_dir.rglob("*metadata*.json"))
        records: List[DeepfakeMetadataRecord] = []
        for meta_file in meta_files:
            payload = json.loads(meta_file.read_text(encoding="utf-8"))
            items = payload if isinstance(payload, list) else payload.get("samples", [])
            for idx, item in enumerate(items):
                rel_video = item.get("file") or item.get("video_path")
                v_path = str(self.dataset_dir / rel_video) if rel_video else None
                records.append(
                    DeepfakeMetadataRecord(
                        sample_id=f"avdf1m-{idx:07d}",
                        dataset_id=self.dataset_id,
                        original_video_id=item.get("original") or item.get("source_id"),
                        video_path=v_path,
                        audio_path=None,
                        visual_label=item.get("visual_label"),
                        audio_label=item.get("audio_label"),
                        audiovisual_label=item.get("label"),
                        manipulation_type=item.get("modify_type"),
                        start_timestamp=item.get("fake_segments", [[None, None]])[0][0]
                        if item.get("fake_segments")
                        else None,
                        end_timestamp=item.get("fake_segments", [[None, None]])[0][1]
                        if item.get("fake_segments")
                        else None,
                        official_split=item.get("split"),
                    )
                )
        return records

    def validate(self) -> DatasetInspectionSummary:
        records = self.parse_records()
        if not records:
            return DatasetInspectionSummary(
                dataset_id=self.dataset_id,
                storage_dir=str(self.dataset_dir),
                available=False,
                total_records=0,
                present_files=0,
                notes=["ACCESS REQUIRED: Hugging Face gated manual approval needed before downloading subsets."],
            )
        missing = [r.video_path for r in records if r.video_path and not Path(r.video_path).exists()]
        return DatasetInspectionSummary(
            dataset_id=self.dataset_id,
            storage_dir=str(self.dataset_dir),
            available=True,
            total_records=len(records),
            present_files=len(records) - len(missing),
            missing_files=missing[:100],
            split_counts=self.count_values([r.official_split for r in records]),
            label_distribution=self.count_values([r.audiovisual_label for r in records]),
        )


class FakeAVCelebAdapter(DatasetAdapter):
    dataset_id = "fakeavceleb"
    domain = "deepfake"

    def parse_records(self) -> List[DeepfakeMetadataRecord]:
        import csv

        csv_files = list(self.dataset_dir.rglob("meta_data*.csv"))
        records: List[DeepfakeMetadataRecord] = []
        for csv_path in csv_files:
            with csv_path.open("r", encoding="utf-8", errors="replace") as fh:
                reader = csv.DictReader(fh)
                for idx, row in enumerate(reader):
                    rel = row.get("path") or row.get("filename")
                    v_path = str(self.dataset_dir / rel) if rel else None
                    cat = row.get("type") or row.get("category")
                    records.append(
                        DeepfakeMetadataRecord(
                            sample_id=f"fakeav-{idx:06d}",
                            dataset_id=self.dataset_id,
                            original_video_id=row.get("source"),
                            video_path=v_path,
                            audio_path=None,
                            visual_label="Fake" if cat and "FakeVideo" in cat else ("Real" if cat else None),
                            audio_label="Fake" if cat and "FakeAudio" in cat else ("Real" if cat else None),
                            audiovisual_label=cat,
                            manipulation_type=row.get("method"),
                            start_timestamp=None,
                            end_timestamp=None,
                            official_split=row.get("split"),
                        )
                    )
        return records

    def validate(self) -> DatasetInspectionSummary:
        records = self.parse_records()
        if not records:
            return DatasetInspectionSummary(
                dataset_id=self.dataset_id,
                storage_dir=str(self.dataset_dir),
                available=False,
                total_records=0,
                present_files=0,
                notes=["ACCESS REQUIRED: Submit FakeAVCeleb Google Form request and await maintainer approval."],
            )
        missing = [r.video_path for r in records if r.video_path and not Path(r.video_path).exists()]
        return DatasetInspectionSummary(
            dataset_id=self.dataset_id,
            storage_dir=str(self.dataset_dir),
            available=True,
            total_records=len(records),
            present_files=len(records) - len(missing),
            missing_files=missing[:100],
            label_distribution=self.count_values([r.audiovisual_label for r in records]),
        )


class FaceForensicsAdapter(DatasetAdapter):
    dataset_id = "faceforensics"
    domain = "deepfake"

    def parse_records(self) -> List[DeepfakeMetadataRecord]:
        videos = sorted(self.dataset_dir.rglob("*.mp4"))
        records: List[DeepfakeMetadataRecord] = []
        for idx, vid in enumerate(videos):
            parts_lower = [p.lower() for p in vid.parts]
            method: Optional[str] = None
            for candidate in ("original", "deepfakes", "face2face", "faceswap", "neuraltextures"):
                if candidate in parts_lower:
                    method = candidate
                    break
            visual_label = "Real" if method == "original" else ("Fake" if method else None)
            split = next((s for s in ("train", "val", "test") if s in parts_lower), None)
            records.append(
                DeepfakeMetadataRecord(
                    sample_id=f"ffpp-{idx:06d}",
                    dataset_id=self.dataset_id,
                    original_video_id=vid.stem.split("_")[0] if "_" in vid.stem else vid.stem,
                    video_path=str(vid),
                    audio_path=None,
                    visual_label=visual_label,
                    audio_label=None,
                    audiovisual_label=None,
                    manipulation_type=method,
                    start_timestamp=None,
                    end_timestamp=None,
                    official_split=split,
                )
            )
        return records

    def validate(self) -> DatasetInspectionSummary:
        records = self.parse_records()
        if not records:
            return DatasetInspectionSummary(
                dataset_id=self.dataset_id,
                storage_dir=str(self.dataset_dir),
                available=False,
                total_records=0,
                present_files=0,
                notes=["ACCESS REQUIRED: Submit FaceForensics++ Google Form and use maintainer download script."],
            )
        corrupt = [r.video_path for r in records if r.video_path and Path(r.video_path).stat().st_size == 0]
        return DatasetInspectionSummary(
            dataset_id=self.dataset_id,
            storage_dir=str(self.dataset_dir),
            available=True,
            total_records=len(records),
            present_files=len(records),
            corrupt_files=corrupt,
            split_counts=self.count_values([r.official_split for r in records]),
            label_distribution=self.count_values([r.manipulation_type for r in records]),
        )


class DFDCAdapter(DatasetAdapter):
    dataset_id = "dfdc"
    domain = "deepfake"

    def parse_records(self) -> List[DeepfakeMetadataRecord]:
        records: List[DeepfakeMetadataRecord] = []
        for meta_file in sorted(self.dataset_dir.rglob("metadata.json")):
            data = json.loads(meta_file.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                continue
            for fname, info in data.items():
                if not isinstance(info, dict):
                    continue
                vid_path = meta_file.parent / fname
                label = info.get("label")
                records.append(
                    DeepfakeMetadataRecord(
                        sample_id=f"dfdc-{len(records):06d}",
                        dataset_id=self.dataset_id,
                        original_video_id=info.get("original"),
                        video_path=str(vid_path),
                        audio_path=None,
                        visual_label=label,
                        audio_label=None,
                        audiovisual_label=label,
                        manipulation_type="dfdc_synthetic" if label == "FAKE" else ("original" if label == "REAL" else None),
                        start_timestamp=None,
                        end_timestamp=None,
                        official_split=info.get("split"),
                    )
                )
        return records

    def validate(self) -> DatasetInspectionSummary:
        records = self.parse_records()
        if not records:
            return DatasetInspectionSummary(
                dataset_id=self.dataset_id,
                storage_dir=str(self.dataset_dir),
                available=False,
                total_records=0,
                present_files=0,
                notes=["ACCESS REQUIRED: Accept Meta/Kaggle DFDC rules and download Preview subset first (~8 GiB)."],
            )
        missing = [r.video_path for r in records if r.video_path and not Path(r.video_path).exists()]
        return DatasetInspectionSummary(
            dataset_id=self.dataset_id,
            storage_dir=str(self.dataset_dir),
            available=True,
            total_records=len(records),
            present_files=len(records) - len(missing),
            missing_files=missing[:100],
            split_counts=self.count_values([r.official_split for r in records]),
            label_distribution=self.count_values([r.audiovisual_label for r in records]),
        )


class ASVspoof2021Adapter(DatasetAdapter):
    dataset_id = "asvspoof2021"
    domain = "deepfake"

    def parse_records(self) -> List[DeepfakeMetadataRecord]:
        # Parse extracted ASVspoof 2021 protocol/key trial metadata files without mixing eval labels into train.
        records: List[DeepfakeMetadataRecord] = []
        trial_files = sorted(self.dataset_dir.rglob("trial_metadata.txt"))
        for tf in trial_files:
            track = "DF" if "DF" in str(tf) else ("LA" if "LA" in str(tf) else "PA")
            with tf.open("r", encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    cols = line.strip().split()
                    if len(cols) < 6:
                        continue
                    speaker_id, utt_id = cols[0], cols[1]
                    attack_id = cols[4] if len(cols) > 4 else None
                    key_label = cols[5] if len(cols) > 5 else None
                    split = cols[7] if len(cols) > 7 else "eval"
                    flac_candidate = next(self.dataset_dir.rglob(f"{utt_id}.flac"), None)
                    records.append(
                        DeepfakeMetadataRecord(
                            sample_id=f"asv21-{track.lower()}-{utt_id}",
                            dataset_id=self.dataset_id,
                            original_video_id=speaker_id,
                            video_path=None,
                            audio_path=str(flac_candidate) if flac_candidate else None,
                            visual_label=None,
                            audio_label=key_label,
                            audiovisual_label=None,
                            manipulation_type=f"{track}:{attack_id}" if attack_id else track,
                            start_timestamp=None,
                            end_timestamp=None,
                            official_split=split,
                        )
                    )
        return records

    def validate(self) -> DatasetInspectionSummary:
        records = self.parse_records()
        audio_present = sum(1 for r in records if r.audio_path and Path(r.audio_path).exists())
        notes: List[str] = []
        if records and audio_present == 0:
            notes.append(
                "Protocol & key metadata parsed and verified; multi-GB FLAC eval audio archives awaiting --confirm-large download."
            )
        if not records:
            notes.append("No ASVspoof 2021 protocol files found in storage directory.")
        return DatasetInspectionSummary(
            dataset_id=self.dataset_id,
            storage_dir=str(self.dataset_dir),
            available=len(records) > 0,
            total_records=len(records),
            present_files=audio_present,
            split_counts=self.count_values([r.official_split for r in records]),
            label_distribution=self.count_values([r.audio_label for r in records]),
            notes=notes,
        )
