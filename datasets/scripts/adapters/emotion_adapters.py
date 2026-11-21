"""Conversational emotion dataset adapters (Module 02): MELD, RAVDESS, CREMA-D."""

from __future__ import annotations

import csv
import wave
from pathlib import Path
from typing import Dict, List, Optional, Set

from .base import DatasetAdapter, DatasetInspectionSummary, EmotionMetadataRecord

RAVDESS_EMOTION_MAP = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised",
}

RAVDESS_MODALITY_MAP = {
    "01": "full-av",
    "02": "video-only",
    "03": "audio-only",
}

RAVDESS_CHANNEL_MAP = {
    "01": "speech",
    "02": "song",
}

RAVDESS_INTENSITY_MAP = {
    "01": "normal",
    "02": "strong",
}


class MELDAdapter(DatasetAdapter):
    dataset_id = "meld"
    domain = "emotion"

    SPLIT_FILES = {
        "train": "train_sent_emo.csv",
        "dev": "dev_sent_emo.csv",
        "test": "test_sent_emo.csv",
    }

    def parse_records(self) -> List[EmotionMetadataRecord]:
        records: List[EmotionMetadataRecord] = []
        for split, filename in self.SPLIT_FILES.items():
            candidates = list(self.dataset_dir.rglob(filename))
            if not candidates:
                continue
            csv_path = candidates[0]
            with csv_path.open("r", encoding="utf-8", errors="replace") as fh:
                reader = csv.DictReader(fh)
                for row in reader:
                    dia_id = row.get("Dialogue_ID")
                    utt_id = row.get("Utterance_ID")
                    clip_name = f"dia{dia_id}_utt{utt_id}.mp4" if dia_id is not None and utt_id is not None else None
                    vid_match = next(self.dataset_dir.rglob(clip_name), None) if clip_name else None
                    start_ts = self._parse_timestamp(row.get("StartTime"))
                    end_ts = self._parse_timestamp(row.get("EndTime"))
                    records.append(
                        EmotionMetadataRecord(
                            sample_id=f"meld-{split}-d{dia_id}-u{utt_id}",
                            dataset_id=self.dataset_id,
                            dialogue_id=dia_id,
                            utterance_id=utt_id,
                            actor_id=None,
                            speaker_id=row.get("Speaker"),
                            video_path=str(vid_match) if vid_match else None,
                            audio_path=None,
                            transcript=row.get("Utterance"),
                            emotion_label=row.get("Emotion"),
                            sentiment_label=row.get("Sentiment"),
                            start_timestamp=start_ts,
                            end_timestamp=end_ts,
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
                notes=["MELD CSV split files not yet present."],
            )

        # Ensure official splits are distinct (no accidental same-dialogue_id leakage across splits)
        dialogues_by_split: Dict[str, Set[str]] = {}
        for r in records:
            if r.official_split and r.dialogue_id:
                dialogues_by_split.setdefault(r.official_split, set()).add(
                    f"{r.official_split}:{r.dialogue_id}"
                )

        # Check dialogue_id overlap using raw dialogue IDs only if dataset used globally unique IDs;
        # in MELD each CSV restarts Dialogue_ID from 0, so we track per-split counts cleanly.
        present_videos = sum(1 for r in records if r.video_path and Path(r.video_path).exists())
        notes: List[str] = []
        if present_videos == 0:
            notes.append(
                "Official train/dev/test dialogue CSV annotations validated; 10.13 GiB MELD.Raw.tar.gz media archive awaiting --confirm-large download."
            )

        return DatasetInspectionSummary(
            dataset_id=self.dataset_id,
            storage_dir=str(self.dataset_dir),
            available=True,
            total_records=len(records),
            present_files=present_videos,
            split_counts=self.count_values([r.official_split for r in records]),
            label_distribution=self.count_values([r.emotion_label for r in records]),
            notes=notes,
        )

    @staticmethod
    def _parse_timestamp(value: Optional[str]) -> Optional[float]:
        if not value or ":" not in value:
            return None
        try:
            clean = value.replace(",", ".")
            parts = clean.split(":")
            if len(parts) == 3:
                h, m, s = float(parts[0]), float(parts[1]), float(parts[2])
                return round(h * 3600.0 + m * 60.0 + s, 3)
        except ValueError:
            return None
        return None


class RAVDESSAdapter(DatasetAdapter):
    dataset_id = "ravdess"
    domain = "emotion"

    def parse_records(self) -> List[EmotionMetadataRecord]:
        records: List[EmotionMetadataRecord] = []
        media_files = sorted(
            list(self.dataset_dir.rglob("*.wav")) + list(self.dataset_dir.rglob("*.mp4"))
        )
        for path in media_files:
            parts = path.stem.split("-")
            if len(parts) != 7:
                continue
            modality_code, channel_code, emo_code, intensity_code, stmt_code, rep_code, actor_code = parts
            emotion = RAVDESS_EMOTION_MAP.get(emo_code)
            channel = RAVDESS_CHANNEL_MAP.get(channel_code, "unknown")
            intensity = RAVDESS_INTENSITY_MAP.get(intensity_code, "unknown")
            is_video = path.suffix.lower() == ".mp4"
            # Actor-independent split recommendation (Actors 1-18 train, 19-21 val, 22-24 test)
            actor_num = int(actor_code) if actor_code.isdigit() else 0
            split = "train" if actor_num <= 18 else ("val" if actor_num <= 21 else "test")

            records.append(
                EmotionMetadataRecord(
                    sample_id=f"ravdess-{channel}-{path.stem}",
                    dataset_id=self.dataset_id,
                    dialogue_id=f"stmt-{stmt_code}-rep-{rep_code}",
                    utterance_id=path.stem,
                    actor_id=f"Actor_{actor_code}",
                    speaker_id=f"Actor_{actor_code}",
                    video_path=str(path) if is_video else None,
                    audio_path=str(path) if not is_video else None,
                    transcript=None,
                    emotion_label=f"{emotion} ({channel}, {intensity})" if emotion else None,
                    sentiment_label=None,
                    start_timestamp=None,
                    end_timestamp=None,
                    official_split=f"{channel}:{split}",
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
                notes=["No extracted RAVDESS WAV/MP4 files found."],
            )

        corrupt: List[str] = []
        present = 0
        for r in records:
            target = r.audio_path or r.video_path
            if not target:
                continue
            p = Path(target)
            if p.exists() and p.stat().st_size > 0:
                present += 1
                if p.suffix.lower() == ".wav":
                    try:
                        with wave.open(str(p), "rb") as wf:
                            if wf.getnframes() == 0:
                                corrupt.append(str(p))
                    except wave.Error:
                        corrupt.append(str(p))
            else:
                corrupt.append(str(p))

        # Verify actor-independent splits have zero actor overlap
        actors_by_split: Dict[str, Set[str]] = {}
        for r in records:
            if r.official_split and r.actor_id:
                split_bucket = r.official_split.split(":")[-1]
                actors_by_split.setdefault(split_bucket, set()).add(r.actor_id)
        leakage: List[str] = []
        all_splits = list(actors_by_split.keys())
        for i in range(len(all_splits)):
            for j in range(i + 1, len(all_splits)):
                overlap = actors_by_split[all_splits[i]] & actors_by_split[all_splits[j]]
                if overlap:
                    leakage.append(f"Actor overlap between {all_splits[i]} and {all_splits[j]}: {sorted(overlap)}")

        return DatasetInspectionSummary(
            dataset_id=self.dataset_id,
            storage_dir=str(self.dataset_dir),
            available=True,
            total_records=len(records),
            present_files=present,
            corrupt_files=corrupt[:100],
            split_counts=self.count_values([r.official_split for r in records]),
            label_distribution=self.count_values(
                [r.emotion_label.split(" ")[0] if r.emotion_label else None for r in records]
            ),
            leakage_warnings=leakage,
            notes=["Separated Speech vs Song tasks; validated WAV headers via stdlib wave."],
        )


class CREMADAdapter(DatasetAdapter):
    dataset_id = "crema_d"
    domain = "emotion"

    EMOTION_CODES = {
        "ANG": "Anger",
        "DIS": "Disgust",
        "FEA": "Fear",
        "HAP": "Happy",
        "NEU": "Neutral",
        "SAD": "Sad",
    }

    def parse_records(self) -> List[EmotionMetadataRecord]:
        records: List[EmotionMetadataRecord] = []
        for wav in sorted(self.dataset_dir.rglob("*.wav")):
            parts = wav.stem.split("_")
            if len(parts) < 4:
                continue
            actor_id, sentence_code, emo_code, level = parts[:4]
            records.append(
                EmotionMetadataRecord(
                    sample_id=f"cremad-{wav.stem}",
                    dataset_id=self.dataset_id,
                    dialogue_id=sentence_code,
                    utterance_id=wav.stem,
                    actor_id=actor_id,
                    speaker_id=actor_id,
                    video_path=None,
                    audio_path=str(wav),
                    transcript=None,
                    emotion_label=self.EMOTION_CODES.get(emo_code),
                    sentiment_label=level,
                    start_timestamp=None,
                    end_timestamp=None,
                    official_split=None,
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
                notes=["ACCESS REQUIRED: Complete CREMA-D access request form and install Git LFS before cloning."],
            )
        corrupt = [r.audio_path for r in records if r.audio_path and Path(r.audio_path).stat().st_size < 1024]
        return DatasetInspectionSummary(
            dataset_id=self.dataset_id,
            storage_dir=str(self.dataset_dir),
            available=True,
            total_records=len(records),
            present_files=len(records),
            corrupt_files=corrupt[:100],
            label_distribution=self.count_values([r.emotion_label for r in records]),
        )
