"""Unified CSV manifest builder for ECHO-MIND's four intelligence modules."""

from __future__ import annotations

import csv
import dataclasses
from pathlib import Path
from typing import Any, Dict, List, Sequence

from .adapters import (
    ASVspoof2021Adapter,
    AVDeepfake1MAdapter,
    CIFAR100Adapter,
    CIFAR10Adapter,
    CORe50Adapter,
    CREMADAdapter,
    DFDCAdapter,
    FaceForensicsAdapter,
    FakeAVCelebAdapter,
    MELDAdapter,
    MUBenchAdapter,
    RAVDESSAdapter,
    TOFUAdapter,
)

DEEPFAKE_FIELDS = [
    "sample_id",
    "dataset_id",
    "original_video_id",
    "video_path",
    "audio_path",
    "visual_label",
    "audio_label",
    "audiovisual_label",
    "manipulation_type",
    "start_timestamp",
    "end_timestamp",
    "official_split",
]

EMOTION_FIELDS = [
    "sample_id",
    "dataset_id",
    "dialogue_id",
    "utterance_id",
    "actor_id",
    "speaker_id",
    "video_path",
    "audio_path",
    "transcript",
    "emotion_label",
    "sentiment_label",
    "start_timestamp",
    "end_timestamp",
    "official_split",
]

CONTINUAL_FIELDS = [
    "sample_id",
    "dataset_id",
    "task_id",
    "class_id",
    "source_id",
    "session_id",
    "file_path",
    "official_split",
    "incremental_order",
]

UNLEARNING_FIELDS = [
    "sample_id",
    "dataset_id",
    "original_sample_id",
    "file_path",
    "content_hash",
    "training_split",
    "forget_set_status",
    "retain_set_status",
    "model_version_reference",
    "replay_memory_reference",
    "derived_feature_reference",
]


def _write_csv(path: Path, fieldnames: Sequence[str], records: Sequence[Any]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for rec in records:
            row = dataclasses.asdict(rec)
            # Keep unavailable fields empty/null without inference
            clean = {k: ("" if row.get(k) is None else row.get(k)) for k in fieldnames}
            writer.writerow(clean)
    return len(records)


def build_all_manifests(datasets_root: str | Path = "datasets") -> Dict[str, int]:
    root = Path(datasets_root)
    raw = root / "raw"
    manifests = root / "manifests"
    manifests.mkdir(parents=True, exist_ok=True)

    deepfake_records = (
        AVDeepfake1MAdapter(raw / "deepfake" / "av_deepfake1m").parse_records()
        + FakeAVCelebAdapter(raw / "deepfake" / "fakeavceleb").parse_records()
        + FaceForensicsAdapter(raw / "deepfake" / "faceforensics").parse_records()
        + DFDCAdapter(raw / "deepfake" / "dfdc").parse_records()
        + ASVspoof2021Adapter(raw / "deepfake" / "asvspoof").parse_records()
    )

    emotion_records = (
        MELDAdapter(raw / "emotion" / "meld").parse_records()
        + RAVDESSAdapter(raw / "emotion" / "ravdess").parse_records()
        + CREMADAdapter(raw / "emotion" / "crema_d").parse_records()
    )

    continual_records = (
        CORe50Adapter(raw / "continual_learning" / "core50").parse_records()
        + CIFAR10Adapter(raw / "continual_learning" / "cifar10").parse_records()
        + CIFAR100Adapter(raw / "continual_learning" / "cifar100").parse_records()
    )

    unlearning_records = (
        MUBenchAdapter(raw / "unlearning" / "mu_bench", raw / "continual_learning" / "cifar100").parse_records()
        + TOFUAdapter(raw / "unlearning" / "tofu").parse_records()
    )

    counts = {
        "deepfake_manifest.csv": _write_csv(manifests / "deepfake_manifest.csv", DEEPFAKE_FIELDS, deepfake_records),
        "emotion_manifest.csv": _write_csv(manifests / "emotion_manifest.csv", EMOTION_FIELDS, emotion_records),
        "continual_learning_manifest.csv": _write_csv(
            manifests / "continual_learning_manifest.csv", CONTINUAL_FIELDS, continual_records
        ),
        "unlearning_manifest.csv": _write_csv(
            manifests / "unlearning_manifest.csv", UNLEARNING_FIELDS, unlearning_records
        ),
    }

    master_fields = ["sample_id", "dataset_id", "module_domain", "official_split", "primary_label", "media_or_file_path"]
    with (manifests / "master_manifest.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=master_fields)
        writer.writeheader()
        total = 0
        for r in deepfake_records:
            writer.writerow(
                {
                    "sample_id": r.sample_id,
                    "dataset_id": r.dataset_id,
                    "module_domain": "deepfake",
                    "official_split": r.official_split or "",
                    "primary_label": r.audiovisual_label or r.visual_label or r.audio_label or "",
                    "media_or_file_path": r.video_path or r.audio_path or "",
                }
            )
            total += 1
        for r in emotion_records:
            writer.writerow(
                {
                    "sample_id": r.sample_id,
                    "dataset_id": r.dataset_id,
                    "module_domain": "emotion",
                    "official_split": r.official_split or "",
                    "primary_label": r.emotion_label or "",
                    "media_or_file_path": r.video_path or r.audio_path or "",
                }
            )
            total += 1
        for r in continual_records:
            writer.writerow(
                {
                    "sample_id": r.sample_id,
                    "dataset_id": r.dataset_id,
                    "module_domain": "continual_learning",
                    "official_split": r.official_split or "",
                    "primary_label": r.class_id or "",
                    "media_or_file_path": r.file_path or "",
                }
            )
            total += 1
        for r in unlearning_records:
            writer.writerow(
                {
                    "sample_id": r.sample_id,
                    "dataset_id": r.dataset_id,
                    "module_domain": "unlearning",
                    "official_split": r.training_split or "",
                    "primary_label": r.forget_set_status or r.retain_set_status or "",
                    "media_or_file_path": r.file_path or "",
                }
            )
            total += 1
        counts["master_manifest.csv"] = total

    return counts


if __name__ == "__main__":
    import json

    print(json.dumps(build_all_manifests(), indent=2))
