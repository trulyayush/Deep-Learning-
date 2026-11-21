"""Dataset validation runner generating integrity, missing-file, and label distribution reports."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

from .adapters import (
    ASVspoof2021Adapter,
    AVDeepfake1MAdapter,
    CIFAR100Adapter,
    CIFAR10Adapter,
    CORe50Adapter,
    CREMADAdapter,
    DatasetAdapter,
    DFDCAdapter,
    FaceForensicsAdapter,
    FakeAVCelebAdapter,
    MELDAdapter,
    MUBenchAdapter,
    RAVDESSAdapter,
    TOFUAdapter,
)
from .download_manager import DatasetDownloadManager


def build_adapters(root: Path) -> List[DatasetAdapter]:
    raw = root / "raw"
    return [
        AVDeepfake1MAdapter(raw / "deepfake" / "av_deepfake1m"),
        FakeAVCelebAdapter(raw / "deepfake" / "fakeavceleb"),
        FaceForensicsAdapter(raw / "deepfake" / "faceforensics"),
        DFDCAdapter(raw / "deepfake" / "dfdc"),
        ASVspoof2021Adapter(raw / "deepfake" / "asvspoof"),
        MELDAdapter(raw / "emotion" / "meld"),
        RAVDESSAdapter(raw / "emotion" / "ravdess"),
        CREMADAdapter(raw / "emotion" / "crema_d"),
        CORe50Adapter(raw / "continual_learning" / "core50"),
        CIFAR10Adapter(raw / "continual_learning" / "cifar10"),
        CIFAR100Adapter(raw / "continual_learning" / "cifar100"),
        MUBenchAdapter(raw / "unlearning" / "mu_bench", raw / "continual_learning" / "cifar100"),
        TOFUAdapter(raw / "unlearning" / "tofu"),
    ]


def validate_all(datasets_root: str | Path = "datasets") -> Dict[str, dict]:
    root = Path(datasets_root)
    val_dir = root / "validation"
    reports_dir = val_dir / "validation_reports"
    logs_dir = root / "logs" / "validation_logs"
    reports_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    mgr = DatasetDownloadManager(root)
    integrity: Dict[str, dict] = {}
    missing_report: Dict[str, list] = {}
    label_dist: Dict[str, dict] = {}
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    for adapter in build_adapters(root):
        summary = adapter.validate()
        data = summary.to_dict()
        data["validated_at"] = now

        (reports_dir / f"{adapter.dataset_id}_report.json").write_text(
            json.dumps(data, indent=2) + "\n", encoding="utf-8"
        )
        integrity[adapter.dataset_id] = {
            "available": summary.available,
            "total_records": summary.total_records,
            "present_files": summary.present_files,
            "corrupt_count": len(summary.corrupt_files),
            "corrupt_files": summary.corrupt_files,
            "leakage_warnings": summary.leakage_warnings,
            "notes": summary.notes,
        }
        missing_report[adapter.dataset_id] = summary.missing_files
        label_dist[adapter.dataset_id] = summary.label_distribution

        entry = mgr.get_dataset_entry(adapter.dataset_id)
        if entry["download_status"] == "ACCESS REQUIRED":
            continue
        if summary.available and not summary.corrupt_files and not summary.leakage_warnings:
            # Mark READY if full subset media is present, or VALIDATED if metadata/partial subset validated cleanly
            new_val = "VALIDATED"
            if adapter.dataset_id in ("cifar10", "cifar100", "ravdess", "tofu", "mu_bench"):
                mgr.update_dataset_status(adapter.dataset_id, download_status="READY", validation_status=new_val)
            else:
                mgr.update_dataset_status(adapter.dataset_id, validation_status=new_val)
        elif summary.corrupt_files or summary.leakage_warnings:
            mgr.update_dataset_status(adapter.dataset_id, validation_status="FAILED")

    (val_dir / "file_integrity.json").write_text(json.dumps(integrity, indent=2) + "\n", encoding="utf-8")
    (val_dir / "missing_files.json").write_text(json.dumps(missing_report, indent=2) + "\n", encoding="utf-8")
    (val_dir / "label_distribution.json").write_text(json.dumps(label_dist, indent=2) + "\n", encoding="utf-8")
    (logs_dir / "latest_validation.json").write_text(
        json.dumps({"validated_at": now, "datasets": integrity}, indent=2) + "\n", encoding="utf-8"
    )
    return integrity


if __name__ == "__main__":
    result = validate_all()
    print(json.dumps(result, indent=2))
    sys.exit(0)
