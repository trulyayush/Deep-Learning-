"""Storage usage report generator for the ECHO-MIND dataset repository."""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict

from .download_manager import format_bytes


def dir_size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def generate_storage_report(datasets_root: str | Path = "datasets") -> Dict[str, object]:
    root = Path(datasets_root)
    disk = shutil.disk_usage(root)
    raw_dir = root / "raw"

    per_dataset: Dict[str, Dict[str, object]] = {}
    if raw_dir.exists():
        for category_dir in sorted(p for p in raw_dir.iterdir() if p.is_dir()):
            for ds_dir in sorted(p for p in category_dir.iterdir() if p.is_dir()):
                size = dir_size_bytes(ds_dir)
                file_count = sum(1 for p in ds_dir.rglob("*") if p.is_file())
                per_dataset[ds_dir.name] = {
                    "category": category_dir.name,
                    "path": str(ds_dir),
                    "bytes": size,
                    "human": format_bytes(size),
                    "file_count": file_count,
                }

    total_dataset_bytes = dir_size_bytes(root)
    report = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "disk_total_bytes": disk.total,
        "disk_free_bytes": disk.free,
        "disk_total_human": format_bytes(disk.total),
        "disk_free_human": format_bytes(disk.free),
        "datasets_total_bytes": total_dataset_bytes,
        "datasets_total_human": format_bytes(total_dataset_bytes),
        "datasets": per_dataset,
    }

    out_path = root / "validation" / "storage_report.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    print(json.dumps(generate_storage_report(), indent=2))
