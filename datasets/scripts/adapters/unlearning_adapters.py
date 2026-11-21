"""Selective machine unlearning adapters (Module 04): MU-Bench & TOFU."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

from .base import DatasetAdapter, DatasetInspectionSummary, UnlearningMetadataRecord, sha256_file


class MUBenchAdapter(DatasetAdapter):
    dataset_id = "mu_bench"
    domain = "unlearning"

    def __init__(self, dataset_dir: str | Path, local_cifar100_dir: str | Path = "datasets/raw/continual_learning/cifar100") -> None:
        super().__init__(dataset_dir)
        self.local_cifar100_dir = Path(local_cifar100_dir)

    def parse_records(self) -> List[UnlearningMetadataRecord]:
        records: List[UnlearningMetadataRecord] = []
        # Register MU-Bench framework task definitions & detect reusable local CIFAR-100 without duplicating files.
        config_files = sorted(
            list(self.dataset_dir.rglob("*.py"))
            + list(self.dataset_dir.rglob("*.yaml"))
            + list(self.dataset_dir.rglob("*.json"))
        )
        cifar_available = any(self.local_cifar100_dir.rglob("train"))
        for idx, cfg in enumerate(config_files[:100]):
            if "cifar" in cfg.name.lower() or "unlearn" in cfg.name.lower() or "forget" in cfg.name.lower():
                records.append(
                    UnlearningMetadataRecord(
                        sample_id=f"mubench-cfg-{idx:04d}",
                        dataset_id=self.dataset_id,
                        original_sample_id=cfg.name,
                        file_path=str(cfg),
                        content_hash=sha256_file(cfg),
                        training_split="benchmark_config",
                        forget_set_status="definition_ready",
                        retain_set_status="links_local_cifar100" if cifar_available else "pending_task_data",
                        model_version_reference=None,
                        replay_memory_reference=None,
                        derived_feature_reference=None,
                    )
                )
        return records

    def validate(self) -> DatasetInspectionSummary:
        records = self.parse_records()
        repo_present = any(self.dataset_dir.rglob("README.md"))
        cifar_linked = any(self.local_cifar100_dir.rglob("train"))
        notes: List[str] = []
        if repo_present:
            notes.append(
                "MU-Bench benchmark framework extracted cleanly; links to local CIFAR-100 without duplicate storage."
                if cifar_linked
                else "MU-Bench repository extracted; local CIFAR-100 not yet extracted."
            )
        else:
            notes.append("MU-Bench repository not yet downloaded.")
        return DatasetInspectionSummary(
            dataset_id=self.dataset_id,
            storage_dir=str(self.dataset_dir),
            available=repo_present,
            total_records=len(records),
            present_files=len(records),
            split_counts=self.count_values([r.retain_set_status for r in records]),
            notes=notes,
        )


class TOFUAdapter(DatasetAdapter):
    dataset_id = "tofu"
    domain = "unlearning"

    def parse_records(self) -> List[UnlearningMetadataRecord]:
        records: List[UnlearningMetadataRecord] = []
        json_files = sorted(self.dataset_dir.glob("*.json"))
        for jf in json_files:
            split_name = jf.stem
            # TOFU files may be JSON arrays or JSONL
            raw_text = jf.read_text(encoding="utf-8", errors="replace").strip()
            if not raw_text:
                continue
            try:
                parsed = json.loads(raw_text)
                rows = parsed if isinstance(parsed, list) else [parsed]
            except json.JSONDecodeError:
                rows = [json.loads(line) for line in raw_text.splitlines() if line.strip()]

            file_hash = sha256_file(jf)
            is_forget = split_name.startswith("forget")
            is_retain = split_name.startswith("retain")
            for idx, _row in enumerate(rows):
                records.append(
                    UnlearningMetadataRecord(
                        sample_id=f"tofu-{split_name}-{idx:04d}",
                        dataset_id=self.dataset_id,
                        original_sample_id=f"{split_name}:{idx}",
                        file_path=str(jf),
                        content_hash=file_hash,
                        training_split=split_name,
                        forget_set_status="forget" if is_forget else ("holdout" if split_name.startswith("holdout") else "none"),
                        retain_set_status="retain" if is_retain else "none",
                        model_version_reference=None,
                        replay_memory_reference=None,
                        derived_feature_reference=None,
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
                notes=["TOFU JSON split files not yet downloaded."],
            )
        return DatasetInspectionSummary(
            dataset_id=self.dataset_id,
            storage_dir=str(self.dataset_dir),
            available=True,
            total_records=len(records),
            present_files=len({r.file_path for r in records if r.file_path}),
            split_counts=self.count_values([r.training_split for r in records]),
            label_distribution=self.count_values([r.forget_set_status for r in records]),
            notes=["Text-only unlearning benchmark kept strictly isolated from audio/video unlearning datasets."],
        )
