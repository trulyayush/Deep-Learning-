"""Continual learning dataset adapters (Module 03): CORe50, CIFAR-10, CIFAR-100."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, List

from .base import ContinualLearningMetadataRecord, DatasetAdapter, DatasetInspectionSummary


class CORe50Adapter(DatasetAdapter):
    dataset_id = "core50"
    domain = "continual_learning"

    def parse_records(self) -> List[ContinualLearningMetadataRecord]:
        records: List[ContinualLearningMetadataRecord] = []
        image_files = sorted(self.dataset_dir.rglob("*.png"))
        for img in image_files:
            # Expected CORe50 path convention: .../s<session>/o<object>/C_<sess>_<obj>_<frame>.png
            session_id = next((p for p in img.parts if p.startswith("s") and p[1:].isdigit()), None)
            object_id = next((p for p in img.parts if p.startswith("o") and p[1:].isdigit()), None)
            obj_num = int(object_id[1:]) if object_id else 0
            category_id = f"cat_{(obj_num - 1) // 5 + 1}" if obj_num > 0 else None
            split = "test" if session_id in ("s3", "s7", "s10") else ("train" if session_id else None)
            task_idx = ((obj_num - 1) // 5) if obj_num > 0 else 0
            records.append(
                ContinualLearningMetadataRecord(
                    sample_id=f"core50-{img.stem}",
                    dataset_id=self.dataset_id,
                    task_id=f"nc_task_{task_idx}",
                    class_id=category_id,
                    source_id=object_id,
                    session_id=session_id,
                    file_path=str(img),
                    official_split=split,
                    incremental_order=task_idx,
                )
            )
        return records

    def validate(self) -> DatasetInspectionSummary:
        records = self.parse_records()
        scripts_present = any(self.dataset_dir.rglob("*.py"))
        notes: List[str] = []
        if not records:
            notes.append(
                "Benchmark repository/filelists ready; 5.49 GiB core50_128x128.zip RGB archive awaiting --confirm-large download."
                if scripts_present
                else "CORe50 archive not yet downloaded."
            )
        return DatasetInspectionSummary(
            dataset_id=self.dataset_id,
            storage_dir=str(self.dataset_dir),
            available=len(records) > 0,
            total_records=len(records),
            present_files=len(records),
            split_counts=self.count_values([r.official_split for r in records]),
            label_distribution=self.count_values([r.class_id for r in records]),
            notes=notes,
        )


class _CIFARBaseAdapter(DatasetAdapter):
    domain = "continual_learning"
    tasks_per_split: int = 5

    def _load_pickle(self, path: Path) -> Dict:
        with path.open("rb") as fh:
            return pickle.load(fh, encoding="bytes")

    def parse_records(self) -> List[ContinualLearningMetadataRecord]:
        return []


class CIFAR10Adapter(_CIFARBaseAdapter):
    dataset_id = "cifar10"

    def parse_records(self) -> List[ContinualLearningMetadataRecord]:
        records: List[ContinualLearningMetadataRecord] = []
        meta_path = next(self.dataset_dir.rglob("batches.meta"), None)
        if meta_path is None:
            return records
        meta = self._load_pickle(meta_path)
        label_names = [lbl.decode("utf-8") for lbl in meta.get(b"label_names", [])]

        batch_files = sorted(self.dataset_dir.rglob("data_batch_*")) + sorted(
            self.dataset_dir.rglob("test_batch")
        )
        for bf in batch_files:
            split = "test" if bf.name == "test_batch" else "train"
            batch = self._load_pickle(bf)
            labels = batch.get(b"labels", [])
            filenames = batch.get(b"filenames", [])
            for idx, lbl_idx in enumerate(labels):
                cname = label_names[lbl_idx] if lbl_idx < len(label_names) else str(lbl_idx)
                task_num = lbl_idx // 2
                fname = filenames[idx].decode("utf-8") if idx < len(filenames) else f"{bf.name}_{idx}"
                records.append(
                    ContinualLearningMetadataRecord(
                        sample_id=f"cifar10-{split}-{bf.name}-{idx:05d}",
                        dataset_id=self.dataset_id,
                        task_id=f"task_{task_num}",
                        class_id=cname,
                        source_id=fname,
                        session_id=bf.name,
                        file_path=str(bf),
                        official_split=split,
                        incremental_order=task_num,
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
                notes=["CIFAR-10 Python batches not found."],
            )
        splits = self.count_values([r.official_split for r in records])
        notes: List[str] = []
        if splits.get("train") != 50000 or splits.get("test") != 10000:
            notes.append(f"Unexpected split counts: {splits} (expected train=50000, test=10000).")
        return DatasetInspectionSummary(
            dataset_id=self.dataset_id,
            storage_dir=str(self.dataset_dir),
            available=True,
            total_records=len(records),
            present_files=len({r.file_path for r in records if r.file_path}),
            split_counts=splits,
            label_distribution=self.count_values([r.class_id for r in records]),
            notes=notes,
        )


class CIFAR100Adapter(_CIFARBaseAdapter):
    dataset_id = "cifar100"

    def parse_records(self, num_tasks: int = 10) -> List[ContinualLearningMetadataRecord]:
        records: List[ContinualLearningMetadataRecord] = []
        meta_path = next(self.dataset_dir.rglob("meta"), None)
        if meta_path is None:
            return records
        meta = self._load_pickle(meta_path)
        fine_names = [lbl.decode("utf-8") for lbl in meta.get(b"fine_label_names", [])]
        classes_per_task = max(1, 100 // max(1, num_tasks))

        for split in ("train", "test"):
            split_file = next((p for p in self.dataset_dir.rglob(split) if p.is_file()), None)
            if split_file is None:
                continue
            batch = self._load_pickle(split_file)
            fine_labels = batch.get(b"fine_labels", [])
            coarse_labels = batch.get(b"coarse_labels", [])
            filenames = batch.get(b"filenames", [])
            for idx, lbl_idx in enumerate(fine_labels):
                cname = fine_names[lbl_idx] if lbl_idx < len(fine_names) else str(lbl_idx)
                task_num = min(num_tasks - 1, lbl_idx // classes_per_task)
                fname = filenames[idx].decode("utf-8") if idx < len(filenames) else f"{split}_{idx}"
                coarse = coarse_labels[idx] if idx < len(coarse_labels) else 0
                records.append(
                    ContinualLearningMetadataRecord(
                        sample_id=f"cifar100-{split}-{idx:05d}",
                        dataset_id=self.dataset_id,
                        task_id=f"task_{task_num}",
                        class_id=cname,
                        source_id=fname,
                        session_id=f"superclass_{coarse}",
                        file_path=str(split_file),
                        official_split=split,
                        incremental_order=task_num,
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
                notes=["CIFAR-100 Python batches not found."],
            )
        splits = self.count_values([r.official_split for r in records])
        notes: List[str] = []
        if splits.get("train") != 50000 or splits.get("test") != 10000:
            notes.append(f"Unexpected split counts: {splits} (expected train=50000, test=10000).")
        return DatasetInspectionSummary(
            dataset_id=self.dataset_id,
            storage_dir=str(self.dataset_dir),
            available=True,
            total_records=len(records),
            present_files=len({r.file_path for r in records if r.file_path}),
            split_counts=splits,
            label_distribution=self.count_values([r.task_id for r in records]),
            notes=notes,
        )
