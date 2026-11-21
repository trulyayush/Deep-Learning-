"""ECHO-MIND Dataset Acquisition & Resumable Download Manager.

Supports:
- Source and size inspection (HTTP HEAD)
- Available storage & estimated extracted-size checks
- Large archive confirmation gate (> 1 GiB requires --confirm-large)
- Strict access-gate enforcement (never bypasses gated HF/Google Form datasets)
- Resumable HTTP downloads via Range headers + retry with backoff
- Duplicate detection & MD5 checksum verification
- Safe path-confined archive extraction (.tar.gz / .zip)
- Structured download/extraction logging and registry state updates
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tarfile
import time
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

LARGE_ARCHIVE_THRESHOLD_BYTES = 1024 * 1024 * 1024  # 1 GiB


def _utc_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _assert_within_dir(target: Path, root: Path) -> Path:
    resolved_root = root.resolve()
    resolved_target = target.resolve()
    if resolved_target != resolved_root and not str(resolved_target).startswith(str(resolved_root) + os.sep):
        raise ValueError(f"Unsafe path traversal detected: {target} outside {root}")
    return resolved_target


def compute_md5(file_path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.md5()
    with file_path.open("rb") as fh:
        while True:
            chunk = fh.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


@dataclass
class StorageCheckResult:
    destination: str
    free_bytes: int
    required_bytes: int
    sufficient: bool
    free_human: str
    required_human: str


def format_bytes(num_bytes: int) -> str:
    size = float(num_bytes)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if size < 1024.0 or unit == "TiB":
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{num_bytes} B"


class DatasetDownloadManager:
    """Manages dataset acquisition, storage safety checks, extraction, and registry tracking."""

    def __init__(self, datasets_root: str | Path = "datasets") -> None:
        self.root = Path(datasets_root)
        self.registry_path = self.root / "registry" / "dataset_registry.json"
        self.sources_path = self.root / "registry" / "dataset_sources.json"
        self.licences_path = self.root / "registry" / "dataset_licences.json"
        self.download_logs_dir = self.root / "logs" / "download_logs"
        self.extraction_logs_dir = self.root / "logs" / "extraction_logs"
        self.provenance_dir = self.root / "metadata" / "provenance_records"
        self.source_meta_dir = self.root / "metadata" / "source_metadata"
        self.licence_records_dir = self.root / "metadata" / "licence_records"
        for d in (
            self.download_logs_dir,
            self.extraction_logs_dir,
            self.provenance_dir,
            self.source_meta_dir,
            self.licence_records_dir,
        ):
            d.mkdir(parents=True, exist_ok=True)

    def load_registry(self) -> Dict[str, Any]:
        return json.loads(self.registry_path.read_text(encoding="utf-8"))

    def save_registry(self, data: Dict[str, Any]) -> None:
        data["updated_at"] = _utc_iso()
        self.registry_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")

    def load_sources(self) -> Dict[str, Any]:
        return json.loads(self.sources_path.read_text(encoding="utf-8"))

    def get_dataset_entry(self, dataset_id: str) -> Dict[str, Any]:
        registry = self.load_registry()
        for entry in registry.get("datasets", []):
            if entry["dataset_id"] == dataset_id:
                return entry
        raise KeyError(f"Unknown dataset_id: {dataset_id}")

    def update_dataset_status(
        self,
        dataset_id: str,
        *,
        download_status: Optional[str] = None,
        extraction_status: Optional[str] = None,
        validation_status: Optional[str] = None,
    ) -> None:
        registry = self.load_registry()
        for entry in registry.get("datasets", []):
            if entry["dataset_id"] == dataset_id:
                if download_status is not None:
                    entry["download_status"] = download_status
                if extraction_status is not None:
                    entry["extraction_status"] = extraction_status
                if validation_status is not None:
                    entry["validation_status"] = validation_status
                break
        self.save_registry(registry)

    def check_storage(
        self,
        destination: Path,
        estimated_download_bytes: int,
        estimated_extracted_bytes: int = 0,
    ) -> StorageCheckResult:
        destination.mkdir(parents=True, exist_ok=True)
        usage = shutil.disk_usage(destination)
        # Keep at least 2 GiB safety headroom
        required = estimated_download_bytes + estimated_extracted_bytes + (2 * 1024**3)
        return StorageCheckResult(
            destination=str(destination),
            free_bytes=usage.free,
            required_bytes=required,
            sufficient=usage.free >= required,
            free_human=format_bytes(usage.free),
            required_human=format_bytes(required),
        )

    def inspect_remote_size(self, url: str, timeout: int = 10) -> Optional[int]:
        req = urllib.request.Request(
            url,
            method="HEAD",
            headers={"User-Agent": "ECHO-MIND-DatasetManager/1.0"},
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                length = resp.headers.get("Content-Length")
                return int(length) if length and length.isdigit() else None
        except Exception:
            return None

    def resumable_download(
        self,
        url: str,
        dest_file: Path,
        *,
        expected_size: Optional[int] = None,
        expected_md5: Optional[str] = None,
        max_retries: int = 3,
        chunk_size: int = 1024 * 512,
    ) -> Dict[str, Any]:
        """Download a URL with HTTP Range resume, duplicate detection, and MD5 verification."""
        dest_file.parent.mkdir(parents=True, exist_ok=True)
        part_file = dest_file.with_name(dest_file.name + ".part")

        # Duplicate detection: if destination already exists and matches size/md5, skip cleanly
        if dest_file.exists():
            actual_size = dest_file.stat().st_size
            size_ok = expected_size is None or actual_size == expected_size
            md5_ok = expected_md5 is None or compute_md5(dest_file) == expected_md5
            if size_ok and md5_ok and actual_size > 0:
                return {
                    "file": str(dest_file),
                    "url": url,
                    "status": "duplicate_verified_skip",
                    "bytes": actual_size,
                    "md5_verified": expected_md5 is not None,
                    "timestamp": _utc_iso(),
                }

        remote_size = self.inspect_remote_size(url) or expected_size
        attempt = 0
        last_error: Optional[str] = None
        force_restart = False

        while attempt < max_retries:
            attempt += 1
            if force_restart:
                part_file.unlink(missing_ok=True)
                force_restart = False

            existing_bytes = part_file.stat().st_size if part_file.exists() else 0
            if remote_size is not None and existing_bytes > remote_size:
                part_file.unlink(missing_ok=True)
                existing_bytes = 0

            headers = {"User-Agent": "ECHO-MIND-DatasetManager/1.0"}
            mode = "wb"
            if existing_bytes > 0 and (remote_size is None or existing_bytes < remote_size):
                headers["Range"] = f"bytes={existing_bytes}-"
                mode = "ab"

            req = urllib.request.Request(url, headers=headers)
            try:
                with urllib.request.urlopen(req, timeout=60) as resp:
                    status_code = getattr(resp, "status", 200)
                    if status_code == 200 and mode == "ab":
                        # Server ignored Range header; restart clean
                        mode = "wb"

                    content_len_hdr = resp.headers.get("Content-Length")
                    resp_len = int(content_len_hdr) if content_len_hdr and content_len_hdr.isdigit() else None
                    if status_code == 200 and resp_len is not None:
                        remote_size = resp_len

                    with part_file.open(mode) as fh:
                        while True:
                            chunk = resp.read(chunk_size)
                            if not chunk:
                                break
                            fh.write(chunk)

                actual_size = part_file.stat().st_size
                if remote_size is not None and actual_size != remote_size:
                    if actual_size > remote_size:
                        part_file.unlink(missing_ok=True)
                    raise IOError(f"Size mismatch: got {actual_size} of {remote_size} bytes")

                if expected_md5 is not None:
                    actual_md5 = compute_md5(part_file)
                    if actual_md5 != expected_md5:
                        part_file.unlink(missing_ok=True)
                        raise ValueError(f"MD5 mismatch: expected {expected_md5}, got {actual_md5}")

                part_file.replace(dest_file)
                return {
                    "file": str(dest_file),
                    "url": url,
                    "status": "downloaded",
                    "bytes": actual_size,
                    "attempts": attempt,
                    "md5_verified": expected_md5 is not None,
                    "timestamp": _utc_iso(),
                }
            except urllib.error.HTTPError as exc:
                last_error = f"HTTPError: HTTP Error {exc.code}: {exc.reason}"
                if exc.code == 416:
                    force_restart = True
                time.sleep(min(4, attempt))
            except (urllib.error.URLError, IOError, ValueError, TimeoutError) as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                time.sleep(min(4, attempt))

        raise RuntimeError(f"Failed to download {url} after {max_retries} attempts: {last_error}")

    def extract_archive(self, archive_path: Path, extract_dir: Path) -> Dict[str, Any]:
        """Safely extract .tar.gz or .zip archives while preserving the original archive."""
        extract_dir.mkdir(parents=True, exist_ok=True)
        extracted_files = 0

        if archive_path.name.endswith((".tar.gz", ".tgz")):
            with tarfile.open(archive_path, "r:gz") as tf:
                members = tf.getmembers()
                for m in members:
                    target = extract_dir / m.name
                    _assert_within_dir(target, extract_dir)
                tf.extractall(path=extract_dir)
                extracted_files = sum(1 for m in members if m.isfile())
        elif archive_path.suffix.lower() == ".zip":
            with zipfile.ZipFile(archive_path, "r") as zf:
                infos = zf.infolist()
                for info in infos:
                    target = extract_dir / info.filename
                    _assert_within_dir(target, extract_dir)
                zf.extractall(path=extract_dir)
                extracted_files = sum(1 for i in infos if not i.is_dir())
        else:
            raise ValueError(f"Unsupported archive format: {archive_path}")

        record = {
            "archive": str(archive_path),
            "extract_dir": str(extract_dir),
            "extracted_files": extracted_files,
            "timestamp": _utc_iso(),
        }
        log_file = self.extraction_logs_dir / f"{archive_path.stem}_extract.json"
        log_file.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        return record

    def acquire_dataset(
        self,
        dataset_id: str,
        *,
        confirm_large: bool = False,
        include_optional_large: bool = False,
        storage_root: Optional[str | Path] = None,
        extract: bool = True,
    ) -> Dict[str, Any]:
        """Acquire manageable artifacts for `dataset_id`, respecting access gates & size guards."""
        entry = self.get_dataset_entry(dataset_id)
        sources = self.load_sources()["sources"].get(dataset_id, {})

        if entry["download_status"] == "ACCESS REQUIRED":
            record = {
                "dataset_id": dataset_id,
                "status": "ACCESS REQUIRED",
                "official_download_url": entry["official_download_url"],
                "required_access_permissions": entry["required_access_permissions"],
                "message": "Restricted dataset; complete official approval before downloading.",
                "timestamp": _utc_iso(),
            }
            self._write_download_log(dataset_id, record)
            return record

        dest_dir = Path(storage_root) / dataset_id if storage_root else Path(entry["storage_destination"])
        archives_dir = dest_dir / "archives"
        extracted_dir = dest_dir / "extracted"
        archives_dir.mkdir(parents=True, exist_ok=True)

        artifacts = sources.get("artifacts", [])
        selected = [
            art for art in artifacts
            if art.get("auto_download", False) or include_optional_large
        ]

        if not selected:
            return {
                "dataset_id": dataset_id,
                "status": entry["download_status"],
                "message": "No auto-downloadable artifacts selected.",
            }

        total_est = sum(int(art.get("size_bytes") or 0) for art in selected)
        if total_est > LARGE_ARCHIVE_THRESHOLD_BYTES and not confirm_large:
            return {
                "dataset_id": dataset_id,
                "status": "READY TO DOWNLOAD",
                "estimated_bytes": total_est,
                "estimated_human": format_bytes(total_est),
                "message": "Archive exceeds 1 GiB threshold; re-run with --confirm-large to proceed.",
            }

        storage = self.check_storage(dest_dir, total_est, int(total_est * 1.25))
        if not storage.sufficient:
            return {
                "dataset_id": dataset_id,
                "status": "FAILED",
                "message": f"Insufficient disk space: free {storage.free_human}, required {storage.required_human}.",
            }

        self.update_dataset_status(dataset_id, download_status="DOWNLOADING")
        results: List[Dict[str, Any]] = []
        extracted_count = 0

        try:
            for art in selected:
                name = art["name"]
                url = art["url"]
                target_file = archives_dir / name if name.endswith((".tar.gz", ".tgz", ".zip")) else dest_dir / name
                dl_res = self.resumable_download(
                    url,
                    target_file,
                    expected_size=art.get("size_bytes"),
                    expected_md5=art.get("md5"),
                )
                results.append(dl_res)

                if extract and target_file.name.endswith((".tar.gz", ".tgz", ".zip")):
                    ext_res = self.extract_archive(target_file, extracted_dir)
                    extracted_count += ext_res["extracted_files"]

            has_pending_large = any(not art.get("auto_download", False) for art in artifacts) and not include_optional_large
            dl_status = "DOWNLOADED"
            ext_status = "EXTRACTED" if extracted_count > 0 else ("NOT REQUIRED" if not any(a["name"].endswith((".zip", ".tar.gz")) for a in selected) else "EXTRACTED")
            self.update_dataset_status(
                dataset_id,
                download_status=dl_status,
                extraction_status=ext_status,
            )

            summary = {
                "dataset_id": dataset_id,
                "status": dl_status,
                "partial_subset_only": has_pending_large,
                "artifacts": results,
                "extracted_files": extracted_count,
                "storage_free_after": format_bytes(shutil.disk_usage(dest_dir).free),
                "timestamp": _utc_iso(),
            }
            self._write_download_log(dataset_id, summary)
            self._write_provenance(entry, sources, summary)
            return summary
        except Exception as exc:
            self.update_dataset_status(dataset_id, download_status="FAILED")
            err = {
                "dataset_id": dataset_id,
                "status": "FAILED",
                "error": str(exc),
                "timestamp": _utc_iso(),
            }
            self._write_download_log(dataset_id, err)
            raise

    def _write_download_log(self, dataset_id: str, payload: Dict[str, Any]) -> None:
        log_path = self.download_logs_dir / f"{dataset_id}_download.json"
        log_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    def _write_provenance(
        self,
        entry: Dict[str, Any],
        sources: Dict[str, Any],
        dl_summary: Dict[str, Any],
    ) -> None:
        prov_path = self.provenance_dir / f"{entry['dataset_id']}_provenance.json"
        prov_path.write_text(
            json.dumps(
                {
                    "dataset_id": entry["dataset_id"],
                    "name": entry["name"],
                    "official_website": entry["official_website"],
                    "licence": entry["licence"],
                    "commercial_eligible": entry["commercial_eligible"],
                    "subset_strategy": sources.get("subset_strategy"),
                    "last_acquisition": dl_summary,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="ECHO-MIND Dataset Acquisition Manager")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("status", help="Show registry status and available disk storage")

    p_dl = sub.add_parser("download", help="Download manageable artifacts for a dataset")
    p_dl.add_argument("dataset_id", help="Dataset ID or 'manageable' for priority open subsets")
    p_dl.add_argument("--confirm-large", action="store_true", help="Confirm archives > 1 GiB")
    p_dl.add_argument("--include-large", action="store_true", help="Include optional large media archives")
    p_dl.add_argument("--dest", default=None, help="Optional external storage directory")
    p_dl.add_argument("--no-extract", action="store_true", help="Download without extracting archives")

    args = parser.parse_args(argv)
    mgr = DatasetDownloadManager()

    if args.command == "status":
        usage = shutil.disk_usage(mgr.root)
        print(f"Free storage: {format_bytes(usage.free)} / {format_bytes(usage.total)}")
        for ds in mgr.load_registry()["datasets"]:
            print(
                f"{ds['dataset_id']:15s} | {ds['download_status']:17s} | "
                f"{ds['extraction_status']:12s} | {ds['validation_status']:12s} | {ds['dataset_size_human']}"
            )
        return 0

    if args.command == "download":
        targets = (
            [
                ds["dataset_id"]
                for ds in mgr.load_registry()["datasets"]
                if ds["download_status"] != "ACCESS REQUIRED"
            ]
            if args.dataset_id == "manageable"
            else [args.dataset_id]
        )
        for tid in targets:
            res = mgr.acquire_dataset(
                tid,
                confirm_large=args.confirm_large,
                include_optional_large=args.include_large,
                storage_root=args.dest,
                extract=not args.no_extract,
            )
            print(json.dumps(res, indent=2))
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
