"""SQL-compatible SQLAlchemy ORM models and Pydantic v2 validation schemas for ECHO-MIND.

Aligned with `database/schema.sql` (PostgreSQL / Supabase) and `server/db.ts` (SQLite WAL).
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _new_uuid() -> str:
    return str(uuid4())


class Base(DeclarativeBase):
    """Declarative base for all ECHO-MIND SQL-compatible ORM entities."""


# ---------------------------------------------------------------------------
# SQLAlchemy 2.0 ORM Entities
# ---------------------------------------------------------------------------


class UserORM(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=_new_uuid)
    email: Mapped[str] = mapped_column(String(320), unique=True, nullable=False)
    display_name: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    role: Mapped[str] = mapped_column(String(32), nullable=False, default="analyst")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=_utcnow)

    media_uploads: Mapped[List["MediaUploadORM"]] = relationship(back_populates="user")
    analysis_jobs: Mapped[List["AnalysisJobORM"]] = relationship(back_populates="user")


class MediaUploadORM(Base):
    __tablename__ = "media_uploads"
    __table_args__ = (
        CheckConstraint("media_type IN ('video', 'audio')", name="ck_media_uploads_type"),
        Index("idx_media_uploads_user", "user_id", "created_at"),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=_new_uuid)
    user_id: Mapped[Optional[str]] = mapped_column(
        String(64), ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    original_name: Mapped[str] = mapped_column(String(255), nullable=False)
    mime_type: Mapped[str] = mapped_column(String(128), nullable=False)
    size_bytes: Mapped[int] = mapped_column(BigInteger, nullable=False)
    media_type: Mapped[str] = mapped_column(String(16), nullable=False)
    storage_path: Mapped[str] = mapped_column(Text, nullable=False)
    duration_sec: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    width: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    height: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    fps: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    video_codec: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    audio_codec: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    sample_rate: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    has_video: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    has_audio: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    is_sample: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    preview_url: Mapped[str] = mapped_column(Text, nullable=False, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=_utcnow)

    user: Mapped[Optional["UserORM"]] = relationship(back_populates="media_uploads")
    analysis_jobs: Mapped[List["AnalysisJobORM"]] = relationship(
        back_populates="upload", cascade="all, delete-orphan"
    )


class AnalysisJobORM(Base):
    __tablename__ = "analysis_jobs"
    __table_args__ = (
        CheckConstraint("job_type IN ('forensics', 'emotion')", name="ck_analysis_jobs_type"),
        CheckConstraint(
            "status IN ('queued', 'processing', 'completed', 'failed')",
            name="ck_analysis_jobs_status",
        ),
        Index("idx_analysis_jobs_status", "status", "created_at"),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=_new_uuid)
    user_id: Mapped[Optional[str]] = mapped_column(
        String(64), ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )
    upload_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("media_uploads.id", ondelete="CASCADE"), nullable=False
    )
    job_type: Mapped[str] = mapped_column(String(32), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="queued")
    progress: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    stage: Mapped[str] = mapped_column(String(128), nullable=False, default="queued")
    error: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    result_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow, onupdate=_utcnow
    )

    user: Mapped[Optional["UserORM"]] = relationship(back_populates="analysis_jobs")
    upload: Mapped["MediaUploadORM"] = relationship(back_populates="analysis_jobs")


class AnalysisResultORM(Base):
    __tablename__ = "analysis_results"
    __table_args__ = (
        CheckConstraint(
            "overall_label IN ('LIKELY AUTHENTIC', 'POTENTIALLY MANIPULATED', 'INCONCLUSIVE')",
            name="ck_analysis_results_label",
        ),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=_new_uuid)
    job_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("analysis_jobs.id", ondelete="CASCADE"), nullable=False
    )
    upload_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("media_uploads.id", ondelete="CASCADE"), nullable=False
    )
    visual_authenticity_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    audio_authenticity_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    av_consistency_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    overall_authenticity_score: Mapped[float] = mapped_column(Float, nullable=False)
    overall_label: Mapped[str] = mapped_column(String(64), nullable=False)
    model_pipeline: Mapped[str] = mapped_column(String(255), nullable=False)
    execution_backend: Mapped[str] = mapped_column(String(64), nullable=False)
    payload: Mapped[Dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    report_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=_utcnow)


class EmotionSegmentORM(Base):
    __tablename__ = "emotion_segments"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=_new_uuid)
    analysis_id: Mapped[str] = mapped_column(String(64), nullable=False)
    upload_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("media_uploads.id", ondelete="CASCADE"), nullable=False
    )
    speaker: Mapped[str] = mapped_column(String(128), nullable=False)
    start_sec: Mapped[float] = mapped_column(Float, nullable=False)
    end_sec: Mapped[float] = mapped_column(Float, nullable=False)
    transcript: Mapped[str] = mapped_column(Text, nullable=False, default="")
    audio_emotion: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    visual_emotion: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    text_emotion: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    combined_prediction: Mapped[str] = mapped_column(String(32), nullable=False)
    combined_confidence: Mapped[float] = mapped_column(Float, nullable=False)
    modality_disagreement: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    disagreement_note: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    payload: Mapped[Dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=_utcnow)


class DatasetORM(Base):
    __tablename__ = "datasets"
    __table_args__ = (
        CheckConstraint("task_domain IN ('forensics', 'emotion')", name="ck_datasets_task_domain"),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=_new_uuid)
    name: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    task_domain: Mapped[str] = mapped_column(String(32), nullable=False)
    classes: Mapped[List[str]] = mapped_column(JSON, nullable=False, default=list)
    approved_for_training: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    provenance_source: Mapped[str] = mapped_column(String(255), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=_utcnow)

    samples: Mapped[List["DatasetSampleORM"]] = relationship(
        back_populates="dataset", cascade="all, delete-orphan"
    )


class DatasetSampleORM(Base):
    __tablename__ = "dataset_samples"
    __table_args__ = (
        CheckConstraint("split IN ('train', 'val')", name="ck_dataset_samples_split"),
        Index("idx_dataset_samples_dataset", "dataset_id", "shard_index"),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=_new_uuid)
    dataset_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("datasets.id", ondelete="CASCADE"), nullable=False
    )
    sample_code: Mapped[str] = mapped_column(String(64), nullable=False)
    label: Mapped[str] = mapped_column(String(64), nullable=False)
    task_domain: Mapped[str] = mapped_column(String(32), nullable=False)
    modality: Mapped[str] = mapped_column(String(32), nullable=False)
    feature_vector: Mapped[List[float]] = mapped_column(JSON, nullable=False, default=list)
    split: Mapped[str] = mapped_column(String(16), nullable=False)
    shard_index: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    in_replay_memory: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    source_name: Mapped[str] = mapped_column(String(128), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=_utcnow)

    dataset: Mapped["DatasetORM"] = relationship(back_populates="samples")


class ModelRegistryORM(Base):
    __tablename__ = "model_registry"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    model_type: Mapped[str] = mapped_column(String(64), nullable=False)
    active_version: Mapped[str] = mapped_column(String(64), nullable=False)
    architecture: Mapped[str] = mapped_column(Text, nullable=False)
    input_requirements: Mapped[str] = mapped_column(Text, nullable=False)
    output_requirements: Mapped[str] = mapped_column(Text, nullable=False)
    checkpoint_location: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    health_note: Mapped[str] = mapped_column(Text, nullable=False)
    last_eval_accuracy: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=_utcnow)

    versions: Mapped[List["ModelVersionORM"]] = relationship(
        back_populates="model", cascade="all, delete-orphan"
    )


class ModelVersionORM(Base):
    __tablename__ = "model_versions"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=_new_uuid)
    model_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("model_registry.id", ondelete="CASCADE"), nullable=False
    )
    version: Mapped[str] = mapped_column(String(64), nullable=False)
    parent_version: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    strategy: Mapped[str] = mapped_column(String(64), nullable=False)
    dataset_id: Mapped[Optional[str]] = mapped_column(
        String(64), ForeignKey("datasets.id", ondelete="SET NULL"), nullable=True
    )
    validation_accuracy: Mapped[float] = mapped_column(Float, nullable=False)
    knowledge_retention: Mapped[float] = mapped_column(Float, nullable=False)
    forgetting_score: Mapped[float] = mapped_column(Float, nullable=False)
    checkpoint_path: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=_utcnow)

    model: Mapped["ModelRegistryORM"] = relationship(back_populates="versions")


class TrainingJobORM(Base):
    __tablename__ = "training_jobs"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=_new_uuid)
    model_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("model_registry.id", ondelete="CASCADE"), nullable=False
    )
    dataset_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("datasets.id", ondelete="CASCADE"), nullable=False
    )
    task_name: Mapped[str] = mapped_column(String(255), nullable=False)
    strategy: Mapped[str] = mapped_column(String(64), nullable=False)
    config: Mapped[Dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    progress: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    previous_version: Mapped[str] = mapped_column(String(64), nullable=False)
    new_version: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    metrics: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)
    error: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=_utcnow)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)


class ContinualLearningHistoryORM(Base):
    __tablename__ = "continual_learning_history"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=_new_uuid)
    training_job_id: Mapped[Optional[str]] = mapped_column(
        String(64), ForeignKey("training_jobs.id", ondelete="CASCADE"), nullable=True
    )
    model_id: Mapped[str] = mapped_column(String(64), nullable=False)
    dataset_id: Mapped[str] = mapped_column(String(64), nullable=False)
    strategy: Mapped[str] = mapped_column(String(64), nullable=False)
    prev_task_acc_before: Mapped[float] = mapped_column(Float, nullable=False)
    prev_task_acc_after: Mapped[float] = mapped_column(Float, nullable=False)
    new_task_val_acc: Mapped[float] = mapped_column(Float, nullable=False)
    knowledge_retention: Mapped[float] = mapped_column(Float, nullable=False)
    forgetting_measure: Mapped[float] = mapped_column(Float, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=_utcnow)


class ReplayMemoryRecordORM(Base):
    __tablename__ = "replay_memory_records"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=_new_uuid)
    sample_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("dataset_samples.id", ondelete="CASCADE"), nullable=False
    )
    dataset_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("datasets.id", ondelete="CASCADE"), nullable=False
    )
    priority_weight: Mapped[float] = mapped_column(Float, nullable=False, default=1.0)
    added_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=_utcnow)


class UnlearningRequestORM(Base):
    __tablename__ = "unlearning_requests"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=_new_uuid)
    model_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("model_registry.id", ondelete="CASCADE"), nullable=False
    )
    dataset_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("datasets.id", ondelete="CASCADE"), nullable=False
    )
    strategy: Mapped[str] = mapped_column(String(64), nullable=False)
    sample_ids: Mapped[List[str]] = mapped_column(JSON, nullable=False, default=list)
    purge_raw_media: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    purge_cached_features: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    purge_replay_memory: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    original_version: Mapped[str] = mapped_column(String(64), nullable=False)
    updated_version: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=_utcnow)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

    audit: Mapped[Optional["UnlearningAuditORM"]] = relationship(
        back_populates="request", uselist=False, cascade="all, delete-orphan"
    )


class UnlearningAuditORM(Base):
    __tablename__ = "unlearning_audit"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=_new_uuid)
    request_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("unlearning_requests.id", ondelete="CASCADE"), nullable=False
    )
    forget_acc_before: Mapped[float] = mapped_column(Float, nullable=False)
    forget_acc_after: Mapped[float] = mapped_column(Float, nullable=False)
    retain_acc_before: Mapped[float] = mapped_column(Float, nullable=False)
    retain_acc_after: Mapped[float] = mapped_column(Float, nullable=False)
    overall_utility_after: Mapped[float] = mapped_column(Float, nullable=False)
    execution_time_ms: Mapped[int] = mapped_column(Integer, nullable=False)
    affected_shards: Mapped[List[int]] = mapped_column(JSON, nullable=False, default=list)
    validation_notes: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=_utcnow)

    request: Mapped["UnlearningRequestORM"] = relationship(back_populates="audit")


class ReportORM(Base):
    __tablename__ = "reports"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=_new_uuid)
    report_type: Mapped[str] = mapped_column(String(64), nullable=False)
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    reference_id: Mapped[str] = mapped_column(String(64), nullable=False)
    summary: Mapped[str] = mapped_column(Text, nullable=False)
    payload: Mapped[Dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=_utcnow)


class ActivityLogORM(Base):
    __tablename__ = "activity_logs"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=_new_uuid)
    category: Mapped[str] = mapped_column(String(64), nullable=False)
    action: Mapped[str] = mapped_column(String(128), nullable=False)
    detail: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=_utcnow)


class SystemSettingORM(Base):
    __tablename__ = "system_settings"

    key: Mapped[str] = mapped_column(String(128), primary_key=True)
    value: Mapped[str] = mapped_column(Text, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow, onupdate=_utcnow
    )


# ---------------------------------------------------------------------------
# Pydantic v2 API & Validation Schemas
# ---------------------------------------------------------------------------


class ORMBaseSchema(BaseModel):
    model_config = ConfigDict(from_attributes=True)


class UserCreate(BaseModel):
    email: str = Field(..., max_length=320)
    display_name: Optional[str] = Field(default=None, max_length=128)
    role: str = Field(default="analyst", max_length=32)


class UserRead(ORMBaseSchema):
    id: str
    email: str
    display_name: Optional[str]
    role: str
    created_at: datetime


class MediaUploadCreate(BaseModel):
    user_id: Optional[str] = None
    filename: str
    original_name: str
    mime_type: str
    size_bytes: int = Field(..., ge=0)
    media_type: Literal["video", "audio"]
    storage_path: str
    duration_sec: float = 0.0
    width: int = 0
    height: int = 0
    fps: float = 0.0
    video_codec: Optional[str] = None
    audio_codec: Optional[str] = None
    sample_rate: Optional[int] = None
    has_video: bool = False
    has_audio: bool = False
    is_sample: bool = False
    preview_url: str = ""


class MediaUploadRead(ORMBaseSchema):
    id: str
    user_id: Optional[str]
    filename: str
    original_name: str
    mime_type: str
    size_bytes: int
    media_type: Literal["video", "audio"]
    storage_path: str
    duration_sec: float
    width: int
    height: int
    fps: float
    video_codec: Optional[str]
    audio_codec: Optional[str]
    sample_rate: Optional[int]
    has_video: bool
    has_audio: bool
    is_sample: bool
    preview_url: str
    created_at: datetime


class AnalysisJobCreate(BaseModel):
    user_id: Optional[str] = None
    upload_id: str
    job_type: Literal["forensics", "emotion"]


class AnalysisJobRead(ORMBaseSchema):
    id: str
    user_id: Optional[str]
    upload_id: str
    job_type: Literal["forensics", "emotion"]
    status: Literal["queued", "processing", "completed", "failed"]
    progress: int
    stage: str
    error: Optional[str]
    result_id: Optional[str]
    created_at: datetime
    updated_at: datetime


class AnalysisResultRead(ORMBaseSchema):
    id: str
    job_id: str
    upload_id: str
    visual_authenticity_score: Optional[float]
    audio_authenticity_score: Optional[float]
    av_consistency_score: Optional[float]
    overall_authenticity_score: float
    overall_label: Literal["LIKELY AUTHENTIC", "POTENTIALLY MANIPULATED", "INCONCLUSIVE"]
    model_pipeline: str
    execution_backend: str
    payload: Dict[str, Any]
    report_id: Optional[str]
    created_at: datetime


class EmotionSegmentRead(ORMBaseSchema):
    id: str
    analysis_id: str
    upload_id: str
    speaker: str
    start_sec: float
    end_sec: float
    transcript: str
    audio_emotion: Optional[str]
    visual_emotion: Optional[str]
    text_emotion: Optional[str]
    combined_prediction: str
    combined_confidence: float
    modality_disagreement: bool
    disagreement_note: Optional[str]
    payload: Dict[str, Any]
    created_at: datetime


class DatasetRead(ORMBaseSchema):
    id: str
    name: str
    description: str
    task_domain: Literal["forensics", "emotion"]
    classes: List[str]
    approved_for_training: bool
    provenance_source: str
    created_at: datetime


class DatasetSampleRead(ORMBaseSchema):
    id: str
    dataset_id: str
    sample_code: str
    label: str
    task_domain: str
    modality: str
    feature_vector: List[float]
    split: Literal["train", "val"]
    shard_index: int
    in_replay_memory: bool
    source_name: str
    created_at: datetime


class ModelRegistryRead(ORMBaseSchema):
    id: str
    name: str
    model_type: str
    active_version: str
    architecture: str
    input_requirements: str
    output_requirements: str
    checkpoint_location: str
    status: str
    health_note: str
    last_eval_accuracy: Optional[float]
    created_at: datetime


class ModelVersionRead(ORMBaseSchema):
    id: str
    model_id: str
    version: str
    parent_version: Optional[str]
    strategy: str
    dataset_id: Optional[str]
    validation_accuracy: float
    knowledge_retention: float
    forgetting_score: float
    checkpoint_path: str
    created_at: datetime


class TrainingJobCreate(BaseModel):
    model_id: str
    dataset_id: str
    task_name: str
    strategy: Literal["sequential_finetuning", "ewc", "experience_replay", "ewc_replay"]
    config: Dict[str, Any] = Field(default_factory=dict)


class TrainingJobRead(ORMBaseSchema):
    id: str
    model_id: str
    dataset_id: str
    task_name: str
    strategy: str
    config: Dict[str, Any]
    status: str
    progress: int
    previous_version: str
    new_version: Optional[str]
    metrics: Optional[Dict[str, Any]]
    error: Optional[str]
    created_at: datetime
    completed_at: Optional[datetime]


class ContinualLearningHistoryRead(ORMBaseSchema):
    id: str
    training_job_id: Optional[str]
    model_id: str
    dataset_id: str
    strategy: str
    prev_task_acc_before: float
    prev_task_acc_after: float
    new_task_val_acc: float
    knowledge_retention: float
    forgetting_measure: float
    created_at: datetime


class ReplayMemoryRecordRead(ORMBaseSchema):
    id: str
    sample_id: str
    dataset_id: str
    priority_weight: float
    added_at: datetime


class UnlearningRequestCreate(BaseModel):
    model_id: str
    dataset_id: str
    strategy: Literal["exact_retrain", "sisa_sharded", "gradient_ascent"]
    sample_ids: List[str] = Field(..., min_length=1)
    purge_raw_media: bool = False
    purge_cached_features: bool = True
    purge_replay_memory: bool = True


class UnlearningRequestRead(ORMBaseSchema):
    id: str
    model_id: str
    dataset_id: str
    strategy: str
    sample_ids: List[str]
    purge_raw_media: bool
    purge_cached_features: bool
    purge_replay_memory: bool
    original_version: str
    updated_version: Optional[str]
    status: str
    created_at: datetime
    completed_at: Optional[datetime]


class UnlearningAuditRead(ORMBaseSchema):
    id: str
    request_id: str
    forget_acc_before: float
    forget_acc_after: float
    retain_acc_before: float
    retain_acc_after: float
    overall_utility_after: float
    execution_time_ms: int
    affected_shards: List[int]
    validation_notes: str
    created_at: datetime


class ReportRead(ORMBaseSchema):
    id: str
    report_type: str
    title: str
    reference_id: str
    summary: str
    payload: Dict[str, Any]
    created_at: datetime


class ActivityLogRead(ORMBaseSchema):
    id: str
    category: str
    action: str
    detail: str
    status: str
    created_at: datetime


class SystemSettingRead(ORMBaseSchema):
    key: str
    value: str
    updated_at: datetime
