-- ECHO-MIND PostgreSQL / Supabase Production Schema
-- Supports Row-Level Security (RLS) for private user media and audit trails.

CREATE EXTENSION IF NOT EXISTS "pgcrypto";

CREATE TABLE IF NOT EXISTS users (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  email TEXT UNIQUE NOT NULL,
  display_name TEXT,
  role TEXT NOT NULL DEFAULT 'analyst',
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS media_uploads (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID REFERENCES users(id) ON DELETE SET NULL,
  filename TEXT NOT NULL,
  original_name TEXT NOT NULL,
  mime_type TEXT NOT NULL,
  size_bytes BIGINT NOT NULL,
  media_type TEXT NOT NULL CHECK (media_type IN ('video', 'audio')),
  storage_path TEXT NOT NULL,
  duration_sec DOUBLE PRECISION NOT NULL DEFAULT 0,
  width INTEGER NOT NULL DEFAULT 0,
  height INTEGER NOT NULL DEFAULT 0,
  fps DOUBLE PRECISION NOT NULL DEFAULT 0,
  video_codec TEXT,
  audio_codec TEXT,
  sample_rate INTEGER,
  has_video BOOLEAN NOT NULL DEFAULT FALSE,
  has_audio BOOLEAN NOT NULL DEFAULT FALSE,
  is_sample BOOLEAN NOT NULL DEFAULT FALSE,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_media_uploads_user ON media_uploads(user_id, created_at DESC);

CREATE TABLE IF NOT EXISTS analysis_jobs (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID REFERENCES users(id) ON DELETE SET NULL,
  upload_id UUID NOT NULL REFERENCES media_uploads(id) ON DELETE CASCADE,
  job_type TEXT NOT NULL CHECK (job_type IN ('forensics', 'emotion')),
  status TEXT NOT NULL CHECK (status IN ('queued', 'processing', 'completed', 'failed')),
  progress INTEGER NOT NULL DEFAULT 0,
  stage TEXT NOT NULL DEFAULT 'queued',
  error TEXT,
  result_id UUID,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_analysis_jobs_status ON analysis_jobs(status, created_at DESC);

CREATE TABLE IF NOT EXISTS analysis_results (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  job_id UUID NOT NULL REFERENCES analysis_jobs(id) ON DELETE CASCADE,
  upload_id UUID NOT NULL REFERENCES media_uploads(id) ON DELETE CASCADE,
  visual_authenticity_score DOUBLE PRECISION,
  audio_authenticity_score DOUBLE PRECISION,
  av_consistency_score DOUBLE PRECISION,
  overall_authenticity_score DOUBLE PRECISION NOT NULL,
  overall_label TEXT NOT NULL CHECK (overall_label IN ('LIKELY AUTHENTIC', 'POTENTIALLY MANIPULATED', 'INCONCLUSIVE')),
  model_pipeline TEXT NOT NULL,
  execution_backend TEXT NOT NULL,
  payload JSONB NOT NULL,
  report_id UUID,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS emotion_segments (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  analysis_id UUID NOT NULL,
  upload_id UUID NOT NULL REFERENCES media_uploads(id) ON DELETE CASCADE,
  speaker TEXT NOT NULL,
  start_sec DOUBLE PRECISION NOT NULL,
  end_sec DOUBLE PRECISION NOT NULL,
  transcript TEXT NOT NULL,
  audio_emotion TEXT,
  visual_emotion TEXT,
  text_emotion TEXT,
  combined_prediction TEXT NOT NULL,
  combined_confidence DOUBLE PRECISION NOT NULL,
  modality_disagreement BOOLEAN NOT NULL DEFAULT FALSE,
  disagreement_note TEXT,
  payload JSONB NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS datasets (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  name TEXT UNIQUE NOT NULL,
  description TEXT NOT NULL,
  task_domain TEXT NOT NULL CHECK (task_domain IN ('forensics', 'emotion')),
  classes JSONB NOT NULL,
  approved_for_training BOOLEAN NOT NULL DEFAULT FALSE,
  provenance_source TEXT NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS dataset_samples (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  dataset_id UUID NOT NULL REFERENCES datasets(id) ON DELETE CASCADE,
  sample_code TEXT NOT NULL,
  label TEXT NOT NULL,
  task_domain TEXT NOT NULL,
  modality TEXT NOT NULL,
  feature_vector JSONB NOT NULL,
  split TEXT NOT NULL CHECK (split IN ('train', 'val')),
  shard_index INTEGER NOT NULL DEFAULT 0,
  in_replay_memory BOOLEAN NOT NULL DEFAULT FALSE,
  source_name TEXT NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_dataset_samples_dataset ON dataset_samples(dataset_id, shard_index);

CREATE TABLE IF NOT EXISTS model_registry (
  id TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  model_type TEXT NOT NULL,
  active_version TEXT NOT NULL,
  architecture TEXT NOT NULL,
  input_requirements TEXT NOT NULL,
  output_requirements TEXT NOT NULL,
  checkpoint_location TEXT NOT NULL,
  status TEXT NOT NULL,
  health_note TEXT NOT NULL,
  last_eval_accuracy DOUBLE PRECISION,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS model_versions (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  model_id TEXT NOT NULL REFERENCES model_registry(id) ON DELETE CASCADE,
  version TEXT NOT NULL,
  parent_version TEXT,
  strategy TEXT NOT NULL,
  dataset_id UUID REFERENCES datasets(id) ON DELETE SET NULL,
  validation_accuracy DOUBLE PRECISION NOT NULL,
  knowledge_retention DOUBLE PRECISION NOT NULL,
  forgetting_score DOUBLE PRECISION NOT NULL,
  checkpoint_path TEXT NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS training_jobs (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  model_id TEXT NOT NULL REFERENCES model_registry(id) ON DELETE CASCADE,
  dataset_id UUID NOT NULL REFERENCES datasets(id) ON DELETE CASCADE,
  task_name TEXT NOT NULL,
  strategy TEXT NOT NULL,
  config JSONB NOT NULL,
  status TEXT NOT NULL,
  progress INTEGER NOT NULL DEFAULT 0,
  previous_version TEXT NOT NULL,
  new_version TEXT,
  metrics JSONB,
  error TEXT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  completed_at TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS continual_learning_history (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  training_job_id UUID REFERENCES training_jobs(id) ON DELETE CASCADE,
  model_id TEXT NOT NULL,
  dataset_id UUID NOT NULL,
  strategy TEXT NOT NULL,
  prev_task_acc_before DOUBLE PRECISION NOT NULL,
  prev_task_acc_after DOUBLE PRECISION NOT NULL,
  new_task_val_acc DOUBLE PRECISION NOT NULL,
  knowledge_retention DOUBLE PRECISION NOT NULL,
  forgetting_measure DOUBLE PRECISION NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS replay_memory_records (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  sample_id UUID NOT NULL REFERENCES dataset_samples(id) ON DELETE CASCADE,
  dataset_id UUID NOT NULL REFERENCES datasets(id) ON DELETE CASCADE,
  priority_weight DOUBLE PRECISION NOT NULL DEFAULT 1.0,
  added_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS unlearning_requests (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  model_id TEXT NOT NULL REFERENCES model_registry(id) ON DELETE CASCADE,
  dataset_id UUID NOT NULL REFERENCES datasets(id) ON DELETE CASCADE,
  strategy TEXT NOT NULL,
  sample_ids JSONB NOT NULL,
  purge_raw_media BOOLEAN NOT NULL DEFAULT FALSE,
  purge_cached_features BOOLEAN NOT NULL DEFAULT TRUE,
  purge_replay_memory BOOLEAN NOT NULL DEFAULT TRUE,
  original_version TEXT NOT NULL,
  updated_version TEXT,
  status TEXT NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  completed_at TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS unlearning_audit (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  request_id UUID NOT NULL REFERENCES unlearning_requests(id) ON DELETE CASCADE,
  forget_acc_before DOUBLE PRECISION NOT NULL,
  forget_acc_after DOUBLE PRECISION NOT NULL,
  retain_acc_before DOUBLE PRECISION NOT NULL,
  retain_acc_after DOUBLE PRECISION NOT NULL,
  overall_utility_after DOUBLE PRECISION NOT NULL,
  execution_time_ms INTEGER NOT NULL,
  affected_shards JSONB NOT NULL,
  validation_notes TEXT NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS reports (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  report_type TEXT NOT NULL,
  title TEXT NOT NULL,
  reference_id TEXT NOT NULL,
  summary TEXT NOT NULL,
  payload JSONB NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS activity_logs (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  category TEXT NOT NULL,
  action TEXT NOT NULL,
  detail TEXT NOT NULL,
  status TEXT NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Supabase Row Level Security (private user media isolation)
ALTER TABLE media_uploads ENABLE ROW LEVEL SECURITY;
ALTER TABLE analysis_jobs ENABLE ROW LEVEL SECURITY;
ALTER TABLE analysis_results ENABLE ROW LEVEL SECURITY;

CREATE POLICY media_owner_policy ON media_uploads
  FOR ALL USING (auth.uid() = user_id OR is_sample = TRUE);

CREATE POLICY analysis_jobs_owner_policy ON analysis_jobs
  FOR ALL USING (auth.uid() = user_id);
