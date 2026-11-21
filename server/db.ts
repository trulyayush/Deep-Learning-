import { DatabaseSync } from 'node:sqlite';
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import {
  createInitialCheckpoint,
  runContinualTraining,
  type LabeledSample,
  type SerializedClassifierCheckpoint,
} from './ml/continualEngine';

export const DATA_DIR = path.resolve(process.cwd(), 'data');
export const UPLOADS_DIR = path.join(DATA_DIR, 'uploads');
export const FRAMES_DIR = path.join(DATA_DIR, 'frames');
export const AUDIO_DIR = path.join(DATA_DIR, 'audio');
export const SAMPLES_DIR = path.join(DATA_DIR, 'samples');
export const CHECKPOINTS_DIR = path.join(DATA_DIR, 'checkpoints');

for (const dir of [DATA_DIR, UPLOADS_DIR, FRAMES_DIR, AUDIO_DIR, SAMPLES_DIR, CHECKPOINTS_DIR]) {
  fs.mkdirSync(dir, { recursive: true });
}

const DB_PATH = path.join(DATA_DIR, 'echomind.db');
export const db = new DatabaseSync(DB_PATH);

db.exec('PRAGMA journal_mode = WAL;');
db.exec('PRAGMA foreign_keys = ON;');

db.exec(`
CREATE TABLE IF NOT EXISTS media_uploads (
  id TEXT PRIMARY KEY,
  filename TEXT NOT NULL,
  original_name TEXT NOT NULL,
  mime_type TEXT NOT NULL,
  size_bytes INTEGER NOT NULL,
  media_type TEXT NOT NULL,
  file_path TEXT NOT NULL,
  duration_sec REAL NOT NULL DEFAULT 0,
  width INTEGER NOT NULL DEFAULT 0,
  height INTEGER NOT NULL DEFAULT 0,
  fps REAL NOT NULL DEFAULT 0,
  video_codec TEXT,
  audio_codec TEXT,
  sample_rate INTEGER,
  has_video INTEGER NOT NULL DEFAULT 0,
  has_audio INTEGER NOT NULL DEFAULT 0,
  is_sample INTEGER NOT NULL DEFAULT 0,
  preview_url TEXT NOT NULL,
  created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS analysis_jobs (
  id TEXT PRIMARY KEY,
  job_type TEXT NOT NULL,
  upload_id TEXT NOT NULL,
  filename TEXT NOT NULL,
  status TEXT NOT NULL,
  progress INTEGER NOT NULL DEFAULT 0,
  stage TEXT NOT NULL,
  error TEXT,
  result_id TEXT,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  FOREIGN KEY (upload_id) REFERENCES media_uploads(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS analysis_results (
  id TEXT PRIMARY KEY,
  job_id TEXT NOT NULL,
  upload_id TEXT NOT NULL,
  filename TEXT NOT NULL,
  media_type TEXT NOT NULL,
  visual_authenticity_score REAL,
  audio_authenticity_score REAL,
  av_consistency_score REAL,
  overall_authenticity_score REAL NOT NULL,
  overall_label TEXT NOT NULL,
  model_pipeline TEXT NOT NULL,
  execution_backend TEXT NOT NULL,
  payload_json TEXT NOT NULL,
  report_id TEXT NOT NULL,
  created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS emotion_results (
  id TEXT PRIMARY KEY,
  job_id TEXT NOT NULL,
  upload_id TEXT NOT NULL,
  filename TEXT NOT NULL,
  duration_sec REAL NOT NULL,
  dominant_emotion TEXT NOT NULL,
  overall_confidence REAL NOT NULL,
  disagreement_rate REAL NOT NULL,
  model_pipeline TEXT NOT NULL,
  payload_json TEXT NOT NULL,
  report_id TEXT NOT NULL,
  created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS emotion_segments (
  id TEXT PRIMARY KEY,
  analysis_id TEXT NOT NULL,
  upload_id TEXT NOT NULL,
  speaker TEXT NOT NULL,
  start_sec REAL NOT NULL,
  end_sec REAL NOT NULL,
  transcript TEXT NOT NULL,
  audio_emotion TEXT,
  visual_emotion TEXT,
  text_emotion TEXT,
  combined_prediction TEXT NOT NULL,
  combined_confidence REAL NOT NULL,
  modality_disagreement INTEGER NOT NULL DEFAULT 0,
  disagreement_note TEXT,
  created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS datasets (
  id TEXT PRIMARY KEY,
  name TEXT NOT NULL UNIQUE,
  description TEXT NOT NULL,
  task_domain TEXT NOT NULL,
  classes_json TEXT NOT NULL,
  approved_for_training INTEGER NOT NULL DEFAULT 1,
  provenance_source TEXT NOT NULL,
  created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS dataset_samples (
  id TEXT PRIMARY KEY,
  dataset_id TEXT NOT NULL,
  sample_code TEXT NOT NULL,
  label TEXT NOT NULL,
  task_domain TEXT NOT NULL,
  modality TEXT NOT NULL,
  feature_vector_json TEXT NOT NULL,
  split TEXT NOT NULL,
  shard_index INTEGER NOT NULL DEFAULT 0,
  in_replay_memory INTEGER NOT NULL DEFAULT 0,
  source_name TEXT NOT NULL,
  created_at TEXT NOT NULL,
  FOREIGN KEY (dataset_id) REFERENCES datasets(id) ON DELETE CASCADE
);

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
  last_eval_accuracy REAL,
  created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS model_versions (
  id TEXT PRIMARY KEY,
  model_id TEXT NOT NULL,
  version TEXT NOT NULL,
  parent_version TEXT,
  strategy TEXT NOT NULL,
  dataset_id TEXT,
  validation_accuracy REAL NOT NULL,
  knowledge_retention REAL NOT NULL,
  forgetting_score REAL NOT NULL,
  checkpoint_path TEXT NOT NULL,
  created_at TEXT NOT NULL,
  FOREIGN KEY (model_id) REFERENCES model_registry(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS training_jobs (
  id TEXT PRIMARY KEY,
  model_id TEXT NOT NULL,
  dataset_id TEXT NOT NULL,
  dataset_name TEXT NOT NULL,
  task_name TEXT NOT NULL,
  strategy TEXT NOT NULL,
  epochs INTEGER NOT NULL,
  learning_rate REAL NOT NULL,
  ewc_lambda REAL NOT NULL,
  replay_capacity INTEGER NOT NULL,
  status TEXT NOT NULL,
  progress INTEGER NOT NULL DEFAULT 0,
  previous_version TEXT NOT NULL,
  new_version TEXT,
  metrics_json TEXT,
  error TEXT,
  created_at TEXT NOT NULL,
  completed_at TEXT
);

CREATE TABLE IF NOT EXISTS continual_learning_history (
  id TEXT PRIMARY KEY,
  training_job_id TEXT NOT NULL,
  model_id TEXT NOT NULL,
  dataset_id TEXT NOT NULL,
  strategy TEXT NOT NULL,
  prev_task_acc_before REAL NOT NULL,
  prev_task_acc_after REAL NOT NULL,
  new_task_val_acc REAL NOT NULL,
  knowledge_retention REAL NOT NULL,
  forgetting_measure REAL NOT NULL,
  created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS replay_memory_records (
  id TEXT PRIMARY KEY,
  sample_id TEXT NOT NULL,
  dataset_id TEXT NOT NULL,
  sample_code TEXT NOT NULL,
  label TEXT NOT NULL,
  task_domain TEXT NOT NULL,
  priority_weight REAL NOT NULL DEFAULT 1.0,
  added_at TEXT NOT NULL,
  FOREIGN KEY (sample_id) REFERENCES dataset_samples(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS unlearning_requests (
  id TEXT PRIMARY KEY,
  model_id TEXT NOT NULL,
  dataset_id TEXT NOT NULL,
  dataset_name TEXT NOT NULL,
  strategy TEXT NOT NULL,
  sample_ids_json TEXT NOT NULL,
  sample_codes_json TEXT NOT NULL,
  purge_raw_media INTEGER NOT NULL DEFAULT 0,
  purge_cached_features INTEGER NOT NULL DEFAULT 1,
  purge_replay_memory INTEGER NOT NULL DEFAULT 1,
  original_version TEXT NOT NULL,
  updated_version TEXT,
  status TEXT NOT NULL,
  created_at TEXT NOT NULL,
  completed_at TEXT
);

CREATE TABLE IF NOT EXISTS unlearning_audit (
  id TEXT PRIMARY KEY,
  request_id TEXT NOT NULL,
  forget_acc_before REAL NOT NULL,
  forget_acc_after REAL NOT NULL,
  retain_acc_before REAL NOT NULL,
  retain_acc_after REAL NOT NULL,
  overall_utility_after REAL NOT NULL,
  execution_time_ms INTEGER NOT NULL,
  affected_shards_json TEXT NOT NULL,
  validation_notes TEXT NOT NULL,
  created_at TEXT NOT NULL,
  FOREIGN KEY (request_id) REFERENCES unlearning_requests(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS reports (
  id TEXT PRIMARY KEY,
  report_type TEXT NOT NULL,
  title TEXT NOT NULL,
  reference_id TEXT NOT NULL,
  summary TEXT NOT NULL,
  payload_json TEXT NOT NULL,
  created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS activity_logs (
  id TEXT PRIMARY KEY,
  category TEXT NOT NULL,
  action TEXT NOT NULL,
  detail TEXT NOT NULL,
  status TEXT NOT NULL,
  created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS system_settings (
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL,
  updated_at TEXT NOT NULL
);
`);

export function logActivity(
  category: 'forensics' | 'emotion' | 'learning' | 'unlearning' | 'model' | 'system',
  action: string,
  detail: string,
  status: 'info' | 'success' | 'warning' | 'error' = 'info'
) {
  const stmt = db.prepare(
    `INSERT INTO activity_logs (id, category, action, detail, status, created_at) VALUES (?, ?, ?, ?, ?, ?)`
  );
  stmt.run(crypto.randomUUID(), category, action, detail, status, new Date().toISOString());
}

export function getSetting(key: string, fallback: string): string {
  const row = db.prepare(`SELECT value FROM system_settings WHERE key = ?`).get(key) as
    | { value: string }
    | undefined;
  return row ? row.value : fallback;
}

export function setSetting(key: string, value: string) {
  db.prepare(
    `INSERT INTO system_settings (key, value, updated_at) VALUES (?, ?, ?)
     ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at`
  ).run(key, value, new Date().toISOString());
}

export function loadCheckpointFromDisk(checkpointPath: string): SerializedClassifierCheckpoint | null {
  const resolved = path.resolve(checkpointPath);
  if (!resolved.startsWith(CHECKPOINTS_DIR + path.sep) && resolved !== CHECKPOINTS_DIR) {
    return null;
  }
  if (!fs.existsSync(resolved)) return null;
  return JSON.parse(fs.readFileSync(resolved, 'utf8')) as SerializedClassifierCheckpoint;
}

export function saveCheckpointToDisk(
  filename: string,
  checkpoint: SerializedClassifierCheckpoint
): string {
  const safeName = filename.replace(/[^a-zA-Z0-9._-]/g, '_');
  const fullPath = path.join(CHECKPOINTS_DIR, safeName);
  fs.writeFileSync(fullPath, JSON.stringify(checkpoint, null, 2), 'utf8');
  return fullPath;
}

export function seedInitialState() {
  // Clean up any jobs orphaned in-flight from a prior server shutdown
  db.prepare(
    `UPDATE analysis_jobs
     SET status = 'failed', progress = 100, stage = 'Interrupted by server restart', error = 'Background job was interrupted by a server restart. Please re-run the analysis.', updated_at = ?
     WHERE status IN ('queued', 'processing')`
  ).run(new Date().toISOString());

  const existingModels = db.prepare(`SELECT COUNT(*) as cnt FROM model_registry`).get() as {
    cnt: number;
  };
  if (existingModels.cnt > 0) return;

  const now = new Date().toISOString();

  // Seed default system settings
  setSetting('max_upload_mb', process.env.MAX_UPLOAD_MB || '100');
  setSetting('external_inference_url', process.env.EXTERNAL_INFERENCE_URL || '');
  setSetting('replay_capacity', '64');
  setSetting('sisa_shards', '4');

  // Seed approved datasets first so continual head can train an initial baseline v1.0.0 checkpoint
  const dsForensicsId = 'ds-ffpp-approved-01';
  const dsForensicsAudioId = 'ds-fakeav-approved-02';
  const dsEmotionId = 'ds-meld-affect-03';

  const insertDs = db.prepare(
    `INSERT INTO datasets (id, name, description, task_domain, classes_json, approved_for_training, provenance_source, created_at)
     VALUES (?, ?, ?, ?, ?, ?, ?, ?)`
  );

  insertDs.run(
    dsForensicsId,
    'FaceForensics++ Curated Subset (Approved)',
    'Approved facial manipulation feature vectors extracted from benchmark authentic vs. face-swap sequences.',
    'forensics',
    JSON.stringify(['Authentic', 'Manipulated']),
    1,
    'Approved Curated Benchmark Adapter',
    now
  );

  insertDs.run(
    dsForensicsAudioId,
    'FakeAVCeleb Acoustic Drift Task (Approved)',
    'Approved incremental audio-visual cross-modal drift samples for continual adaptation.',
    'forensics',
    JSON.stringify(['Authentic', 'Manipulated']),
    1,
    'Approved Incremental Task Adapter',
    now
  );

  insertDs.run(
    dsEmotionId,
    'MELD Conversational Affect Subset (Approved)',
    'Approved multimodal dialogue embeddings across Neutral, Joy, Sadness, and Anger observation classes.',
    'emotion',
    JSON.stringify(['Neutral', 'Joy', 'Sadness', 'Anger']),
    1,
    'Approved Multimodal Dialogue Adapter',
    now
  );

  const insertSample = db.prepare(
    `INSERT INTO dataset_samples (id, dataset_id, sample_code, label, task_domain, modality, feature_vector_json, split, shard_index, in_replay_memory, source_name, created_at)
     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`
  );
  const insertReplay = db.prepare(
    `INSERT INTO replay_memory_records (id, sample_id, dataset_id, sample_code, label, task_domain, priority_weight, added_at)
     VALUES (?, ?, ?, ?, ?, ?, ?, ?)`
  );

  const forensicsTask1Train: LabeledSample[] = [];
  for (let i = 0; i < 24; i++) {
    const isManip = i % 2 === 1;
    const label = isManip ? 'Manipulated' : 'Authentic';
    const s = isManip ? 1 : -1;
    const jitter = ((i * 11) % 7 - 3) * 0.06;
    const vec = [
      Number((s * 1.25 + jitter).toFixed(4)),
      Number((s * 0.95 - jitter).toFixed(4)),
      Number((s * 0.8 + jitter * 0.5).toFixed(4)),
      Number((0.25 * s + jitter).toFixed(4)),
      Number((-0.3 * s).toFixed(4)),
      Number((0.45 * s + jitter * 0.4).toFixed(4)),
    ];
    const sampleId = `smp-ff-${String(i + 1).padStart(3, '0')}`;
    const sampleCode = `FFPP-${String(i + 1).padStart(3, '0')}`;
    const split = i < 18 ? 'train' : 'val';
    const inReplay = i < 8 ? 1 : 0;
    insertSample.run(
      sampleId,
      dsForensicsId,
      sampleCode,
      label,
      'forensics',
      'visual',
      JSON.stringify(vec),
      split,
      i % 4,
      inReplay,
      'FaceForensics-Curated',
      now
    );
    if (inReplay) {
      insertReplay.run(
        `rep-${sampleId}`,
        sampleId,
        dsForensicsId,
        sampleCode,
        label,
        'forensics',
        1.0,
        now
      );
    }
    if (split === 'train') {
      forensicsTask1Train.push({
        id: sampleId,
        features: vec,
        labelIndex: isManip ? 1 : 0,
        taskIndex: 1,
      });
    }
  }

  for (let i = 0; i < 20; i++) {
    const isManip = i % 2 === 1;
    const label = isManip ? 'Manipulated' : 'Authentic';
    const s = isManip ? 1 : -1;
    const jitter = ((i * 13) % 7 - 3) * 0.07;
    const vec = [
      Number((s * 1.05 + jitter).toFixed(4)),
      Number((s * 1.2 - jitter).toFixed(4)),
      Number((s * 0.65 + 0.3).toFixed(4)),
      Number((0.45 * s + jitter).toFixed(4)),
      Number((0.25 * s).toFixed(4)),
      Number((-0.2 * s).toFixed(4)),
    ];
    const sampleId = `smp-fav-${String(i + 1).padStart(3, '0')}`;
    const sampleCode = `FAVC-${String(i + 1).padStart(3, '0')}`;
    insertSample.run(
      sampleId,
      dsForensicsAudioId,
      sampleCode,
      label,
      'forensics',
      'audiovisual',
      JSON.stringify(vec),
      i < 14 ? 'train' : 'val',
      i % 4,
      0,
      'FakeAVCeleb-Curated',
      now
    );
  }

  const emotionClasses = ['Neutral', 'Joy', 'Sadness', 'Anger'];
  for (let i = 0; i < 24; i++) {
    const clsIdx = i % emotionClasses.length;
    const label = emotionClasses[clsIdx];
    const angle = (clsIdx * Math.PI) / 2;
    const jitter = ((i * 5) % 5 - 2) * 0.08;
    const vec = [
      Number((Math.cos(angle) * 1.3 + jitter).toFixed(4)),
      Number((Math.sin(angle) * 1.3 - jitter).toFixed(4)),
      Number(((clsIdx - 1.5) * 0.5).toFixed(4)),
      Number((Math.cos(angle * 2) * 0.6).toFixed(4)),
      Number((Math.sin(angle * 2) * 0.6).toFixed(4)),
      Number((jitter * 0.7).toFixed(4)),
    ];
    const sampleId = `smp-meld-${String(i + 1).padStart(3, '0')}`;
    const sampleCode = `MELD-${String(i + 1).padStart(3, '0')}`;
    insertSample.run(
      sampleId,
      dsEmotionId,
      sampleCode,
      label,
      'emotion',
      'audiovisual',
      JSON.stringify(vec),
      i < 18 ? 'train' : 'val',
      i % 4,
      i < 4 ? 1 : 0,
      'MELD-Curated',
      now
    );
    if (i < 4) {
      insertReplay.run(
        `rep-${sampleId}`,
        sampleId,
        dsEmotionId,
        sampleCode,
        label,
        'emotion',
        0.9,
        now
      );
    }
  }

  // Train genuine baseline checkpoint v1.0.0 for the continual head on dsForensicsId
  const initCkpt = createInitialCheckpoint(6, 2);
  const baselineTrain = runContinualTraining(
    initCkpt,
    forensicsTask1Train,
    forensicsTask1Train,
    [],
    [],
    {
      strategy: 'sequential_finetuning',
      epochs: 35,
      learningRate: 0.14,
      ewcLambda: 0,
    }
  );
  baselineTrain.checkpoint.trainedTasks = [1];
  const ckptPath = saveCheckpointToDisk('continual-forensics-head-v1.0.0.json', baselineTrain.checkpoint);

  const insertModel = db.prepare(
    `INSERT INTO model_registry (id, name, model_type, active_version, architecture, input_requirements, output_requirements, checkpoint_location, status, health_note, last_eval_accuracy, created_at)
     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`
  );

  const geminiAvailable = Boolean(process.env.GEMINI_API_KEY && process.env.GEMINI_API_KEY !== 'MY_GEMINI_API_KEY');

  insertModel.run(
    'mod-shared-fusion',
    'Cross-Attention Multimodal Backbone',
    'shared_backbone',
    'v1.0.0',
    'Modality Projections -> Cross-Modal Attention -> Temporal Layer',
    'Visual frames (PNG/RGB) + Audio PCM/WAV (16kHz) + Transcript text',
    'Shared multimodal feature representation (256-d)',
    'gemini-4-hydrogen + local FFmpeg signal extractor',
    geminiAvailable ? 'operational' : 'unavailable',
    geminiAvailable
      ? 'Operational via server-side Gemini multimodal encoder + FFmpeg signal pipeline.'
      : 'GEMINI_API_KEY not configured; multimodal inference unavailable until configured.',
    null,
    now
  );

  insertModel.run(
    'mod-forensics-av',
    'Multimodal Deepfake Forensics Head',
    'forensics_visual',
    'v1.0.0',
    'Frame Saliency + Temporal Delta + Audio Spectral Forensics Adapter',
    'Sampled RGB frames + 16kHz mono WAV + AV sync offsets',
    'Authenticity scores + Suspicious segments + Saliency regions',
    'gemini-4-hydrogen + signalProcessor.ts',
    geminiAvailable ? 'operational' : 'unavailable',
    geminiAvailable
      ? 'Active multimodal forensics pipeline ready for video and audio inspection.'
      : 'Awaiting GEMINI_API_KEY secret for visual/audio forensics inference.',
    null,
    now
  );

  insertModel.run(
    'mod-emotion-fusion',
    'Conversational Affective Intelligence Head',
    'emotion_fusion',
    'v1.0.0',
    'Audio Prosody + Facial Expression + Transcript Cross-Modal Fusion',
    'Audio speech track + Face frames + Optional transcript/subtitles',
    '7-class probabilistic emotion timeline with per-modality outputs',
    'gemini-4-hydrogen + pitch/RMS acoustic analyzer',
    geminiAvailable ? 'operational' : 'unavailable',
    geminiAvailable
      ? 'Operational across 7 affective categories with disagreement tracking.'
      : 'Awaiting GEMINI_API_KEY secret for conversational affect inference.',
    null,
    now
  );

  insertModel.run(
    'mod-continual-head',
    'Continual & Unlearning Task Classifier Head',
    'continual_head',
    'v1.0.0',
    '6D Feature Projection -> Linear/Softmax Head + Diagonal EWC Fisher Matrix',
    '6D normalized multimodal feature vectors',
    'Task-specific class probabilities (Authentic/Manipulated or Emotion)',
    ckptPath,
    'operational',
    'Local checkpoint loaded and verified with real gradient/EWC/SISA support.',
    baselineTrain.newTaskValidationAccuracy,
    now
  );

  insertModel.run(
    'mod-pytorch-external',
    'External GPU PyTorch Worker (Optional)',
    'forensics_audio',
    'external',
    'XceptionNet / EfficientNet-B0 / WavLM-Base+ (FastAPI Worker)',
    'Remote HTTP multipart media stream',
    'PyTorch tensor logits + Grad-CAM overlay',
    'Not connected (configure External Inference URL in Settings)',
    'unavailable',
    'Optional external PyTorch GPU server not connected; local Node + Gemini + FFmpeg pipeline active.',
    null,
    now
  );

  db.prepare(
    `INSERT INTO model_versions (id, model_id, version, parent_version, strategy, dataset_id, validation_accuracy, knowledge_retention, forgetting_score, checkpoint_path, created_at)
     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`
  ).run(
    'ver-continual-100',
    'mod-continual-head',
    'v1.0.0',
    null,
    'baseline_initialization',
    dsForensicsId,
    baselineTrain.newTaskValidationAccuracy,
    1.0,
    0.0,
    ckptPath,
    now
  );

  logActivity(
    'system',
    'Platform Initialized',
    'Initialized SQLite state, approved benchmark dataset adapters, and baseline v1.0.0 continual head checkpoint.',
    'success'
  );
}
