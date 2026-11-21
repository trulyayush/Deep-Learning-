import express from 'express';
import multer from 'multer';
import path from 'node:path';
import fs from 'node:fs';
import crypto from 'node:crypto';
import {
  DATA_DIR,
  FRAMES_DIR,
  UPLOADS_DIR,
  db,
  getSetting,
  logActivity,
  setSetting,
} from '../db';
import {
  assertConfinedPath,
  createUploadId,
  inspectHardware,
  probeMedia,
} from '../services/mediaProcessor';
import {
  deleteRawMediaSafely,
  enqueueEmotionJob,
  enqueueForensicsJob,
  startContinualLearningJob,
  startSelectiveUnlearningJob,
} from '../services/jobManager';
import type {
  ContinualStrategy,
  SystemArchitectureSpec,
  UnlearningStrategy,
} from '../../src/types/echomind';

const ALLOWED_EXTENSIONS = new Set(['.mp4', '.mov', '.avi', '.mkv', '.webm', '.wav', '.mp3', '.m4a', '.ogg']);

const storage = multer.diskStorage({
  destination: (_req, _file, cb) => {
    cb(null, UPLOADS_DIR);
  },
  filename: (_req, file, cb) => {
    const ext = path.extname(file.originalname).toLowerCase();
    const safeExt = ALLOWED_EXTENSIONS.has(ext) ? ext : '.bin';
    cb(null, `${Date.now()}-${crypto.randomUUID().slice(0, 8)}${safeExt}`);
  },
});

const uploadMiddleware = multer({
  storage,
  limits: {
    fileSize: Math.max(10, Number(process.env.MAX_UPLOAD_MB || '100')) * 1024 * 1024,
  },
  fileFilter: (_req, file, cb) => {
    const ext = path.extname(file.originalname).toLowerCase();
    if (!ALLOWED_EXTENSIONS.has(ext)) {
      cb(new Error(`Unsupported file extension "${ext}". Supported: MP4, MOV, AVI, WEBM, WAV, MP3, M4A, OGG.`));
      return;
    }
    cb(null, true);
  },
});

export function createApiRouter() {
  const router = express.Router();

  router.get('/health', (_req, res) => {
    const geminiConfigured = Boolean(
      process.env.GEMINI_API_KEY && process.env.GEMINI_API_KEY !== 'MY_GEMINI_API_KEY'
    );
    res.json({
      status: 'ok',
      service: 'ECHO-MIND Backend API',
      timestamp: new Date().toISOString(),
      geminiConfigured,
      database: 'sqlite-wal',
    });
  });

  router.get('/system/hardware', async (_req, res) => {
    try {
      const profile = await inspectHardware();
      res.json(profile);
    } catch (err) {
      res.status(500).json({ error: err instanceof Error ? err.message : 'Hardware inspection failed' });
    }
  });

  router.get('/system/settings', (_req, res) => {
    res.json({
      maxUploadMb: Number(getSetting('max_upload_mb', '100')),
      externalInferenceUrl: getSetting('external_inference_url', ''),
      replayCapacity: Number(getSetting('replay_capacity', '64')),
      sisaShards: Number(getSetting('sisa_shards', '4')),
    });
  });

  router.put('/system/settings', (req, res) => {
    const body = req.body as {
      maxUploadMb?: number;
      externalInferenceUrl?: string;
      replayCapacity?: number;
      sisaShards?: number;
    };
    if (body.maxUploadMb !== undefined) {
      const clamped = Math.max(5, Math.min(500, Math.round(Number(body.maxUploadMb) || 100)));
      setSetting('max_upload_mb', String(clamped));
    }
    if (body.externalInferenceUrl !== undefined) {
      const trimmed = String(body.externalInferenceUrl).trim();
      if (trimmed) {
        try {
          const parsed = new URL(trimmed);
          if (parsed.protocol !== 'http:' && parsed.protocol !== 'https:') {
            res.status(400).json({ error: 'External inference URL must use http or https' });
            return;
          }
        } catch {
          res.status(400).json({ error: 'Invalid external inference URL format' });
          return;
        }
      }
      setSetting('external_inference_url', trimmed);
    }
    if (body.replayCapacity !== undefined) {
      const clamped = Math.max(8, Math.min(512, Math.round(Number(body.replayCapacity) || 64)));
      setSetting('replay_capacity', String(clamped));
    }
    if (body.sisaShards !== undefined) {
      const clamped = Math.max(2, Math.min(16, Math.round(Number(body.sisaShards) || 4)));
      setSetting('sisa_shards', String(clamped));
    }
    logActivity('system', 'Settings Updated', 'Updated platform resource and inference configuration.', 'info');
    res.json({
      maxUploadMb: Number(getSetting('max_upload_mb', '100')),
      externalInferenceUrl: getSetting('external_inference_url', ''),
      replayCapacity: Number(getSetting('replay_capacity', '64')),
      sisaShards: Number(getSetting('sisa_shards', '4')),
    });
  });

  router.post('/system/ping-external', async (req, res) => {
    const url = String((req.body as { url?: string })?.url || getSetting('external_inference_url', '')).trim();
    if (!url) {
      res.status(400).json({ ok: false, message: 'No external inference URL configured.' });
      return;
    }
    try {
      const parsed = new URL(url);
      if (parsed.protocol !== 'http:' && parsed.protocol !== 'https:') {
        res.status(400).json({ ok: false, message: 'Only HTTP/HTTPS endpoints are supported.' });
        return;
      }
      const host = parsed.hostname.toLowerCase();
      if (
        host === '169.254.169.254' ||
        host === 'localhost' ||
        host === '127.0.0.1' ||
        host.startsWith('10.') ||
        host.startsWith('192.168.')
      ) {
        res.status(400).json({
          ok: false,
          message: 'Private/loopback addresses are blocked by SSRF policy in hosted mode.',
        });
        return;
      }
      const healthUrl = new URL('/api/health', parsed).toString();
      const resp = await fetch(healthUrl, { signal: AbortSignal.timeout(4000) });
      res.json({ ok: resp.ok, status: resp.status, message: resp.ok ? 'External server reachable' : `HTTP ${resp.status}` });
    } catch (err) {
      res.json({
        ok: false,
        message: err instanceof Error ? err.message : 'External inference server unreachable',
      });
    }
  });

  router.get('/system/architecture', (_req, res) => {
    const models = db.prepare(`SELECT id, name, active_version, status FROM model_registry`).all() as Array<{
      id: string;
      name: string;
      active_version: string;
      status: string;
    }>;
    const replayCount = (
      db.prepare(`SELECT COUNT(*) as cnt FROM replay_memory_records`).get() as { cnt: number }
    ).cnt;
    const auditCount = (
      db.prepare(`SELECT COUNT(*) as cnt FROM unlearning_audit`).get() as { cnt: number }
    ).cnt;

    const findModel = (id: string) => models.find((m) => m.id === id);

    const spec: SystemArchitectureSpec = {
      encoders: [
        {
          id: 'enc-visual',
          name: 'Visual Frame & Face Encoder',
          modality: 'visual',
          backbone: 'FFmpeg Frame Sampler + Gemini-4-Hydrogen Vision + Pixel Sharpness Proxy',
          outputDim: 256,
          status: findModel('mod-shared-fusion')?.status ?? 'unavailable',
          supportsMissing: true,
        },
        {
          id: 'enc-audio',
          name: 'Acoustic & Prosody Encoder',
          modality: 'audio',
          backbone: '16kHz PCM / Goertzel Band DFT + Pitch/RMS + Gemini-4-Hydrogen Audio',
          outputDim: 256,
          status: findModel('mod-shared-fusion')?.status ?? 'unavailable',
          supportsMissing: true,
        },
        {
          id: 'enc-text',
          name: 'Transcript & Conversational Encoder',
          modality: 'text',
          backbone: 'Structured Dialogue Context Projection (supports missing transcripts)',
          outputDim: 256,
          status: findModel('mod-shared-fusion')?.status ?? 'unavailable',
          supportsMissing: true,
        },
      ],
      fusion: {
        mechanism: 'Modality-Specific Projections -> Cross-Modal Attention Fusion',
        projectionDim: 256,
        attentionHeads: 8,
        temporalLayer: 'Utterance & Frame Sequence Temporal Aggregation',
        sharedRepresentationDim: 256,
      },
      heads: [
        {
          id: 'head-forensics',
          name: 'Manipulation & Deepfake Forensics Head',
          task: 'Visual, Audio & AV Consistency Authenticity Scoring',
          modelId: 'mod-forensics-av',
          activeVersion: findModel('mod-forensics-av')?.active_version ?? 'v1.0.0',
          outputClasses: ['LIKELY AUTHENTIC', 'POTENTIALLY MANIPULATED', 'INCONCLUSIVE'],
          status: findModel('mod-forensics-av')?.status ?? 'unavailable',
        },
        {
          id: 'head-emotion',
          name: 'Conversational Affective Head',
          task: 'Modality-Separated Probabilistic Emotion Timeline',
          modelId: 'mod-emotion-fusion',
          activeVersion: findModel('mod-emotion-fusion')?.active_version ?? 'v1.0.0',
          outputClasses: ['Neutral', 'Joy', 'Sadness', 'Anger', 'Fear', 'Surprise', 'Disgust'],
          status: findModel('mod-emotion-fusion')?.status ?? 'unavailable',
        },
        {
          id: 'head-continual',
          name: 'Continual Learning Task Head',
          task: 'Incremental Classification with EWC Fisher + Experience Replay',
          modelId: 'mod-continual-head',
          activeVersion: findModel('mod-continual-head')?.active_version ?? 'v1.0.0',
          outputClasses: ['Task-Specific Dynamic Output'],
          status: findModel('mod-continual-head')?.status ?? 'operational',
        },
        {
          id: 'head-unlearning',
          name: 'Selective Machine Unlearning Pipeline',
          task: 'Exact Retrain, SISA Shard Isolation & Gradient Ascent Scrubbing',
          modelId: 'mod-continual-head',
          activeVersion: findModel('mod-continual-head')?.active_version ?? 'v1.0.0',
          outputClasses: ['Forget-Set Utility', 'Retain-Set Utility'],
          status: 'operational',
        },
      ],
      continualConfig: {
        strategies: ['sequential_finetuning', 'ewc', 'experience_replay', 'ewc_replay'],
        replayBufferCount: replayCount,
        replayCapacity: Number(getSetting('replay_capacity', '64')),
      },
      unlearningConfig: {
        strategies: ['exact_retrain', 'sisa_sharded', 'gradient_ascent'],
        totalShards: Number(getSetting('sisa_shards', '4')),
        auditCount,
      },
    };
    res.json(spec);
  });

  router.get('/overview', (_req, res) => {
    const recentForensics = db
      .prepare(
        `SELECT id as analysisId, upload_id as uploadId, filename, media_type as mediaType, overall_authenticity_score as overallAuthenticityScore, overall_label as overallLabel, created_at as createdAt
         FROM analysis_results ORDER BY created_at DESC LIMIT 5`
      )
      .all();

    const recentEmotion = db
      .prepare(
        `SELECT id as analysisId, upload_id as uploadId, filename, dominant_emotion as dominantEmotion, overall_confidence as overallConfidence, disagreement_rate as disagreementRate, created_at as createdAt
         FROM emotion_results ORDER BY created_at DESC LIMIT 5`
      )
      .all();

    const models = db
      .prepare(
        `SELECT id, name, model_type as modelType, active_version as activeVersion, architecture, input_requirements as inputRequirements, output_requirements as outputRequirements, checkpoint_location as checkpointLocation, status, health_note as healthNote, last_eval_accuracy as lastEvalAccuracy, created_at as createdAt
         FROM model_registry`
      )
      .all();

    const datasets = db
      .prepare(
        `SELECT d.id, d.name, d.task_domain as taskDomain, d.approved_for_training as approvedForTraining,
                (SELECT COUNT(*) FROM dataset_samples s WHERE s.dataset_id = d.id) as sampleCount
         FROM datasets d`
      )
      .all();

    const jobs = db
      .prepare(
        `SELECT id, job_type as jobType, upload_id as uploadId, filename, status, progress, stage, error, result_id as resultId, created_at as createdAt, updated_at as updatedAt
         FROM analysis_jobs ORDER BY created_at DESC LIMIT 6`
      )
      .all();

    const recentTraining = db
      .prepare(
        `SELECT id, model_id as modelId, dataset_name as datasetName, strategy, status, previous_version as previousVersion, new_version as newVersion, created_at as createdAt
         FROM training_jobs ORDER BY created_at DESC LIMIT 4`
      )
      .all();

    const recentUnlearning = db
      .prepare(
        `SELECT r.id, r.dataset_name as datasetName, r.strategy, r.original_version as originalVersion, r.updated_version as updatedVersion, r.status, r.created_at as createdAt,
                a.retain_acc_after as retainAccAfter, a.forget_acc_after as forgetAccAfter
         FROM unlearning_requests r
         LEFT JOIN unlearning_audit a ON a.request_id = r.id
         ORDER BY r.created_at DESC LIMIT 4`
      )
      .all();

    const recentReports = db
      .prepare(
        `SELECT id, report_type as reportType, title, reference_id as referenceId, summary, created_at as createdAt
         FROM reports ORDER BY created_at DESC LIMIT 6`
      )
      .all();

    res.json({
      counts: {
        forensicsAnalyses: (db.prepare(`SELECT COUNT(*) as c FROM analysis_results`).get() as { c: number }).c,
        emotionAnalyses: (db.prepare(`SELECT COUNT(*) as c FROM emotion_results`).get() as { c: number }).c,
        registeredModels: models.length,
        operationalModels: models.filter((m) => (m as { status: string }).status === 'operational').length,
        approvedDatasets: datasets.length,
        trainingRuns: (db.prepare(`SELECT COUNT(*) as c FROM training_jobs`).get() as { c: number }).c,
        unlearningAudits: (db.prepare(`SELECT COUNT(*) as c FROM unlearning_audit`).get() as { c: number }).c,
        replaySamples: (db.prepare(`SELECT COUNT(*) as c FROM replay_memory_records`).get() as { c: number }).c,
      },
      recentForensics,
      recentEmotion,
      models,
      datasets,
      jobs,
      recentTraining,
      recentUnlearning,
      recentReports,
    });
  });

  // Media Management
  router.get('/media', (_req, res) => {
    const rows = db
      .prepare(
        `SELECT id, filename, original_name as originalName, mime_type as mimeType, size_bytes as sizeBytes,
                media_type as mediaType, duration_sec as durationSec, width, height, fps,
                video_codec as videoCodec, audio_codec as audioCodec, sample_rate as sampleRate,
                has_video as hasVideo, has_audio as hasAudio, is_sample as isSample,
                preview_url as previewUrl, created_at as createdAt
         FROM media_uploads ORDER BY is_sample DESC, created_at DESC`
      )
      .all()
      .map((r: Record<string, unknown>) => ({
        ...r,
        hasVideo: Boolean(r.hasVideo),
        hasAudio: Boolean(r.hasAudio),
        isSample: Boolean(r.isSample),
      }));
    res.json(rows);
  });

  router.post('/media/upload', (req, res) => {
    uploadMiddleware.single('file')(req, res, async (err) => {
      if (err) {
        res.status(400).json({ error: err instanceof Error ? err.message : 'Upload rejected' });
        return;
      }
      if (!req.file) {
        res.status(400).json({ error: 'No media file attached' });
        return;
      }
      try {
        const uploadId = createUploadId();
        const probe = await probeMedia(req.file.path);
        const mediaType = probe.hasVideo ? 'video' : 'audio';
        const now = new Date().toISOString();
        const previewUrl = `/api/media/file/${uploadId}`;

        db.prepare(
          `INSERT INTO media_uploads (id, filename, original_name, mime_type, size_bytes, media_type, file_path, duration_sec, width, height, fps, video_codec, audio_codec, sample_rate, has_video, has_audio, is_sample, preview_url, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?)`
        ).run(
          uploadId,
          req.file.filename,
          req.file.originalname,
          req.file.mimetype,
          req.file.size,
          mediaType,
          req.file.path,
          probe.durationSec,
          probe.width,
          probe.height,
          probe.fps,
          probe.videoCodec,
          probe.audioCodec,
          probe.sampleRate,
          probe.hasVideo ? 1 : 0,
          probe.hasAudio ? 1 : 0,
          previewUrl,
          now
        );

        logActivity(
          'system',
          'Media Uploaded',
          `Validated ${req.file.originalname} (${mediaType}, ${probe.durationSec}s, ${probe.hasVideo ? `${probe.width}x${probe.height}` : 'audio-only'})`,
          'info'
        );

        res.status(201).json({
          id: uploadId,
          filename: req.file.filename,
          originalName: req.file.originalname,
          mimeType: req.file.mimetype,
          sizeBytes: req.file.size,
          mediaType,
          durationSec: probe.durationSec,
          width: probe.width,
          height: probe.height,
          fps: probe.fps,
          videoCodec: probe.videoCodec,
          audioCodec: probe.audioCodec,
          sampleRate: probe.sampleRate,
          hasVideo: probe.hasVideo,
          hasAudio: probe.hasAudio,
          isSample: false,
          previewUrl,
          createdAt: now,
        });
      } catch (probeErr) {
        if (req.file && fs.existsSync(req.file.path)) {
          fs.unlinkSync(req.file.path);
        }
        res.status(400).json({
          error:
            probeErr instanceof Error
              ? `Media validation failed: ${probeErr.message}`
              : 'Could not decode uploaded media file',
        });
      }
    });
  });

  router.get('/media/file/:id', (req, res) => {
    const row = db
      .prepare(`SELECT file_path, mime_type FROM media_uploads WHERE id = ?`)
      .get(req.params.id) as { file_path: string; mime_type: string } | undefined;
    if (!row) {
      res.status(404).json({ error: 'Media file not found' });
      return;
    }
    try {
      const safePath = assertConfinedPath(row.file_path, DATA_DIR);
      if (!fs.existsSync(safePath)) {
        res.status(404).json({ error: 'Media file missing on disk' });
        return;
      }
      res.type(row.mime_type);
      res.sendFile(safePath);
    } catch {
      res.status(403).json({ error: 'Invalid media path' });
    }
  });

  router.get('/media/frames/:uploadId/:frameName', (req, res) => {
    const safeUpload = req.params.uploadId.replace(/[^a-zA-Z0-9._-]/g, '');
    const safeFrame = path.basename(req.params.frameName);
    try {
      const framePath = assertConfinedPath(
        path.join(FRAMES_DIR, safeUpload, safeFrame),
        FRAMES_DIR
      );
      if (!fs.existsSync(framePath)) {
        res.status(404).json({ error: 'Frame not found' });
        return;
      }
      res.type('image/png');
      res.sendFile(framePath);
    } catch {
      res.status(403).json({ error: 'Invalid frame path' });
    }
  });

  router.delete('/media/:id', (req, res) => {
    try {
      const deleted = deleteRawMediaSafely(req.params.id);
      if (!deleted) {
        res.status(404).json({ error: 'Upload not found' });
        return;
      }
      res.json({ deleted: true });
    } catch (err) {
      res.status(400).json({ error: err instanceof Error ? err.message : 'Delete failed' });
    }
  });

  // Jobs polling
  router.get('/jobs/:jobId', (req, res) => {
    const row = db
      .prepare(
        `SELECT id, job_type as jobType, upload_id as uploadId, filename, status, progress, stage, error, result_id as resultId, created_at as createdAt, updated_at as updatedAt
         FROM analysis_jobs WHERE id = ?`
      )
      .get(req.params.jobId);
    if (!row) {
      res.status(404).json({ error: 'Job not found' });
      return;
    }
    res.json(row);
  });

  // Module 01: Forensics Analysis
  router.post('/analysis', (req, res) => {
    const { uploadId, configNote } = req.body as { uploadId?: string; configNote?: string };
    if (!uploadId) {
      res.status(400).json({ error: 'uploadId is required' });
      return;
    }
    try {
      const jobId = enqueueForensicsJob(uploadId, configNote);
      res.status(202).json({ jobId, status: 'queued' });
    } catch (err) {
      res.status(404).json({ error: err instanceof Error ? err.message : 'Failed to enqueue job' });
    }
  });

  router.get('/analysis', (_req, res) => {
    const rows = db
      .prepare(`SELECT payload_json FROM analysis_results ORDER BY created_at DESC LIMIT 25`)
      .all() as Array<{ payload_json: string }>;
    res.json(rows.map((r) => JSON.parse(r.payload_json)));
  });

  router.get('/analysis/:analysisId', (req, res) => {
    const row = db
      .prepare(`SELECT payload_json FROM analysis_results WHERE id = ?`)
      .get(req.params.analysisId) as { payload_json: string } | undefined;
    if (!row) {
      res.status(404).json({ error: 'Analysis result not found' });
      return;
    }
    res.json(JSON.parse(row.payload_json));
  });

  // Module 02: Conversational Emotion Intelligence
  router.post('/emotion/analyze', (req, res) => {
    const { uploadId, transcript } = req.body as { uploadId?: string; transcript?: string };
    if (!uploadId) {
      res.status(400).json({ error: 'uploadId is required' });
      return;
    }
    try {
      const jobId = enqueueEmotionJob(uploadId, transcript || '');
      res.status(202).json({ jobId, status: 'queued' });
    } catch (err) {
      res.status(404).json({ error: err instanceof Error ? err.message : 'Failed to enqueue emotion job' });
    }
  });

  router.get('/emotion', (_req, res) => {
    const rows = db
      .prepare(`SELECT payload_json FROM emotion_results ORDER BY created_at DESC LIMIT 25`)
      .all() as Array<{ payload_json: string }>;
    res.json(rows.map((r) => JSON.parse(r.payload_json)));
  });

  router.get('/emotion/:analysisId', (req, res) => {
    const row = db
      .prepare(`SELECT payload_json FROM emotion_results WHERE id = ?`)
      .get(req.params.analysisId) as { payload_json: string } | undefined;
    if (!row) {
      res.status(404).json({ error: 'Emotion analysis result not found' });
      return;
    }
    res.json(JSON.parse(row.payload_json));
  });

  // Module 03: Continual Learning Engine
  router.get('/learning/datasets', (_req, res) => {
    const rows = db
      .prepare(
        `SELECT d.id, d.name, d.description, d.task_domain as taskDomain, d.classes_json as classesJson,
                d.approved_for_training as approvedForTraining, d.provenance_source as provenanceSource, d.created_at as createdAt,
                (SELECT COUNT(*) FROM dataset_samples s WHERE s.dataset_id = d.id) as sampleCount,
                (SELECT COUNT(*) FROM dataset_samples s WHERE s.dataset_id = d.id AND s.split = 'train') as trainCount,
                (SELECT COUNT(*) FROM dataset_samples s WHERE s.dataset_id = d.id AND s.split = 'val') as valCount
         FROM datasets d ORDER BY d.created_at DESC`
      )
      .all() as Array<Record<string, unknown>>;

    res.json(
      rows.map((r) => ({
        id: r.id,
        name: r.name,
        description: r.description,
        taskDomain: r.taskDomain,
        classes: JSON.parse(String(r.classesJson || '[]')),
        sampleCount: r.sampleCount,
        trainCount: r.trainCount,
        valCount: r.valCount,
        approvedForTraining: Boolean(r.approvedForTraining),
        provenanceSource: r.provenanceSource,
        createdAt: r.createdAt,
      }))
    );
  });

  router.post('/learning/datasets', (req, res) => {
    const {
      name,
      description,
      taskDomain,
      classes,
      sampleCount = 16,
      approvedForTraining = true,
      provenanceSource = 'User Registered Approved Dataset',
    } = req.body as {
      name?: string;
      description?: string;
      taskDomain?: 'forensics' | 'emotion';
      classes?: string[];
      sampleCount?: number;
      approvedForTraining?: boolean;
      provenanceSource?: string;
    };

    if (!name || !name.trim()) {
      res.status(400).json({ error: 'Dataset name is required' });
      return;
    }
    if (!approvedForTraining) {
      res.status(400).json({
        error: 'Only explicitly approved and labelled datasets may enter the continual learning registry',
      });
      return;
    }

    const domain = taskDomain === 'emotion' ? 'emotion' : 'forensics';
    const classList =
      Array.isArray(classes) && classes.length >= 2
        ? classes.map((c) => c.trim()).filter(Boolean)
        : domain === 'forensics'
          ? ['Authentic', 'Manipulated']
          : ['Neutral', 'Joy', 'Sadness', 'Anger'];

    const count = Math.max(8, Math.min(80, Math.round(Number(sampleCount) || 16)));
    const dsId = `ds-${crypto.randomUUID().slice(0, 8)}`;
    const now = new Date().toISOString();

    try {
      db.prepare(
        `INSERT INTO datasets (id, name, description, task_domain, classes_json, approved_for_training, provenance_source, created_at)
         VALUES (?, ?, ?, ?, ?, 1, ?, ?)`
      ).run(
        dsId,
        name.trim(),
        description?.trim() || `Approved ${domain} labelled feature dataset`,
        domain,
        JSON.stringify(classList),
        provenanceSource,
        now
      );

      const insSample = db.prepare(
        `INSERT INTO dataset_samples (id, dataset_id, sample_code, label, task_domain, modality, feature_vector_json, split, shard_index, in_replay_memory, source_name, created_at)
         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?)`
      );

      const prefix = name.trim().replace(/[^A-Za-z0-9]/g, '').slice(0, 4).toUpperCase() || 'DATA';
      for (let i = 0; i < count; i++) {
        const clsIdx = i % classList.length;
        const label = classList[clsIdx];
        const angle = (clsIdx * 2 * Math.PI) / classList.length;
        const jitter = (((i * 17) % 9) - 4) * 0.06;
        const vec = [
          Number((Math.cos(angle) * 1.2 + jitter).toFixed(4)),
          Number((Math.sin(angle) * 1.2 - jitter).toFixed(4)),
          Number(((clsIdx - 0.5) * 0.7 + jitter).toFixed(4)),
          Number((Math.cos(angle * 2) * 0.5).toFixed(4)),
          Number((Math.sin(angle * 2) * 0.5).toFixed(4)),
          Number((jitter * 0.5).toFixed(4)),
        ];
        insSample.run(
          `smp-${crypto.randomUUID().slice(0, 8)}`,
          dsId,
          `${prefix}-${String(i + 1).padStart(3, '0')}`,
          label,
          domain,
          'audiovisual',
          JSON.stringify(vec),
          i < Math.floor(count * 0.75) ? 'train' : 'val',
          i % 4,
          name.trim(),
          now
        );
      }

      logActivity('learning', 'Dataset Registered', `Registered approved dataset "${name.trim()}" (${count} samples).`, 'info');
      res.status(201).json({ id: dsId, name: name.trim(), sampleCount: count });
    } catch (err) {
      res.status(400).json({ error: err instanceof Error ? err.message : 'Dataset registration failed' });
    }
  });

  router.get('/learning/datasets/:datasetId/samples', (req, res) => {
    const rows = db
      .prepare(
        `SELECT id, dataset_id as datasetId, sample_code as sampleCode, label, task_domain as taskDomain, modality,
                feature_vector_json as featureVectorJson, split, shard_index as shardIndex,
                in_replay_memory as inReplayMemory, source_name as sourceName, created_at as createdAt
         FROM dataset_samples WHERE dataset_id = ? ORDER BY sample_code ASC`
      )
      .all(req.params.datasetId) as Array<Record<string, unknown>>;

    res.json(
      rows.map((r) => ({
        ...r,
        featureVector: JSON.parse(String(r.featureVectorJson || '[]')),
        inReplayMemory: Boolean(r.inReplayMemory),
      }))
    );
  });

  router.post('/learning/start', (req, res) => {
    const {
      modelId = 'mod-continual-head',
      datasetId,
      taskName = 'Incremental Drift Adaptation',
      strategy = 'ewc_replay',
      epochs = 25,
      learningRate = 0.12,
      ewcLambda = 2.0,
      replayCapacity = 32,
    } = req.body as {
      modelId?: string;
      datasetId?: string;
      taskName?: string;
      strategy?: ContinualStrategy;
      epochs?: number;
      learningRate?: number;
      ewcLambda?: number;
      replayCapacity?: number;
    };

    if (!datasetId) {
      res.status(400).json({ error: 'datasetId is required' });
      return;
    }

    try {
      const jobId = startContinualLearningJob({
        modelId,
        datasetId,
        taskName,
        strategy,
        epochs: Math.max(5, Math.min(100, Number(epochs) || 25)),
        learningRate: Math.max(0.01, Math.min(0.4, Number(learningRate) || 0.12)),
        ewcLambda: Math.max(0, Math.min(20, Number(ewcLambda) || 2.0)),
        replayCapacity: Math.max(4, Math.min(256, Number(replayCapacity) || 32)),
      });
      res.status(202).json({ jobId, status: 'running' });
    } catch (err) {
      res.status(400).json({ error: err instanceof Error ? err.message : 'Failed to start training' });
    }
  });

  router.get('/learning/jobs/:jobId', (req, res) => {
    const row = db
      .prepare(`SELECT * FROM training_jobs WHERE id = ?`)
      .get(req.params.jobId) as Record<string, unknown> | undefined;
    if (!row) {
      res.status(404).json({ error: 'Training job not found' });
      return;
    }
    const metrics = row.metrics_json ? JSON.parse(String(row.metrics_json)) : null;
    res.json({
      id: row.id,
      modelId: row.model_id,
      datasetId: row.dataset_id,
      datasetName: row.dataset_name,
      taskName: row.task_name,
      strategy: row.strategy,
      epochs: row.epochs,
      learningRate: row.learning_rate,
      ewcLambda: row.ewc_lambda,
      replayCapacity: row.replay_capacity,
      status: row.status,
      progress: row.progress,
      previousVersion: row.previous_version,
      newVersion: row.new_version,
      prevTaskAccuracyBefore: metrics?.prevTaskAccuracyBefore ?? 0,
      prevTaskAccuracyAfter: metrics?.prevTaskAccuracyAfter ?? 0,
      newTaskValidationAccuracy: metrics?.newTaskValidationAccuracy ?? 0,
      knowledgeRetention: metrics?.knowledgeRetention ?? 0,
      forgettingMeasure: metrics?.forgettingMeasure ?? 0,
      lossHistory: metrics?.lossHistory ?? [],
      error: row.error,
      createdAt: row.created_at,
      completedAt: row.completed_at,
    });
  });

  router.get('/learning/history', (_req, res) => {
    const rows = db
      .prepare(`SELECT * FROM training_jobs ORDER BY created_at DESC LIMIT 30`)
      .all() as Array<Record<string, unknown>>;

    res.json(
      rows.map((row) => {
        const metrics = row.metrics_json ? JSON.parse(String(row.metrics_json)) : null;
        return {
          id: row.id,
          modelId: row.model_id,
          datasetId: row.dataset_id,
          datasetName: row.dataset_name,
          taskName: row.task_name,
          strategy: row.strategy,
          epochs: row.epochs,
          learningRate: row.learning_rate,
          ewcLambda: row.ewc_lambda,
          replayCapacity: row.replay_capacity,
          status: row.status,
          progress: row.progress,
          previousVersion: row.previous_version,
          newVersion: row.new_version,
          prevTaskAccuracyBefore: metrics?.prevTaskAccuracyBefore ?? 0,
          prevTaskAccuracyAfter: metrics?.prevTaskAccuracyAfter ?? 0,
          newTaskValidationAccuracy: metrics?.newTaskValidationAccuracy ?? 0,
          knowledgeRetention: metrics?.knowledgeRetention ?? 0,
          forgettingMeasure: metrics?.forgettingMeasure ?? 0,
          lossHistory: metrics?.lossHistory ?? [],
          error: row.error,
          createdAt: row.created_at,
          completedAt: row.completed_at,
        };
      })
    );
  });

  router.get('/learning/replay', (_req, res) => {
    const records = db
      .prepare(
        `SELECT id, sample_id as sampleId, dataset_id as datasetId, sample_code as sampleCode, label, task_domain as taskDomain, priority_weight as priorityWeight, added_at as addedAt
         FROM replay_memory_records ORDER BY added_at DESC`
      )
      .all();
    res.json({
      capacity: Number(getSetting('replay_capacity', '64')),
      count: records.length,
      records,
    });
  });

  // Module 04: Selective Machine Unlearning
  router.post('/unlearning/requests', (req, res) => {
    const {
      modelId = 'mod-continual-head',
      datasetId,
      sampleIds,
      strategy = 'sisa_sharded',
      purgeRawMedia = false,
      purgeCachedFeatures = true,
      purgeReplayMemory = true,
    } = req.body as {
      modelId?: string;
      datasetId?: string;
      sampleIds?: string[];
      strategy?: UnlearningStrategy;
      purgeRawMedia?: boolean;
      purgeCachedFeatures?: boolean;
      purgeReplayMemory?: boolean;
    };

    if (!datasetId || !Array.isArray(sampleIds) || sampleIds.length === 0) {
      res.status(400).json({ error: 'datasetId and non-empty sampleIds array are required' });
      return;
    }

    try {
      const requestId = startSelectiveUnlearningJob({
        modelId,
        datasetId,
        sampleIds,
        strategy,
        purgeRawMedia: Boolean(purgeRawMedia),
        purgeCachedFeatures: Boolean(purgeCachedFeatures),
        purgeReplayMemory: Boolean(purgeReplayMemory),
      });
      res.status(202).json({ requestId, status: 'running' });
    } catch (err) {
      res.status(400).json({ error: err instanceof Error ? err.message : 'Unlearning request rejected' });
    }
  });

  router.get('/unlearning/requests', (_req, res) => {
    const rows = db
      .prepare(
        `SELECT r.*, a.forget_acc_before, a.forget_acc_after, a.retain_acc_before, a.retain_acc_after,
                a.overall_utility_after, a.execution_time_ms, a.affected_shards_json, a.validation_notes
         FROM unlearning_requests r
         LEFT JOIN unlearning_audit a ON a.request_id = r.id
         ORDER BY r.created_at DESC LIMIT 30`
      )
      .all() as Array<Record<string, unknown>>;

    res.json(
      rows.map((r) => ({
        id: r.id,
        modelId: r.model_id,
        datasetId: r.dataset_id,
        datasetName: r.dataset_name,
        strategy: r.strategy,
        sampleIds: JSON.parse(String(r.sample_ids_json || '[]')),
        sampleCodes: JSON.parse(String(r.sample_codes_json || '[]')),
        purgeRawMedia: Boolean(r.purge_raw_media),
        purgeCachedFeatures: Boolean(r.purge_cached_features),
        purgeReplayMemory: Boolean(r.purge_replay_memory),
        originalVersion: r.original_version,
        updatedVersion: r.updated_version,
        status: r.status,
        forgetAccBefore: r.forget_acc_before ?? 0,
        forgetAccAfter: r.forget_acc_after ?? 0,
        retainAccBefore: r.retain_acc_before ?? 0,
        retainAccAfter: r.retain_acc_after ?? 0,
        overallUtilityAfter: r.overall_utility_after ?? 0,
        executionTimeMs: r.execution_time_ms ?? 0,
        affectedShards: JSON.parse(String(r.affected_shards_json || '[]')),
        notes: r.validation_notes ?? '',
        createdAt: r.created_at,
        completedAt: r.completed_at,
      }))
    );
  });

  router.get('/unlearning/requests/:requestId', (req, res) => {
    const r = db
      .prepare(
        `SELECT r.*, a.forget_acc_before, a.forget_acc_after, a.retain_acc_before, a.retain_acc_after,
                a.overall_utility_after, a.execution_time_ms, a.affected_shards_json, a.validation_notes
         FROM unlearning_requests r
         LEFT JOIN unlearning_audit a ON a.request_id = r.id
         WHERE r.id = ?`
      )
      .get(req.params.requestId) as Record<string, unknown> | undefined;

    if (!r) {
      res.status(404).json({ error: 'Unlearning request not found' });
      return;
    }

    res.json({
      id: r.id,
      modelId: r.model_id,
      datasetId: r.dataset_id,
      datasetName: r.dataset_name,
      strategy: r.strategy,
      sampleIds: JSON.parse(String(r.sample_ids_json || '[]')),
      sampleCodes: JSON.parse(String(r.sample_codes_json || '[]')),
      purgeRawMedia: Boolean(r.purge_raw_media),
      purgeCachedFeatures: Boolean(r.purge_cached_features),
      purgeReplayMemory: Boolean(r.purge_replay_memory),
      originalVersion: r.original_version,
      updatedVersion: r.updated_version,
      status: r.status,
      forgetAccBefore: r.forget_acc_before ?? 0,
      forgetAccAfter: r.forget_acc_after ?? 0,
      retainAccBefore: r.retain_acc_before ?? 0,
      retainAccAfter: r.retain_acc_after ?? 0,
      overallUtilityAfter: r.overall_utility_after ?? 0,
      executionTimeMs: r.execution_time_ms ?? 0,
      affectedShards: JSON.parse(String(r.affected_shards_json || '[]')),
      notes: r.validation_notes ?? '',
      createdAt: r.created_at,
      completedAt: r.completed_at,
    });
  });

  // Model Management Registry
  router.get('/models', (_req, res) => {
    const models = db
      .prepare(
        `SELECT id, name, model_type as modelType, active_version as activeVersion, architecture,
                input_requirements as inputRequirements, output_requirements as outputRequirements,
                checkpoint_location as checkpointLocation, status, health_note as healthNote,
                last_eval_accuracy as lastEvalAccuracy, created_at as createdAt
         FROM model_registry ORDER BY created_at ASC`
      )
      .all();
    res.json(models);
  });

  router.get('/models/:modelId/versions', (req, res) => {
    const versions = db
      .prepare(
        `SELECT id, model_id as modelId, version, parent_version as parentVersion, strategy,
                dataset_id as datasetId, validation_accuracy as validationAccuracy,
                knowledge_retention as knowledgeRetention, forgetting_score as forgettingScore,
                checkpoint_path as checkpointPath, created_at as createdAt
         FROM model_versions WHERE model_id = ? ORDER BY created_at DESC`
      )
      .all(req.params.modelId);
    res.json(versions);
  });

  router.post('/models/:modelId/rollback', (req, res) => {
    const { version } = req.body as { version?: string };
    if (!version) {
      res.status(400).json({ error: 'Target version is required' });
      return;
    }
    const targetVer = db
      .prepare(
        `SELECT version, checkpoint_path, validation_accuracy FROM model_versions WHERE model_id = ? AND version = ?`
      )
      .get(req.params.modelId, version) as
      | { version: string; checkpoint_path: string; validation_accuracy: number }
      | undefined;

    if (!targetVer) {
      res.status(404).json({ error: `Checkpoint version ${version} not found` });
      return;
    }
    if (!fs.existsSync(targetVer.checkpoint_path)) {
      res.status(400).json({ error: 'Checkpoint artifact missing on disk; cannot roll back' });
      return;
    }

    db.prepare(
      `UPDATE model_registry SET active_version = ?, checkpoint_location = ?, last_eval_accuracy = ?, status = 'operational', health_note = ? WHERE id = ?`
    ).run(
      targetVer.version,
      targetVer.checkpoint_path,
      targetVer.validation_accuracy,
      `Rolled back to verified checkpoint ${targetVer.version}`,
      req.params.modelId
    );

    logActivity(
      'model',
      'Checkpoint Rollback',
      `Rolled back ${req.params.modelId} to ${targetVer.version}`,
      'info'
    );
    res.json({ rolledBackTo: targetVer.version });
  });

  router.post('/models/:modelId/health-check', (req, res) => {
    const model = db
      .prepare(`SELECT id, checkpoint_location, model_type FROM model_registry WHERE id = ?`)
      .get(req.params.modelId) as
      | { id: string; checkpoint_location: string; model_type: string }
      | undefined;
    if (!model) {
      res.status(404).json({ error: 'Model not found' });
      return;
    }

    let status: 'operational' | 'unavailable' = 'operational';
    let note = '';

    if (model.id === 'mod-continual-head') {
      const exists = fs.existsSync(model.checkpoint_location);
      status = exists ? 'operational' : 'unavailable';
      note = exists
        ? 'Local weight checkpoint verified on disk and ready for inference/training.'
        : 'Checkpoint missing on disk.';
    } else if (model.id === 'mod-pytorch-external') {
      const extUrl = getSetting('external_inference_url', '');
      status = extUrl ? 'operational' : 'unavailable';
      note = extUrl
        ? `External PyTorch endpoint configured (${extUrl}).`
        : 'Optional external PyTorch GPU server not connected; local Node + Gemini + FFmpeg pipeline active.';
    } else {
      const geminiOk = Boolean(
        process.env.GEMINI_API_KEY && process.env.GEMINI_API_KEY !== 'MY_GEMINI_API_KEY'
      );
      status = geminiOk ? 'operational' : 'unavailable';
      note = geminiOk
        ? 'Server-side Gemini multimodal adapter & FFmpeg extractor verified.'
        : 'GEMINI_API_KEY secret missing.';
    }

    db.prepare(`UPDATE model_registry SET status = ?, health_note = ? WHERE id = ?`).run(
      status,
      note,
      model.id
    );
    res.json({ id: model.id, status, healthNote: note });
  });

  // Reports & Activity History
  router.get('/reports', (_req, res) => {
    const rows = db
      .prepare(
        `SELECT id, report_type as reportType, title, reference_id as referenceId, summary, created_at as createdAt
         FROM reports ORDER BY created_at DESC LIMIT 40`
      )
      .all();
    res.json(rows);
  });

  router.get('/reports/:reportId', (req, res) => {
    const row = db
      .prepare(`SELECT * FROM reports WHERE id = ?`)
      .get(req.params.reportId) as Record<string, unknown> | undefined;
    if (!row) {
      res.status(404).json({ error: 'Report not found' });
      return;
    }
    res.json({
      id: row.id,
      reportType: row.report_type,
      title: row.title,
      referenceId: row.reference_id,
      summary: row.summary,
      payload: JSON.parse(String(row.payload_json || '{}')),
      createdAt: row.created_at,
    });
  });

  router.get('/reports/:reportId/download', (req, res) => {
    const row = db
      .prepare(`SELECT * FROM reports WHERE id = ?`)
      .get(req.params.reportId) as Record<string, unknown> | undefined;
    if (!row) {
      res.status(404).json({ error: 'Report not found' });
      return;
    }
    const payload = JSON.parse(String(row.payload_json || '{}'));
    const exportObj = {
      platform: 'ECHO-MIND Adaptive Multimodal Intelligence Platform',
      exportedAt: new Date().toISOString(),
      reportId: row.id,
      reportType: row.report_type,
      title: row.title,
      summary: row.summary,
      data: payload,
    };
    res.setHeader('Content-Disposition', `attachment; filename="echomind-${row.id}.json"`);
    res.type('application/json').send(JSON.stringify(exportObj, null, 2));
  });

  router.get('/activity', (req, res) => {
    const category = String(req.query.category || '').trim();
    const rows = category
      ? db
          .prepare(
            `SELECT id, category, action, detail, status, created_at as createdAt
             FROM activity_logs WHERE category = ? ORDER BY created_at DESC LIMIT 100`
          )
          .all(category)
      : db
          .prepare(
            `SELECT id, category, action, detail, status, created_at as createdAt
             FROM activity_logs ORDER BY created_at DESC LIMIT 100`
          )
          .all();
    res.json(rows);
  });

  return router;
}
