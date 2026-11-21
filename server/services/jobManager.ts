import crypto from 'node:crypto';
import fs from 'node:fs';
import {
  db,
  getSetting,
  loadCheckpointFromDisk,
  logActivity,
  saveCheckpointToDisk,
  setSetting,
} from '../db';
import { extractMediaArtifacts } from './mediaProcessor';
import {
  buildExtractedFrames,
  runGeminiEmotionInference,
  runGeminiForensicsInference,
} from './geminiInference';
import {
  createInitialCheckpoint,
  runContinualTraining,
  selectReplayBuffer,
  type LabeledSample,
} from '../ml/continualEngine';
import { executeSelectiveUnlearning, type ShardSample } from '../ml/unlearningEngine';
import type {
  ContinualStrategy,
  EmotionCategory,
  EmotionResult,
  ForensicsResult,
  UnlearningStrategy,
} from '../../src/types/echomind';

function bumpSemver(version: string): string {
  const clean = version.replace(/^v/, '');
  const parts = clean.split('.').map((n) => parseInt(n, 10) || 0);
  const [maj = 1, min = 0, pat = 0] = parts;
  return `v${maj}.${min + 1}.${pat}`;
}

export function enqueueForensicsJob(uploadId: string, configNote?: string): string {
  const upload = db
    .prepare(`SELECT id, original_name, file_path, media_type FROM media_uploads WHERE id = ?`)
    .get(uploadId) as
    | { id: string; original_name: string; file_path: string; media_type: 'video' | 'audio' }
    | undefined;

  if (!upload) {
    throw new Error('Media upload not found');
  }

  const jobId = `job-for-${crypto.randomUUID().slice(0, 8)}`;
  const now = new Date().toISOString();

  db.prepare(
    `INSERT INTO analysis_jobs (id, job_type, upload_id, filename, status, progress, stage, error, result_id, created_at, updated_at)
     VALUES (?, 'forensics', ?, ?, 'queued', 5, 'Queued for media extraction', NULL, NULL, ?, ?)`
  ).run(jobId, upload.id, upload.original_name, now, now);

  setImmediate(() => {
    void processForensicsJob(jobId, upload, configNote);
  });

  return jobId;
}

async function processForensicsJob(
  jobId: string,
  upload: { id: string; original_name: string; file_path: string; media_type: 'video' | 'audio' },
  configNote?: string
) {
  const t0 = performance.now();
  const updateStage = (status: string, progress: number, stage: string, error: string | null = null, resultId: string | null = null) => {
    db.prepare(
      `UPDATE analysis_jobs SET status = ?, progress = ?, stage = ?, error = ?, result_id = ?, updated_at = ? WHERE id = ?`
    ).run(status, progress, stage, error, resultId, new Date().toISOString(), jobId);
  };

  try {
    updateStage('processing', 20, 'Decoding media streams & extracting frames/PCM via FFmpeg');
    const artifacts = await extractMediaArtifacts(upload.id, upload.file_path);

    updateStage('processing', 55, 'Running multimodal visual, acoustic & AV consistency inference');
    const inference = await runGeminiForensicsInference(upload.original_name, artifacts);

    updateStage('processing', 85, 'Computing saliency overlays & compiling forensics report');
    const extractedFrames = buildExtractedFrames(artifacts, inference.frameAnnotations);

    const analysisId = `for-${crypto.randomUUID().slice(0, 8)}`;
    const reportId = `rep-${crypto.randomUUID().slice(0, 8)}`;
    const elapsedMs = Math.round(performance.now() - t0);
    const createdAt = new Date().toISOString();

    const modalitiesPresent: string[] = [];
    if (artifacts.probe.hasVideo) modalitiesPresent.push('video');
    if (artifacts.probe.hasAudio) modalitiesPresent.push('audio');

    const result: ForensicsResult = {
      analysisId,
      uploadId: upload.id,
      filename: upload.original_name,
      mediaType: upload.media_type,
      durationSec: artifacts.probe.durationSec,
      visualAuthenticityScore: inference.visualAuthenticityScore,
      audioAuthenticityScore: inference.audioAuthenticityScore,
      avConsistencyScore: inference.avConsistencyScore,
      overallAuthenticityScore: inference.overallAuthenticityScore,
      overallLabel: inference.overallLabel,
      confidenceNote: configNote
        ? `${inference.confidenceNote} (Config: ${configNote})`
        : inference.confidenceNote,
      modelPipeline: 'Cross-Attention Multimodal Backbone + Gemini-4-Hydrogen + FFmpeg Signal Analyzer',
      executionBackend: 'gemini-multimodal-signal',
      suspiciousSegments: inference.suspiciousSegments,
      extractedFrames,
      waveform: artifacts.signalSummary.waveform,
      spectralBands: artifacts.signalSummary.spectralBands,
      processingMetadata: {
        framesAnalyzed: extractedFrames.length,
        audioSegmentsAnalyzed: artifacts.signalSummary.waveform.length,
        sampleRateHz: artifacts.probe.sampleRate,
        videoResolution: artifacts.probe.hasVideo
          ? `${artifacts.probe.width}x${artifacts.probe.height}`
          : null,
        elapsedMs,
        modalitiesPresent,
      },
      limitationsNotice:
        'Model scores represent probabilistic pattern indicators across sampled frames and acoustic bands. They do not constitute forensic proof of authenticity or manipulation.',
      reportId,
      createdAt,
    };

    db.prepare(
      `INSERT INTO analysis_results (id, job_id, upload_id, filename, media_type, visual_authenticity_score, audio_authenticity_score, av_consistency_score, overall_authenticity_score, overall_label, model_pipeline, execution_backend, payload_json, report_id, created_at)
       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`
    ).run(
      analysisId,
      jobId,
      upload.id,
      upload.original_name,
      upload.media_type,
      result.visualAuthenticityScore,
      result.audioAuthenticityScore,
      result.avConsistencyScore,
      result.overallAuthenticityScore,
      result.overallLabel,
      result.modelPipeline,
      result.executionBackend,
      JSON.stringify(result),
      reportId,
      createdAt
    );

    db.prepare(
      `INSERT INTO reports (id, report_type, title, reference_id, summary, payload_json, created_at)
       VALUES (?, 'forensics', ?, ?, ?, ?, ?)`
    ).run(
      reportId,
      `Forensics Report — ${upload.original_name}`,
      analysisId,
      `${result.overallLabel} (Overall Score ${(result.overallAuthenticityScore * 100).toFixed(1)}%)`,
      JSON.stringify(result),
      createdAt
    );

    updateStage('completed', 100, 'Forensics pipeline completed', null, analysisId);
    logActivity(
      'forensics',
      'Forensics Analysis Completed',
      `${upload.original_name}: ${result.overallLabel} (${(result.overallAuthenticityScore * 100).toFixed(1)}%)`,
      result.overallLabel === 'POTENTIALLY MANIPULATED' ? 'warning' : 'success'
    );
  } catch (err) {
    const msg = err instanceof Error ? err.message : 'Forensics job failed';
    updateStage('failed', 100, 'Forensics pipeline failed', msg, null);
    logActivity('forensics', 'Forensics Analysis Failed', `${upload.original_name}: ${msg}`, 'error');
  }
}

export function enqueueEmotionJob(uploadId: string, transcript = ''): string {
  const upload = db
    .prepare(`SELECT id, original_name, file_path, media_type FROM media_uploads WHERE id = ?`)
    .get(uploadId) as
    | { id: string; original_name: string; file_path: string; media_type: 'video' | 'audio' }
    | undefined;

  if (!upload) {
    throw new Error('Media upload not found');
  }

  const jobId = `job-emo-${crypto.randomUUID().slice(0, 8)}`;
  const now = new Date().toISOString();

  db.prepare(
    `INSERT INTO analysis_jobs (id, job_type, upload_id, filename, status, progress, stage, error, result_id, created_at, updated_at)
     VALUES (?, 'emotion', ?, ?, 'queued', 5, 'Queued for affective segmentation', NULL, NULL, ?, ?)`
  ).run(jobId, upload.id, upload.original_name, now, now);

  setImmediate(() => {
    void processEmotionJob(jobId, upload, transcript);
  });

  return jobId;
}

async function processEmotionJob(
  jobId: string,
  upload: { id: string; original_name: string; file_path: string; media_type: 'video' | 'audio' },
  transcript: string
) {
  const updateStage = (status: string, progress: number, stage: string, error: string | null = null, resultId: string | null = null) => {
    db.prepare(
      `UPDATE analysis_jobs SET status = ?, progress = ?, stage = ?, error = ?, result_id = ?, updated_at = ? WHERE id = ?`
    ).run(status, progress, stage, error, resultId, new Date().toISOString(), jobId);
  };

  try {
    updateStage('processing', 25, 'Extracting facial frames & acoustic prosody features');
    const artifacts = await extractMediaArtifacts(upload.id, upload.file_path);

    updateStage('processing', 60, 'Running modality-separated affective observation encoders');
    const emoOut = await runGeminiEmotionInference(upload.original_name, artifacts, transcript);

    const analysisId = `emo-${crypto.randomUUID().slice(0, 8)}`;
    const reportId = `rep-${crypto.randomUUID().slice(0, 8)}`;
    const createdAt = new Date().toISOString();

    const modalitiesUsed: string[] = [];
    if (artifacts.probe.hasAudio) modalitiesUsed.push('audio');
    if (artifacts.probe.hasVideo) modalitiesUsed.push('visual');
    if (transcript.trim() || emoOut.segments.some((s) => s.textPrediction)) {
      modalitiesUsed.push('text');
    }

    const counts = new Map<EmotionCategory, number>();
    for (const seg of emoOut.segments) {
      counts.set(seg.combinedPrediction, (counts.get(seg.combinedPrediction) ?? 0) + 1);
    }
    const totalSegs = Math.max(1, emoOut.segments.length);
    const categories: EmotionCategory[] = [
      'Neutral',
      'Joy',
      'Sadness',
      'Anger',
      'Fear',
      'Surprise',
      'Disgust',
    ];
    const distribution = categories.map((emotion) => {
      const count = counts.get(emotion) ?? 0;
      return { emotion, count, share: Number((count / totalSegs).toFixed(3)) };
    });

    const disagreementCount = emoOut.segments.filter((s) => s.modalityDisagreement).length;
    const disagreementRate = Number((disagreementCount / totalSegs).toFixed(3));

    const result: EmotionResult = {
      analysisId,
      uploadId: upload.id,
      filename: upload.original_name,
      durationSec: artifacts.probe.durationSec,
      dominantEmotion: emoOut.dominantEmotion,
      overallConfidence: emoOut.overallConfidence,
      disagreementRate,
      modalitiesUsed,
      modelPipeline: 'Audio Prosody + Facial Expression + Transcript Cross-Attention Fusion',
      segments: emoOut.segments,
      distribution,
      probabilisticNotice:
        'All outputs are probabilistic affective observations derived from external vocal, facial, and textual cues. They do not measure internal psychological states, intentions, or personality.',
      reportId,
      createdAt,
    };

    db.prepare(
      `INSERT INTO emotion_results (id, job_id, upload_id, filename, duration_sec, dominant_emotion, overall_confidence, disagreement_rate, model_pipeline, payload_json, report_id, created_at)
       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`
    ).run(
      analysisId,
      jobId,
      upload.id,
      upload.original_name,
      result.durationSec,
      result.dominantEmotion,
      result.overallConfidence,
      result.disagreementRate,
      result.modelPipeline,
      JSON.stringify(result),
      reportId,
      createdAt
    );

    const insertSeg = db.prepare(
      `INSERT INTO emotion_segments (id, analysis_id, upload_id, speaker, start_sec, end_sec, transcript, audio_emotion, visual_emotion, text_emotion, combined_prediction, combined_confidence, modality_disagreement, disagreement_note, created_at)
       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`
    );
    for (const seg of result.segments) {
      insertSeg.run(
        `${analysisId}-${seg.id}`,
        analysisId,
        upload.id,
        seg.speaker,
        seg.startSec,
        seg.endSec,
        seg.transcript,
        seg.audioPrediction?.emotion ?? null,
        seg.visualPrediction?.emotion ?? null,
        seg.textPrediction?.emotion ?? null,
        seg.combinedPrediction,
        seg.combinedConfidence,
        seg.modalityDisagreement ? 1 : 0,
        seg.disagreementNote,
        createdAt
      );
    }

    db.prepare(
      `INSERT INTO reports (id, report_type, title, reference_id, summary, payload_json, created_at)
       VALUES (?, 'emotion', ?, ?, ?, ?, ?)`
    ).run(
      reportId,
      `Emotion Intelligence Report — ${upload.original_name}`,
      analysisId,
      `Dominant: ${result.dominantEmotion} (${(result.overallConfidence * 100).toFixed(0)}% conf · ${Math.round(disagreementRate * 100)}% modality disagreement)`,
      JSON.stringify(result),
      createdAt
    );

    updateStage('completed', 100, 'Conversational emotion analysis completed', null, analysisId);
    logActivity(
      'emotion',
      'Emotion Analysis Completed',
      `${upload.original_name}: Dominant ${result.dominantEmotion}, ${result.segments.length} segments`,
      'success'
    );
  } catch (err) {
    const msg = err instanceof Error ? err.message : 'Emotion analysis failed';
    updateStage('failed', 100, 'Emotion pipeline failed', msg, null);
    logActivity('emotion', 'Emotion Analysis Failed', `${upload.original_name}: ${msg}`, 'error');
  }
}

export function startContinualLearningJob(params: {
  modelId: string;
  datasetId: string;
  taskName: string;
  strategy: ContinualStrategy;
  epochs: number;
  learningRate: number;
  ewcLambda: number;
  replayCapacity: number;
}): string {
  const model = db
    .prepare(`SELECT id, active_version, checkpoint_location FROM model_registry WHERE id = ?`)
    .get(params.modelId) as
    | { id: string; active_version: string; checkpoint_location: string }
    | undefined;
  if (!model) throw new Error('Target continual model not found');

  const dataset = db
    .prepare(`SELECT id, name, task_domain, classes_json, approved_for_training FROM datasets WHERE id = ?`)
    .get(params.datasetId) as
    | { id: string; name: string; task_domain: string; classes_json: string; approved_for_training: number }
    | undefined;
  if (!dataset) throw new Error('Dataset not found');
  if (!dataset.approved_for_training) {
    throw new Error('Dataset is not explicitly approved for continual learning');
  }

  const jobId = `trn-${crypto.randomUUID().slice(0, 8)}`;
  const now = new Date().toISOString();

  db.prepare(
    `INSERT INTO training_jobs (id, model_id, dataset_id, dataset_name, task_name, strategy, epochs, learning_rate, ewc_lambda, replay_capacity, status, progress, previous_version, new_version, metrics_json, error, created_at, completed_at)
     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'running', 25, ?, NULL, NULL, NULL, ?, NULL)`
  ).run(
    jobId,
    model.id,
    dataset.id,
    dataset.name,
    params.taskName,
    params.strategy,
    params.epochs,
    params.learningRate,
    params.ewcLambda,
    params.replayCapacity,
    model.active_version,
    now
  );

  db.prepare(`UPDATE model_registry SET status = 'training' WHERE id = ?`).run(model.id);

  setImmediate(() => {
    try {
      const classes = JSON.parse(dataset.classes_json) as string[];
      const samplesRaw = db
        .prepare(
          `SELECT id, sample_code, label, feature_vector_json, split FROM dataset_samples WHERE dataset_id = ?`
        )
        .all(dataset.id) as Array<{
        id: string;
        sample_code: string;
        label: string;
        feature_vector_json: string;
        split: 'train' | 'val';
      }>;

      const labelToIdx = new Map<string, number>();
      classes.forEach((c, idx) => labelToIdx.set(c, idx));

      const toLabeled = (rows: typeof samplesRaw, taskIdx: number): LabeledSample[] =>
        rows.map((r) => ({
          id: r.id,
          features: JSON.parse(r.feature_vector_json) as number[],
          labelIndex: labelToIdx.get(r.label) ?? 0,
          taskIndex: taskIdx,
        }));

      const newTrain = toLabeled(samplesRaw.filter((s) => s.split === 'train'), 2);
      const newVal = toLabeled(samplesRaw.filter((s) => s.split === 'val'), 2);

      // Load previous task validation samples (same task domain, other datasets) for retention / forgetting evaluation
      const prevRows = db
        .prepare(
          `SELECT ds.id, ds.sample_code, ds.label, ds.feature_vector_json, ds.split
           FROM dataset_samples ds
           JOIN datasets d ON d.id = ds.dataset_id
           WHERE ds.dataset_id != ? AND d.task_domain = ? AND ds.split = 'val'`
        )
        .all(dataset.id, dataset.task_domain) as Array<{
        id: string;
        sample_code: string;
        label: string;
        feature_vector_json: string;
        split: 'train' | 'val';
      }>;

      const prevVal = toLabeled(prevRows, 1);

      // Load replay memory samples compatible with same task domain
      const replayRows = db
        .prepare(
          `SELECT ds.id, ds.sample_code, ds.label, ds.feature_vector_json, ds.split
           FROM replay_memory_records rm
           JOIN dataset_samples ds ON ds.id = rm.sample_id
           WHERE rm.task_domain = ?`
        )
        .all(dataset.task_domain) as Array<{
        id: string;
        sample_code: string;
        label: string;
        feature_vector_json: string;
        split: 'train' | 'val';
      }>;

      const replaySamples = toLabeled(replayRows, 1);

      let baseCkpt = loadCheckpointFromDisk(model.checkpoint_location);
      const featureDim = newTrain[0]?.features.length || 6;
      if (!baseCkpt || baseCkpt.numClasses !== classes.length || baseCkpt.inputDim !== featureDim) {
        baseCkpt = createInitialCheckpoint(featureDim, classes.length);
      }

      const outcome = runContinualTraining(
        baseCkpt,
        newTrain,
        newVal.length > 0 ? newVal : newTrain,
        prevVal,
        replaySamples,
        {
          strategy: params.strategy,
          epochs: params.epochs,
          learningRate: params.learningRate,
          ewcLambda: params.ewcLambda,
        }
      );

      const newVersion = bumpSemver(model.active_version);
      const ckptFilename = `${model.id}-${newVersion}.json`;
      const newCkptPath = saveCheckpointToDisk(ckptFilename, outcome.checkpoint);
      const completedAt = new Date().toISOString();

      // Update replay memory respecting configured capacity
      setSetting('replay_capacity', String(params.replayCapacity));
      const allReplayCandidates = [
        ...replayRows.map((r) => ({ id: r.id, sample_code: r.sample_code, label: r.label, dataset_id: dataset.id })),
        ...samplesRaw
          .filter((s) => s.split === 'train')
          .map((s) => ({ id: s.id, sample_code: s.sample_code, label: s.label, dataset_id: dataset.id })),
      ];
      const uniqueCandidates = Array.from(
        new Map(allReplayCandidates.map((c) => [c.id, c])).values()
      );
      const selectedReplay = selectReplayBuffer(uniqueCandidates, params.replayCapacity);

      db.prepare(`DELETE FROM replay_memory_records WHERE task_domain = ?`).run(dataset.task_domain);
      db.prepare(`UPDATE dataset_samples SET in_replay_memory = 0 WHERE task_domain = ?`).run(dataset.task_domain);

      const insReplay = db.prepare(
        `INSERT INTO replay_memory_records (id, sample_id, dataset_id, sample_code, label, task_domain, priority_weight, added_at)
         VALUES (?, ?, ?, ?, ?, ?, ?, ?)`
      );
      const markSample = db.prepare(`UPDATE dataset_samples SET in_replay_memory = 1 WHERE id = ?`);

      for (const item of selectedReplay) {
        const dsLookup = db
          .prepare(`SELECT dataset_id FROM dataset_samples WHERE id = ?`)
          .get(item.id) as { dataset_id: string } | undefined;
        insReplay.run(
          `rep-${crypto.randomUUID().slice(0, 8)}`,
          item.id,
          dsLookup?.dataset_id ?? dataset.id,
          item.sample_code,
          item.label,
          dataset.task_domain,
          1.0,
          completedAt
        );
        markSample.run(item.id);
      }

      db.prepare(
        `INSERT INTO model_versions (id, model_id, version, parent_version, strategy, dataset_id, validation_accuracy, knowledge_retention, forgetting_score, checkpoint_path, created_at)
         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`
      ).run(
        `ver-${crypto.randomUUID().slice(0, 8)}`,
        model.id,
        newVersion,
        model.active_version,
        params.strategy,
        dataset.id,
        outcome.newTaskValidationAccuracy,
        outcome.knowledgeRetention,
        outcome.forgettingMeasure,
        newCkptPath,
        completedAt
      );

      db.prepare(
        `UPDATE model_registry SET active_version = ?, checkpoint_location = ?, status = 'operational', last_eval_accuracy = ?, health_note = ? WHERE id = ?`
      ).run(
        newVersion,
        newCkptPath,
        outcome.newTaskValidationAccuracy,
        `Trained ${params.strategy} on ${dataset.name}`,
        model.id
      );

      const metricsPayload = {
        prevTaskAccuracyBefore: outcome.prevTaskAccuracyBefore,
        prevTaskAccuracyAfter: outcome.prevTaskAccuracyAfter,
        newTaskValidationAccuracy: outcome.newTaskValidationAccuracy,
        knowledgeRetention: outcome.knowledgeRetention,
        forgettingMeasure: outcome.forgettingMeasure,
        lossHistory: outcome.lossHistory,
      };

      db.prepare(
        `UPDATE training_jobs SET status = 'completed', progress = 100, new_version = ?, metrics_json = ?, completed_at = ? WHERE id = ?`
      ).run(newVersion, JSON.stringify(metricsPayload), completedAt, jobId);

      db.prepare(
        `INSERT INTO continual_learning_history (id, training_job_id, model_id, dataset_id, strategy, prev_task_acc_before, prev_task_acc_after, new_task_val_acc, knowledge_retention, forgetting_measure, created_at)
         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`
      ).run(
        `clh-${crypto.randomUUID().slice(0, 8)}`,
        jobId,
        model.id,
        dataset.id,
        params.strategy,
        outcome.prevTaskAccuracyBefore,
        outcome.prevTaskAccuracyAfter,
        outcome.newTaskValidationAccuracy,
        outcome.knowledgeRetention,
        outcome.forgettingMeasure,
        completedAt
      );

      logActivity(
        'learning',
        'Incremental Learning Completed',
        `${model.id} updated ${model.active_version} -> ${newVersion} (${params.strategy}, Val Acc ${(outcome.newTaskValidationAccuracy * 100).toFixed(1)}%, Retention ${(outcome.knowledgeRetention * 100).toFixed(1)}%)`,
        'success'
      );
    } catch (err) {
      const msg = err instanceof Error ? err.message : 'Training failed';
      db.prepare(
        `UPDATE training_jobs SET status = 'failed', progress = 100, error = ?, completed_at = ? WHERE id = ?`
      ).run(msg, new Date().toISOString(), jobId);
      db.prepare(`UPDATE model_registry SET status = 'operational' WHERE id = ?`).run(model.id);
      logActivity('learning', 'Continual Training Failed', msg, 'error');
    }
  });

  return jobId;
}

export function startSelectiveUnlearningJob(params: {
  modelId: string;
  datasetId: string;
  sampleIds: string[];
  strategy: UnlearningStrategy;
  purgeRawMedia: boolean;
  purgeCachedFeatures: boolean;
  purgeReplayMemory: boolean;
}): string {
  if (params.sampleIds.length === 0) {
    throw new Error('Select at least one sample for the forget-set');
  }

  const model = db
    .prepare(`SELECT id, active_version, checkpoint_location FROM model_registry WHERE id = ?`)
    .get(params.modelId) as
    | { id: string; active_version: string; checkpoint_location: string }
    | undefined;
  if (!model) throw new Error('Target model not found');

  const dataset = db
    .prepare(`SELECT id, name, classes_json FROM datasets WHERE id = ?`)
    .get(params.datasetId) as
    | { id: string; name: string; classes_json: string }
    | undefined;
  if (!dataset) throw new Error('Dataset not found');

  const placeholders = params.sampleIds.map(() => '?').join(', ');
  const selectedRows = db
    .prepare(`SELECT id, sample_code FROM dataset_samples WHERE dataset_id = ? AND id IN (${placeholders})`)
    .all(dataset.id, ...params.sampleIds) as Array<{ id: string; sample_code: string }>;

  if (selectedRows.length === 0) {
    throw new Error('No matching samples found in selected dataset');
  }

  const requestId = `unl-${crypto.randomUUID().slice(0, 8)}`;
  const now = new Date().toISOString();

  db.prepare(
    `INSERT INTO unlearning_requests (id, model_id, dataset_id, dataset_name, strategy, sample_ids_json, sample_codes_json, purge_raw_media, purge_cached_features, purge_replay_memory, original_version, updated_version, status, created_at, completed_at)
     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, 'running', ?, NULL)`
  ).run(
    requestId,
    model.id,
    dataset.id,
    dataset.name,
    params.strategy,
    JSON.stringify(selectedRows.map((r) => r.id)),
    JSON.stringify(selectedRows.map((r) => r.sample_code)),
    params.purgeRawMedia ? 1 : 0,
    params.purgeCachedFeatures ? 1 : 0,
    params.purgeReplayMemory ? 1 : 0,
    model.active_version,
    now
  );

  db.prepare(`UPDATE model_registry SET status = 'unlearning' WHERE id = ?`).run(model.id);

  setImmediate(() => {
    try {
      const classes = JSON.parse(dataset.classes_json) as string[];
      const labelToIdx = new Map<string, number>();
      classes.forEach((c, idx) => labelToIdx.set(c, idx));

      const allRows = db
        .prepare(
          `SELECT id, sample_code, label, feature_vector_json, shard_index FROM dataset_samples WHERE dataset_id = ?`
        )
        .all(dataset.id) as Array<{
        id: string;
        sample_code: string;
        label: string;
        feature_vector_json: string;
        shard_index: number;
      }>;

      const shardSamples: ShardSample[] = allRows.map((r) => ({
        id: r.id,
        features: JSON.parse(r.feature_vector_json) as number[],
        labelIndex: labelToIdx.get(r.label) ?? 0,
        taskIndex: 1,
        shardIndex: r.shard_index,
      }));

      let ckpt = loadCheckpointFromDisk(model.checkpoint_location);
      const featureDim = shardSamples[0]?.features.length || 6;
      if (!ckpt || ckpt.numClasses !== classes.length || ckpt.inputDim !== featureDim) {
        ckpt = createInitialCheckpoint(featureDim, classes.length);
      }

      const forgetSetIds = new Set(selectedRows.map((r) => r.id));
      const numShards = Number(getSetting('sisa_shards', '4')) || 4;

      const outcome = executeSelectiveUnlearning(
        ckpt,
        shardSamples,
        forgetSetIds,
        params.strategy,
        numShards
      );

      // Perform explicit requested data hygiene actions
      if (params.purgeReplayMemory) {
        for (const id of forgetSetIds) {
          db.prepare(`DELETE FROM replay_memory_records WHERE sample_id = ?`).run(id);
          db.prepare(`UPDATE dataset_samples SET in_replay_memory = 0 WHERE id = ?`).run(id);
        }
      }
      if (params.purgeCachedFeatures) {
        for (const id of forgetSetIds) {
          db.prepare(`DELETE FROM dataset_samples WHERE id = ?`).run(id);
        }
      }

      const updatedVersion = bumpSemver(model.active_version);
      const ckptFilename = `${model.id}-${updatedVersion}-unlearned.json`;
      const newCkptPath = saveCheckpointToDisk(ckptFilename, outcome.updatedCheckpoint);
      const completedAt = new Date().toISOString();

      db.prepare(
        `INSERT INTO model_versions (id, model_id, version, parent_version, strategy, dataset_id, validation_accuracy, knowledge_retention, forgetting_score, checkpoint_path, created_at)
         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`
      ).run(
        `ver-${crypto.randomUUID().slice(0, 8)}`,
        model.id,
        updatedVersion,
        model.active_version,
        `unlearn:${params.strategy}`,
        dataset.id,
        outcome.retainAccAfter,
        outcome.overallUtilityAfter,
        Number(Math.max(0, outcome.forgetAccBefore - outcome.forgetAccAfter).toFixed(4)),
        newCkptPath,
        completedAt
      );

      db.prepare(
        `UPDATE model_registry SET active_version = ?, checkpoint_location = ?, status = 'operational', last_eval_accuracy = ?, health_note = ? WHERE id = ?`
      ).run(
        updatedVersion,
        newCkptPath,
        outcome.retainAccAfter,
        `Unlearned ${selectedRows.length} sample(s) via ${params.strategy}`,
        model.id
      );

      db.prepare(
        `UPDATE unlearning_requests SET updated_version = ?, status = 'completed', completed_at = ? WHERE id = ?`
      ).run(updatedVersion, completedAt, requestId);

      db.prepare(
        `INSERT INTO unlearning_audit (id, request_id, forget_acc_before, forget_acc_after, retain_acc_before, retain_acc_after, overall_utility_after, execution_time_ms, affected_shards_json, validation_notes, created_at)
         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`
      ).run(
        `aud-${crypto.randomUUID().slice(0, 8)}`,
        requestId,
        outcome.forgetAccBefore,
        outcome.forgetAccAfter,
        outcome.retainAccBefore,
        outcome.retainAccAfter,
        outcome.overallUtilityAfter,
        outcome.executionTimeMs,
        JSON.stringify(outcome.affectedShards),
        outcome.notes,
        completedAt
      );

      const reportId = `rep-${crypto.randomUUID().slice(0, 8)}`;
      const auditPayload = {
        requestId,
        modelId: model.id,
        datasetName: dataset.name,
        strategy: params.strategy,
        sampleCodes: selectedRows.map((r) => r.sample_code),
        originalVersion: model.active_version,
        updatedVersion,
        purgeActions: {
          rawMedia: params.purgeRawMedia,
          cachedFeatures: params.purgeCachedFeatures,
          replayMemory: params.purgeReplayMemory,
        },
        metrics: {
          forgetAccBefore: outcome.forgetAccBefore,
          forgetAccAfter: outcome.forgetAccAfter,
          retainAccBefore: outcome.retainAccBefore,
          retainAccAfter: outcome.retainAccAfter,
          overallUtilityAfter: outcome.overallUtilityAfter,
          executionTimeMs: outcome.executionTimeMs,
          affectedShards: outcome.affectedShards,
        },
        notes: outcome.notes,
      };

      db.prepare(
        `INSERT INTO reports (id, report_type, title, reference_id, summary, payload_json, created_at)
         VALUES (?, 'unlearning', ?, ?, ?, ?, ?)`
      ).run(
        reportId,
        `Unlearning Audit Report — ${requestId}`,
        requestId,
        `${params.strategy}: Retain ${(outcome.retainAccAfter * 100).toFixed(1)}%, Forget ${(outcome.forgetAccBefore * 100).toFixed(1)}% -> ${(outcome.forgetAccAfter * 100).toFixed(1)}%`,
        JSON.stringify(auditPayload),
        completedAt
      );

      logActivity(
        'unlearning',
        'Selective Unlearning Executed',
        `${selectedRows.length} sample(s) unlearned via ${params.strategy} (${model.active_version} -> ${updatedVersion})`,
        'success'
      );
    } catch (err) {
      const msg = err instanceof Error ? err.message : 'Unlearning execution failed';
      db.prepare(
        `UPDATE unlearning_requests SET status = 'failed', completed_at = ? WHERE id = ?`
      ).run(new Date().toISOString(), requestId);
      db.prepare(`UPDATE model_registry SET status = 'operational' WHERE id = ?`).run(model.id);
      logActivity('unlearning', 'Unlearning Failed', msg, 'error');
    }
  });

  return requestId;
}

export function deleteRawMediaSafely(uploadId: string): boolean {
  const row = db
    .prepare(`SELECT id, original_name, file_path, is_sample FROM media_uploads WHERE id = ?`)
    .get(uploadId) as
    | { id: string; original_name: string; file_path: string; is_sample: number }
    | undefined;
  if (!row) return false;
  if (row.is_sample) {
    throw new Error('Built-in sample media probes cannot be deleted');
  }
  if (fs.existsSync(row.file_path)) {
    fs.unlinkSync(row.file_path);
  }
  db.prepare(`DELETE FROM media_uploads WHERE id = ?`).run(uploadId);
  logActivity(
    'system',
    'Raw Media Deleted',
    `Deleted raw uploaded file "${row.original_name}". Note: Deleting raw files does not alter trained model parameters unless selective unlearning is executed.`,
    'info'
  );
  return true;
}
