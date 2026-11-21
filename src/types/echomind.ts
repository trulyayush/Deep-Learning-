export type NavSection =
  | 'overview'
  | 'forensics'
  | 'emotion'
  | 'learning'
  | 'unlearning'
  | 'architecture'
  | 'models'
  | 'history'
  | 'settings';

export type AuthenticityLabel =
  | 'LIKELY AUTHENTIC'
  | 'POTENTIALLY MANIPULATED'
  | 'INCONCLUSIVE';

export type EmotionCategory =
  | 'Neutral'
  | 'Joy'
  | 'Sadness'
  | 'Anger'
  | 'Fear'
  | 'Surprise'
  | 'Disgust';

export type ContinualStrategy =
  | 'sequential_finetuning'
  | 'ewc'
  | 'experience_replay'
  | 'ewc_replay';

export type UnlearningStrategy =
  | 'exact_retrain'
  | 'sisa_sharded'
  | 'gradient_ascent';

export interface MediaUpload {
  id: string;
  filename: string;
  originalName: string;
  mimeType: string;
  sizeBytes: number;
  mediaType: 'video' | 'audio';
  durationSec: number;
  width: number;
  height: number;
  fps: number;
  videoCodec: string | null;
  audioCodec: string | null;
  sampleRate: number | null;
  hasVideo: boolean;
  hasAudio: boolean;
  isSample: boolean;
  previewUrl: string;
  createdAt: string;
}

export interface SuspiciousSegment {
  id: string;
  startSec: number;
  endSec: number;
  modality: 'visual' | 'audio' | 'audiovisual';
  severity: 'low' | 'medium' | 'high';
  anomalyScore: number;
  summary: string;
  frameUrl?: string;
  heatmapRegion?: {
    x: number; // normalized 0-1
    y: number; // normalized 0-1
    w: number; // normalized 0-1
    h: number; // normalized 0-1
    intensity: number; // 0-1
    label: string;
  };
}

export interface WaveformPoint {
  timeSec: number;
  amplitude: number;
  rmsEnergy: number;
  spectralCentroidHz: number;
  anomalyFlag: boolean;
}

export interface ExtractedFrame {
  index: number;
  timestampSec: number;
  url: string;
  sharpnessScore: number;
  temporalDelta: number;
  anomalyScore: number;
  heatmapRegion?: {
    x: number;
    y: number;
    w: number;
    h: number;
    intensity: number;
    label: string;
  };
}

export interface ForensicsResult {
  analysisId: string;
  uploadId: string;
  filename: string;
  mediaType: 'video' | 'audio';
  durationSec: number;
  visualAuthenticityScore: number | null;
  audioAuthenticityScore: number | null;
  avConsistencyScore: number | null;
  overallAuthenticityScore: number;
  overallLabel: AuthenticityLabel;
  confidenceNote: string;
  modelPipeline: string;
  executionBackend: 'gemini-multimodal-signal' | 'external-pytorch' | 'signal-only';
  suspiciousSegments: SuspiciousSegment[];
  extractedFrames: ExtractedFrame[];
  waveform: WaveformPoint[];
  spectralBands: { band: string; energyDb: number; baselineDb: number }[];
  processingMetadata: {
    framesAnalyzed: number;
    audioSegmentsAnalyzed: number;
    sampleRateHz: number | null;
    videoResolution: string | null;
    elapsedMs: number;
    modalitiesPresent: string[];
  };
  limitationsNotice: string;
  reportId: string;
  createdAt: string;
}

export interface EmotionModalityPrediction {
  emotion: EmotionCategory;
  confidence: number;
  rationale: string;
}

export interface EmotionSegment {
  id: string;
  speaker: string;
  startSec: number;
  endSec: number;
  transcript: string;
  audioPrediction: EmotionModalityPrediction | null;
  visualPrediction: EmotionModalityPrediction | null;
  textPrediction: EmotionModalityPrediction | null;
  combinedPrediction: EmotionCategory;
  combinedConfidence: number;
  modalityDisagreement: boolean;
  disagreementNote: string | null;
  acousticFeatures: {
    rmsEnergy: number;
    estimatedPitchHz: number;
    zeroCrossingRate: number;
  };
}

export interface EmotionResult {
  analysisId: string;
  uploadId: string;
  filename: string;
  durationSec: number;
  dominantEmotion: EmotionCategory;
  overallConfidence: number;
  disagreementRate: number;
  modalitiesUsed: string[];
  modelPipeline: string;
  segments: EmotionSegment[];
  distribution: { emotion: EmotionCategory; share: number; count: number }[];
  probabilisticNotice: string;
  reportId: string;
  createdAt: string;
}

export interface AnalysisJob {
  id: string;
  jobType: 'forensics' | 'emotion';
  uploadId: string;
  filename: string;
  status: 'queued' | 'processing' | 'completed' | 'failed';
  progress: number;
  stage: string;
  error: string | null;
  resultId: string | null;
  createdAt: string;
  updatedAt: string;
}

export interface DatasetSample {
  id: string;
  datasetId: string;
  sampleCode: string;
  label: string;
  taskDomain: 'forensics' | 'emotion';
  modality: 'audiovisual' | 'audio' | 'visual' | 'text';
  featureVector: number[];
  split: 'train' | 'val';
  shardIndex: number;
  inReplayMemory: boolean;
  sourceName: string;
  createdAt: string;
}

export interface TrainingDataset {
  id: string;
  name: string;
  description: string;
  taskDomain: 'forensics' | 'emotion';
  classes: string[];
  sampleCount: number;
  trainCount: number;
  valCount: number;
  approvedForTraining: boolean;
  provenanceSource: string;
  createdAt: string;
}

export interface ModelRecord {
  id: string;
  name: string;
  modelType: 'forensics_visual' | 'forensics_audio' | 'emotion_fusion' | 'continual_head' | 'shared_backbone';
  activeVersion: string;
  architecture: string;
  inputRequirements: string;
  outputRequirements: string;
  checkpointLocation: string;
  status: 'operational' | 'unavailable' | 'training' | 'unlearning';
  healthNote: string;
  lastEvalAccuracy: number | null;
  createdAt: string;
}

export interface ModelVersionRecord {
  id: string;
  modelId: string;
  version: string;
  parentVersion: string | null;
  strategy: string;
  datasetId: string | null;
  validationAccuracy: number;
  knowledgeRetention: number;
  forgettingScore: number;
  checkpointPath: string;
  createdAt: string;
}

export interface LearningJobRecord {
  id: string;
  modelId: string;
  datasetId: string;
  datasetName: string;
  taskName: string;
  strategy: ContinualStrategy;
  epochs: number;
  learningRate: number;
  ewcLambda: number;
  replayCapacity: number;
  status: 'queued' | 'running' | 'completed' | 'failed';
  progress: number;
  previousVersion: string;
  newVersion: string | null;
  prevTaskAccuracyBefore: number;
  prevTaskAccuracyAfter: number;
  newTaskValidationAccuracy: number;
  knowledgeRetention: number;
  forgettingMeasure: number;
  lossHistory: { epoch: number; trainLoss: number; valAccuracy: number; ewcPenalty: number }[];
  error: string | null;
  createdAt: string;
  completedAt: string | null;
}

export interface ReplayMemoryRecord {
  id: string;
  sampleId: string;
  datasetId: string;
  sampleCode: string;
  label: string;
  taskDomain: string;
  priorityWeight: number;
  addedAt: string;
}

export interface UnlearningRequestRecord {
  id: string;
  modelId: string;
  datasetId: string;
  datasetName: string;
  strategy: UnlearningStrategy;
  sampleIds: string[];
  sampleCodes: string[];
  purgeRawMedia: boolean;
  purgeCachedFeatures: boolean;
  purgeReplayMemory: boolean;
  originalVersion: string;
  updatedVersion: string | null;
  status: 'queued' | 'running' | 'completed' | 'failed';
  forgetAccBefore: number;
  forgetAccAfter: number;
  retainAccBefore: number;
  retainAccAfter: number;
  overallUtilityAfter: number;
  executionTimeMs: number;
  affectedShards: number[];
  notes: string;
  createdAt: string;
  completedAt: string | null;
}

export interface ReportRecord {
  id: string;
  reportType: 'forensics' | 'emotion' | 'unlearning';
  title: string;
  referenceId: string;
  summary: string;
  payload: unknown;
  createdAt: string;
}

export interface ActivityLogRecord {
  id: string;
  category: 'forensics' | 'emotion' | 'learning' | 'unlearning' | 'model' | 'system';
  action: string;
  detail: string;
  status: 'info' | 'success' | 'warning' | 'error';
  createdAt: string;
}

export interface HardwareProfile {
  osPlatform: string;
  osRelease: string;
  arch: string;
  cpuModel: string;
  cpuCores: number;
  totalMemoryMb: number;
  freeMemoryMb: number;
  nodeVersion: string;
  pythonVersion: string | null;
  cudaAvailable: boolean;
  gpuName: string | null;
  ffmpegAvailable: boolean;
  geminiConfigured: boolean;
  externalInferenceUrl: string;
  executionMode: string;
  recommendedBatchSize: number;
  mixedPrecisionSupported: boolean;
  diskFreeSummary: string;
}

export interface SystemArchitectureSpec {
  encoders: {
    id: string;
    name: string;
    modality: 'visual' | 'audio' | 'text';
    backbone: string;
    outputDim: number;
    status: string;
    supportsMissing: boolean;
  }[];
  fusion: {
    mechanism: string;
    projectionDim: number;
    attentionHeads: number;
    temporalLayer: string;
    sharedRepresentationDim: number;
  };
  heads: {
    id: string;
    name: string;
    task: string;
    modelId: string;
    activeVersion: string;
    outputClasses: string[];
    status: string;
  }[];
  continualConfig: {
    strategies: ContinualStrategy[];
    replayBufferCount: number;
    replayCapacity: number;
  };
  unlearningConfig: {
    strategies: UnlearningStrategy[];
    totalShards: number;
    auditCount: number;
  };
}
