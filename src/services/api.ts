import type {
  ActivityLogRecord,
  AnalysisJob,
  ContinualStrategy,
  DatasetSample,
  EmotionResult,
  ForensicsResult,
  HardwareProfile,
  LearningJobRecord,
  MediaUpload,
  ModelRecord,
  ModelVersionRecord,
  ReportRecord,
  SystemArchitectureSpec,
  TrainingDataset,
  UnlearningRequestRecord,
  UnlearningStrategy,
} from '../types/echomind';

async function requestJson<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(url, init);
  if (!res.ok) {
    const errBody = await res.json().catch(() => ({ error: `HTTP ${res.status}` }));
    throw new Error((errBody as { error?: string }).error || `Request failed (${res.status})`);
  }
  return (await res.json()) as T;
}

export interface OverviewPayload {
  counts: {
    forensicsAnalyses: number;
    emotionAnalyses: number;
    registeredModels: number;
    operationalModels: number;
    approvedDatasets: number;
    trainingRuns: number;
    unlearningAudits: number;
    replaySamples: number;
  };
  recentForensics: Array<{
    analysisId: string;
    uploadId: string;
    filename: string;
    mediaType: 'video' | 'audio';
    overallAuthenticityScore: number;
    overallLabel: string;
    createdAt: string;
  }>;
  recentEmotion: Array<{
    analysisId: string;
    uploadId: string;
    filename: string;
    dominantEmotion: string;
    overallConfidence: number;
    disagreementRate: number;
    createdAt: string;
  }>;
  models: ModelRecord[];
  datasets: Array<{
    id: string;
    name: string;
    taskDomain: string;
    approvedForTraining: number;
    sampleCount: number;
  }>;
  jobs: AnalysisJob[];
  recentTraining: Array<{
    id: string;
    modelId: string;
    datasetName: string;
    strategy: string;
    status: string;
    previousVersion: string;
    newVersion: string | null;
    createdAt: string;
  }>;
  recentUnlearning: Array<{
    id: string;
    datasetName: string;
    strategy: string;
    originalVersion: string;
    updatedVersion: string | null;
    status: string;
    retainAccAfter: number | null;
    forgetAccAfter: number | null;
    createdAt: string;
  }>;
  recentReports: ReportRecord[];
}

export const api = {
  getHealth: () =>
    requestJson<{ status: string; geminiConfigured: boolean; database: string }>('/api/health'),
  getOverview: () => requestJson<OverviewPayload>('/api/overview'),
  getHardware: () => requestJson<HardwareProfile>('/api/system/hardware'),
  getArchitecture: () => requestJson<SystemArchitectureSpec>('/api/system/architecture'),
  getSettings: () =>
    requestJson<{
      maxUploadMb: number;
      externalInferenceUrl: string;
      replayCapacity: number;
      sisaShards: number;
    }>('/api/system/settings'),
  updateSettings: (payload: {
    maxUploadMb?: number;
    externalInferenceUrl?: string;
    replayCapacity?: number;
    sisaShards?: number;
  }) =>
    requestJson<{
      maxUploadMb: number;
      externalInferenceUrl: string;
      replayCapacity: number;
      sisaShards: number;
    }>('/api/system/settings', {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    }),
  pingExternalServer: (url: string) =>
    requestJson<{ ok: boolean; message: string }>('/api/system/ping-external', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ url }),
    }),

  // Media
  getMedia: () => requestJson<MediaUpload[]>('/api/media'),
  uploadMedia: async (file: File): Promise<MediaUpload> => {
    const fd = new FormData();
    fd.append('file', file);
    const res = await fetch('/api/media/upload', { method: 'POST', body: fd });
    if (!res.ok) {
      const errBody = await res.json().catch(() => ({ error: 'Upload failed' }));
      throw new Error((errBody as { error?: string }).error || 'Upload failed');
    }
    return (await res.json()) as MediaUpload;
  },
  deleteMedia: (id: string) =>
    requestJson<{ deleted: boolean }>(`/api/media/${encodeURIComponent(id)}`, { method: 'DELETE' }),

  // Jobs
  getJob: (jobId: string) => requestJson<AnalysisJob>(`/api/jobs/${encodeURIComponent(jobId)}`),

  // Forensics
  startForensics: (uploadId: string, configNote?: string) =>
    requestJson<{ jobId: string; status: string }>('/api/analysis', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ uploadId, configNote }),
    }),
  getForensicsList: () => requestJson<ForensicsResult[]>('/api/analysis'),
  getForensicsResult: (analysisId: string) =>
    requestJson<ForensicsResult>(`/api/analysis/${encodeURIComponent(analysisId)}`),

  // Emotion
  startEmotion: (uploadId: string, transcript?: string) =>
    requestJson<{ jobId: string; status: string }>('/api/emotion/analyze', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ uploadId, transcript }),
    }),
  getEmotionList: () => requestJson<EmotionResult[]>('/api/emotion'),
  getEmotionResult: (analysisId: string) =>
    requestJson<EmotionResult>(`/api/emotion/${encodeURIComponent(analysisId)}`),

  // Continual Learning
  getDatasets: () => requestJson<TrainingDataset[]>('/api/learning/datasets'),
  registerDataset: (payload: {
    name: string;
    description: string;
    taskDomain: 'forensics' | 'emotion';
    classes: string[];
    sampleCount: number;
    approvedForTraining: boolean;
    provenanceSource: string;
  }) =>
    requestJson<{ id: string; name: string; sampleCount: number }>('/api/learning/datasets', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    }),
  getDatasetSamples: (datasetId: string) =>
    requestJson<DatasetSample[]>(`/api/learning/datasets/${encodeURIComponent(datasetId)}/samples`),
  startLearning: (payload: {
    modelId: string;
    datasetId: string;
    taskName: string;
    strategy: ContinualStrategy;
    epochs: number;
    learningRate: number;
    ewcLambda: number;
    replayCapacity: number;
  }) =>
    requestJson<{ jobId: string; status: string }>('/api/learning/start', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    }),
  getLearningJob: (jobId: string) =>
    requestJson<LearningJobRecord>(`/api/learning/jobs/${encodeURIComponent(jobId)}`),
  getLearningHistory: () => requestJson<LearningJobRecord[]>('/api/learning/history'),
  getReplayMemory: () =>
    requestJson<{
      capacity: number;
      count: number;
      records: Array<{
        id: string;
        sampleId: string;
        datasetId: string;
        sampleCode: string;
        label: string;
        taskDomain: string;
        priorityWeight: number;
        addedAt: string;
      }>;
    }>('/api/learning/replay'),

  // Machine Unlearning
  startUnlearning: (payload: {
    modelId: string;
    datasetId: string;
    sampleIds: string[];
    strategy: UnlearningStrategy;
    purgeRawMedia: boolean;
    purgeCachedFeatures: boolean;
    purgeReplayMemory: boolean;
  }) =>
    requestJson<{ requestId: string; status: string }>('/api/unlearning/requests', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    }),
  getUnlearningRequests: () => requestJson<UnlearningRequestRecord[]>('/api/unlearning/requests'),
  getUnlearningRequest: (requestId: string) =>
    requestJson<UnlearningRequestRecord>(`/api/unlearning/requests/${encodeURIComponent(requestId)}`),

  // Models
  getModels: () => requestJson<ModelRecord[]>('/api/models'),
  getModelVersions: (modelId: string) =>
    requestJson<ModelVersionRecord[]>(`/api/models/${encodeURIComponent(modelId)}/versions`),
  rollbackModel: (modelId: string, version: string) =>
    requestJson<{ rolledBackTo: string }>(`/api/models/${encodeURIComponent(modelId)}/rollback`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ version }),
    }),
  healthCheckModel: (modelId: string) =>
    requestJson<{ id: string; status: string; healthNote: string }>(
      `/api/models/${encodeURIComponent(modelId)}/health-check`,
      { method: 'POST' }
    ),

  // Reports & Activity
  getReports: () => requestJson<ReportRecord[]>('/api/reports'),
  getActivity: (category?: string) =>
    requestJson<ActivityLogRecord[]>(
      category ? `/api/activity?category=${encodeURIComponent(category)}` : '/api/activity'
    ),
};
