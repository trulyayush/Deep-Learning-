import { useEffect, useState } from 'react';
import { Loader2, Plus, RotateCcw } from 'lucide-react';
import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import { api } from '../services/api';
import type {
  ContinualStrategy,
  LearningJobRecord,
  ModelRecord,
  ModelVersionRecord,
  TrainingDataset,
} from '../types/echomind';

export function LearningLabView() {
  const [datasets, setDatasets] = useState<TrainingDataset[]>([]);
  const [models, setModels] = useState<ModelRecord[]>([]);
  const [versions, setVersions] = useState<ModelVersionRecord[]>([]);
  const [history, setHistory] = useState<LearningJobRecord[]>([]);
  const [replay, setReplay] = useState<{
    capacity: number;
    count: number;
    records: Array<{
      id: string;
      sampleCode: string;
      label: string;
      taskDomain: string;
      priorityWeight: number;
    }>;
  } | null>(null);

  const [selectedDatasetId, setSelectedDatasetId] = useState('');
  const [taskName, setTaskName] = useState('Incremental Drift Adaptation');
  const [strategy, setStrategy] = useState<ContinualStrategy>('ewc_replay');
  const [epochs, setEpochs] = useState(25);
  const [learningRate, setLearningRate] = useState(0.12);
  const [ewcLambda, setEwcLambda] = useState(2.0);
  const [replayCapacity, setReplayCapacity] = useState(32);

  const [activeJob, setActiveJob] = useState<LearningJobRecord | null>(null);
  const [showNewDatasetForm, setShowNewDatasetForm] = useState(false);
  const [newDsName, setNewDsName] = useState('');
  const [newDsDomain, setNewDsDomain] = useState<'forensics' | 'emotion'>('forensics');
  const [newDsClasses, setNewDsClasses] = useState('Authentic, Manipulated');
  const [newDsSamples, setNewDsSamples] = useState(20);
  const [newDsApproved, setNewDsApproved] = useState(true);

  const [error, setError] = useState<string | null>(null);
  const [statusMessage, setStatusMessage] = useState<string | null>(null);

  const loadAll = async () => {
    try {
      const [ds, mods, vers, hist, rep] = await Promise.all([
        api.getDatasets(),
        api.getModels(),
        api.getModelVersions('mod-continual-head'),
        api.getLearningHistory(),
        api.getReplayMemory(),
      ]);
      setDatasets(ds);
      setModels(mods);
      setVersions(vers);
      setHistory(hist);
      setReplay(rep);
      if (!selectedDatasetId && ds.length > 0) {
        const secondOrFirst = ds.find((d) => d.id.includes('fakeav')) ?? ds[0];
        setSelectedDatasetId(secondOrFirst.id);
      }
      if (!activeJob && hist.length > 0) {
        setActiveJob(hist[0]);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load learning lab');
    }
  };

  useEffect(() => {
    void loadAll();
  }, []);

  useEffect(() => {
    if (!activeJob || activeJob.status === 'completed' || activeJob.status === 'failed') return;
    const timer = setInterval(async () => {
      try {
        const job = await api.getLearningJob(activeJob.id);
        setActiveJob(job);
        if (job.status === 'completed' || job.status === 'failed') {
          await loadAll();
          if (job.status === 'failed') setError(job.error || 'Training failed');
        }
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Training poll failed');
      }
    }, 600);
    return () => clearInterval(timer);
  }, [activeJob]);

  const continualModel = models.find((m) => m.id === 'mod-continual-head') ?? null;

  const handleStartTraining = async () => {
    if (!selectedDatasetId) {
      setError('Select an approved labelled dataset first.');
      return;
    }
    setError(null);
    setStatusMessage(null);
    try {
      const { jobId } = await api.startLearning({
        modelId: 'mod-continual-head',
        datasetId: selectedDatasetId,
        taskName,
        strategy,
        epochs,
        learningRate,
        ewcLambda,
        replayCapacity,
      });
      const job = await api.getLearningJob(jobId);
      setActiveJob(job);
      if (job.status === 'completed' || job.status === 'failed') {
        await loadAll();
        if (job.status === 'failed') setError(job.error || 'Training failed');
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Could not start incremental learning');
    }
  };

  const handleRollback = async (version: string) => {
    setError(null);
    try {
      await api.rollbackModel('mod-continual-head', version);
      setStatusMessage(`Rolled back continual classifier head to ${version}.`);
      await loadAll();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Rollback failed');
    }
  };

  const handleRegisterDataset = async () => {
    setError(null);
    try {
      const classes = newDsClasses
        .split(',')
        .map((c) => c.trim())
        .filter(Boolean);
      const created = await api.registerDataset({
        name: newDsName,
        description: `Registered ${newDsDomain} labelled incremental learning task`,
        taskDomain: newDsDomain,
        classes,
        sampleCount: newDsSamples,
        approvedForTraining: newDsApproved,
        provenanceSource: 'Workstation Approved Dataset Registry',
      });
      setShowNewDatasetForm(false);
      setNewDsName('');
      await loadAll();
      setSelectedDatasetId(created.id);
      setStatusMessage(`Registered approved dataset "${created.name}" (${created.sampleCount} samples).`);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Dataset registration failed');
    }
  };

  return (
    <div className="space-y-8">
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div className="min-w-0 flex-1 max-w-2xl">
          <h2 className="text-lg font-semibold tracking-tight text-[#F3F4F6]">
            03. Continual Learning Engine
          </h2>
          <p className="text-sm text-[#9CA3AF]">
            Incrementally train task heads on explicitly approved datasets using EWC Fisher regularization and Experience Replay without overwriting prior knowledge.
          </p>
        </div>

        <div className="flex shrink-0 items-center gap-3">
          <button
            type="button"
            onClick={() => setShowNewDatasetForm((prev) => !prev)}
            className="inline-flex items-center gap-1.5 whitespace-nowrap rounded-md border border-white/[0.12] bg-[#12151A] px-3.5 py-2 text-xs font-medium text-[#F3F4F6] hover:bg-[#181C23]"
          >
            <Plus className="h-3.5 w-3.5" />
            Register Approved Dataset
          </button>
          <button
            type="button"
            onClick={() => void handleStartTraining()}
            disabled={activeJob?.status === 'running' || activeJob?.status === 'queued'}
            className="whitespace-nowrap rounded-md bg-[#3B82F6] px-4 py-2 text-xs font-medium text-white hover:bg-[#2563EB] disabled:opacity-50"
          >
            {activeJob?.status === 'running' ? 'Training...' : 'Start Incremental Training'}
          </button>
        </div>
      </div>

      {error && (
        <div role="alert" className="rounded-lg border border-red-500/30 bg-red-950/20 px-4 py-3 text-xs text-red-200">
          {error}
        </div>
      )}
      {statusMessage && (
        <div className="rounded-lg border border-emerald-500/30 bg-emerald-950/20 px-4 py-3 text-xs text-emerald-200">
          {statusMessage}
        </div>
      )}

      {/* Optional Dataset Registration Drawer */}
      {showNewDatasetForm && (
        <section className="rounded-lg border border-[#3B82F6]/30 bg-[#12151A] p-5 space-y-4">
          <h3 className="text-sm font-semibold text-[#F3F4F6]">Register Approved Labelled Dataset</h3>
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
            <div>
              <label htmlFor="ds-name" className="block text-xs text-[#9CA3AF]">Dataset Name</label>
              <input
                id="ds-name"
                type="text"
                value={newDsName}
                onChange={(e) => setNewDsName(e.target.value)}
                placeholder="e.g., DFDC-Calibration-Batch-04"
                className="mt-1 w-full rounded-md border border-white/[0.12] bg-[#0B0D10] px-3 py-2 text-xs text-[#F3F4F6]"
              />
            </div>
            <div>
              <label htmlFor="ds-domain" className="block text-xs text-[#9CA3AF]">Task Domain</label>
              <select
                id="ds-domain"
                value={newDsDomain}
                onChange={(e) => {
                  const d = e.target.value as 'forensics' | 'emotion';
                  setNewDsDomain(d);
                  setNewDsClasses(
                    d === 'forensics' ? 'Authentic, Manipulated' : 'Neutral, Joy, Sadness, Anger'
                  );
                }}
                className="mt-1 w-full rounded-md border border-white/[0.12] bg-[#0B0D10] px-3 py-2 text-xs text-[#F3F4F6]"
              >
                <option value="forensics">Forensics (Binary)</option>
                <option value="emotion">Emotion (Multi-Class)</option>
              </select>
            </div>
            <div>
              <label htmlFor="ds-classes" className="block text-xs text-[#9CA3AF]">Class Labels (CSV)</label>
              <input
                id="ds-classes"
                type="text"
                value={newDsClasses}
                onChange={(e) => setNewDsClasses(e.target.value)}
                className="mt-1 w-full rounded-md border border-white/[0.12] bg-[#0B0D10] px-3 py-2 text-xs text-[#F3F4F6]"
              />
            </div>
            <div>
              <label htmlFor="ds-count" className="block text-xs text-[#9CA3AF]">Sample Count</label>
              <input
                id="ds-count"
                type="number"
                min={8}
                max={80}
                value={newDsSamples}
                onChange={(e) => setNewDsSamples(Number(e.target.value))}
                className="mt-1 w-full rounded-md border border-white/[0.12] bg-[#0B0D10] px-3 py-2 text-xs text-[#F3F4F6]"
              />
            </div>
          </div>

          <div className="flex flex-wrap items-center justify-between gap-4 border-t border-white/[0.06] pt-3">
            <label className="flex items-center gap-2 text-xs text-[#E5E7EB]">
              <input
                type="checkbox"
                checked={newDsApproved}
                onChange={(e) => setNewDsApproved(e.target.checked)}
                className="rounded border-white/20 bg-[#0B0D10]"
              />
              Confirm dataset is explicitly labelled and approved for continual training (private uploads never auto-enter).
            </label>
            <div className="flex items-center gap-2">
              <button
                type="button"
                onClick={() => setShowNewDatasetForm(false)}
                className="rounded-md border border-white/[0.12] px-3 py-1.5 text-xs text-[#9CA3AF] hover:text-white"
              >
                Cancel
              </button>
              <button
                type="button"
                onClick={() => void handleRegisterDataset()}
                className="rounded-md bg-[#3B82F6] px-4 py-1.5 text-xs font-medium text-white hover:bg-[#2563EB]"
              >
                Save Dataset
              </button>
            </div>
          </div>
        </section>
      )}

      {/* Configuration & Active Checkpoint Summary */}
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-12">
        <section className="lg:col-span-7 rounded-lg border border-white/[0.08] bg-[#12151A] p-5 space-y-4">
          <div className="flex items-center justify-between border-b border-white/[0.06] pb-3">
            <h3 className="text-sm font-semibold text-[#F3F4F6]">Training Session Configuration</h3>
            <span className="font-mono-tabular text-xs text-[#60A5FA]">
              Active Head: {continualModel?.activeVersion ?? 'v1.0.0'}
            </span>
          </div>

          <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
            <div>
              <label htmlFor="cl-dataset" className="block text-xs text-[#9CA3AF]">
                Approved Dataset
              </label>
              <select
                id="cl-dataset"
                value={selectedDatasetId}
                onChange={(e) => setSelectedDatasetId(e.target.value)}
                className="mt-1 w-full rounded-md border border-white/[0.12] bg-[#0B0D10] px-3 py-2 text-xs text-[#F3F4F6]"
              >
                {datasets.map((d) => (
                  <option key={d.id} value={d.id}>
                    {d.name} ({d.sampleCount} samples · {d.taskDomain})
                  </option>
                ))}
              </select>
            </div>

            <div>
              <label htmlFor="cl-strategy" className="block text-xs text-[#9CA3AF]">
                Continual Strategy
              </label>
              <select
                id="cl-strategy"
                value={strategy}
                onChange={(e) => setStrategy(e.target.value as ContinualStrategy)}
                className="mt-1 w-full rounded-md border border-white/[0.12] bg-[#0B0D10] px-3 py-2 text-xs text-[#F3F4F6]"
              >
                <option value="ewc_replay">EWC + Experience Replay (Recommended)</option>
                <option value="ewc">Elastic Weight Consolidation (EWC)</option>
                <option value="experience_replay">Experience Replay Only</option>
                <option value="sequential_finetuning">Sequential Fine-Tuning (Baseline)</option>
              </select>
            </div>

            <div>
              <label htmlFor="cl-task" className="block text-xs text-[#9CA3AF]">
                Task Identifier
              </label>
              <input
                id="cl-task"
                type="text"
                value={taskName}
                onChange={(e) => setTaskName(e.target.value)}
                className="mt-1 w-full rounded-md border border-white/[0.12] bg-[#0B0D10] px-3 py-2 text-xs text-[#F3F4F6]"
              />
            </div>

            <div className="grid grid-cols-3 gap-2">
              <div>
                <label htmlFor="cl-epochs" className="block text-xs text-[#9CA3AF]">Epochs</label>
                <input
                  id="cl-epochs"
                  type="number"
                  min={5}
                  max={100}
                  value={epochs}
                  onChange={(e) => setEpochs(Number(e.target.value))}
                  className="mt-1 w-full rounded-md border border-white/[0.12] bg-[#0B0D10] px-2.5 py-2 font-mono-tabular text-xs text-[#F3F4F6]"
                />
              </div>
              <div>
                <label htmlFor="cl-lambda" className="block text-xs text-[#9CA3AF]">EWC λ</label>
                <input
                  id="cl-lambda"
                  type="number"
                  step={0.5}
                  min={0}
                  max={20}
                  value={ewcLambda}
                  onChange={(e) => setEwcLambda(Number(e.target.value))}
                  className="mt-1 w-full rounded-md border border-white/[0.12] bg-[#0B0D10] px-2.5 py-2 font-mono-tabular text-xs text-[#F3F4F6]"
                />
              </div>
              <div>
                <label htmlFor="cl-replay" className="block text-xs text-[#9CA3AF]">Replay Cap</label>
                <input
                  id="cl-replay"
                  type="number"
                  min={4}
                  max={256}
                  value={replayCapacity}
                  onChange={(e) => setReplayCapacity(Number(e.target.value))}
                  className="mt-1 w-full rounded-md border border-white/[0.12] bg-[#0B0D10] px-2.5 py-2 font-mono-tabular text-xs text-[#F3F4F6]"
                />
              </div>
            </div>
          </div>

          <div className="flex items-center justify-between border-t border-white/[0.06] pt-3 text-xs text-[#9CA3AF]">
            <span>Learning Rate: {learningRate}</span>
            <input
              aria-label="Learning Rate"
              type="range"
              min={0.02}
              max={0.3}
              step={0.02}
              value={learningRate}
              onChange={(e) => setLearningRate(Number(e.target.value))}
              className="w-40 accent-[#3B82F6]"
            />
          </div>
        </section>

        <section className="lg:col-span-5 rounded-lg border border-white/[0.08] bg-[#12151A] p-5 space-y-4">
          <div className="flex items-center justify-between border-b border-white/[0.06] pb-3">
            <h3 className="text-sm font-semibold text-[#F3F4F6]">Episodic Replay Memory</h3>
            <span className="font-mono-tabular text-xs text-[#9CA3AF]">
              {replay?.count ?? 0} / {replay?.capacity ?? 64} slots
            </span>
          </div>

          <div className="h-2 w-full overflow-hidden rounded-full bg-white/[0.08]">
            <div
              className="h-full bg-[#3B82F6]"
              style={{
                width: `${Math.min(
                  100,
                  ((replay?.count ?? 0) / Math.max(1, replay?.capacity ?? 64)) * 100
                )}%`,
              }}
            />
          </div>

          <div className="max-h-[165px] divide-y divide-white/[0.06] overflow-y-auto text-xs">
            {(replay?.records ?? []).slice(0, 12).map((r) => (
              <div key={r.id} className="flex items-center justify-between py-2">
                <div>
                  <span className="font-mono-tabular font-medium text-[#E5E7EB]">{r.sampleCode}</span>
                  <span className="ml-2 text-[#9CA3AF]">· {r.taskDomain}</span>
                </div>
                <span className="text-[#9CA3AF]">{r.label}</span>
              </div>
            ))}
          </div>
        </section>
      </div>

      {/* Training Progress & Retention Curves */}
      {activeJob && (
        <section className="rounded-lg border border-white/[0.08] bg-[#12151A] p-5 space-y-5">
          <div className="flex flex-wrap items-center justify-between gap-4 border-b border-white/[0.06] pb-3">
            <div className="flex items-center gap-2.5">
              {activeJob.status === 'running' && (
                <Loader2 className="h-4 w-4 animate-spin text-[#60A5FA]" />
              )}
              <h3 className="text-sm font-semibold text-[#F3F4F6]">
                Session {activeJob.id} · {activeJob.datasetName} ({activeJob.strategy})
              </h3>
            </div>
            <div className="font-mono-tabular text-xs text-[#60A5FA]">
              {activeJob.previousVersion} → {activeJob.newVersion ?? activeJob.status}
            </div>
          </div>

          <div className="grid grid-cols-2 gap-4 lg:grid-cols-4">
            <div className="rounded bg-[#0B0D10] p-3">
              <div className="text-xs text-[#9CA3AF]">New Task Val Accuracy</div>
              <div className="mt-1 font-mono-tabular text-lg font-semibold text-[#F3F4F6]">
                {(activeJob.newTaskValidationAccuracy * 100).toFixed(1)}%
              </div>
            </div>
            <div className="rounded bg-[#0B0D10] p-3">
              <div className="text-xs text-[#9CA3AF]">Knowledge Retention</div>
              <div className="mt-1 font-mono-tabular text-lg font-semibold text-emerald-400">
                {(activeJob.knowledgeRetention * 100).toFixed(1)}%
              </div>
            </div>
            <div className="rounded bg-[#0B0D10] p-3">
              <div className="text-xs text-[#9CA3AF]">Forgetting Measure</div>
              <div className="mt-1 font-mono-tabular text-lg font-semibold text-amber-400">
                {(activeJob.forgettingMeasure * 100).toFixed(1)}%
              </div>
            </div>
            <div className="rounded bg-[#0B0D10] p-3">
              <div className="text-xs text-[#9CA3AF]">Prior Task Accuracy</div>
              <div className="mt-1 font-mono-tabular text-lg font-semibold text-[#F3F4F6]">
                {(activeJob.prevTaskAccuracyBefore * 100).toFixed(0)}% →{' '}
                {(activeJob.prevTaskAccuracyAfter * 100).toFixed(0)}%
              </div>
            </div>
          </div>

          {activeJob.lossHistory.length > 0 && (
            <div className="h-[220px] w-full">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={activeJob.lossHistory}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" />
                  <XAxis dataKey="epoch" stroke="#9CA3AF" fontSize={11} />
                  <YAxis stroke="#9CA3AF" fontSize={11} />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: '#12151A',
                      borderColor: 'rgba(255,255,255,0.12)',
                      fontSize: '12px',
                    }}
                  />
                  <Line type="monotone" dataKey="trainLoss" stroke="#3B82F6" strokeWidth={2} dot={false} name="Training Loss" />
                  <Line type="monotone" dataKey="valAccuracy" stroke="#34D399" strokeWidth={2} dot={false} name="Val Accuracy" />
                  <Line type="monotone" dataKey="ewcPenalty" stroke="#A78BFA" strokeWidth={1.5} dot={false} name="EWC Penalty" />
                </LineChart>
              </ResponsiveContainer>
            </div>
          )}
        </section>
      )}

      {/* Version Checkpoints & Rollback + Training History */}
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-12">
        <section className="lg:col-span-6 rounded-lg border border-white/[0.08] bg-[#12151A] p-5 space-y-3">
          <h3 className="text-sm font-semibold text-[#F3F4F6]">
            Model Checkpoints & Version Rollback
          </h3>
          <div className="divide-y divide-white/[0.06] text-xs">
            {versions.map((v) => {
              const isActive = continualModel?.activeVersion === v.version;
              return (
                <div key={v.id} className="flex items-center justify-between gap-3 py-2.5">
                  <div className="min-w-0">
                    <div className="flex items-center gap-2 font-mono-tabular font-semibold text-[#F3F4F6]">
                      <span>{v.version}</span>
                      {isActive && <span className="text-[#60A5FA]">· Active</span>}
                    </div>
                    <div className="truncate text-[11px] text-[#9CA3AF]">
                      {v.strategy} · Val {(v.validationAccuracy * 100).toFixed(1)}% · Retention{' '}
                      {(v.knowledgeRetention * 100).toFixed(1)}%
                    </div>
                  </div>
                  {!isActive && (
                    <button
                      type="button"
                      onClick={() => void handleRollback(v.version)}
                      className="inline-flex items-center gap-1 whitespace-nowrap rounded border border-white/[0.12] px-2.5 py-1 text-[11px] font-medium text-[#E5E7EB] hover:bg-white/[0.06]"
                    >
                      <RotateCcw className="h-3 w-3" />
                      Roll Back
                    </button>
                  )}
                </div>
              );
            })}
          </div>
        </section>

        <section className="lg:col-span-6 rounded-lg border border-white/[0.08] bg-[#12151A] p-5 space-y-3">
          <h3 className="text-sm font-semibold text-[#F3F4F6]">Incremental Training History</h3>
          {history.length === 0 ? (
            <p className="py-6 text-xs text-[#9CA3AF]">
              No incremental training jobs executed yet. Configure a dataset and strategy above to begin.
            </p>
          ) : (
            <div className="divide-y divide-white/[0.06] text-xs">
              {history.map((h) => (
                <button
                  key={h.id}
                  type="button"
                  onClick={() => setActiveJob(h)}
                  className="flex w-full items-center justify-between gap-3 py-2.5 text-left hover:bg-white/[0.02]"
                >
                  <div className="min-w-0">
                    <div className="truncate font-medium text-[#F3F4F6]">
                      {h.datasetName} · {h.strategy}
                    </div>
                    <div className="text-[11px] text-[#9CA3AF]">
                      Val {(h.newTaskValidationAccuracy * 100).toFixed(1)}% · Forgetting{' '}
                      {(h.forgettingMeasure * 100).toFixed(1)}%
                    </div>
                  </div>
                  <span className="shrink-0 font-mono-tabular text-[#60A5FA]">
                    {h.previousVersion} → {h.newVersion ?? h.status}
                  </span>
                </button>
              ))}
            </div>
          )}
        </section>
      </div>
    </div>
  );
}
