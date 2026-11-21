import { useEffect, useState } from 'react';
import { AlertTriangle, CheckCircle2, Download, ShieldOff } from 'lucide-react';
import { api } from '../services/api';
import type {
  DatasetSample,
  TrainingDataset,
  UnlearningRequestRecord,
  UnlearningStrategy,
} from '../types/echomind';

export function UnlearningLabView() {
  const [datasets, setDatasets] = useState<TrainingDataset[]>([]);
  const [selectedDatasetId, setSelectedDatasetId] = useState('');
  const [samples, setSamples] = useState<DatasetSample[]>([]);
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set());
  const [strategy, setStrategy] = useState<UnlearningStrategy>('sisa_sharded');
  const [purgeCachedFeatures, setPurgeCachedFeatures] = useState(false);
  const [purgeReplayMemory, setPurgeReplayMemory] = useState(true);
  const [purgeRawMedia, setPurgeRawMedia] = useState(false);
  const [confirmOpen, setConfirmOpen] = useState(false);

  const [requests, setRequests] = useState<UnlearningRequestRecord[]>([]);
  const [activeRequest, setActiveRequest] = useState<UnlearningRequestRecord | null>(null);
  const [error, setError] = useState<string | null>(null);

  const loadWorkspace = async (dsId?: string) => {
    try {
      const [dsList, reqList] = await Promise.all([api.getDatasets(), api.getUnlearningRequests()]);
      setDatasets(dsList);
      setRequests(reqList);
      const targetDs = dsId || selectedDatasetId || dsList[0]?.id || '';
      if (targetDs) {
        setSelectedDatasetId(targetDs);
        const smps = await api.getDatasetSamples(targetDs);
        setSamples(smps);
      }
      if (!activeRequest && reqList.length > 0) {
        setActiveRequest(reqList[0]);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load unlearning workspace');
    }
  };

  useEffect(() => {
    void loadWorkspace();
  }, []);

  const handleDatasetChange = async (dsId: string) => {
    setSelectedDatasetId(dsId);
    setSelectedIds(new Set());
    try {
      const smps = await api.getDatasetSamples(dsId);
      setSamples(smps);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load samples');
    }
  };

  const toggleSample = (id: string) => {
    setSelectedIds((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  };

  useEffect(() => {
    if (!activeRequest || activeRequest.status === 'completed' || activeRequest.status === 'failed') {
      return;
    }
    const timer = setInterval(async () => {
      try {
        const updated = await api.getUnlearningRequest(activeRequest.id);
        setActiveRequest(updated);
        if (updated.status === 'completed' || updated.status === 'failed') {
          await loadWorkspace(selectedDatasetId);
        }
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Unlearning poll failed');
      }
    }, 500);
    return () => clearInterval(timer);
  }, [activeRequest, selectedDatasetId]);

  const handleExecuteUnlearning = async () => {
    setConfirmOpen(false);
    setError(null);
    try {
      const { requestId } = await api.startUnlearning({
        modelId: 'mod-continual-head',
        datasetId: selectedDatasetId,
        sampleIds: Array.from(selectedIds),
        strategy,
        purgeRawMedia,
        purgeCachedFeatures,
        purgeReplayMemory,
      });
      setSelectedIds(new Set());
      const req = await api.getUnlearningRequest(requestId);
      setActiveRequest(req);
      if (req.status === 'completed' || req.status === 'failed') {
        await loadWorkspace(selectedDatasetId);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to execute unlearning');
    }
  };

  const selectedSamples = samples.filter((s) => selectedIds.has(s.id));
  const replayAffectedCount = selectedSamples.filter((s) => s.inReplayMemory).length;
  const affectedShardSet = Array.from(new Set(selectedSamples.map((s) => s.shardIndex))).sort();

  return (
    <div className="space-y-8">
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div className="min-w-0 flex-1 max-w-2xl">
          <h2 className="text-lg font-semibold tracking-tight text-[#F3F4F6]">
            04. Selective Machine Unlearning Engine
          </h2>
          <p className="text-sm text-[#9CA3AF]">
            Select specific training samples, build a forget-set manifest, scrub replay/cached entries, and execute SISA, Exact Retrain, or Gradient Ascent unlearning.
          </p>
        </div>

        <button
          type="button"
          disabled={selectedIds.size === 0}
          onClick={() => setConfirmOpen(true)}
          className="inline-flex shrink-0 items-center gap-2 whitespace-nowrap rounded-md bg-[#3B82F6] px-4 py-2 text-xs font-medium text-white hover:bg-[#2563EB] disabled:opacity-50"
        >
          <ShieldOff className="h-3.5 w-3.5" />
          Configure & Unlearn ({selectedIds.size} selected)
        </button>
      </div>

      {error && (
        <div role="alert" className="rounded-lg border border-red-500/30 bg-red-950/20 px-4 py-3 text-xs text-red-200">
          {error}
        </div>
      )}

      {/* Confirmation Dialog */}
      {confirmOpen && (
        <div
          role="dialog"
          aria-modal="true"
          aria-labelledby="unlearn-confirm-title"
          className="rounded-lg border border-amber-500/40 bg-[#12151A] p-5 space-y-4"
        >
          <div className="flex items-start gap-3">
            <AlertTriangle className="mt-0.5 h-5 w-5 shrink-0 text-amber-400" />
            <div>
              <h3 id="unlearn-confirm-title" className="text-sm font-semibold text-[#F3F4F6]">
                Confirm Selective Machine Unlearning Operation
              </h3>
              <p className="mt-1 text-xs leading-relaxed text-[#9CA3AF]">
                This operation will produce a new model checkpoint attempting to remove the influence of{' '}
                <strong className="text-[#F3F4F6]">{selectedIds.size}</strong> sample(s) using{' '}
                <strong className="text-[#F3F4F6]">{strategy}</strong>.
                {strategy === 'gradient_ascent' &&
                  ' Note: Approximate gradient unlearning reduces sample influence and does not guarantee exact forgetting.'}
              </p>
              <div className="mt-2 font-mono-tabular text-[11px] text-[#9CA3AF]">
                Replay entries affected: {replayAffectedCount} · SISA Shards impacted: [
                {affectedShardSet.join(', ')}] · Purge Cached Features:{' '}
                {purgeCachedFeatures ? 'Yes' : 'No'} · Purge Replay: {purgeReplayMemory ? 'Yes' : 'No'}
              </div>
            </div>
          </div>
          <div className="flex justify-end gap-2">
            <button
              type="button"
              onClick={() => setConfirmOpen(false)}
              className="rounded-md border border-white/[0.12] px-3.5 py-1.5 text-xs text-[#9CA3AF] hover:text-white"
            >
              Cancel
            </button>
            <button
              type="button"
              onClick={() => void handleExecuteUnlearning()}
              className="rounded-md bg-amber-600 px-4 py-1.5 text-xs font-medium text-white hover:bg-amber-500"
            >
              Confirm & Execute Unlearning
            </button>
          </div>
        </div>
      )}

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-12">
        {/* Dataset Browser & Sample Selection */}
        <section className="lg:col-span-7 rounded-lg border border-white/[0.08] bg-[#12151A] p-5 space-y-4">
          <div className="flex flex-wrap items-center justify-between gap-3 border-b border-white/[0.06] pb-3">
            <h3 className="text-sm font-semibold text-[#F3F4F6]">Registered Dataset Samples</h3>
            <select
              aria-label="Select dataset for unlearning"
              value={selectedDatasetId}
              onChange={(e) => void handleDatasetChange(e.target.value)}
              className="rounded-md border border-white/[0.12] bg-[#0B0D10] px-3 py-1.5 text-xs text-[#F3F4F6]"
            >
              {datasets.map((d) => (
                <option key={d.id} value={d.id}>
                  {d.name} ({d.sampleCount} samples)
                </option>
              ))}
            </select>
          </div>

          <div className="max-h-[340px] overflow-y-auto">
            <table className="w-full text-left text-xs">
              <thead className="sticky top-0 bg-[#12151A] text-[#9CA3AF]">
                <tr className="border-b border-white/[0.08]">
                  <th className="py-2 pr-3 font-medium">Select</th>
                  <th className="py-2 pr-3 font-medium">Sample ID</th>
                  <th className="py-2 pr-3 font-medium">Label</th>
                  <th className="py-2 pr-3 font-medium">Split</th>
                  <th className="py-2 pr-3 font-medium">Shard</th>
                  <th className="py-2 font-medium">Replay Memory</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-white/[0.06]">
                {samples.map((s) => (
                  <tr key={s.id} className="hover:bg-white/[0.02]">
                    <td className="py-2 pr-3">
                      <input
                        type="checkbox"
                        aria-label={`Select sample ${s.sampleCode}`}
                        checked={selectedIds.has(s.id)}
                        onChange={() => toggleSample(s.id)}
                        className="rounded border-white/20 bg-[#0B0D10]"
                      />
                    </td>
                    <td className="py-2 pr-3 font-mono-tabular font-medium text-[#F3F4F6]">
                      {s.sampleCode}
                    </td>
                    <td className="py-2 pr-3 text-[#D1D5DB]">{s.label}</td>
                    <td className="py-2 pr-3 text-[#9CA3AF]">{s.split}</td>
                    <td className="py-2 pr-3 font-mono-tabular text-[#9CA3AF]">#{s.shardIndex}</td>
                    <td className="py-2 text-[#9CA3AF]">
                      {s.inReplayMemory ? 'Stored in Replay' : 'Not Cached'}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>

        {/* Forget-Set Manifest & Strategy Selector */}
        <section className="lg:col-span-5 rounded-lg border border-white/[0.08] bg-[#12151A] p-5 space-y-4">
          <div className="border-b border-white/[0.06] pb-3">
            <h3 className="text-sm font-semibold text-[#F3F4F6]">
              Forget-Set Manifest & Scope
            </h3>
            <p className="mt-0.5 text-xs text-[#9CA3AF]">
              Explicitly separates file/cache deletion from model parameter unlearning.
            </p>
          </div>

          <div className="space-y-3 text-xs">
            <div className="rounded bg-[#0B0D10] p-3 space-y-1">
              <div className="flex justify-between">
                <span className="text-[#9CA3AF]">Selected Samples:</span>
                <span className="font-mono-tabular text-[#F3F4F6]">{selectedIds.size}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-[#9CA3AF]">Replay Buffer Entries Affected:</span>
                <span className="font-mono-tabular text-[#F3F4F6]">{replayAffectedCount}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-[#9CA3AF]">SISA Shards Impacted:</span>
                <span className="font-mono-tabular text-[#F3F4F6]">
                  {affectedShardSet.length > 0 ? affectedShardSet.join(', ') : 'None'}
                </span>
              </div>
            </div>

            <div>
              <label htmlFor="unlearn-strategy" className="block text-xs text-[#9CA3AF]">
                Unlearning Algorithm
              </label>
              <select
                id="unlearn-strategy"
                value={strategy}
                onChange={(e) => setStrategy(e.target.value as UnlearningStrategy)}
                className="mt-1 w-full rounded-md border border-white/[0.12] bg-[#0B0D10] px-3 py-2 text-xs text-[#F3F4F6]"
              >
                <option value="sisa_sharded">SISA-Style Shard Retraining (Targeted Shards)</option>
                <option value="exact_retrain">Exact Retraining Without Forget-Set (Baseline)</option>
                <option value="gradient_ascent">Approximate Gradient Ascent + Retain Repair</option>
              </select>
            </div>

            <div className="space-y-2 border-t border-white/[0.06] pt-3">
              <div className="text-[11px] font-medium text-[#9CA3AF]">Data Hygiene Actions</div>
              <label className="flex items-center gap-2 text-xs text-[#E5E7EB]">
                <input
                  type="checkbox"
                  checked={purgeReplayMemory}
                  onChange={(e) => setPurgeReplayMemory(e.target.checked)}
                />
                Remove selected samples from Replay Memory buffer
              </label>
              <label className="flex items-center gap-2 text-xs text-[#E5E7EB]">
                <input
                  type="checkbox"
                  checked={purgeCachedFeatures}
                  onChange={(e) => setPurgeCachedFeatures(e.target.checked)}
                />
                Purge extracted feature vectors from dataset cache
              </label>
              <label className="flex items-center gap-2 text-xs text-[#E5E7EB]">
                <input
                  type="checkbox"
                  checked={purgeRawMedia}
                  onChange={(e) => setPurgeRawMedia(e.target.checked)}
                />
                Flag associated raw media references for deletion
              </label>
            </div>
          </div>
        </section>
      </div>

      {/* Before / After Comparison & Audit Trail */}
      {activeRequest && (
        <section className="rounded-lg border border-white/[0.08] bg-[#12151A] p-5 space-y-4">
          <div className="flex flex-wrap items-center justify-between gap-4 border-b border-white/[0.06] pb-3">
            <div className="flex items-center gap-2">
              <CheckCircle2 className="h-4 w-4 text-emerald-400" />
              <h3 className="text-sm font-semibold text-[#F3F4F6]">
                Unlearning Validation — {activeRequest.id} ({activeRequest.strategy})
              </h3>
            </div>
            <div className="flex items-center gap-3 font-mono-tabular text-xs">
              <span className="text-[#9CA3AF]">
                {activeRequest.originalVersion} → {activeRequest.updatedVersion ?? activeRequest.status}
              </span>
              <span className="text-[#60A5FA]">{activeRequest.executionTimeMs} ms</span>
            </div>
          </div>

          <div className="grid grid-cols-2 gap-4 lg:grid-cols-4">
            <div className="rounded bg-[#0B0D10] p-3">
              <div className="text-xs text-[#9CA3AF]">Forget-Set Accuracy</div>
              <div className="mt-1 font-mono-tabular text-lg font-semibold text-[#F3F4F6]">
                {(activeRequest.forgetAccBefore * 100).toFixed(1)}% →{' '}
                {(activeRequest.forgetAccAfter * 100).toFixed(1)}%
              </div>
            </div>
            <div className="rounded bg-[#0B0D10] p-3">
              <div className="text-xs text-[#9CA3AF]">Retain-Set Accuracy</div>
              <div className="mt-1 font-mono-tabular text-lg font-semibold text-emerald-400">
                {(activeRequest.retainAccBefore * 100).toFixed(1)}% →{' '}
                {(activeRequest.retainAccAfter * 100).toFixed(1)}%
              </div>
            </div>
            <div className="rounded bg-[#0B0D10] p-3">
              <div className="text-xs text-[#9CA3AF]">Overall Model Utility</div>
              <div className="mt-1 font-mono-tabular text-lg font-semibold text-[#60A5FA]">
                {(activeRequest.overallUtilityAfter * 100).toFixed(1)}%
              </div>
            </div>
            <div className="rounded bg-[#0B0D10] p-3">
              <div className="text-xs text-[#9CA3AF]">Samples Forgotten</div>
              <div className="mt-1 font-mono-tabular text-lg font-semibold text-[#F3F4F6]">
                {activeRequest.sampleCodes.length}
              </div>
              <div className="truncate font-mono-tabular text-[11px] text-[#9CA3AF]">
                {activeRequest.sampleCodes.join(', ')}
              </div>
            </div>
          </div>

          <p className="text-xs text-[#9CA3AF]">{activeRequest.notes}</p>
        </section>
      )}

      {/* Immutable Audit Log */}
      <section className="rounded-lg border border-white/[0.08] bg-[#12151A] p-5 space-y-3">
        <div className="flex items-center justify-between border-b border-white/[0.06] pb-3">
          <h3 className="text-sm font-semibold text-[#F3F4F6]">Unlearning Audit History</h3>
          <span className="font-mono-tabular text-xs text-[#9CA3AF]">
            {requests.length} recorded requests
          </span>
        </div>

        {requests.length === 0 ? (
          <p className="py-6 text-xs text-[#9CA3AF]">
            No unlearning operations recorded yet. Select one or more dataset samples above to initiate an unlearning audit.
          </p>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs">
              <thead>
                <tr className="border-b border-white/[0.08] text-[#9CA3AF]">
                  <th className="py-2 pr-4 font-medium">Request ID</th>
                  <th className="py-2 pr-4 font-medium">Strategy</th>
                  <th className="py-2 pr-4 font-medium">Samples</th>
                  <th className="py-2 pr-4 font-medium">Version Delta</th>
                  <th className="py-2 pr-4 font-medium">Retain After</th>
                  <th className="py-2 pr-4 font-medium">Time</th>
                  <th className="py-2 font-medium">Inspect</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-white/[0.06]">
                {requests.map((r) => (
                  <tr key={r.id} className="hover:bg-white/[0.02]">
                    <td className="py-2.5 pr-4 font-mono-tabular text-[#F3F4F6]">{r.id}</td>
                    <td className="py-2.5 pr-4 text-[#D1D5DB]">{r.strategy}</td>
                    <td className="py-2.5 pr-4 font-mono-tabular text-[#9CA3AF]">
                      {r.sampleCodes.length}
                    </td>
                    <td className="py-2.5 pr-4 font-mono-tabular text-[#60A5FA]">
                      {r.originalVersion} → {r.updatedVersion ?? r.status}
                    </td>
                    <td className="py-2.5 pr-4 font-mono-tabular text-emerald-400">
                      {(r.retainAccAfter * 100).toFixed(1)}%
                    </td>
                    <td className="py-2.5 pr-4 font-mono-tabular text-[#9CA3AF]">
                      {r.executionTimeMs} ms
                    </td>
                    <td className="py-2.5">
                      <button
                        type="button"
                        onClick={() => setActiveRequest(r)}
                        className="inline-flex items-center gap-1 text-[#60A5FA] hover:underline"
                      >
                        <Download className="h-3 w-3" />
                        Inspect
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>
    </div>
  );
}
