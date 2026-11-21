import { useEffect, useState } from 'react';
import { Activity, RotateCcw } from 'lucide-react';
import { api } from '../services/api';
import type { ModelRecord, ModelVersionRecord } from '../types/echomind';

export function ModelsView() {
  const [models, setModels] = useState<ModelRecord[]>([]);
  const [selectedModelId, setSelectedModelId] = useState<string>('mod-continual-head');
  const [versions, setVersions] = useState<ModelVersionRecord[]>([]);
  const [notice, setNotice] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const loadModels = async () => {
    try {
      const list = await api.getModels();
      setModels(list);
      if (selectedModelId) {
        const vers = await api.getModelVersions(selectedModelId);
        setVersions(vers);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load model registry');
    }
  };

  useEffect(() => {
    void loadModels();
  }, [selectedModelId]);

  const handleHealthCheck = async (modelId: string) => {
    setError(null);
    setNotice(null);
    try {
      const res = await api.healthCheckModel(modelId);
      setNotice(`Health check (${res.id}): ${res.status} — ${res.healthNote}`);
      await loadModels();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Health check failed');
    }
  };

  const handleRollback = async (modelId: string, version: string) => {
    setError(null);
    setNotice(null);
    try {
      await api.rollbackModel(modelId, version);
      setNotice(`Rolled back ${modelId} to ${version}.`);
      await loadModels();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Rollback failed');
    }
  };

  return (
    <div className="space-y-8">
      <div>
        <h2 className="text-lg font-semibold tracking-tight text-[#F3F4F6]">
          Central Model Registry
        </h2>
        <p className="text-sm text-[#9CA3AF]">
          Inspect registered encoders, task heads, checkpoint locations, health status, and version rollbacks. Unavailable checkpoints are reported explicitly.
        </p>
      </div>

      {error && (
        <div role="alert" className="rounded-lg border border-red-500/30 bg-red-950/20 px-4 py-3 text-xs text-red-200">
          {error}
        </div>
      )}
      {notice && (
        <div className="rounded-lg border border-[#3B82F6]/30 bg-[#12151A] px-4 py-3 text-xs text-[#E5E7EB]">
          {notice}
        </div>
      )}

      <section className="rounded-lg border border-white/[0.08] bg-[#12151A] p-5">
        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs">
            <thead>
              <tr className="border-b border-white/[0.08] text-[#9CA3AF]">
                <th className="py-2 pr-4 font-medium">Model</th>
                <th className="py-2 pr-4 font-medium">Type</th>
                <th className="py-2 pr-4 font-medium">Version</th>
                <th className="py-2 pr-4 font-medium">Status</th>
                <th className="py-2 pr-4 font-medium">Eval</th>
                <th className="py-2 font-medium">Actions</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-white/[0.06]">
              {models.map((m) => (
                <tr key={m.id} className="hover:bg-white/[0.02]">
                  <td className="py-3 pr-4">
                    <button
                      type="button"
                      onClick={() => setSelectedModelId(m.id)}
                      className="text-left font-medium text-[#F3F4F6] hover:text-[#60A5FA]"
                    >
                      {m.name}
                    </button>
                    <div className="mt-0.5 text-[11px] text-[#9CA3AF]">{m.architecture}</div>
                  </td>
                  <td className="py-3 pr-4 font-mono-tabular text-[#9CA3AF]">{m.modelType}</td>
                  <td className="py-3 pr-4 font-mono-tabular text-[#E5E7EB]">{m.activeVersion}</td>
                  <td className="py-3 pr-4">
                    <span
                      className={`font-medium ${
                        m.status === 'operational'
                          ? 'text-emerald-400'
                          : m.status === 'unavailable'
                            ? 'text-amber-400'
                            : 'text-[#60A5FA]'
                      }`}
                    >
                      {m.status}
                    </span>
                    <div className="mt-0.5 max-w-xs truncate text-[11px] text-[#9CA3AF]">
                      {m.healthNote}
                    </div>
                  </td>
                  <td className="py-3 pr-4 font-mono-tabular text-[#E5E7EB]">
                    {m.lastEvalAccuracy !== null
                      ? `${(m.lastEvalAccuracy * 100).toFixed(1)}%`
                      : '—'}
                  </td>
                  <td className="py-3">
                    <button
                      type="button"
                      onClick={() => void handleHealthCheck(m.id)}
                      className="inline-flex items-center gap-1 whitespace-nowrap rounded border border-white/[0.12] px-2.5 py-1 text-[11px] text-[#E5E7EB] hover:bg-white/[0.06]"
                    >
                      <Activity className="h-3 w-3" />
                      Health Check
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      {/* Selected Model Details & Version Checkpoints */}
      {(() => {
        const current = models.find((m) => m.id === selectedModelId);
        if (!current) return null;
        return (
          <section className="rounded-lg border border-white/[0.08] bg-[#12151A] p-5 space-y-4">
            <div className="border-b border-white/[0.06] pb-3">
              <h3 className="text-sm font-semibold text-[#F3F4F6]">
                {current.name} — Specification & Version History
              </h3>
              <p className="mt-0.5 font-mono-tabular text-xs text-[#9CA3AF]">
                ID: {current.id} · Checkpoint: {current.checkpointLocation}
              </p>
            </div>

            <div className="grid grid-cols-1 gap-4 text-xs sm:grid-cols-2">
              <div className="rounded bg-[#0B0D10] p-3">
                <div className="text-[#9CA3AF]">Input Requirements</div>
                <div className="mt-1 text-[#E5E7EB]">{current.inputRequirements}</div>
              </div>
              <div className="rounded bg-[#0B0D10] p-3">
                <div className="text-[#9CA3AF]">Output Requirements</div>
                <div className="mt-1 text-[#E5E7EB]">{current.outputRequirements}</div>
              </div>
            </div>

            {versions.length === 0 ? (
              <p className="text-xs text-[#9CA3AF]">
                This model uses a managed multimodal adapter without local incremental weight versions.
              </p>
            ) : (
              <div className="divide-y divide-white/[0.06] text-xs">
                {versions.map((v) => (
                  <div key={v.id} className="flex items-center justify-between gap-4 py-2.5">
                    <div>
                      <span className="font-mono-tabular font-semibold text-[#F3F4F6]">
                        {v.version}
                      </span>
                      <span className="ml-2 text-[#9CA3AF]">
                        ({v.strategy}) · Val {(v.validationAccuracy * 100).toFixed(1)}% · Retention{' '}
                        {(v.knowledgeRetention * 100).toFixed(1)}%
                      </span>
                    </div>
                    {current.activeVersion !== v.version ? (
                      <button
                        type="button"
                        onClick={() => void handleRollback(current.id, v.version)}
                        className="inline-flex items-center gap-1 rounded border border-white/[0.12] px-2.5 py-1 text-[11px] text-[#E5E7EB] hover:bg-white/[0.06]"
                      >
                        <RotateCcw className="h-3 w-3" />
                        Roll Back
                      </button>
                    ) : (
                      <span className="font-mono-tabular text-[11px] text-[#60A5FA]">Active</span>
                    )}
                  </div>
                ))}
              </div>
            )}
          </section>
        );
      })()}
    </div>
  );
}
