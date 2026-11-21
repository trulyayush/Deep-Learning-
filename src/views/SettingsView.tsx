import { useEffect, useState } from 'react';
import { api } from '../services/api';
import type { HardwareProfile } from '../types/echomind';

export function SettingsView() {
  const [hardware, setHardware] = useState<HardwareProfile | null>(null);
  const [maxUploadMb, setMaxUploadMb] = useState(100);
  const [externalUrl, setExternalUrl] = useState('');
  const [replayCapacity, setReplayCapacity] = useState(64);
  const [sisaShards, setSisaShards] = useState(4);
  const [pingResult, setPingResult] = useState<string | null>(null);
  const [savedNotice, setSavedNotice] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    Promise.all([api.getHardware(), api.getSettings()])
      .then(([hw, cfg]) => {
        setHardware(hw);
        setMaxUploadMb(cfg.maxUploadMb);
        setExternalUrl(cfg.externalInferenceUrl);
        setReplayCapacity(cfg.replayCapacity);
        setSisaShards(cfg.sisaShards);
      })
      .catch((err) => setError(err instanceof Error ? err.message : 'Failed to load settings'));
  }, []);

  const handleSave = async () => {
    setError(null);
    setSavedNotice(null);
    try {
      const updated = await api.updateSettings({
        maxUploadMb,
        externalInferenceUrl: externalUrl,
        replayCapacity,
        sisaShards,
      });
      setMaxUploadMb(updated.maxUploadMb);
      setExternalUrl(updated.externalInferenceUrl);
      setReplayCapacity(updated.replayCapacity);
      setSisaShards(updated.sisaShards);
      setSavedNotice('Settings saved to persistent SQLite configuration.');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to save settings');
    }
  };

  const handlePing = async () => {
    setPingResult(null);
    setError(null);
    try {
      const res = await api.pingExternalServer(externalUrl);
      setPingResult(res.ok ? `Connected: ${res.message}` : `Unavailable: ${res.message}`);
    } catch (err) {
      setPingResult(err instanceof Error ? err.message : 'Ping failed');
    }
  };

  return (
    <div className="space-y-8">
      <div>
        <h2 className="text-lg font-semibold tracking-tight text-[#F3F4F6]">
          Environment & Hardware Configuration
        </h2>
        <p className="text-sm text-[#9CA3AF]">
          Live startup inspection of local compute resources, execution settings, upload thresholds, and optional external GPU server routing.
        </p>
      </div>

      {error && (
        <div role="alert" className="rounded-lg border border-red-500/30 bg-red-950/20 px-4 py-3 text-xs text-red-200">
          {error}
        </div>
      )}
      {savedNotice && (
        <div className="rounded-lg border border-emerald-500/30 bg-emerald-950/20 px-4 py-3 text-xs text-emerald-200">
          {savedNotice}
        </div>
      )}

      {/* Hardware Telemetry */}
      {hardware && (
        <section className="rounded-lg border border-white/[0.08] bg-[#12151A] p-5 space-y-4">
          <h3 className="text-sm font-semibold text-[#F3F4F6]">Detected Runtime Environment</h3>
          <div className="grid grid-cols-2 gap-4 text-xs sm:grid-cols-3 lg:grid-cols-4">
            <div className="rounded bg-[#0B0D10] p-3">
              <div className="text-[#9CA3AF]">Operating System</div>
              <div className="mt-1 font-mono-tabular text-[#F3F4F6]">
                {hardware.osPlatform} ({hardware.arch})
              </div>
            </div>
            <div className="rounded bg-[#0B0D10] p-3">
              <div className="text-[#9CA3AF]">CPU Cores</div>
              <div className="mt-1 font-mono-tabular text-[#F3F4F6]">
                {hardware.cpuCores} cores
              </div>
            </div>
            <div className="rounded bg-[#0B0D10] p-3">
              <div className="text-[#9CA3AF]">Memory (Free / Total)</div>
              <div className="mt-1 font-mono-tabular text-[#F3F4F6]">
                {hardware.freeMemoryMb} / {hardware.totalMemoryMb} MB
              </div>
            </div>
            <div className="rounded bg-[#0B0D10] p-3">
              <div className="text-[#9CA3AF]">GPU / CUDA</div>
              <div className="mt-1 font-mono-tabular text-[#F3F4F6]">
                {hardware.cudaAvailable ? hardware.gpuName || 'CUDA Ready' : 'CPU Fallback'}
              </div>
            </div>
            <div className="rounded bg-[#0B0D10] p-3">
              <div className="text-[#9CA3AF]">Node / Python</div>
              <div className="mt-1 font-mono-tabular text-[#F3F4F6]">
                {hardware.nodeVersion} · {hardware.pythonVersion || 'N/A'}
              </div>
            </div>
            <div className="rounded bg-[#0B0D10] p-3">
              <div className="text-[#9CA3AF]">FFmpeg Media Engine</div>
              <div className="mt-1 font-mono-tabular text-[#F3F4F6]">
                {hardware.ffmpegAvailable ? 'Available' : 'Missing'}
              </div>
            </div>
            <div className="rounded bg-[#0B0D10] p-3">
              <div className="text-[#9CA3AF]">Disk Storage</div>
              <div className="mt-1 font-mono-tabular text-[#F3F4F6]">
                {hardware.diskFreeSummary}
              </div>
            </div>
            <div className="rounded bg-[#0B0D10] p-3">
              <div className="text-[#9CA3AF]">Recommended Batch / FP16</div>
              <div className="mt-1 font-mono-tabular text-[#F3F4F6]">
                Batch {hardware.recommendedBatchSize} ·{' '}
                {hardware.mixedPrecisionSupported ? 'FP16' : 'FP32'}
              </div>
            </div>
          </div>
          <div className="text-xs text-[#9CA3AF]">
            Active Execution Mode: <span className="text-[#E5E7EB]">{hardware.executionMode}</span>
          </div>
        </section>
      )}

      {/* Editable Platform Settings */}
      <section className="rounded-lg border border-white/[0.08] bg-[#12151A] p-5 space-y-5">
        <h3 className="text-sm font-semibold text-[#F3F4F6]">Execution & Resource Limits</h3>

        <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
          <div>
            <label htmlFor="cfg-upload" className="block text-xs text-[#9CA3AF]">
              Max Media Upload Size (MB)
            </label>
            <input
              id="cfg-upload"
              type="number"
              min={5}
              max={500}
              value={maxUploadMb}
              onChange={(e) => setMaxUploadMb(Number(e.target.value))}
              className="mt-1 w-full rounded-md border border-white/[0.12] bg-[#0B0D10] px-3 py-2 font-mono-tabular text-xs text-[#F3F4F6]"
            />
          </div>

          <div>
            <label htmlFor="cfg-replay" className="block text-xs text-[#9CA3AF]">
              Default Replay Buffer Capacity
            </label>
            <input
              id="cfg-replay"
              type="number"
              min={8}
              max={512}
              value={replayCapacity}
              onChange={(e) => setReplayCapacity(Number(e.target.value))}
              className="mt-1 w-full rounded-md border border-white/[0.12] bg-[#0B0D10] px-3 py-2 font-mono-tabular text-xs text-[#F3F4F6]"
            />
          </div>

          <div>
            <label htmlFor="cfg-shards" className="block text-xs text-[#9CA3AF]">
              SISA Shard Partitions (K)
            </label>
            <input
              id="cfg-shards"
              type="number"
              min={2}
              max={16}
              value={sisaShards}
              onChange={(e) => setSisaShards(Number(e.target.value))}
              className="mt-1 w-full rounded-md border border-white/[0.12] bg-[#0B0D10] px-3 py-2 font-mono-tabular text-xs text-[#F3F4F6]"
            />
          </div>
        </div>

        <div className="space-y-2 border-t border-white/[0.06] pt-4">
          <label htmlFor="cfg-external-url" className="block text-xs text-[#9CA3AF]">
            Optional External FastAPI / PyTorch GPU Server URL
          </label>
          <div className="flex flex-wrap gap-2">
            <input
              id="cfg-external-url"
              type="url"
              value={externalUrl}
              onChange={(e) => setExternalUrl(e.target.value)}
              placeholder="https://gpu-worker.example.com"
              className="min-w-[260px] flex-1 rounded-md border border-white/[0.12] bg-[#0B0D10] px-3 py-2 text-xs text-[#F3F4F6]"
            />
            <button
              type="button"
              onClick={() => void handlePing()}
              className="rounded-md border border-white/[0.12] bg-[#0B0D10] px-3.5 py-2 text-xs font-medium text-[#E5E7EB] hover:bg-white/[0.06]"
            >
              Test Connection
            </button>
          </div>
          {pingResult && <p className="text-xs text-[#9CA3AF]">{pingResult}</p>}
        </div>

        <div className="flex justify-end border-t border-white/[0.06] pt-4">
          <button
            type="button"
            onClick={() => void handleSave()}
            className="rounded-md bg-[#3B82F6] px-4 py-2 text-xs font-medium text-white hover:bg-[#2563EB]"
          >
            Save Configuration
          </button>
        </div>
      </section>
    </div>
  );
}
