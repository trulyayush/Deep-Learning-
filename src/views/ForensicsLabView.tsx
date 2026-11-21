import { useEffect, useState } from 'react';
import {
  AlertTriangle,
  CheckCircle2,
  Download,
  Eye,
  Film,
  Loader2,
  Music,
  Trash2,
  Upload,
} from 'lucide-react';
import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import { api } from '../services/api';
import type { AnalysisJob, ForensicsResult, MediaUpload } from '../types/echomind';

export function ForensicsLabView() {
  const [mediaList, setMediaList] = useState<MediaUpload[]>([]);
  const [selectedUploadId, setSelectedUploadId] = useState<string>('');
  const [results, setResults] = useState<ForensicsResult[]>([]);
  const [activeResult, setActiveResult] = useState<ForensicsResult | null>(null);
  const [activeJob, setActiveJob] = useState<AnalysisJob | null>(null);
  const [configPreset, setConfigPreset] = useState<'balanced' | 'high_sensitivity' | 'audio_focus'>(
    'balanced'
  );
  const [showHeatmaps, setShowHeatmaps] = useState(true);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const loadData = async () => {
    try {
      const [media, forensics] = await Promise.all([api.getMedia(), api.getForensicsList()]);
      setMediaList(media);
      setResults(forensics);
      if (!selectedUploadId && media.length > 0) {
        setSelectedUploadId(media[0].id);
      }
      if (!activeResult && forensics.length > 0) {
        setActiveResult(forensics[0]);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load forensics workspace');
    }
  };

  useEffect(() => {
    void loadData();
  }, []);

  // Poll running job until complete or failed
  useEffect(() => {
    if (!activeJob || activeJob.status === 'completed' || activeJob.status === 'failed') return;
    const timer = setInterval(async () => {
      try {
        const updated = await api.getJob(activeJob.id);
        setActiveJob(updated);
        if (updated.status === 'completed' && updated.resultId) {
          const res = await api.getForensicsResult(updated.resultId);
          setActiveResult(res);
          const list = await api.getForensicsList();
          setResults(list);
        } else if (updated.status === 'failed') {
          setError(updated.error || 'Forensics job failed');
        }
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Job status check failed');
      }
    }, 900);
    return () => clearInterval(timer);
  }, [activeJob]);

  const selectedMedia = mediaList.find((m) => m.id === selectedUploadId) ?? null;

  const handleFileUpload = async (file: File | undefined) => {
    if (!file) return;
    setError(null);
    setUploading(true);
    try {
      const uploaded = await api.uploadMedia(file);
      const list = await api.getMedia();
      setMediaList(list);
      setSelectedUploadId(uploaded.id);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Upload failed');
    } finally {
      setUploading(false);
    }
  };

  const handleRunAnalysis = async () => {
    if (!selectedUploadId) {
      setError('Please select or upload a media file first.');
      return;
    }
    setError(null);
    try {
      const { jobId } = await api.startForensics(selectedUploadId, configPreset);
      const initialJob = await api.getJob(jobId);
      setActiveJob(initialJob);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Could not start forensics job');
    }
  };

  const handleDeleteRawMedia = async (id: string) => {
    setError(null);
    try {
      await api.deleteMedia(id);
      const list = await api.getMedia();
      setMediaList(list);
      if (selectedUploadId === id) {
        setSelectedUploadId(list[0]?.id ?? '');
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Could not delete raw file');
    }
  };

  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <h2 className="text-lg font-semibold tracking-tight text-[#F3F4F6]">
            01. Multimodal Deepfake Forensics
          </h2>
          <p className="text-sm text-[#9CA3AF]">
            Inspect visual frames, acoustic bands, and audiovisual alignment for synthetic manipulation indicators.
          </p>
        </div>

        <div className="flex items-center gap-3">
          <label className="cursor-pointer whitespace-nowrap rounded-md border border-white/[0.12] bg-[#12151A] px-4 py-2 text-xs font-medium text-[#F3F4F6] hover:bg-[#181C23] transition-colors">
            <span className="inline-flex items-center gap-2">
              <Upload className="h-3.5 w-3.5" />
              {uploading ? 'Validating Media...' : 'Upload Video / Audio'}
            </span>
            <input
              type="file"
              accept=".mp4,.mov,.avi,.mkv,.webm,.wav,.mp3,.m4a,.ogg"
              className="sr-only"
              onChange={(e) => void handleFileUpload(e.target.files?.[0])}
            />
          </label>
          <button
            type="button"
            onClick={() => void handleRunAnalysis()}
            disabled={!selectedUploadId || activeJob?.status === 'queued' || activeJob?.status === 'processing'}
            className="whitespace-nowrap rounded-md bg-[#3B82F6] px-4 py-2 text-xs font-medium text-white hover:bg-[#2563EB] disabled:opacity-50 transition-colors"
          >
            {activeJob?.status === 'processing' || activeJob?.status === 'queued'
              ? 'Analyzing Media...'
              : 'Run Forensics Analysis'}
          </button>
        </div>
      </div>

      {error && (
        <div
          role="alert"
          className="flex items-start justify-between gap-4 rounded-lg border border-red-500/30 bg-red-950/20 px-4 py-3 text-xs text-red-200"
        >
          <div>{error}</div>
          <button
            type="button"
            onClick={() => setError(null)}
            className="text-red-300 underline hover:text-white"
          >
            Dismiss
          </button>
        </div>
      )}

      {/* Media Selection & Preview */}
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-12">
        <section className="lg:col-span-5 rounded-lg border border-white/[0.08] bg-[#12151A] p-5 space-y-4">
          <div className="flex items-center justify-between border-b border-white/[0.06] pb-3">
            <h3 className="text-sm font-semibold text-[#F3F4F6]">Media Queue & Probes</h3>
            <span className="font-mono-tabular text-xs text-[#9CA3AF]">
              {mediaList.length} files
            </span>
          </div>

          <div className="max-h-[250px] space-y-1.5 overflow-y-auto pr-1">
            {mediaList.map((m) => (
              <div
                key={m.id}
                className={`flex items-center justify-between gap-2 rounded-md border px-3 py-2 text-left transition-colors ${
                  selectedUploadId === m.id
                    ? 'border-[#3B82F6] bg-[#3B82F6]/10'
                    : 'border-white/[0.06] bg-[#0F1217] hover:border-white/[0.14]'
                }`}
              >
                <button
                  type="button"
                  onClick={() => setSelectedUploadId(m.id)}
                  className="flex min-w-0 flex-1 items-center gap-2.5 text-left"
                >
                  {m.mediaType === 'video' ? (
                    <Film className="h-4 w-4 shrink-0 text-[#60A5FA]" />
                  ) : (
                    <Music className="h-4 w-4 shrink-0 text-[#A78BFA]" />
                  )}
                  <div className="min-w-0">
                    <div className="truncate text-xs font-medium text-[#F3F4F6]">
                      {m.originalName}
                    </div>
                    <div className="font-mono-tabular text-[11px] text-[#9CA3AF]">
                      {m.durationSec}s · {m.hasVideo ? `${m.width}×${m.height}` : 'Audio-only'}
                      {m.isSample ? ' · Sample Probe' : ''}
                    </div>
                  </div>
                </button>
                {!m.isSample && (
                  <button
                    type="button"
                    aria-label={`Delete raw media ${m.originalName}`}
                    onClick={() => void handleDeleteRawMedia(m.id)}
                    className="rounded p-1 text-[#9CA3AF] hover:bg-white/[0.08] hover:text-red-400"
                  >
                    <Trash2 className="h-3.5 w-3.5" />
                  </button>
                )}
              </div>
            ))}
          </div>

          <div className="space-y-2 border-t border-white/[0.06] pt-3">
            <label htmlFor="forensics-preset" className="block text-xs font-medium text-[#9CA3AF]">
              Analysis Configuration Profile
            </label>
            <select
              id="forensics-preset"
              value={configPreset}
              onChange={(e) => setConfigPreset(e.target.value as typeof configPreset)}
              className="w-full rounded-md border border-white/[0.12] bg-[#0B0D10] px-3 py-2 text-xs text-[#F3F4F6] focus:border-[#3B82F6] focus:outline-none"
            >
              <option value="balanced">Balanced Multimodal (Visual + Acoustic + AV Sync)</option>
              <option value="high_sensitivity">High-Sensitivity Facial Artifact Inspection</option>
              <option value="audio_focus">Acoustic Vocoder & Spectral Band Focus</option>
            </select>
          </div>
        </section>

        <section className="lg:col-span-7 rounded-lg border border-white/[0.08] bg-[#12151A] p-5 space-y-4">
          <div className="flex items-center justify-between border-b border-white/[0.06] pb-3">
            <h3 className="text-sm font-semibold text-[#F3F4F6]">Media Preview & Stream Metadata</h3>
            {selectedMedia && (
              <span className="font-mono-tabular text-xs text-[#9CA3AF]">
                {selectedMedia.videoCodec || selectedMedia.audioCodec || 'media'} ·{' '}
                {(selectedMedia.sizeBytes / 1024).toFixed(0)} KB
              </span>
            )}
          </div>

          {selectedMedia ? (
            <div className="space-y-3">
              {selectedMedia.hasVideo ? (
                <video
                  key={selectedMedia.previewUrl}
                  src={selectedMedia.previewUrl}
                  controls
                  className="max-h-[250px] w-full rounded-md border border-white/[0.08] bg-black object-contain"
                />
              ) : (
                <div className="rounded-md border border-white/[0.08] bg-[#0B0D10] p-6">
                  <div className="mb-3 text-xs text-[#9CA3AF]">
                    Audio-Only Input ({selectedMedia.originalName}) — Visual & AV sync heads will remain inactive without crashing.
                  </div>
                  <audio key={selectedMedia.previewUrl} src={selectedMedia.previewUrl} controls className="w-full" />
                </div>
              )}

              <div className="grid grid-cols-2 gap-3 text-xs sm:grid-cols-4">
                <div className="rounded bg-[#0B0D10] px-3 py-2">
                  <div className="text-[#9CA3AF]">Duration</div>
                  <div className="mt-0.5 font-mono-tabular font-medium text-[#F3F4F6]">
                    {selectedMedia.durationSec}s
                  </div>
                </div>
                <div className="rounded bg-[#0B0D10] px-3 py-2">
                  <div className="text-[#9CA3AF]">Resolution</div>
                  <div className="mt-0.5 font-mono-tabular font-medium text-[#F3F4F6]">
                    {selectedMedia.hasVideo ? `${selectedMedia.width}×${selectedMedia.height}` : 'N/A'}
                  </div>
                </div>
                <div className="rounded bg-[#0B0D10] px-3 py-2">
                  <div className="text-[#9CA3AF]">Frame Rate</div>
                  <div className="mt-0.5 font-mono-tabular font-medium text-[#F3F4F6]">
                    {selectedMedia.hasVideo ? `${selectedMedia.fps} fps` : 'N/A'}
                  </div>
                </div>
                <div className="rounded bg-[#0B0D10] px-3 py-2">
                  <div className="text-[#9CA3AF]">Sample Rate</div>
                  <div className="mt-0.5 font-mono-tabular font-medium text-[#F3F4F6]">
                    {selectedMedia.sampleRate ? `${selectedMedia.sampleRate} Hz` : 'No Audio'}
                  </div>
                </div>
              </div>
            </div>
          ) : (
            <div className="py-12 text-center text-sm text-[#9CA3AF]">
              Select or upload a media file to inspect streams.
            </div>
          )}
        </section>
      </div>

      {/* Job Progress Banner */}
      {activeJob && (activeJob.status === 'queued' || activeJob.status === 'processing') && (
        <div className="rounded-lg border border-[#3B82F6]/40 bg-[#12151A] p-4">
          <div className="flex items-center justify-between text-xs">
            <span className="inline-flex items-center gap-2 font-medium text-[#F3F4F6]">
              <Loader2 className="h-3.5 w-3.5 animate-spin text-[#60A5FA]" />
              {activeJob.stage}
            </span>
            <span className="font-mono-tabular text-[#9CA3AF]">{activeJob.progress}%</span>
          </div>
          <div className="mt-2 h-1.5 w-full overflow-hidden rounded-full bg-white/[0.08]">
            <div
              className="h-full bg-[#3B82F6] transition-all duration-300"
              style={{ width: `${activeJob.progress}%` }}
            />
          </div>
        </div>
      )}

      {/* Results Area */}
      {activeResult ? (
        <div className="space-y-6">
          {/* Authenticity Assessment Bar */}
          <section className="rounded-lg border border-white/[0.08] bg-[#12151A] p-5 space-y-4">
            <div className="flex flex-wrap items-center justify-between gap-4 border-b border-white/[0.06] pb-4">
              <div>
                <div className="text-xs text-[#9CA3AF]">
                  Analysis {activeResult.analysisId} · {activeResult.filename}
                </div>
                <div className="mt-1 flex items-center gap-2.5">
                  {activeResult.overallLabel === 'LIKELY AUTHENTIC' ? (
                    <CheckCircle2 className="h-5 w-5 text-emerald-400" />
                  ) : (
                    <AlertTriangle className="h-5 w-5 text-amber-400" />
                  )}
                  <h3 className="text-base font-semibold text-[#F3F4F6]">
                    {activeResult.overallLabel}
                  </h3>
                  <span className="font-mono-tabular text-sm text-[#60A5FA]">
                    {(activeResult.overallAuthenticityScore * 100).toFixed(1)}%
                  </span>
                </div>
              </div>

              <div className="flex items-center gap-3">
                {results.length > 1 && (
                  <select
                    aria-label="Select historical forensics analysis"
                    value={activeResult.analysisId}
                    onChange={(e) => {
                      const found = results.find((r) => r.analysisId === e.target.value);
                      if (found) setActiveResult(found);
                    }}
                    className="rounded-md border border-white/[0.12] bg-[#0B0D10] px-3 py-1.5 text-xs text-[#F3F4F6]"
                  >
                    {results.map((r) => (
                      <option key={r.analysisId} value={r.analysisId}>
                        {r.filename} ({r.overallLabel})
                      </option>
                    ))}
                  </select>
                )}
                <a
                  href={`/api/reports/${encodeURIComponent(activeResult.reportId)}/download`}
                  className="inline-flex items-center gap-1.5 whitespace-nowrap rounded-md border border-white/[0.12] bg-[#0B0D10] px-3.5 py-1.5 text-xs font-medium text-[#F3F4F6] hover:bg-white/[0.06]"
                >
                  <Download className="h-3.5 w-3.5" />
                  Download Report
                </a>
              </div>
            </div>

            <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
              <div className="rounded-md bg-[#0B0D10] p-4">
                <div className="text-xs text-[#9CA3AF]">Visual Authenticity</div>
                <div className="mt-1 font-mono-tabular text-xl font-semibold text-[#F3F4F6]">
                  {activeResult.visualAuthenticityScore !== null
                    ? `${(activeResult.visualAuthenticityScore * 100).toFixed(1)}%`
                    : 'N/A (No Video)'}
                </div>
              </div>
              <div className="rounded-md bg-[#0B0D10] p-4">
                <div className="text-xs text-[#9CA3AF]">Audio Authenticity</div>
                <div className="mt-1 font-mono-tabular text-xl font-semibold text-[#F3F4F6]">
                  {activeResult.audioAuthenticityScore !== null
                    ? `${(activeResult.audioAuthenticityScore * 100).toFixed(1)}%`
                    : 'N/A (No Audio)'}
                </div>
              </div>
              <div className="rounded-md bg-[#0B0D10] p-4">
                <div className="text-xs text-[#9CA3AF]">AV Cross-Modal Consistency</div>
                <div className="mt-1 font-mono-tabular text-xl font-semibold text-[#F3F4F6]">
                  {activeResult.avConsistencyScore !== null
                    ? `${(activeResult.avConsistencyScore * 100).toFixed(1)}%`
                    : 'N/A (Single Modality)'}
                </div>
              </div>
            </div>

            <p className="text-xs leading-relaxed text-[#9CA3AF]">
              <strong className="text-[#E5E7EB]">Assessment Note:</strong> {activeResult.confidenceNote}{' '}
              {activeResult.limitationsNotice}
            </p>
          </section>

          {/* Waveform & Spectral Energy */}
          {activeResult.waveform.length > 0 && (
            <div className="grid grid-cols-1 gap-6 lg:grid-cols-12">
              <section className="lg:col-span-7 rounded-lg border border-white/[0.08] bg-[#12151A] p-5">
                <h4 className="mb-3 text-sm font-semibold text-[#F3F4F6]">
                  Audio RMS Waveform & Transient Envelope
                </h4>
                <div className="h-[220px] w-full">
                  <ResponsiveContainer width="100%" height="100%">
                    <AreaChart data={activeResult.waveform}>
                      <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" />
                      <XAxis dataKey="timeSec" stroke="#9CA3AF" fontSize={11} tickFormatter={(v) => `${v}s`} />
                      <YAxis stroke="#9CA3AF" fontSize={11} />
                      <Tooltip
                        contentStyle={{
                          backgroundColor: '#12151A',
                          borderColor: 'rgba(255,255,255,0.12)',
                          fontSize: '12px',
                        }}
                      />
                      <Area
                        type="monotone"
                        dataKey="amplitude"
                        stroke="#3B82F6"
                        fill="#3B82F6"
                        fillOpacity={0.22}
                        name="Peak Amplitude"
                      />
                      <Area
                        type="monotone"
                        dataKey="rmsEnergy"
                        stroke="#8B5CF6"
                        fill="#8B5CF6"
                        fillOpacity={0.18}
                        name="RMS Energy"
                      />
                    </AreaChart>
                  </ResponsiveContainer>
                </div>
              </section>

              <section className="lg:col-span-5 rounded-lg border border-white/[0.08] bg-[#12151A] p-5">
                <h4 className="mb-3 text-sm font-semibold text-[#F3F4F6]">
                  Acoustic Spectral Band Energy (dB)
                </h4>
                <div className="h-[220px] w-full">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={activeResult.spectralBands}>
                      <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" />
                      <XAxis dataKey="band" stroke="#9CA3AF" fontSize={10} />
                      <YAxis stroke="#9CA3AF" fontSize={11} domain={[-72, 0]} />
                      <Tooltip
                        contentStyle={{
                          backgroundColor: '#12151A',
                          borderColor: 'rgba(255,255,255,0.12)',
                          fontSize: '12px',
                        }}
                      />
                      <Bar dataKey="energyDb" fill="#3B82F6" name="Measured (dB)" radius={[3, 3, 0, 0]} />
                      <Bar dataKey="baselineDb" fill="#4B5563" name="Speech Baseline (dB)" radius={[3, 3, 0, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </section>
            </div>
          )}

          {/* Extracted Frame Gallery + Grad-CAM Saliency Overlays */}
          {activeResult.extractedFrames.length > 0 && (
            <section className="rounded-lg border border-white/[0.08] bg-[#12151A] p-5 space-y-4">
              <div className="flex items-center justify-between border-b border-white/[0.06] pb-3">
                <div>
                  <h4 className="text-sm font-semibold text-[#F3F4F6]">
                    Extracted Frame Gallery & Saliency Attribution
                  </h4>
                  <p className="text-xs text-[#9CA3AF]">
                    Frames extracted via FFmpeg with localized saliency overlays and temporal delta scores.
                  </p>
                </div>
                <button
                  type="button"
                  onClick={() => setShowHeatmaps((prev) => !prev)}
                  className="inline-flex items-center gap-1.5 rounded-md border border-white/[0.12] bg-[#0B0D10] px-3 py-1.5 text-xs font-medium text-[#F3F4F6] hover:bg-white/[0.06]"
                >
                  <Eye className="h-3.5 w-3.5" />
                  {showHeatmaps ? 'Hide Saliency Overlay' : 'Show Saliency Overlay'}
                </button>
              </div>

              <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
                {activeResult.extractedFrames.map((frame) => (
                  <div
                    key={frame.index}
                    className="overflow-hidden rounded-md border border-white/[0.08] bg-[#0B0D10]"
                  >
                    <div className="relative aspect-video bg-black">
                      <img
                        src={frame.url}
                        alt={`Extracted frame at ${frame.timestampSec}s`}
                        referrerPolicy="no-referrer"
                        className="h-full w-full object-cover"
                      />
                      {showHeatmaps && frame.heatmapRegion && (
                        <div
                          aria-label={frame.heatmapRegion.label}
                          style={{
                            left: `${frame.heatmapRegion.x * 100}%`,
                            top: `${frame.heatmapRegion.y * 100}%`,
                            width: `${frame.heatmapRegion.w * 100}%`,
                            height: `${frame.heatmapRegion.h * 100}%`,
                          }}
                          className="pointer-events-none absolute rounded border border-amber-400/80 bg-radial from-red-500/45 via-amber-500/20 to-transparent"
                        />
                      )}
                    </div>
                    <div className="flex items-center justify-between px-3 py-2 text-[11px] text-[#9CA3AF]">
                      <span className="font-mono-tabular text-[#E5E7EB]">
                        t = {frame.timestampSec}s
                      </span>
                      <span className="font-mono-tabular">
                        Sharp {frame.sharpnessScore} · Anomaly {(frame.anomalyScore * 100).toFixed(0)}%
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            </section>
          )}

          {/* Suspicious Segments Table */}
          <section className="rounded-lg border border-white/[0.08] bg-[#12151A] p-5 space-y-3">
            <h4 className="text-sm font-semibold text-[#F3F4F6]">
              Timestamped Segment Inspection
            </h4>
            {activeResult.suspiciousSegments.length === 0 ? (
              <p className="text-xs text-[#9CA3AF]">
                No anomalous timestamps flagged above threshold.
              </p>
            ) : (
              <div className="overflow-x-auto">
                <table className="w-full text-left text-xs">
                  <thead>
                    <tr className="border-b border-white/[0.08] text-[#9CA3AF]">
                      <th className="py-2 pr-4 font-medium">Window</th>
                      <th className="py-2 pr-4 font-medium">Modality</th>
                      <th className="py-2 pr-4 font-medium">Severity</th>
                      <th className="py-2 pr-4 font-medium">Anomaly</th>
                      <th className="py-2 font-medium">Observation</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-white/[0.06]">
                    {activeResult.suspiciousSegments.map((seg) => (
                      <tr key={seg.id}>
                        <td className="py-2.5 pr-4 font-mono-tabular whitespace-nowrap text-[#E5E7EB]">
                          {seg.startSec}s – {seg.endSec}s
                        </td>
                        <td className="py-2.5 pr-4 capitalize text-[#9CA3AF]">{seg.modality}</td>
                        <td className="py-2.5 pr-4 capitalize text-[#F3F4F6]">{seg.severity}</td>
                        <td className="py-2.5 pr-4 font-mono-tabular text-[#60A5FA]">
                          {(seg.anomalyScore * 100).toFixed(0)}%
                        </td>
                        <td className="py-2.5 text-[#D1D5DB]">{seg.summary}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </section>
        </div>
      ) : (
        <section className="rounded-lg border border-white/[0.08] bg-[#12151A] p-8 text-center">
          <h3 className="text-sm font-semibold text-[#F3F4F6]">Ready for Forensics Inspection</h3>
          <p className="mx-auto mt-1 max-w-lg text-xs text-[#9CA3AF]">
            Select one of the verified sample probes above or upload an MP4, MOV, AVI, WAV, or MP3 file, then click Run Forensics Analysis to inspect frame saliency, waveforms, and authenticity indicators.
          </p>
        </section>
      )}
    </div>
  );
}
