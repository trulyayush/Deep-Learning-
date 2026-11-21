import { useEffect, useState } from 'react';
import { Download, Loader2, MessageSquare, Split, Upload } from 'lucide-react';
import {
  Bar,
  BarChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import { api } from '../services/api';
import type { AnalysisJob, EmotionResult, MediaUpload } from '../types/echomind';

export function EmotionLabView() {
  const [mediaList, setMediaList] = useState<MediaUpload[]>([]);
  const [selectedUploadId, setSelectedUploadId] = useState('');
  const [transcript, setTranscript] = useState(
    '00:00-00:03 Speaker 1: We have reviewed the telemetry anomaly and need to isolate the affected shard immediately.\n00:03-00:06 Speaker 2: Understood, I am calm about the containment plan, though the latency spike was unexpected.'
  );
  const [results, setResults] = useState<EmotionResult[]>([]);
  const [activeResult, setActiveResult] = useState<EmotionResult | null>(null);
  const [activeJob, setActiveJob] = useState<AnalysisJob | null>(null);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const init = async () => {
      try {
        const [media, emoList] = await Promise.all([api.getMedia(), api.getEmotionList()]);
        setMediaList(media);
        setResults(emoList);
        if (media.length > 0) setSelectedUploadId(media[0].id);
        if (emoList.length > 0) setActiveResult(emoList[0]);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to load emotion intelligence data');
      }
    };
    void init();
  }, []);

  useEffect(() => {
    if (!activeJob || activeJob.status === 'completed' || activeJob.status === 'failed') return;
    const timer = setInterval(async () => {
      try {
        const updated = await api.getJob(activeJob.id);
        setActiveJob(updated);
        if (updated.status === 'completed' && updated.resultId) {
          const res = await api.getEmotionResult(updated.resultId);
          setActiveResult(res);
          const list = await api.getEmotionList();
          setResults(list);
        } else if (updated.status === 'failed') {
          setError(updated.error || 'Emotion analysis failed');
        }
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Polling failed');
      }
    }, 900);
    return () => clearInterval(timer);
  }, [activeJob]);

  const handleUpload = async (file: File | undefined) => {
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

  const handleAnalyze = async () => {
    if (!selectedUploadId) {
      setError('Select or upload a conversation media file first.');
      return;
    }
    setError(null);
    try {
      const { jobId } = await api.startEmotion(selectedUploadId, transcript);
      const job = await api.getJob(jobId);
      setActiveJob(job);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to enqueue emotion job');
    }
  };

  const selectedMedia = mediaList.find((m) => m.id === selectedUploadId) ?? null;

  return (
    <div className="space-y-8">
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div className="min-w-0 flex-1 max-w-2xl">
          <h2 className="text-lg font-semibold tracking-tight text-[#F3F4F6]">
            02. Conversational Emotion Intelligence
          </h2>
          <p className="text-sm text-[#9CA3AF]">
            Probabilistic affective observation across audio prosody, facial expressions, and transcript context with explicit modality disagreement tracking.
          </p>
        </div>
        <div className="flex shrink-0 items-center gap-3">
          <label className="cursor-pointer whitespace-nowrap rounded-md border border-white/[0.12] bg-[#12151A] px-4 py-2 text-xs font-medium text-[#F3F4F6] hover:bg-[#181C23]">
            <span className="inline-flex items-center gap-2">
              <Upload className="h-3.5 w-3.5" />
              {uploading ? 'Uploading...' : 'Upload Conversation'}
            </span>
            <input
              type="file"
              accept=".mp4,.mov,.avi,.webm,.wav,.mp3,.m4a,.ogg"
              className="sr-only"
              onChange={(e) => void handleUpload(e.target.files?.[0])}
            />
          </label>
          <button
            type="button"
            onClick={() => void handleAnalyze()}
            disabled={!selectedUploadId || activeJob?.status === 'queued' || activeJob?.status === 'processing'}
            className="whitespace-nowrap rounded-md bg-[#3B82F6] px-4 py-2 text-xs font-medium text-white hover:bg-[#2563EB] disabled:opacity-50"
          >
            {activeJob?.status === 'processing' || activeJob?.status === 'queued'
              ? 'Analyzing Affect...'
              : 'Analyze Conversation'}
          </button>
        </div>
      </div>

      {error && (
        <div
          role="alert"
          className="flex items-center justify-between rounded-lg border border-red-500/30 bg-red-950/20 px-4 py-3 text-xs text-red-200"
        >
          <span>{error}</span>
          <button type="button" onClick={() => setError(null)} className="underline">
            Dismiss
          </button>
        </div>
      )}

      {/* Input Configuration */}
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-12">
        <section className="lg:col-span-5 rounded-lg border border-white/[0.08] bg-[#12151A] p-5 space-y-4">
          <div>
            <label htmlFor="emotion-media-select" className="block text-xs font-medium text-[#9CA3AF]">
              Target Audio / Video Source
            </label>
            <select
              id="emotion-media-select"
              value={selectedUploadId}
              onChange={(e) => setSelectedUploadId(e.target.value)}
              className="mt-1.5 w-full rounded-md border border-white/[0.12] bg-[#0B0D10] px-3 py-2 text-xs text-[#F3F4F6]"
            >
              {mediaList.map((m) => (
                <option key={m.id} value={m.id}>
                  {m.originalName} ({m.mediaType}, {m.durationSec}s)
                </option>
              ))}
            </select>
          </div>

          {selectedMedia && (
            <div>
              {selectedMedia.hasVideo ? (
                <video
                  key={selectedMedia.previewUrl}
                  src={selectedMedia.previewUrl}
                  controls
                  className="max-h-[190px] w-full rounded-md border border-white/[0.08] bg-black object-contain"
                />
              ) : (
                <audio key={selectedMedia.previewUrl} src={selectedMedia.previewUrl} controls className="w-full" />
              )}
            </div>
          )}
        </section>

        <section className="lg:col-span-7 rounded-lg border border-white/[0.08] bg-[#12151A] p-5 space-y-3">
          <div className="flex items-center justify-between">
            <label htmlFor="emotion-transcript" className="text-sm font-semibold text-[#F3F4F6]">
              Optional Transcript / Subtitles
            </label>
            <button
              type="button"
              onClick={() => setTranscript('')}
              className="text-xs text-[#9CA3AF] hover:text-white"
            >
              Clear (Test Missing Text Modality)
            </button>
          </div>
          <textarea
            id="emotion-transcript"
            rows={7}
            value={transcript}
            onChange={(e) => setTranscript(e.target.value)}
            placeholder="Paste optional speaker transcript or subtitles (leave blank to run audio/visual modalities only)..."
            className="w-full rounded-md border border-white/[0.12] bg-[#0B0D10] p-3 text-xs leading-relaxed text-[#F3F4F6] placeholder-[#6B7280] focus:border-[#3B82F6] focus:outline-none"
          />
          <p className="text-[11px] text-[#9CA3AF]">
            Supports missing modalities: if video or transcript is absent, only active modality encoders execute.
          </p>
        </section>
      </div>

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
              className="h-full bg-[#8B5CF6] transition-all duration-300"
              style={{ width: `${activeJob.progress}%` }}
            />
          </div>
        </div>
      )}

      {activeResult ? (
        <div className="space-y-6">
          <section className="rounded-lg border border-white/[0.08] bg-[#12151A] p-5 space-y-4">
            <div className="flex flex-wrap items-center justify-between gap-4 border-b border-white/[0.06] pb-4">
              <div>
                <div className="text-xs text-[#9CA3AF]">
                  {activeResult.filename} · Modalities: {activeResult.modalitiesUsed.join(', ')}
                </div>
                <div className="mt-1 flex items-center gap-3">
                  <h3 className="text-base font-semibold text-[#F3F4F6]">
                    Dominant Observation: {activeResult.dominantEmotion}
                  </h3>
                  <span className="font-mono-tabular text-sm text-[#A78BFA]">
                    {(activeResult.overallConfidence * 100).toFixed(0)}% confidence
                  </span>
                </div>
              </div>

              <div className="flex items-center gap-3">
                {results.length > 1 && (
                  <select
                    aria-label="Select historical emotion analysis"
                    value={activeResult.analysisId}
                    onChange={(e) => {
                      const found = results.find((r) => r.analysisId === e.target.value);
                      if (found) setActiveResult(found);
                    }}
                    className="rounded-md border border-white/[0.12] bg-[#0B0D10] px-3 py-1.5 text-xs text-[#F3F4F6]"
                  >
                    {results.map((r) => (
                      <option key={r.analysisId} value={r.analysisId}>
                        {r.filename} ({r.dominantEmotion})
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

            <div className="grid grid-cols-1 gap-6 lg:grid-cols-12">
              <div className="lg:col-span-7 space-y-2">
                <div className="flex items-center justify-between text-xs text-[#9CA3AF]">
                  <span>7-Class Affective Distribution</span>
                  <span className="font-mono-tabular">
                    Modality Disagreement Rate: {Math.round(activeResult.disagreementRate * 100)}%
                  </span>
                </div>
                <div className="h-[200px] w-full">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={activeResult.distribution}>
                      <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" />
                      <XAxis dataKey="emotion" stroke="#9CA3AF" fontSize={11} />
                      <YAxis stroke="#9CA3AF" fontSize={11} allowDecimals={false} />
                      <Tooltip
                        contentStyle={{
                          backgroundColor: '#12151A',
                          borderColor: 'rgba(255,255,255,0.12)',
                          fontSize: '12px',
                        }}
                      />
                      <Bar dataKey="count" fill="#8B5CF6" name="Segments" radius={[3, 3, 0, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>

              <div className="lg:col-span-5 flex flex-col justify-between rounded-md bg-[#0B0D10] p-4">
                <div>
                  <div className="text-xs font-semibold text-[#F3F4F6]">
                    Interpretive Boundary Notice
                  </div>
                  <p className="mt-2 text-xs leading-relaxed text-[#9CA3AF]">
                    {activeResult.probabilisticNotice}
                  </p>
                </div>
                <div className="mt-4 border-t border-white/[0.06] pt-3 text-xs text-[#9CA3AF]">
                  Pipeline: <span className="text-[#E5E7EB]">{activeResult.modelPipeline}</span>
                </div>
              </div>
            </div>
          </section>

          {/* Utterance Timeline & Modality Comparison */}
          <section className="rounded-lg border border-white/[0.08] bg-[#12151A] p-5 space-y-4">
            <div className="flex items-center justify-between border-b border-white/[0.06] pb-3">
              <h4 className="text-sm font-semibold text-[#F3F4F6]">
                Utterance Timeline & Modality Comparison
              </h4>
              <span className="text-xs text-[#9CA3AF]">
                Audio, Visual & Text predictions shown independently
              </span>
            </div>

            <div className="divide-y divide-white/[0.06]">
              {activeResult.segments.map((seg) => (
                <div key={seg.id} className="py-4 space-y-3">
                  <div className="flex flex-wrap items-center justify-between gap-2">
                    <div className="flex items-center gap-2.5 text-xs">
                      <MessageSquare className="h-3.5 w-3.5 text-[#60A5FA]" />
                      <span className="font-semibold text-[#F3F4F6]">{seg.speaker}</span>
                      <span className="font-mono-tabular text-[#9CA3AF]">
                        {seg.startSec}s – {seg.endSec}s
                      </span>
                      <span className="font-mono-tabular text-[#9CA3AF]">
                        · Pitch {seg.acousticFeatures.estimatedPitchHz} Hz · RMS {seg.acousticFeatures.rmsEnergy}
                      </span>
                    </div>

                    <div className="flex items-center gap-3">
                      {seg.modalityDisagreement && (
                        <span className="inline-flex items-center gap-1 text-xs font-medium text-amber-400">
                          <Split className="h-3.5 w-3.5" />
                          Modality Disagreement
                        </span>
                      )}
                      <span className="font-mono-tabular text-xs font-semibold text-[#A78BFA]">
                        Combined: {seg.combinedPrediction} ({(seg.combinedConfidence * 100).toFixed(0)}%)
                      </span>
                    </div>
                  </div>

                  <p className="text-xs leading-relaxed text-[#E5E7EB]">“{seg.transcript}”</p>

                  <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
                    <div className="rounded bg-[#0B0D10] p-3">
                      <div className="flex items-center justify-between text-[11px] text-[#9CA3AF]">
                        <span>Audio Modality</span>
                        {seg.audioPrediction && (
                          <span className="font-mono-tabular text-[#60A5FA]">
                            {(seg.audioPrediction.confidence * 100).toFixed(0)}%
                          </span>
                        )}
                      </div>
                      <div className="mt-1 text-xs font-medium text-[#F3F4F6]">
                        {seg.audioPrediction ? seg.audioPrediction.emotion : 'Unavailable'}
                      </div>
                      {seg.audioPrediction && (
                        <p className="mt-1 text-[11px] text-[#9CA3AF]">
                          {seg.audioPrediction.rationale}
                        </p>
                      )}
                    </div>

                    <div className="rounded bg-[#0B0D10] p-3">
                      <div className="flex items-center justify-between text-[11px] text-[#9CA3AF]">
                        <span>Visual Modality</span>
                        {seg.visualPrediction && (
                          <span className="font-mono-tabular text-[#60A5FA]">
                            {(seg.visualPrediction.confidence * 100).toFixed(0)}%
                          </span>
                        )}
                      </div>
                      <div className="mt-1 text-xs font-medium text-[#F3F4F6]">
                        {seg.visualPrediction ? seg.visualPrediction.emotion : 'Unavailable (No Video)'}
                      </div>
                      {seg.visualPrediction && (
                        <p className="mt-1 text-[11px] text-[#9CA3AF]">
                          {seg.visualPrediction.rationale}
                        </p>
                      )}
                    </div>

                    <div className="rounded bg-[#0B0D10] p-3">
                      <div className="flex items-center justify-between text-[11px] text-[#9CA3AF]">
                        <span>Text Modality</span>
                        {seg.textPrediction && (
                          <span className="font-mono-tabular text-[#60A5FA]">
                            {(seg.textPrediction.confidence * 100).toFixed(0)}%
                          </span>
                        )}
                      </div>
                      <div className="mt-1 text-xs font-medium text-[#F3F4F6]">
                        {seg.textPrediction ? seg.textPrediction.emotion : 'Unavailable'}
                      </div>
                      {seg.textPrediction && (
                        <p className="mt-1 text-[11px] text-[#9CA3AF]">
                          {seg.textPrediction.rationale}
                        </p>
                      )}
                    </div>
                  </div>

                  {seg.disagreementNote && (
                    <div className="text-[11px] text-amber-300/90">
                      Divergence Note: {seg.disagreementNote}
                    </div>
                  )}
                </div>
              ))}
            </div>
          </section>
        </div>
      ) : (
        <section className="rounded-lg border border-white/[0.08] bg-[#12151A] p-8 text-center">
          <h3 className="text-sm font-semibold text-[#F3F4F6]">No Emotion Analysis Run Yet</h3>
          <p className="mx-auto mt-1 max-w-lg text-xs text-[#9CA3AF]">
            Select an audio/video probe, supply an optional transcript, and click Analyze Conversation to observe separated audio, visual, and text predictions.
          </p>
        </section>
      )}
    </div>
  );
}
