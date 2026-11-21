import { ArrowRight, Download, ShieldAlert, ShieldCheck } from 'lucide-react';
import type { OverviewPayload } from '../services/api';
import type { NavSection } from '../types/echomind';
import { NeuralNetworkCanvas } from '../components/NeuralNetworkCanvas';

interface OverviewViewProps {
  data: OverviewPayload | null;
  loading: boolean;
  error: string | null;
  onRetry: () => void;
  onNavigate: (section: NavSection) => void;
}

export function OverviewView({
  data,
  loading,
  error,
  onRetry,
  onNavigate,
}: OverviewViewProps) {
  if (loading && !data) {
    return (
      <div className="space-y-6">
        <div className="h-[430px] animate-pulse rounded-lg border border-white/[0.08] bg-[#12151A]" />
        <div className="grid grid-cols-2 gap-4 lg:grid-cols-4">
          {[0, 1, 2, 3].map((i) => (
            <div
              key={i}
              className="h-24 animate-pulse rounded-lg border border-white/[0.08] bg-[#12151A]"
            />
          ))}
        </div>
      </div>
    );
  }

  if (error && !data) {
    return (
      <div className="rounded-lg border border-red-500/30 bg-[#12151A] p-6">
        <h2 className="text-base font-semibold text-white">Unable to Load Platform Overview</h2>
        <p className="mt-1 text-sm text-[#9CA3AF]">{error}</p>
        <button
          type="button"
          onClick={onRetry}
          className="mt-4 rounded-md bg-[#3B82F6] px-4 py-2 text-xs font-medium text-white hover:bg-[#2563EB]"
        >
          Retry Connection
        </button>
      </div>
    );
  }

  if (!data) return null;

  return (
    <div className="space-y-8">
      {/* 3D Neural Architecture Visualisation */}
      <section className="space-y-3">
        <div className="flex flex-wrap items-end justify-between gap-4">
          <div>
            <h2 className="text-lg font-semibold tracking-tight text-[#F3F4F6]">
              Adaptive Multimodal Core
            </h2>
            <p className="text-sm text-[#9CA3AF]">
              Modality-specific encoders feeding a shared cross-attention representation and isolated task heads.
            </p>
          </div>
          <div className="flex items-center gap-2">
            <button
              type="button"
              onClick={() => onNavigate('forensics')}
              className="whitespace-nowrap rounded-md bg-[#3B82F6] px-4 py-2 text-xs font-medium text-white hover:bg-[#2563EB] transition-colors"
            >
              Launch Forensics Scan
            </button>
            <button
              type="button"
              onClick={() => onNavigate('emotion')}
              className="whitespace-nowrap rounded-md border border-white/[0.12] bg-[#12151A] px-4 py-2 text-xs font-medium text-[#F3F4F6] hover:bg-[#181C23] transition-colors"
            >
              Emotion Lab
            </button>
          </div>
        </div>

        <NeuralNetworkCanvas onNavigate={onNavigate} />
      </section>

      {/* Primary KPI Strip */}
      <section className="grid grid-cols-2 gap-4 lg:grid-cols-4">
        <div className="rounded-lg border border-white/[0.08] bg-[#12151A] p-4">
          <div className="text-xs text-[#9CA3AF]">Completed Analyses</div>
          <div className="mt-2 font-mono-tabular text-2xl font-semibold text-[#F3F4F6]">
            {data.counts.forensicsAnalyses + data.counts.emotionAnalyses}
          </div>
          <div className="mt-1 text-xs text-[#9CA3AF]">
            {data.counts.forensicsAnalyses} forensics · {data.counts.emotionAnalyses} emotion
          </div>
        </div>

        <div className="rounded-lg border border-white/[0.08] bg-[#12151A] p-4">
          <div className="text-xs text-[#9CA3AF]">Registered Models</div>
          <div className="mt-2 font-mono-tabular text-2xl font-semibold text-[#F3F4F6]">
            {data.counts.operationalModels} / {data.counts.registeredModels}
          </div>
          <div className="mt-1 text-xs text-[#9CA3AF]">Operational checkpoints & adapters</div>
        </div>

        <div className="rounded-lg border border-white/[0.08] bg-[#12151A] p-4">
          <div className="text-xs text-[#9CA3AF]">Continual Learning</div>
          <div className="mt-2 font-mono-tabular text-2xl font-semibold text-[#F3F4F6]">
            {data.counts.trainingRuns}
          </div>
          <div className="mt-1 text-xs text-[#9CA3AF]">
            {data.counts.approvedDatasets} datasets · {data.counts.replaySamples} replay vectors
          </div>
        </div>

        <div className="rounded-lg border border-white/[0.08] bg-[#12151A] p-4">
          <div className="text-xs text-[#9CA3AF]">Unlearning Audits</div>
          <div className="mt-2 font-mono-tabular text-2xl font-semibold text-[#F3F4F6]">
            {data.counts.unlearningAudits}
          </div>
          <div className="mt-1 text-xs text-[#9CA3AF]">Verified forget-set operations</div>
        </div>
      </section>

      {/* Latest Analyses & Model Health */}
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-12">
        <section className="lg:col-span-7 rounded-lg border border-white/[0.08] bg-[#12151A] p-5">
          <div className="flex items-center justify-between border-b border-white/[0.06] pb-3">
            <h3 className="text-sm font-semibold text-[#F3F4F6]">Recent Media Analyses</h3>
            <button
              type="button"
              onClick={() => onNavigate('forensics')}
              className="inline-flex items-center gap-1 text-xs font-medium text-[#60A5FA] hover:text-white"
            >
              Open Lab <ArrowRight className="h-3.5 w-3.5" />
            </button>
          </div>

          {data.recentForensics.length === 0 && data.recentEmotion.length === 0 ? (
            <div className="py-10 text-center">
              <p className="text-sm text-[#9CA3AF]">
                No completed media analyses yet. Run a Forensics or Emotion job using the built-in samples or an uploaded clip.
              </p>
              <button
                type="button"
                onClick={() => onNavigate('forensics')}
                className="mt-4 rounded-md bg-[#3B82F6] px-4 py-2 text-xs font-medium text-white hover:bg-[#2563EB]"
              >
                Run First Forensics Analysis
              </button>
            </div>
          ) : (
            <div className="divide-y divide-white/[0.06]">
              {data.recentForensics.map((item) => (
                <div key={item.analysisId} className="flex items-center justify-between gap-4 py-3">
                  <div className="min-w-0">
                    <div className="truncate text-sm font-medium text-[#F3F4F6]">
                      {item.filename}
                    </div>
                    <div className="mt-0.5 text-xs text-[#9CA3AF]">
                      Forensics · {item.mediaType} ·{' '}
                      {new Date(item.createdAt).toLocaleTimeString()}
                    </div>
                  </div>
                  <div className="flex shrink-0 items-center gap-3 text-right">
                    <span className="inline-flex items-center gap-1.5 text-xs font-medium text-[#E5E7EB]">
                      {item.overallLabel === 'LIKELY AUTHENTIC' ? (
                        <ShieldCheck className="h-3.5 w-3.5 text-emerald-400" />
                      ) : (
                        <ShieldAlert className="h-3.5 w-3.5 text-amber-400" />
                      )}
                      {item.overallLabel}
                    </span>
                    <span className="font-mono-tabular text-xs text-[#9CA3AF]">
                      {(item.overallAuthenticityScore * 100).toFixed(1)}%
                    </span>
                  </div>
                </div>
              ))}
              {data.recentEmotion.map((item) => (
                <div key={item.analysisId} className="flex items-center justify-between gap-4 py-3">
                  <div className="min-w-0">
                    <div className="truncate text-sm font-medium text-[#F3F4F6]">
                      {item.filename}
                    </div>
                    <div className="mt-0.5 text-xs text-[#9CA3AF]">
                      Emotion Intelligence · {Math.round(item.disagreementRate * 100)}% modality disagreement
                    </div>
                  </div>
                  <div className="shrink-0 text-right">
                    <span className="text-xs font-medium text-[#A78BFA]">{item.dominantEmotion}</span>
                    <span className="ml-3 font-mono-tabular text-xs text-[#9CA3AF]">
                      {(item.overallConfidence * 100).toFixed(0)}%
                    </span>
                  </div>
                </div>
              ))}
            </div>
          )}
        </section>

        <section className="lg:col-span-5 rounded-lg border border-white/[0.08] bg-[#12151A] p-5">
          <div className="flex items-center justify-between border-b border-white/[0.06] pb-3">
            <h3 className="text-sm font-semibold text-[#F3F4F6]">Model Registry Health</h3>
            <button
              type="button"
              onClick={() => onNavigate('models')}
              className="inline-flex items-center gap-1 text-xs font-medium text-[#60A5FA] hover:text-white"
            >
              Manage <ArrowRight className="h-3.5 w-3.5" />
            </button>
          </div>
          <div className="divide-y divide-white/[0.06]">
            {data.models.map((m) => (
              <div key={m.id} className="flex items-start justify-between gap-3 py-3">
                <div className="min-w-0">
                  <div className="truncate text-sm font-medium text-[#F3F4F6]">{m.name}</div>
                  <div className="mt-0.5 truncate text-xs text-[#9CA3AF]">{m.healthNote}</div>
                </div>
                <div className="shrink-0 text-right">
                  <div className="font-mono-tabular text-xs text-[#E5E7EB]">{m.activeVersion}</div>
                  <div
                    className={`mt-0.5 text-[11px] font-medium ${
                      m.status === 'operational'
                        ? 'text-emerald-400'
                        : m.status === 'unavailable'
                          ? 'text-amber-400'
                          : 'text-[#60A5FA]'
                    }`}
                  >
                    {m.status}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </section>
      </div>

      {/* Learning, Unlearning & Reports Strip */}
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
        <section className="rounded-lg border border-white/[0.08] bg-[#12151A] p-5">
          <div className="flex items-center justify-between border-b border-white/[0.06] pb-3">
            <h3 className="text-sm font-semibold text-[#F3F4F6]">Learning Activity</h3>
            <button
              type="button"
              onClick={() => onNavigate('learning')}
              className="text-xs font-medium text-[#60A5FA] hover:text-white"
            >
              Open Lab
            </button>
          </div>
          {data.recentTraining.length === 0 ? (
            <p className="py-6 text-xs text-[#9CA3AF]">
              No incremental training sessions recorded yet. Visit Learning Laboratory to run EWC or Experience Replay.
            </p>
          ) : (
            <div className="divide-y divide-white/[0.06]">
              {data.recentTraining.map((t) => (
                <div key={t.id} className="py-2.5">
                  <div className="flex items-center justify-between text-xs font-medium text-[#F3F4F6]">
                    <span className="truncate">{t.datasetName}</span>
                    <span className="font-mono-tabular text-[#60A5FA]">
                      {t.previousVersion} → {t.newVersion ?? t.status}
                    </span>
                  </div>
                  <div className="mt-0.5 text-[11px] text-[#9CA3AF]">
                    {t.strategy} · {t.status}
                  </div>
                </div>
              ))}
            </div>
          )}
        </section>

        <section className="rounded-lg border border-white/[0.08] bg-[#12151A] p-5">
          <div className="flex items-center justify-between border-b border-white/[0.06] pb-3">
            <h3 className="text-sm font-semibold text-[#F3F4F6]">Unlearning Activity</h3>
            <button
              type="button"
              onClick={() => onNavigate('unlearning')}
              className="text-xs font-medium text-[#60A5FA] hover:text-white"
            >
              Open Lab
            </button>
          </div>
          {data.recentUnlearning.length === 0 ? (
            <p className="py-6 text-xs text-[#9CA3AF]">
              No selective unlearning operations run yet. Select samples in Unlearning Laboratory to execute SISA or Exact Retrain.
            </p>
          ) : (
            <div className="divide-y divide-white/[0.06]">
              {data.recentUnlearning.map((u) => (
                <div key={u.id} className="py-2.5">
                  <div className="flex items-center justify-between text-xs font-medium text-[#F3F4F6]">
                    <span className="truncate">{u.datasetName}</span>
                    <span className="font-mono-tabular text-amber-400">
                      {u.originalVersion} → {u.updatedVersion ?? u.status}
                    </span>
                  </div>
                  <div className="mt-0.5 text-[11px] text-[#9CA3AF]">
                    {u.strategy} · Retain{' '}
                    {u.retainAccAfter !== null ? `${(u.retainAccAfter * 100).toFixed(1)}%` : 'pending'}
                  </div>
                </div>
              ))}
            </div>
          )}
        </section>

        <section className="rounded-lg border border-white/[0.08] bg-[#12151A] p-5">
          <div className="flex items-center justify-between border-b border-white/[0.06] pb-3">
            <h3 className="text-sm font-semibold text-[#F3F4F6]">Recent Reports & Jobs</h3>
            <button
              type="button"
              onClick={() => onNavigate('history')}
              className="text-xs font-medium text-[#60A5FA] hover:text-white"
            >
              All Logs
            </button>
          </div>
          {data.recentReports.length === 0 ? (
            <p className="py-6 text-xs text-[#9CA3AF]">
              Downloadable JSON reports appear automatically after Forensics, Emotion, or Unlearning runs.
            </p>
          ) : (
            <div className="divide-y divide-white/[0.06]">
              {data.recentReports.map((rep) => (
                <div key={rep.id} className="flex items-center justify-between gap-2 py-2.5">
                  <div className="min-w-0">
                    <div className="truncate text-xs font-medium text-[#F3F4F6]">{rep.title}</div>
                    <div className="truncate text-[11px] text-[#9CA3AF]">{rep.summary}</div>
                  </div>
                  <a
                    href={`/api/reports/${encodeURIComponent(rep.id)}/download`}
                    aria-label={`Download ${rep.title}`}
                    className="shrink-0 rounded border border-white/[0.1] p-1.5 text-[#9CA3AF] hover:bg-white/[0.06] hover:text-white"
                  >
                    <Download className="h-3.5 w-3.5" />
                  </a>
                </div>
              ))}
            </div>
          )}
        </section>
      </div>
    </div>
  );
}
