import { useEffect, useState } from 'react';
import { Download } from 'lucide-react';
import { api } from '../services/api';
import type { ActivityLogRecord, ReportRecord } from '../types/echomind';

export function HistoryView() {
  const [logs, setLogs] = useState<ActivityLogRecord[]>([]);
  const [reports, setReports] = useState<ReportRecord[]>([]);
  const [categoryFilter, setCategoryFilter] = useState<string>('');
  const [error, setError] = useState<string | null>(null);

  const loadData = async (cat: string) => {
    try {
      const [act, reps] = await Promise.all([api.getActivity(cat || undefined), api.getReports()]);
      setLogs(act);
      setReports(reps);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load activity history');
    }
  };

  useEffect(() => {
    void loadData(categoryFilter);
  }, [categoryFilter]);

  return (
    <div className="space-y-8">
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <h2 className="text-lg font-semibold tracking-tight text-[#F3F4F6]">
            Activity History & Generated Reports
          </h2>
          <p className="text-sm text-[#9CA3AF]">
            Persistent audit log across analyses, dataset registrations, continual learning updates, and selective unlearning operations.
          </p>
        </div>

        <select
          aria-label="Filter activity category"
          value={categoryFilter}
          onChange={(e) => setCategoryFilter(e.target.value)}
          className="rounded-md border border-white/[0.12] bg-[#12151A] px-3 py-2 text-xs text-[#F3F4F6]"
        >
          <option value="">All Categories</option>
          <option value="forensics">Forensics</option>
          <option value="emotion">Emotion</option>
          <option value="learning">Continual Learning</option>
          <option value="unlearning">Machine Unlearning</option>
          <option value="model">Model Registry</option>
          <option value="system">System</option>
        </select>
      </div>

      {error && (
        <div role="alert" className="rounded-lg border border-red-500/30 bg-red-950/20 px-4 py-3 text-xs text-red-200">
          {error}
        </div>
      )}

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-12">
        <section className="lg:col-span-7 rounded-lg border border-white/[0.08] bg-[#12151A] p-5 space-y-3">
          <h3 className="text-sm font-semibold text-[#F3F4F6]">Event Log</h3>
          <div className="max-h-[480px] divide-y divide-white/[0.06] overflow-y-auto text-xs">
            {logs.map((log) => (
              <div key={log.id} className="py-3">
                <div className="flex items-center justify-between gap-2">
                  <span className="font-medium text-[#F3F4F6]">
                    {log.action} · <span className="capitalize text-[#9CA3AF]">{log.category}</span>
                  </span>
                  <span className="font-mono-tabular text-[11px] text-[#9CA3AF]">
                    {new Date(log.createdAt).toLocaleString()}
                  </span>
                </div>
                <p className="mt-1 text-[#9CA3AF]">{log.detail}</p>
              </div>
            ))}
          </div>
        </section>

        <section className="lg:col-span-5 rounded-lg border border-white/[0.08] bg-[#12151A] p-5 space-y-3">
          <h3 className="text-sm font-semibold text-[#F3F4F6]">Downloadable Reports Archive</h3>
          {reports.length === 0 ? (
            <p className="py-8 text-xs text-[#9CA3AF]">
              No analysis or unlearning reports generated yet.
            </p>
          ) : (
            <div className="max-h-[480px] divide-y divide-white/[0.06] overflow-y-auto text-xs">
              {reports.map((rep) => (
                <div key={rep.id} className="flex items-start justify-between gap-3 py-3">
                  <div className="min-w-0">
                    <div className="truncate font-medium text-[#F3F4F6]">{rep.title}</div>
                    <div className="mt-0.5 text-[11px] text-[#9CA3AF]">{rep.summary}</div>
                    <div className="mt-0.5 font-mono-tabular text-[11px] text-[#6B7280]">
                      {new Date(rep.createdAt).toLocaleString()}
                    </div>
                  </div>
                  <a
                    href={`/api/reports/${encodeURIComponent(rep.id)}/download`}
                    aria-label={`Download report ${rep.title}`}
                    className="shrink-0 rounded border border-white/[0.12] p-1.5 text-[#9CA3AF] hover:bg-white/[0.06] hover:text-white"
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
