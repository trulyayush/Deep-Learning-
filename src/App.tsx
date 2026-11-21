import { useEffect, useState } from 'react';
import {
  Activity,
  Brain,
  Cpu,
  Database,
  History,
  LayoutGrid,
  Menu,
  ScanFace,
  Settings,
  ShieldOff,
  Smile,
  X,
} from 'lucide-react';
import { api, type OverviewPayload } from './services/api';
import type { NavSection } from './types/echomind';
import { OverviewView } from './views/OverviewView';
import { ForensicsLabView } from './views/ForensicsLabView';
import { EmotionLabView } from './views/EmotionLabView';
import { LearningLabView } from './views/LearningLabView';
import { UnlearningLabView } from './views/UnlearningLabView';
import { ArchitectureView } from './views/ArchitectureView';
import { ModelsView } from './views/ModelsView';
import { HistoryView } from './views/HistoryView';
import { SettingsView } from './views/SettingsView';

const NAV_ITEMS: { id: NavSection; label: string; icon: typeof LayoutGrid }[] = [
  { id: 'overview', label: 'Overview', icon: LayoutGrid },
  { id: 'forensics', label: 'Forensics Lab', icon: ScanFace },
  { id: 'emotion', label: 'Emotion Intelligence', icon: Smile },
  { id: 'learning', label: 'Learning Laboratory', icon: Brain },
  { id: 'unlearning', label: 'Unlearning Laboratory', icon: ShieldOff },
  { id: 'architecture', label: 'Neural Architecture', icon: Cpu },
  { id: 'models', label: 'Model Management', icon: Database },
  { id: 'history', label: 'Activity History', icon: History },
  { id: 'settings', label: 'Settings', icon: Settings },
];

export default function App() {
  const [section, setSection] = useState<NavSection>('overview');
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const [overviewData, setOverviewData] = useState<OverviewPayload | null>(null);
  const [loadingOverview, setLoadingOverview] = useState(true);
  const [overviewError, setOverviewError] = useState<string | null>(null);

  const fetchOverview = async () => {
    setLoadingOverview(true);
    setOverviewError(null);
    try {
      const data = await api.getOverview();
      setOverviewData(data);
    } catch (err) {
      setOverviewError(err instanceof Error ? err.message : 'Failed to connect to backend');
    } finally {
      setLoadingOverview(false);
    }
  };

  useEffect(() => {
    void fetchOverview();
  }, [section]);

  const currentNav = NAV_ITEMS.find((n) => n.id === section) ?? NAV_ITEMS[0];

  const handleSelectSection = (next: NavSection) => {
    setSection(next);
    setMobileMenuOpen(false);
  };

  return (
    <div className="min-h-screen bg-[#0B0D10] text-[#F3F4F6] lg:flex">
      {/* Desktop Sidebar */}
      <aside className="hidden w-[248px] shrink-0 flex-col border-r border-white/[0.08] bg-[#0E1116] lg:flex">
        <div className="flex h-14 items-center border-b border-white/[0.08] px-5">
          <span className="font-display text-base font-bold tracking-tight text-white whitespace-nowrap">
            ECHO-MIND
          </span>
        </div>

        <nav aria-label="Primary Navigation" className="flex-1 space-y-1 p-3">
          {NAV_ITEMS.map((item) => {
            const Icon = item.icon;
            const active = section === item.id;
            return (
              <button
                key={item.id}
                type="button"
                onClick={() => handleSelectSection(item.id)}
                className={`flex w-full items-center gap-3 rounded-md px-3 py-2.5 text-left text-[13px] font-medium whitespace-nowrap transition-colors ${
                  active
                    ? 'bg-[#3B82F6]/15 text-white'
                    : 'text-[#9CA3AF] hover:bg-white/[0.04] hover:text-[#F3F4F6]'
                }`}
              >
                <Icon className={`h-4 w-4 shrink-0 ${active ? 'text-[#60A5FA]' : 'text-[#6B7280]'}`} />
                <span className="truncate">{item.label}</span>
              </button>
            );
          })}
        </nav>

        <div className="border-t border-white/[0.08] px-5 py-3 text-[11px] text-[#6B7280]">
          Adaptive Multimodal Intelligence
        </div>
      </aside>

      {/* Mobile Drawer */}
      {mobileMenuOpen && (
        <div className="fixed inset-0 z-50 flex lg:hidden">
          <div
            className="fixed inset-0 bg-black/70"
            onClick={() => setMobileMenuOpen(false)}
          />
          <aside className="relative z-10 flex w-64 flex-col border-r border-white/[0.08] bg-[#0E1116]">
            <div className="flex h-14 items-center justify-between border-b border-white/[0.08] px-4">
              <span className="font-display text-base font-bold text-white">ECHO-MIND</span>
              <button
                type="button"
                aria-label="Close navigation menu"
                onClick={() => setMobileMenuOpen(false)}
                className="rounded p-1.5 text-[#9CA3AF] hover:text-white"
              >
                <X className="h-5 w-5" />
              </button>
            </div>
            <nav aria-label="Mobile Navigation" className="flex-1 space-y-1 p-3">
              {NAV_ITEMS.map((item) => {
                const Icon = item.icon;
                const active = section === item.id;
                return (
                  <button
                    key={item.id}
                    type="button"
                    onClick={() => handleSelectSection(item.id)}
                    className={`flex w-full items-center gap-3 rounded-md px-3 py-2.5 text-left text-sm font-medium ${
                      active ? 'bg-[#3B82F6]/15 text-white' : 'text-[#9CA3AF] hover:text-white'
                    }`}
                  >
                    <Icon className="h-4 w-4 shrink-0" />
                    <span>{item.label}</span>
                  </button>
                );
              })}
            </nav>
          </aside>
        </div>
      )}

      {/* Main Content Workspace */}
      <div className="flex min-w-0 flex-1 flex-col">
        {/* Top Bar Contract: Breadcrumb/Title — Context — Primary Action */}
        <header className="sticky top-0 z-30 flex h-14 items-center justify-between gap-4 border-b border-white/[0.08] bg-[#0B0D10] px-4 sm:px-6">
          <div className="flex min-w-0 items-center gap-3">
            <button
              type="button"
              aria-label="Open navigation menu"
              onClick={() => setMobileMenuOpen(true)}
              className="rounded p-1.5 text-[#9CA3AF] hover:bg-white/[0.06] hover:text-white lg:hidden"
            >
              <Menu className="h-5 w-5" />
            </button>
            <h1 className="truncate text-sm font-semibold text-[#F3F4F6] whitespace-nowrap">
              Workstation / {currentNav.label}
            </h1>
          </div>

          <div className="flex items-center gap-4 shrink-0">
            <span className="hidden sm:inline-flex items-center gap-1.5 font-mono-tabular text-xs text-[#9CA3AF] whitespace-nowrap">
              <Activity className="h-3.5 w-3.5 text-[#60A5FA]" />
              {overviewData
                ? `${overviewData.counts.operationalModels}/${overviewData.counts.registeredModels} models ready`
                : 'Connecting'}
            </span>
            <button
              type="button"
              onClick={() => handleSelectSection('forensics')}
              className="whitespace-nowrap shrink-0 rounded-md bg-[#3B82F6] px-3.5 py-1.5 text-xs font-medium text-white hover:bg-[#2563EB] transition-colors"
            >
              New Analysis
            </button>
          </div>
        </header>

        <main className="mx-auto w-full max-w-[1260px] flex-1 px-4 py-6 sm:px-6 lg:px-8">
          {section === 'overview' && (
            <OverviewView
              data={overviewData}
              loading={loadingOverview}
              error={overviewError}
              onRetry={() => void fetchOverview()}
              onNavigate={handleSelectSection}
            />
          )}
          {section === 'forensics' && <ForensicsLabView />}
          {section === 'emotion' && <EmotionLabView />}
          {section === 'learning' && <LearningLabView />}
          {section === 'unlearning' && <UnlearningLabView />}
          {section === 'architecture' && <ArchitectureView onNavigate={handleSelectSection} />}
          {section === 'models' && <ModelsView />}
          {section === 'history' && <HistoryView />}
          {section === 'settings' && <SettingsView />}
        </main>
      </div>
    </div>
  );
}
