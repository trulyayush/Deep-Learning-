import { useEffect, useState } from 'react';
import { ArrowRight } from 'lucide-react';
import { api } from '../services/api';
import type { NavSection, SystemArchitectureSpec } from '../types/echomind';
import { NeuralNetworkCanvas } from '../components/NeuralNetworkCanvas';

interface ArchitectureViewProps {
  onNavigate: (section: NavSection) => void;
}

export function ArchitectureView({ onNavigate }: ArchitectureViewProps) {
  const [spec, setSpec] = useState<SystemArchitectureSpec | null>(null);
  const [selectedStageId, setSelectedStageId] = useState<string>('fusion');
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api
      .getArchitecture()
      .then(setSpec)
      .catch((err) => setError(err instanceof Error ? err.message : 'Failed to load architecture'));
  }, []);

  if (error) {
    return (
      <div className="rounded-lg border border-red-500/30 bg-[#12151A] p-5 text-xs text-red-200">
        {error}
      </div>
    );
  }

  if (!spec) {
    return <div className="h-[480px] animate-pulse rounded-lg border border-white/[0.08] bg-[#12151A]" />;
  }

  return (
    <div className="space-y-8">
      <div>
        <h2 className="text-lg font-semibold tracking-tight text-[#F3F4F6]">
          Shared Multimodal Neural Architecture
        </h2>
        <p className="text-sm text-[#9CA3AF]">
          Live configuration topology showing independent modality encoders, cross-attention fusion, and decoupled task heads.
        </p>
      </div>

      {/* Interactive Architectural Pipeline Map */}
      <section className="rounded-lg border border-white/[0.08] bg-[#12151A] p-5 space-y-6">
        <div className="grid grid-cols-1 gap-6 lg:grid-cols-11 lg:items-center">
          {/* Column 1: Modality Encoders */}
          <div className="lg:col-span-3 space-y-3">
            <div className="text-xs font-semibold text-[#9CA3AF]">1. Modality Encoders</div>
            {spec.encoders.map((enc) => (
              <button
                key={enc.id}
                type="button"
                onClick={() => setSelectedStageId(enc.id)}
                className={`w-full rounded-md border p-3 text-left transition-colors ${
                  selectedStageId === enc.id
                    ? 'border-[#3B82F6] bg-[#3B82F6]/10'
                    : 'border-white/[0.08] bg-[#0B0D10] hover:border-white/[0.2]'
                }`}
              >
                <div className="flex items-center justify-between text-xs font-medium text-[#F3F4F6]">
                  <span>{enc.name}</span>
                  <span className="font-mono-tabular text-[11px] text-[#60A5FA]">
                    {enc.outputDim}-d
                  </span>
                </div>
                <div className="mt-1 truncate text-[11px] text-[#9CA3AF]">{enc.backbone}</div>
                <div className="mt-1 text-[11px] text-[#9CA3AF]">
                  Status: {enc.status} · Missing-safe: {enc.supportsMissing ? 'Yes' : 'No'}
                </div>
              </button>
            ))}
          </div>

          <div className="hidden lg:flex lg:col-span-1 justify-center text-[#4B5563]">
            <ArrowRight className="h-5 w-5" />
          </div>

          {/* Column 2: Cross-Attention Fusion & Temporal Core */}
          <div className="lg:col-span-3 space-y-3">
            <div className="text-xs font-semibold text-[#9CA3AF]">2. Fusion & Memory Core</div>
            <button
              type="button"
              onClick={() => setSelectedStageId('fusion')}
              className={`w-full rounded-md border p-4 text-left transition-colors ${
                selectedStageId === 'fusion'
                  ? 'border-[#8B5CF6] bg-[#8B5CF6]/10'
                  : 'border-white/[0.08] bg-[#0B0D10] hover:border-white/[0.2]'
              }`}
            >
              <div className="text-xs font-semibold text-[#F3F4F6]">Cross-Attention Fusion</div>
              <p className="mt-1 text-[11px] text-[#9CA3AF]">{spec.fusion.mechanism}</p>
              <div className="mt-2 font-mono-tabular text-[11px] text-[#A78BFA]">
                {spec.fusion.attentionHeads} heads · {spec.fusion.sharedRepresentationDim}-d shared latent
              </div>
            </button>

            <button
              type="button"
              onClick={() => setSelectedStageId('continual')}
              className={`w-full rounded-md border p-3 text-left transition-colors ${
                selectedStageId === 'continual'
                  ? 'border-[#3B82F6] bg-[#3B82F6]/10'
                  : 'border-white/[0.08] bg-[#0B0D10] hover:border-white/[0.2]'
              }`}
            >
              <div className="text-xs font-medium text-[#F3F4F6]">Continual Memory & EWC</div>
              <div className="mt-1 font-mono-tabular text-[11px] text-[#9CA3AF]">
                Replay Buffer: {spec.continualConfig.replayBufferCount} /{' '}
                {spec.continualConfig.replayCapacity}
              </div>
            </button>

            <button
              type="button"
              onClick={() => setSelectedStageId('unlearning')}
              className={`w-full rounded-md border p-3 text-left transition-colors ${
                selectedStageId === 'unlearning'
                  ? 'border-[#3B82F6] bg-[#3B82F6]/10'
                  : 'border-white/[0.08] bg-[#0B0D10] hover:border-white/[0.2]'
              }`}
            >
              <div className="text-xs font-medium text-[#F3F4F6]">Unlearning Shards & Audit</div>
              <div className="mt-1 font-mono-tabular text-[11px] text-[#9CA3AF]">
                {spec.unlearningConfig.totalShards} SISA shards · {spec.unlearningConfig.auditCount}{' '}
                audits
              </div>
            </button>
          </div>

          <div className="hidden lg:flex lg:col-span-1 justify-center text-[#4B5563]">
            <ArrowRight className="h-5 w-5" />
          </div>

          {/* Column 3: Task-Specific Heads */}
          <div className="lg:col-span-3 space-y-2.5">
            <div className="text-xs font-semibold text-[#9CA3AF]">3. Task-Specific Heads</div>
            {spec.heads.map((head) => (
              <button
                key={head.id}
                type="button"
                onClick={() => setSelectedStageId(head.id)}
                className={`w-full rounded-md border p-3 text-left transition-colors ${
                  selectedStageId === head.id
                    ? 'border-[#3B82F6] bg-[#3B82F6]/10'
                    : 'border-white/[0.08] bg-[#0B0D10] hover:border-white/[0.2]'
                }`}
              >
                <div className="flex items-center justify-between text-xs font-medium text-[#F3F4F6]">
                  <span className="truncate">{head.name}</span>
                  <span className="font-mono-tabular text-[11px] text-[#60A5FA]">
                    {head.activeVersion}
                  </span>
                </div>
                <div className="mt-1 truncate text-[11px] text-[#9CA3AF]">{head.task}</div>
              </button>
            ))}
          </div>
        </div>

        {/* Selected Component Detail Strip */}
        <div className="flex flex-wrap items-center justify-between gap-4 border-t border-white/[0.06] pt-4 text-xs">
          <div className="text-[#9CA3AF]">
            Selected Stage: <strong className="text-[#F3F4F6]">{selectedStageId}</strong> — Task
            heads remain strictly decoupled; missing modalities degrade gracefully without failing shared execution.
          </div>
          <button
            type="button"
            onClick={() => {
              if (selectedStageId.includes('emotion')) onNavigate('emotion');
              else if (selectedStageId.includes('continual')) onNavigate('learning');
              else if (selectedStageId.includes('unlearn')) onNavigate('unlearning');
              else onNavigate('forensics');
            }}
            className="rounded-md bg-[#3B82F6] px-3.5 py-1.5 font-medium text-white hover:bg-[#2563EB]"
          >
            Inspect in Laboratory
          </button>
        </div>
      </section>

      <NeuralNetworkCanvas onNavigate={onNavigate} compact />
    </div>
  );
}
