import { GoogleGenAI, ThinkingLevel, Type } from '@google/genai';
import fs from 'node:fs';
import type { ExtractedMediaArtifacts } from './mediaProcessor';
import type {
  AuthenticityLabel,
  EmotionCategory,
  EmotionSegment,
  ExtractedFrame,
  SuspiciousSegment,
} from '../../src/types/echomind';

function getGeminiClient(): GoogleGenAI {
  const apiKey = process.env.GEMINI_API_KEY;
  if (!apiKey || apiKey === 'MY_GEMINI_API_KEY') {
    throw new Error(
      'GEMINI_API_KEY is not configured. Configure a valid key in Settings -> Secrets to run multimodal inference.'
    );
  }
  return new GoogleGenAI({
    apiKey,
    httpOptions: {
      headers: {
        'User-Agent': 'aistudio-build',
      },
    },
  });
}

async function generateWithFallback(
  ai: GoogleGenAI,
  parts: Array<{ text: string } | { inlineData: { mimeType: string; data: string } }>,
  responseSchema: Record<string, unknown>
) {
  const attempts: Array<{
    model: string;
    thinking?: { thinkingLevel: ThinkingLevel };
    timeoutMs: number;
  }> = [
    { model: 'gemini-4-hydrogen', thinking: { thinkingLevel: ThinkingLevel.LOW }, timeoutMs: 14000 },
    { model: 'gemini-3.1-flash-lite', thinking: { thinkingLevel: ThinkingLevel.MINIMAL }, timeoutMs: 22000 },
    { model: 'gemini-3.1-flash-lite', thinking: { thinkingLevel: ThinkingLevel.MINIMAL }, timeoutMs: 25000 },
  ];

  let lastError: unknown = null;
  for (let i = 0; i < attempts.length; i++) {
    const attempt = attempts[i];
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(new Error(`Request timed out after ${attempt.timeoutMs}ms`)), attempt.timeoutMs);
    try {
      const response = await Promise.race([
        ai.models.generateContent({
          model: attempt.model,
          contents: { parts },
          config: {
            responseMimeType: 'application/json',
            responseSchema,
            thinkingConfig: attempt.thinking,
            abortSignal: controller.signal,
          },
        }),
        new Promise<never>((_, reject) =>
          setTimeout(() => reject(new Error(`Gemini timeout (${attempt.model} > ${attempt.timeoutMs}ms)`)), attempt.timeoutMs + 500)
        ),
      ]);
      clearTimeout(timer);
      return response;
    } catch (err) {
      clearTimeout(timer);
      lastError = err;
      const msg = err instanceof Error ? err.message : String(err);
      const isTransient =
        msg.includes('503') ||
        msg.includes('UNAVAILABLE') ||
        msg.includes('429') ||
        msg.includes('timeout') ||
        msg.includes('timed out') ||
        msg.includes('AbortError') ||
        msg.includes('aborted') ||
        msg.includes('fetch failed') ||
        msg.includes('ECONNRESET');
      if (!isTransient || i === attempts.length - 1) {
        throw err;
      }
      await new Promise((r) => setTimeout(r, 800 * (i + 1)));
    }
  }
  throw lastError;
}

export interface GeminiForensicsOutput {
  visualAuthenticityScore: number | null;
  audioAuthenticityScore: number | null;
  avConsistencyScore: number | null;
  overallAuthenticityScore: number;
  overallLabel: AuthenticityLabel;
  confidenceNote: string;
  suspiciousSegments: SuspiciousSegment[];
  frameAnnotations: {
    frameIndex: number;
    anomalyScore: number;
    heatmapRegion: {
      x: number;
      y: number;
      w: number;
      h: number;
      intensity: number;
      label: string;
    };
  }[];
}

export async function runGeminiForensicsInference(
  filename: string,
  artifacts: ExtractedMediaArtifacts
): Promise<GeminiForensicsOutput> {
  const ai = getGeminiClient();

  const parts: Array<
    { text: string } | { inlineData: { mimeType: string; data: string } }
  > = [];

  const duration = Math.max(1, artifacts.probe.durationSec || 5);
  const frameSummary = artifacts.frameBuffers.map((_, idx) => {
    const t = Number((((idx + 0.5) / Math.max(1, artifacts.frameBuffers.length)) * duration).toFixed(2));
    return `Frame ${idx} @ ${t}s: sharpness=${artifacts.sharpnessScores[idx] ?? 0.5}, temporalDelta=${artifacts.temporalDeltas[idx] ?? 0}`;
  });

  parts.push({
    text: [
      `You are ECHO-MIND's Multimodal Deepfake Forensics analyzer.`,
      `Inspect the provided media artifacts for file "${filename}".`,
      `Media Metadata: duration=${duration}s, hasVideo=${artifacts.probe.hasVideo}, hasAudio=${artifacts.probe.hasAudio}, resolution=${artifacts.probe.width}x${artifacts.probe.height}, fps=${artifacts.probe.fps}.`,
      `Audio Signal Summary: meanRMS=${artifacts.signalSummary.meanRms}, peak=${artifacts.signalSummary.peakAmplitude}, pitchHz=${artifacts.signalSummary.estimatedPitchHz}, zcr=${artifacts.signalSummary.zeroCrossingRate}, spectralFlatness=${artifacts.signalSummary.spectralFlatness}, silenceRatio=${artifacts.signalSummary.silenceRatio}.`,
      `Frame Signal Summary:\n${frameSummary.join('\n') || 'No video frames (audio-only input).'}`,
      `Instructions:`,
      `- Evaluate visual artifacts (facial boundary blending, lighting/skin texture consistency, temporal jitter) ONLY if video frames are present; otherwise return null for visualAuthenticityScore and avConsistencyScore.`,
      `- Evaluate audio artifacts (harmonic vocoder buzzing, unnatural silence cutoffs, spectral flatness) ONLY if audio is present; otherwise return null for audioAuthenticityScore.`,
      `- Do NOT treat minor AV sync offset alone as proof of manipulation.`,
      `- Never claim forensic proof; frame findings as probabilistic model observations.`,
      `- Provide normalized [0..1] scores where 1.0 = most authentic, 0.0 = strongly anomalous for visualAuthenticityScore, audioAuthenticityScore, avConsistencyScore, and overallAuthenticityScore.`,
      `- Conversely, for suspiciousSegments[].anomalyScore and frameAnnotations[].anomalyScore, use 0.0 = natural/authentic and 1.0 = strong synthetic manipulation artifact (so an authentic frame has anomalyScore <= 0.15).`,
      `- Identify 1 to 3 inspection segments with timestamps within [0, ${duration}] and normalized saliency heatmap boxes (x,y,w,h in 0..1) for any inspected frames.`,
    ].join('\n'),
  });

  // Attach up to 2 representative PNG frames for low-latency multimodal inspection
  const maxFrames = Math.min(2, artifacts.frameBuffers.length);
  for (let i = 0; i < maxFrames; i++) {
    const stepIdx = Math.floor((i * artifacts.frameBuffers.length) / maxFrames);
    parts.push({
      inlineData: {
        mimeType: 'image/png',
        data: artifacts.frameBuffers[stepIdx].toString('base64'),
      },
    });
  }

  // Attach WAV audio if under 4MB
  if (artifacts.wavPath && fs.existsSync(artifacts.wavPath)) {
    const wavStat = fs.statSync(artifacts.wavPath);
    if (wavStat.size > 0 && wavStat.size < 4 * 1024 * 1024) {
      parts.push({
        inlineData: {
          mimeType: 'audio/wav',
          data: fs.readFileSync(artifacts.wavPath).toString('base64'),
        },
      });
    }
  }

  const response = await generateWithFallback(ai, parts, {
    type: Type.OBJECT,
    properties: {
      visualAuthenticityScore: { type: Type.NUMBER, nullable: true },
      audioAuthenticityScore: { type: Type.NUMBER, nullable: true },
      avConsistencyScore: { type: Type.NUMBER, nullable: true },
      overallAuthenticityScore: { type: Type.NUMBER },
      overallLabel: {
        type: Type.STRING,
        enum: ['LIKELY AUTHENTIC', 'POTENTIALLY MANIPULATED', 'INCONCLUSIVE'],
      },
      confidenceNote: { type: Type.STRING },
      suspiciousSegments: {
        type: Type.ARRAY,
        items: {
          type: Type.OBJECT,
          properties: {
            startSec: { type: Type.NUMBER },
            endSec: { type: Type.NUMBER },
            modality: { type: Type.STRING, enum: ['visual', 'audio', 'audiovisual'] },
            severity: { type: Type.STRING, enum: ['low', 'medium', 'high'] },
            anomalyScore: { type: Type.NUMBER },
            summary: { type: Type.STRING },
            heatmapX: { type: Type.NUMBER },
            heatmapY: { type: Type.NUMBER },
            heatmapW: { type: Type.NUMBER },
            heatmapH: { type: Type.NUMBER },
            heatmapIntensity: { type: Type.NUMBER },
            heatmapLabel: { type: Type.STRING },
          },
        },
      },
      frameAnnotations: {
        type: Type.ARRAY,
        items: {
          type: Type.OBJECT,
          properties: {
            frameIndex: { type: Type.INTEGER },
            anomalyScore: { type: Type.NUMBER },
            x: { type: Type.NUMBER },
            y: { type: Type.NUMBER },
            w: { type: Type.NUMBER },
            h: { type: Type.NUMBER },
            intensity: { type: Type.NUMBER },
            label: { type: Type.STRING },
          },
        },
      },
    },
  });

  const raw = JSON.parse((response.text ?? '{}').trim()) as {
    visualAuthenticityScore?: number | null;
    audioAuthenticityScore?: number | null;
    avConsistencyScore?: number | null;
    overallAuthenticityScore?: number;
    overallLabel?: AuthenticityLabel;
    confidenceNote?: string;
    suspiciousSegments?: Array<{
      startSec: number;
      endSec: number;
      modality: 'visual' | 'audio' | 'audiovisual';
      severity: 'low' | 'medium' | 'high';
      anomalyScore: number;
      summary: string;
      heatmapX?: number;
      heatmapY?: number;
      heatmapW?: number;
      heatmapH?: number;
      heatmapIntensity?: number;
      heatmapLabel?: string;
    }>;
    frameAnnotations?: Array<{
      frameIndex: number;
      anomalyScore: number;
      x: number;
      y: number;
      w: number;
      h: number;
      intensity: number;
      label: string;
    }>;
  };

  const clamp01 = (v: number | null | undefined): number | null =>
    v === null || v === undefined || Number.isNaN(v)
      ? null
      : Number(Math.max(0, Math.min(1, v)).toFixed(3));

  const visualScore = artifacts.probe.hasVideo
    ? clamp01(raw.visualAuthenticityScore) ?? 0.65
    : null;
  const audioScore = artifacts.probe.hasAudio
    ? clamp01(raw.audioAuthenticityScore) ?? 0.65
    : null;
  const avScore =
    artifacts.probe.hasVideo && artifacts.probe.hasAudio
      ? clamp01(raw.avConsistencyScore) ?? 0.7
      : null;

  const overall =
    clamp01(raw.overallAuthenticityScore) ??
    Number(
      (
        ([visualScore, audioScore, avScore].filter((v): v is number => v !== null).reduce((a, b) => a + b, 0) /
          Math.max(1, [visualScore, audioScore, avScore].filter((v) => v !== null).length)) ||
        0.5
      ).toFixed(3)
    );

  const label: AuthenticityLabel =
    raw.overallLabel && ['LIKELY AUTHENTIC', 'POTENTIALLY MANIPULATED', 'INCONCLUSIVE'].includes(raw.overallLabel)
      ? raw.overallLabel
      : overall >= 0.72
        ? 'LIKELY AUTHENTIC'
        : overall <= 0.45
          ? 'POTENTIALLY MANIPULATED'
          : 'INCONCLUSIVE';

  const suspiciousSegments: SuspiciousSegment[] = (raw.suspiciousSegments ?? []).map((seg, idx) => {
    const startSec = Math.max(0, Math.min(duration, Number(seg.startSec ?? 0)));
    const endSec = Math.max(startSec + 0.2, Math.min(duration, Number(seg.endSec ?? startSec + 1)));
    const frameIdx =
      artifacts.frameUrls.length > 0
        ? Math.min(
            artifacts.frameUrls.length - 1,
            Math.floor((startSec / duration) * artifacts.frameUrls.length)
          )
        : -1;

    return {
      id: `seg-${idx + 1}`,
      startSec: Number(startSec.toFixed(2)),
      endSec: Number(endSec.toFixed(2)),
      modality: seg.modality ?? (artifacts.probe.hasVideo ? 'visual' : 'audio'),
      severity: seg.severity ?? 'medium',
      anomalyScore: clamp01(seg.anomalyScore) ?? 0.4,
      summary: seg.summary || 'Localized temporal/spectral signal variance observed.',
      frameUrl: frameIdx >= 0 ? artifacts.frameUrls[frameIdx] : undefined,
      heatmapRegion: artifacts.probe.hasVideo
        ? {
            x: clamp01(seg.heatmapX) ?? 0.34,
            y: clamp01(seg.heatmapY) ?? 0.16,
            w: Math.max(0.12, clamp01(seg.heatmapW) ?? 0.32),
            h: Math.max(0.12, clamp01(seg.heatmapH) ?? 0.34),
            intensity: clamp01(seg.heatmapIntensity) ?? 0.65,
            label: seg.heatmapLabel || 'Facial Region Saliency',
          }
        : undefined,
    };
  });

  const frameAnnotations = (raw.frameAnnotations ?? []).map((fa) => ({
    frameIndex: Math.max(0, Math.min(artifacts.frameUrls.length - 1, fa.frameIndex ?? 0)),
    anomalyScore: clamp01(fa.anomalyScore) ?? 0.25,
    heatmapRegion: {
      x: clamp01(fa.x) ?? 0.35,
      y: clamp01(fa.y) ?? 0.15,
      w: Math.max(0.12, clamp01(fa.w) ?? 0.3),
      h: Math.max(0.12, clamp01(fa.h) ?? 0.32),
      intensity: clamp01(fa.intensity) ?? 0.55,
      label: fa.label || 'Saliency Activation',
    },
  }));

  return {
    visualAuthenticityScore: visualScore,
    audioAuthenticityScore: audioScore,
    avConsistencyScore: avScore,
    overallAuthenticityScore: overall,
    overallLabel: label,
    confidenceNote:
      raw.confidenceNote ||
      'Probabilistic multimodal assessment combining visual frame inspection and acoustic signal statistics.',
    suspiciousSegments,
    frameAnnotations,
  };
}

export function buildExtractedFrames(
  artifacts: ExtractedMediaArtifacts,
  annotations: GeminiForensicsOutput['frameAnnotations']
): ExtractedFrame[] {
  const duration = Math.max(1, artifacts.probe.durationSec || 5);
  return artifacts.frameUrls.map((url, idx) => {
    const ann = annotations.find((a) => a.frameIndex === idx);
    return {
      index: idx,
      timestampSec: Number((((idx + 0.5) / Math.max(1, artifacts.frameUrls.length)) * duration).toFixed(2)),
      url,
      sharpnessScore: artifacts.sharpnessScores[idx] ?? 0.72,
      temporalDelta: artifacts.temporalDeltas[idx] ?? 0.08,
      anomalyScore: ann?.anomalyScore ?? Number((artifacts.temporalDeltas[idx] ?? 0.1).toFixed(2)),
      heatmapRegion: ann?.heatmapRegion ?? {
        x: 0.36,
        y: 0.14,
        w: 0.28,
        h: 0.34,
        intensity: Number((0.35 + (artifacts.temporalDeltas[idx] ?? 0.1)).toFixed(2)),
        label: 'Facial Boundary Attention',
      },
    };
  });
}

const VALID_EMOTIONS: EmotionCategory[] = [
  'Neutral',
  'Joy',
  'Sadness',
  'Anger',
  'Fear',
  'Surprise',
  'Disgust',
];

function normalizeEmotion(val: string | undefined): EmotionCategory {
  if (!val) return 'Neutral';
  const match = VALID_EMOTIONS.find((e) => e.toLowerCase() === val.trim().toLowerCase());
  return match ?? 'Neutral';
}

export interface GeminiEmotionOutput {
  dominantEmotion: EmotionCategory;
  overallConfidence: number;
  segments: EmotionSegment[];
}

export async function runGeminiEmotionInference(
  filename: string,
  artifacts: ExtractedMediaArtifacts,
  optionalTranscript: string
): Promise<GeminiEmotionOutput> {
  const ai = getGeminiClient();
  const duration = Math.max(1, artifacts.probe.durationSec || 5);

  const parts: Array<{ text: string } | { inlineData: { mimeType: string; data: string } }> = [
    {
      text: [
        `You are ECHO-MIND's Conversational Emotion Intelligence analyzer.`,
        `Analyze observable affective cues across available modalities for "${filename}".`,
        `Available modalities: video=${artifacts.probe.hasVideo}, audio=${artifacts.probe.hasAudio}, transcriptProvided=${Boolean(optionalTranscript.trim())}.`,
        `Acoustic baseline: meanRMS=${artifacts.signalSummary.meanRms}, estimatedPitchHz=${artifacts.signalSummary.estimatedPitchHz}, zcr=${artifacts.signalSummary.zeroCrossingRate}.`,
        optionalTranscript.trim()
          ? `User-provided transcript/subtitles:\n${optionalTranscript.trim()}`
          : `No external transcript supplied; infer observable spoken or tonal segments from audio/video if present.`,
        `Supported emotion categories ONLY: Neutral, Joy, Sadness, Anger, Fear, Surprise, Disgust.`,
        `CRITICAL RULES:`,
        `- Produce 2 to 4 timestamped utterance segments spanning [0, ${duration}].`,
        `- Provide SEPARATE modality predictions (audioPrediction, visualPrediction, textPrediction) ONLY for modalities that actually exist. If a modality is missing, set it to null.`,
        `- Do NOT force modalities to agree! When audio tone and facial expression or text disagree, set modalityDisagreement=true and explain the divergence in disagreementNote.`,
        `- Frame all predictions strictly as probabilistic observable affective cues, never claims about internal psychological states.`,
      ].join('\n'),
    },
  ];

  const maxFrames = Math.min(2, artifacts.frameBuffers.length);
  for (let i = 0; i < maxFrames; i++) {
    const stepIdx = Math.floor((i * artifacts.frameBuffers.length) / maxFrames);
    parts.push({
      inlineData: {
        mimeType: 'image/png',
        data: artifacts.frameBuffers[stepIdx].toString('base64'),
      },
    });
  }

  if (artifacts.wavPath && fs.existsSync(artifacts.wavPath)) {
    const wavStat = fs.statSync(artifacts.wavPath);
    if (wavStat.size > 0 && wavStat.size < 4 * 1024 * 1024) {
      parts.push({
        inlineData: {
          mimeType: 'audio/wav',
          data: fs.readFileSync(artifacts.wavPath).toString('base64'),
        },
      });
    }
  }

  const requiredModalitySchema = {
    type: Type.OBJECT,
    nullable: false,
    properties: {
      emotion: { type: Type.STRING, enum: VALID_EMOTIONS },
      confidence: { type: Type.NUMBER },
      rationale: { type: Type.STRING },
    },
    required: ['emotion', 'confidence', 'rationale'],
  };

  const nullableModalitySchema = {
    type: Type.OBJECT,
    nullable: true,
    properties: {
      emotion: { type: Type.STRING, enum: VALID_EMOTIONS },
      confidence: { type: Type.NUMBER },
      rationale: { type: Type.STRING },
    },
  };

  const hasTextInput = Boolean(optionalTranscript.trim());
  const requiredSegFields = ['speaker', 'startSec', 'endSec', 'transcript', 'combinedPrediction', 'combinedConfidence', 'modalityDisagreement'];
  if (artifacts.probe.hasAudio) requiredSegFields.push('audioPrediction');
  if (artifacts.probe.hasVideo) requiredSegFields.push('visualPrediction');
  if (hasTextInput) requiredSegFields.push('textPrediction');

  const response = await generateWithFallback(ai, parts, {
    type: Type.OBJECT,
    properties: {
      dominantEmotion: { type: Type.STRING, enum: VALID_EMOTIONS },
      overallConfidence: { type: Type.NUMBER },
      segments: {
        type: Type.ARRAY,
        items: {
          type: Type.OBJECT,
          properties: {
            speaker: { type: Type.STRING },
            startSec: { type: Type.NUMBER },
            endSec: { type: Type.NUMBER },
            transcript: { type: Type.STRING },
            audioPrediction: artifacts.probe.hasAudio ? requiredModalitySchema : nullableModalitySchema,
            visualPrediction: artifacts.probe.hasVideo ? requiredModalitySchema : nullableModalitySchema,
            textPrediction: hasTextInput ? requiredModalitySchema : nullableModalitySchema,
            combinedPrediction: { type: Type.STRING, enum: VALID_EMOTIONS },
            combinedConfidence: { type: Type.NUMBER },
            modalityDisagreement: { type: Type.BOOLEAN },
            disagreementNote: { type: Type.STRING, nullable: true },
          },
          required: requiredSegFields,
        },
      },
    },
    required: ['dominantEmotion', 'overallConfidence', 'segments'],
  });

  const raw = JSON.parse((response.text ?? '{}').trim()) as {
    dominantEmotion?: string;
    overallConfidence?: number;
    segments?: Array<{
      speaker?: string;
      startSec?: number;
      endSec?: number;
      transcript?: string;
      audioPrediction?: { emotion?: string; confidence?: number; rationale?: string } | null;
      visualPrediction?: { emotion?: string; confidence?: number; rationale?: string } | null;
      textPrediction?: { emotion?: string; confidence?: number; rationale?: string } | null;
      combinedPrediction?: string;
      combinedConfidence?: number;
      modalityDisagreement?: boolean;
      disagreementNote?: string | null;
    }>;
  };

  const clamp01 = (v: number | undefined): number =>
    Number(Math.max(0.05, Math.min(0.99, Number.isFinite(v) ? (v as number) : 0.65)).toFixed(3));

  const segments: EmotionSegment[] = (raw.segments ?? []).map((seg, idx) => {
    const startSec = Math.max(0, Math.min(duration, Number(seg.startSec ?? idx * (duration / 2))));
    const endSec = Math.max(startSec + 0.3, Math.min(duration, Number(seg.endSec ?? startSec + duration / 2)));

    const audioPrediction =
      artifacts.probe.hasAudio && seg.audioPrediction
        ? {
            emotion: normalizeEmotion(seg.audioPrediction.emotion),
            confidence: clamp01(seg.audioPrediction.confidence),
            rationale: seg.audioPrediction.rationale || 'Acoustic prosody and energy variation',
          }
        : null;

    const visualPrediction =
      artifacts.probe.hasVideo && seg.visualPrediction
        ? {
            emotion: normalizeEmotion(seg.visualPrediction.emotion),
            confidence: clamp01(seg.visualPrediction.confidence),
            rationale: seg.visualPrediction.rationale || 'Observed facial muscle configuration',
          }
        : null;

    const textPrediction =
      (optionalTranscript.trim() || seg.transcript) && seg.textPrediction
        ? {
            emotion: normalizeEmotion(seg.textPrediction.emotion),
            confidence: clamp01(seg.textPrediction.confidence),
            rationale: seg.textPrediction.rationale || 'Lexical and conversational context',
          }
        : null;

    const activePreds = [audioPrediction?.emotion, visualPrediction?.emotion, textPrediction?.emotion].filter(
      (e): e is EmotionCategory => Boolean(e)
    );
    const hasDisagreement =
      Boolean(seg.modalityDisagreement) || new Set(activePreds).size > 1;

    return {
      id: `emo-seg-${idx + 1}`,
      speaker: seg.speaker || `Speaker ${(idx % 2) + 1}`,
      startSec: Number(startSec.toFixed(2)),
      endSec: Number(endSec.toFixed(2)),
      transcript: seg.transcript || optionalTranscript.trim() || '(Acoustic segment without verbatim transcript)',
      audioPrediction,
      visualPrediction,
      textPrediction,
      combinedPrediction: normalizeEmotion(seg.combinedPrediction),
      combinedConfidence: clamp01(seg.combinedConfidence),
      modalityDisagreement: hasDisagreement,
      disagreementNote: hasDisagreement
        ? seg.disagreementNote ||
          `Divergence observed across modalities (${activePreds.join(' vs ')}).`
        : null,
      acousticFeatures: {
        rmsEnergy: artifacts.signalSummary.meanRms,
        estimatedPitchHz: artifacts.signalSummary.estimatedPitchHz,
        zeroCrossingRate: artifacts.signalSummary.zeroCrossingRate,
      },
    };
  });

  return {
    dominantEmotion: normalizeEmotion(raw.dominantEmotion),
    overallConfidence: clamp01(raw.overallConfidence),
    segments,
  };
}
