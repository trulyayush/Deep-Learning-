import type { WaveformPoint } from '../../src/types/echomind';

export interface SignalSummary {
  waveform: WaveformPoint[];
  spectralBands: { band: string; energyDb: number; baselineDb: number }[];
  meanRms: number;
  peakAmplitude: number;
  estimatedPitchHz: number;
  zeroCrossingRate: number;
  spectralFlatness: number;
  silenceRatio: number;
}

/**
 * Computes real acoustic signal features from raw signed 16-bit little-endian PCM mono samples.
 */
export function analyzePcmSignal(
  pcmBuffer: Buffer,
  sampleRate = 16000,
  targetPoints = 64
): SignalSummary {
  const sampleCount = Math.floor(pcmBuffer.length / 2);
  if (sampleCount < 64) {
    return {
      waveform: [],
      spectralBands: [
        { band: '0–300 Hz', energyDb: -60, baselineDb: -42 },
        { band: '300–1 kHz', energyDb: -60, baselineDb: -24 },
        { band: '1–3 kHz', energyDb: -60, baselineDb: -28 },
        { band: '3–6 kHz', energyDb: -60, baselineDb: -36 },
        { band: '6–8 kHz', energyDb: -60, baselineDb: -48 },
      ],
      meanRms: 0,
      peakAmplitude: 0,
      estimatedPitchHz: 0,
      zeroCrossingRate: 0,
      spectralFlatness: 0,
      silenceRatio: 1,
    };
  }

  const samples = new Float32Array(sampleCount);
  let globalPeak = 0;
  let globalSumSq = 0;
  let zeroCrossings = 0;
  let silentSamples = 0;

  for (let i = 0; i < sampleCount; i++) {
    const val = pcmBuffer.readInt16LE(i * 2) / 32768;
    samples[i] = val;
    const abs = Math.abs(val);
    if (abs > globalPeak) globalPeak = abs;
    globalSumSq += val * val;
    if (abs < 0.015) silentSamples++;
    if (i > 0 && ((samples[i - 1] >= 0 && val < 0) || (samples[i - 1] < 0 && val >= 0))) {
      zeroCrossings++;
    }
  }

  const meanRms = Math.sqrt(globalSumSq / sampleCount);
  const zeroCrossingRate = zeroCrossings / sampleCount;
  const silenceRatio = silentSamples / sampleCount;

  // Estimate fundamental pitch via autocorrelation over a middle 1024-sample window
  const windowSize = Math.min(1024, sampleCount);
  const startOffset = Math.max(0, Math.floor((sampleCount - windowSize) / 2));
  const minLag = Math.floor(sampleRate / 400); // 400 Hz max speech pitch
  const maxLag = Math.min(Math.floor(sampleRate / 70), Math.floor(windowSize / 2)); // 70 Hz min
  let bestLag = 0;
  let bestCorr = -Infinity;

  for (let lag = minLag; lag <= maxLag; lag++) {
    let corr = 0;
    for (let i = 0; i < windowSize - lag; i++) {
      corr += samples[startOffset + i] * samples[startOffset + i + lag];
    }
    if (corr > bestCorr) {
      bestCorr = corr;
      bestLag = lag;
    }
  }
  const estimatedPitchHz = bestLag > 0 && bestCorr > 0.001 ? Math.round(sampleRate / bestLag) : 0;

  // Goertzel-style DFT energy across 5 forensic speech bands on a representative 512-sample frame
  const dftN = Math.min(512, sampleCount);
  const bandDefs = [
    { band: '0–300 Hz', freqs: [120, 220], baselineDb: -34 },
    { band: '300–1 kHz', freqs: [440, 720, 920], baselineDb: -22 },
    { band: '1–3 kHz', freqs: [1300, 2000, 2700], baselineDb: -26 },
    { band: '3–6 kHz', freqs: [3600, 4800], baselineDb: -35 },
    { band: '6–8 kHz', freqs: [6400, 7400], baselineDb: -46 },
  ];

  const bandPowers: number[] = [];
  const spectralBands = bandDefs.map((b) => {
    let totalPower = 0;
    for (const freq of b.freqs) {
      const k = (freq * dftN) / sampleRate;
      const omega = (2 * Math.PI * k) / dftN;
      const coeff = 2 * Math.cos(omega);
      let q0 = 0;
      let q1 = 0;
      let q2 = 0;
      for (let n = 0; n < dftN; n++) {
        const win = 0.54 - 0.46 * Math.cos((2 * Math.PI * n) / (dftN - 1));
        q0 = coeff * q1 - q2 + samples[startOffset + n] * win;
        q2 = q1;
        q1 = q0;
      }
      const power = q1 * q1 + q2 * q2 - q1 * q2 * coeff;
      totalPower += Math.max(power / (dftN * dftN), 1e-10);
    }
    const avgPower = totalPower / b.freqs.length;
    bandPowers.push(avgPower);
    const energyDb = Number(Math.max(-72, Math.min(0, 10 * Math.log10(avgPower))).toFixed(1));
    return {
      band: b.band,
      energyDb,
      baselineDb: b.baselineDb,
    };
  });

  const geoMean = Math.exp(
    bandPowers.reduce((acc, p) => acc + Math.log(Math.max(p, 1e-12)), 0) / bandPowers.length
  );
  const arithMean = bandPowers.reduce((a, b) => a + b, 0) / bandPowers.length;
  const spectralFlatness = arithMean > 0 ? Number((geoMean / arithMean).toFixed(4)) : 0;

  // Build waveform timeline
  const pointsCount = Math.min(targetPoints, Math.max(8, Math.floor(sampleCount / 64)));
  const chunkSize = Math.floor(sampleCount / pointsCount);
  const waveform: WaveformPoint[] = [];

  for (let p = 0; p < pointsCount; p++) {
    const start = p * chunkSize;
    const end = Math.min(sampleCount, start + chunkSize);
    let maxAbs = 0;
    let sumSq = 0;
    let zc = 0;
    for (let i = start; i < end; i++) {
      const abs = Math.abs(samples[i]);
      if (abs > maxAbs) maxAbs = abs;
      sumSq += samples[i] * samples[i];
      if (i > start && ((samples[i - 1] >= 0 && samples[i] < 0) || (samples[i - 1] < 0 && samples[i] >= 0))) {
        zc++;
      }
    }
    const rms = Math.sqrt(sumSq / Math.max(1, end - start));
    const localZcr = zc / Math.max(1, end - start);
    const centroid = Math.round(localZcr * (sampleRate / 2));
    const timeSec = Number(((start + (end - start) / 2) / sampleRate).toFixed(2));
    // Flag sharp transient spikes or unnatural silence dropoffs relative to global RMS
    const anomalyFlag =
      (meanRms > 0.01 && (rms > meanRms * 2.35 || (rms < meanRms * 0.12 && p > 1 && p < pointsCount - 2)));

    waveform.push({
      timeSec,
      amplitude: Number(maxAbs.toFixed(4)),
      rmsEnergy: Number(rms.toFixed(4)),
      spectralCentroidHz: centroid,
      anomalyFlag,
    });
  }

  return {
    waveform,
    spectralBands,
    meanRms: Number(meanRms.toFixed(4)),
    peakAmplitude: Number(globalPeak.toFixed(4)),
    estimatedPitchHz,
    zeroCrossingRate: Number(zeroCrossingRate.toFixed(4)),
    spectralFlatness,
    silenceRatio: Number(silenceRatio.toFixed(4)),
  };
}

/**
 * Computes byte-level sharpness proxy and frame-to-frame temporal delta across extracted PNG buffers.
 */
export function analyzeFrameSignals(frameBuffers: Buffer[]): {
  sharpnessScores: number[];
  temporalDeltas: number[];
} {
  if (frameBuffers.length === 0) {
    return { sharpnessScores: [], temporalDeltas: [] };
  }

  const sharpnessScores: number[] = [];
  const temporalDeltas: number[] = [];

  for (let i = 0; i < frameBuffers.length; i++) {
    const buf = frameBuffers[i];
    // Compressed PNG size relative to buffer entropy reflects high-frequency spatial detail
    const sampleStep = Math.max(1, Math.floor(buf.length / 2048));
    let diffSum = 0;
    let prev = buf[0] ?? 0;
    let count = 0;
    for (let j = sampleStep; j < buf.length; j += sampleStep) {
      diffSum += Math.abs(buf[j] - prev);
      prev = buf[j];
      count++;
    }
    const entropyProxy = count > 0 ? Math.min(1, diffSum / (count * 90)) : 0.5;
    sharpnessScores.push(Number((0.4 + entropyProxy * 0.58).toFixed(3)));

    if (i === 0) {
      temporalDeltas.push(0);
    } else {
      const prevBuf = frameBuffers[i - 1];
      const compareLen = Math.min(buf.length, prevBuf.length);
      const step = Math.max(1, Math.floor(compareLen / 1024));
      let delta = 0;
      let samples = 0;
      for (let k = 0; k < compareLen; k += step) {
        delta += Math.abs(buf[k] - prevBuf[k]);
        samples++;
      }
      const normDelta = samples > 0 ? Math.min(1, delta / (samples * 110)) : 0;
      temporalDeltas.push(Number(normDelta.toFixed(3)));
    }
  }

  return { sharpnessScores, temporalDeltas };
}
