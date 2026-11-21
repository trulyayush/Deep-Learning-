import type { ContinualStrategy } from '../../src/types/echomind';

export interface LabeledSample {
  id: string;
  features: number[];
  labelIndex: number;
  taskIndex: number;
}

export interface SerializedClassifierCheckpoint {
  inputDim: number;
  numClasses: number;
  weights: number[]; // length = numClasses * inputDim
  biases: number[]; // length = numClasses
  fisherWeights?: number[];
  fisherBiases?: number[];
  anchorWeights?: number[];
  anchorBiases?: number[];
  trainedTasks: number[];
}

export interface ContinualTrainConfig {
  strategy: ContinualStrategy;
  epochs: number;
  learningRate: number;
  ewcLambda: number;
  l2Decay?: number;
}

export interface ContinualTrainMetrics {
  checkpoint: SerializedClassifierCheckpoint;
  prevTaskAccuracyBefore: number;
  prevTaskAccuracyAfter: number;
  newTaskValidationAccuracy: number;
  knowledgeRetention: number;
  forgettingMeasure: number;
  lossHistory: {
    epoch: number;
    trainLoss: number;
    valAccuracy: number;
    ewcPenalty: number;
  }[];
}

function deterministicInit(seedIndex: number, fanIn: number): number {
  const x = Math.sin(seedIndex * 12.9898 + 78.233) * 43758.5453;
  const frac = x - Math.floor(x);
  const scale = Math.sqrt(2 / Math.max(1, fanIn));
  return (frac * 2 - 1) * scale;
}

export function createInitialCheckpoint(
  inputDim: number,
  numClasses: number
): SerializedClassifierCheckpoint {
  const weights: number[] = [];
  for (let i = 0; i < numClasses * inputDim; i++) {
    weights.push(deterministicInit(i + 1, inputDim));
  }
  const biases = new Array<number>(numClasses).fill(0);
  return {
    inputDim,
    numClasses,
    weights,
    biases,
    trainedTasks: [],
  };
}

export function softmaxLogits(logits: number[]): number[] {
  let maxLogit = -Infinity;
  for (let i = 0; i < logits.length; i++) {
    if (logits[i] > maxLogit) maxLogit = logits[i];
  }
  const exps = new Array<number>(logits.length);
  let sum = 0;
  for (let i = 0; i < logits.length; i++) {
    const e = Math.exp(Math.max(-40, Math.min(40, logits[i] - maxLogit)));
    exps[i] = e;
    sum += e;
  }
  return exps.map((e) => e / (sum || 1));
}

export function predictProbs(
  ckpt: SerializedClassifierCheckpoint,
  features: number[]
): number[] {
  const logits = new Array<number>(ckpt.numClasses);
  for (let c = 0; c < ckpt.numClasses; c++) {
    let sum = ckpt.biases[c] ?? 0;
    const rowOffset = c * ckpt.inputDim;
    for (let d = 0; d < ckpt.inputDim; d++) {
      sum += (ckpt.weights[rowOffset + d] ?? 0) * (features[d] ?? 0);
    }
    logits[c] = sum;
  }
  return softmaxLogits(logits);
}

export function evaluateAccuracy(
  ckpt: SerializedClassifierCheckpoint,
  samples: LabeledSample[]
): number {
  if (samples.length === 0) return 1;
  let correct = 0;
  for (const s of samples) {
    const probs = predictProbs(ckpt, s.features);
    let bestIdx = 0;
    let bestProb = -1;
    for (let c = 0; c < probs.length; c++) {
      if (probs[c] > bestProb) {
        bestProb = probs[c];
        bestIdx = c;
      }
    }
    if (bestIdx === s.labelIndex) correct++;
  }
  return Number((correct / samples.length).toFixed(4));
}

/**
 * Computes diagonal Fisher Information Matrix over a set of samples for Elastic Weight Consolidation.
 */
export function computeDiagonalFisher(
  ckpt: SerializedClassifierCheckpoint,
  samples: LabeledSample[]
): { fisherWeights: number[]; fisherBiases: number[] } {
  const fisherWeights = new Array<number>(ckpt.weights.length).fill(0);
  const fisherBiases = new Array<number>(ckpt.biases.length).fill(0);
  if (samples.length === 0) {
    return { fisherWeights, fisherBiases };
  }

  for (const s of samples) {
    const probs = predictProbs(ckpt, s.features);
    for (let c = 0; c < ckpt.numClasses; c++) {
      const target = c === s.labelIndex ? 1 : 0;
      const gradB = probs[c] - target;
      fisherBiases[c] += gradB * gradB;
      const rowOffset = c * ckpt.inputDim;
      for (let d = 0; d < ckpt.inputDim; d++) {
        const gradW = gradB * (s.features[d] ?? 0);
        fisherWeights[rowOffset + d] += gradW * gradW;
      }
    }
  }

  for (let i = 0; i < fisherWeights.length; i++) {
    fisherWeights[i] /= samples.length;
  }
  for (let c = 0; c < fisherBiases.length; c++) {
    fisherBiases[c] /= samples.length;
  }
  return { fisherWeights, fisherBiases };
}

/**
 * Runs continual learning on a new task while supporting Sequential Fine-Tuning, EWC, Experience Replay, and EWC+Replay.
 */
export function runContinualTraining(
  baseCheckpoint: SerializedClassifierCheckpoint,
  newTrainSamples: LabeledSample[],
  newValSamples: LabeledSample[],
  prevTaskValSamples: LabeledSample[],
  replaySamples: LabeledSample[],
  config: ContinualTrainConfig
): ContinualTrainMetrics {
  const ckpt: SerializedClassifierCheckpoint = {
    inputDim: baseCheckpoint.inputDim,
    numClasses: baseCheckpoint.numClasses,
    weights: [...baseCheckpoint.weights],
    biases: [...baseCheckpoint.biases],
    fisherWeights: baseCheckpoint.fisherWeights ? [...baseCheckpoint.fisherWeights] : undefined,
    fisherBiases: baseCheckpoint.fisherBiases ? [...baseCheckpoint.fisherBiases] : undefined,
    anchorWeights: [...baseCheckpoint.weights],
    anchorBiases: [...baseCheckpoint.biases],
    trainedTasks: [...baseCheckpoint.trainedTasks],
  };

  const prevTaskAccuracyBefore = evaluateAccuracy(ckpt, prevTaskValSamples);
  const useEwc = config.strategy === 'ewc' || config.strategy === 'ewc_replay';
  const useReplay =
    config.strategy === 'experience_replay' || config.strategy === 'ewc_replay';

  // If EWC is active and no Fisher matrix was stored yet, compute one from prior validation / replay samples
  if (useEwc && (!ckpt.fisherWeights || ckpt.fisherWeights.length !== ckpt.weights.length)) {
    const fisherBasis = prevTaskValSamples.length > 0 ? prevTaskValSamples : replaySamples;
    const computed = computeDiagonalFisher(ckpt, fisherBasis);
    ckpt.fisherWeights = computed.fisherWeights;
    ckpt.fisherBiases = computed.fisherBiases;
  }

  const combinedTrain: LabeledSample[] = useReplay
    ? [...newTrainSamples, ...replaySamples]
    : [...newTrainSamples];

  const l2 = config.l2Decay ?? 0.0005;
  const lr = Math.max(0.001, Math.min(0.5, config.learningRate));
  const epochs = Math.max(1, Math.min(200, config.epochs));
  const lossHistory: ContinualTrainMetrics['lossHistory'] = [];

  for (let epoch = 1; epoch <= epochs; epoch++) {
    const gradW = new Array<number>(ckpt.weights.length).fill(0);
    const gradB = new Array<number>(ckpt.biases.length).fill(0);
    let ceLoss = 0;

    for (const s of combinedTrain) {
      const probs = predictProbs(ckpt, s.features);
      const pTrue = Math.max(1e-9, probs[s.labelIndex] ?? 1e-9);
      ceLoss -= Math.log(pTrue);

      for (let c = 0; c < ckpt.numClasses; c++) {
        const target = c === s.labelIndex ? 1 : 0;
        const err = probs[c] - target;
        gradB[c] += err;
        const rowOffset = c * ckpt.inputDim;
        for (let d = 0; d < ckpt.inputDim; d++) {
          gradW[rowOffset + d] += err * (s.features[d] ?? 0);
        }
      }
    }

    const n = Math.max(1, combinedTrain.length);
    let ewcPenalty = 0;

    for (let i = 0; i < ckpt.weights.length; i++) {
      let g = gradW[i] / n + l2 * ckpt.weights[i];
      if (useEwc && ckpt.fisherWeights && ckpt.anchorWeights) {
        const diff = ckpt.weights[i] - ckpt.anchorWeights[i];
        const f = ckpt.fisherWeights[i];
        ewcPenalty += 0.5 * config.ewcLambda * f * diff * diff;
        g += config.ewcLambda * f * diff;
      }
      ckpt.weights[i] -= lr * g;
    }

    for (let c = 0; c < ckpt.biases.length; c++) {
      let g = gradB[c] / n;
      if (useEwc && ckpt.fisherBiases && ckpt.anchorBiases) {
        const diff = ckpt.biases[c] - ckpt.anchorBiases[c];
        const f = ckpt.fisherBiases[c];
        ewcPenalty += 0.5 * config.ewcLambda * f * diff * diff;
        g += config.ewcLambda * f * diff;
      }
      ckpt.biases[c] -= lr * g;
    }

    const valAcc = evaluateAccuracy(ckpt, newValSamples);
    lossHistory.push({
      epoch,
      trainLoss: Number((ceLoss / n + ewcPenalty).toFixed(4)),
      valAccuracy: valAcc,
      ewcPenalty: Number(ewcPenalty.toFixed(4)),
    });
  }

  // Update consolidated Fisher Information Matrix accumulating old + new task importance
  const newFisher = computeDiagonalFisher(ckpt, [...newTrainSamples, ...newValSamples]);
  if (ckpt.fisherWeights && ckpt.fisherBiases) {
    for (let i = 0; i < ckpt.weights.length; i++) {
      ckpt.fisherWeights[i] = Number(
        (0.6 * ckpt.fisherWeights[i] + 0.4 * newFisher.fisherWeights[i]).toFixed(6)
      );
    }
    for (let c = 0; c < ckpt.biases.length; c++) {
      ckpt.fisherBiases[c] = Number(
        (0.6 * ckpt.fisherBiases[c] + 0.4 * newFisher.fisherBiases[c]).toFixed(6)
      );
    }
  } else {
    ckpt.fisherWeights = newFisher.fisherWeights;
    ckpt.fisherBiases = newFisher.fisherBiases;
  }

  const prevTaskAccuracyAfter = evaluateAccuracy(ckpt, prevTaskValSamples);
  const newTaskValidationAccuracy = evaluateAccuracy(ckpt, newValSamples);
  const forgettingMeasure = Number(
    Math.max(0, prevTaskAccuracyBefore - prevTaskAccuracyAfter).toFixed(4)
  );
  const knowledgeRetention =
    prevTaskValSamples.length > 0 && prevTaskAccuracyBefore > 0
      ? Number(
          Math.min(1, prevTaskAccuracyAfter / Math.max(0.01, prevTaskAccuracyBefore)).toFixed(4)
        )
      : newTaskValidationAccuracy;

  return {
    checkpoint: ckpt,
    prevTaskAccuracyBefore,
    prevTaskAccuracyAfter,
    newTaskValidationAccuracy,
    knowledgeRetention,
    forgettingMeasure,
    lossHistory,
  };
}

/**
 * Selects balanced replay samples respecting capacity limits across classes.
 */
export function selectReplayBuffer<T extends { id: string; label: string }>(
  candidates: T[],
  capacity: number
): T[] {
  if (capacity <= 0 || candidates.length === 0) return [];
  if (candidates.length <= capacity) return [...candidates];

  const byLabel = new Map<string, T[]>();
  for (const item of candidates) {
    const arr = byLabel.get(item.label) ?? [];
    arr.push(item);
    byLabel.set(item.label, arr);
  }

  const selected: T[] = [];
  const labelGroups = Array.from(byLabel.values());
  let idx = 0;
  while (selected.length < capacity) {
    let addedInRound = false;
    for (const group of labelGroups) {
      if (idx < group.length && selected.length < capacity) {
        selected.push(group[idx]);
        addedInRound = true;
      }
    }
    if (!addedInRound) break;
    idx++;
  }
  return selected;
}
