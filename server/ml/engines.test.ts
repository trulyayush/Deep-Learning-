import test from 'node:test';
import assert from 'node:assert/strict';
import {
  createInitialCheckpoint,
  evaluateAccuracy,
  runContinualTraining,
  selectReplayBuffer,
  type LabeledSample,
} from './continualEngine';
import { executeSelectiveUnlearning, type ShardSample } from './unlearningEngine';
import { analyzePcmSignal, analyzeFrameSignals } from './signalProcessor';

function makeTaskSamples(taskIndex: number, offset: number): LabeledSample[] {
  const samples: LabeledSample[] = [];
  for (let i = 0; i < 24; i++) {
    const labelIndex = i % 2;
    const sign = labelIndex === 0 ? -1 : 1;
    const noise = ((i * 7) % 5) * 0.08 - 0.16;
    samples.push({
      id: `t${taskIndex}_s${i}`,
      taskIndex,
      labelIndex,
      features: [
        sign * 1.4 + offset + noise,
        sign * 1.1 - offset + noise,
        sign * 0.8 + noise,
        offset * 0.5,
        0.25 * sign,
        -0.15 * sign,
      ],
    });
  }
  return samples;
}

test('Continual learning EWC + Replay retains prior task better or learns new task cleanly', () => {
  const init = createInitialCheckpoint(6, 2);
  const task1 = makeTaskSamples(1, 0.1);
  const t1Train = task1.slice(0, 16);
  const t1Val = task1.slice(16);

  const stage1 = runContinualTraining(init, t1Train, t1Val, [], [], {
    strategy: 'sequential_finetuning',
    epochs: 30,
    learningRate: 0.15,
    ewcLambda: 0,
  });

  assert.ok(
    stage1.newTaskValidationAccuracy >= 0.85,
    `Expected task 1 val accuracy >= 0.85, got ${stage1.newTaskValidationAccuracy}`
  );

  const task2 = makeTaskSamples(2, -0.2);
  const t2Train = task2.slice(0, 16);
  const t2Val = task2.slice(16);

  const replayCandidates = t1Train.map((s) => ({ ...s, label: String(s.labelIndex) }));
  const replay = selectReplayBuffer(replayCandidates, 8);
  assert.equal(replay.length, 8);

  const stage2 = runContinualTraining(
    stage1.checkpoint,
    t2Train,
    t2Val,
    t1Val,
    replay,
    {
      strategy: 'ewc_replay',
      epochs: 25,
      learningRate: 0.12,
      ewcLambda: 2.5,
    }
  );

  assert.ok(stage2.knowledgeRetention >= 0.75);
  assert.equal(stage2.lossHistory.length, 25);
});

test('Selective machine unlearning supports Exact Retrain, SISA, and Gradient Ascent', () => {
  const init = createInitialCheckpoint(6, 2);
  const baseSamples = makeTaskSamples(1, 0).map((s, idx): ShardSample => ({
    ...s,
    shardIndex: idx % 4,
  }));

  // Flip 4 samples to simulate memorized outlier / target forget-set
  const forgetIds = new Set<string>();
  for (let i = 0; i < 4; i++) {
    baseSamples[i].labelIndex = baseSamples[i].labelIndex === 0 ? 1 : 0;
    forgetIds.add(baseSamples[i].id);
  }

  const trained = runContinualTraining(init, baseSamples, baseSamples, [], [], {
    strategy: 'sequential_finetuning',
    epochs: 40,
    learningRate: 0.15,
    ewcLambda: 0,
  });

  const exactOutcome = executeSelectiveUnlearning(
    trained.checkpoint,
    baseSamples,
    forgetIds,
    'exact_retrain',
    4
  );
  assert.ok(exactOutcome.retainAccAfter >= 0.85);
  assert.equal(exactOutcome.affectedShards.length, 4);

  const sisaOutcome = executeSelectiveUnlearning(
    trained.checkpoint,
    baseSamples,
    forgetIds,
    'sisa_sharded',
    4
  );
  assert.ok(sisaOutcome.affectedShards.length >= 1);
  assert.ok(sisaOutcome.retainAccAfter >= 0.8);

  const gradOutcome = executeSelectiveUnlearning(
    trained.checkpoint,
    baseSamples,
    forgetIds,
    'gradient_ascent',
    4
  );
  assert.ok(gradOutcome.retainAccAfter >= 0.7);
  assert.ok(gradOutcome.executionTimeMs >= 1);
});

test('Signal processor extracts valid waveform, pitch, and spectral bands from PCM sine wave', () => {
  const sampleRate = 16000;
  const durationSec = 0.5;
  const totalSamples = Math.floor(sampleRate * durationSec);
  const buf = Buffer.alloc(totalSamples * 2);
  const freq = 220; // A3 vocal pitch
  for (let i = 0; i < totalSamples; i++) {
    const v = Math.sin((2 * Math.PI * freq * i) / sampleRate) * 0.5;
    buf.writeInt16LE(Math.round(v * 32767), i * 2);
  }

  const summary = analyzePcmSignal(buf, sampleRate, 32);
  assert.equal(summary.waveform.length, 32);
  assert.equal(summary.spectralBands.length, 5);
  assert.ok(summary.meanRms > 0.3 && summary.meanRms < 0.45);
  assert.ok(
    Math.abs(summary.estimatedPitchHz - freq) <= 15,
    `Expected pitch near ${freq}Hz, got ${summary.estimatedPitchHz}Hz`
  );

  const frames = analyzeFrameSignals([Buffer.from([10, 20, 30, 40]), Buffer.from([12, 22, 35, 45])]);
  assert.equal(frames.sharpnessScores.length, 2);
  assert.equal(frames.temporalDeltas[0], 0);
  assert.ok(frames.temporalDeltas[1] > 0);
});
