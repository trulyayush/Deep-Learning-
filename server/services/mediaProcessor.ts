import { execFile } from 'node:child_process';
import { promisify } from 'node:util';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import crypto from 'node:crypto';
import {
  AUDIO_DIR,
  DATA_DIR,
  FRAMES_DIR,
  SAMPLES_DIR,
  db,
  getSetting,
  logActivity,
} from '../db';
import { analyzeFrameSignals, analyzePcmSignal, type SignalSummary } from '../ml/signalProcessor';
import type { HardwareProfile } from '../../src/types/echomind';

const execFileAsync = promisify(execFile);

export interface MediaProbeInfo {
  durationSec: number;
  width: number;
  height: number;
  fps: number;
  videoCodec: string | null;
  audioCodec: string | null;
  sampleRate: number | null;
  hasVideo: boolean;
  hasAudio: boolean;
}

export interface ExtractedMediaArtifacts {
  probe: MediaProbeInfo;
  framePaths: string[];
  frameUrls: string[];
  frameBuffers: Buffer[];
  wavPath: string | null;
  signalSummary: SignalSummary;
  sharpnessScores: number[];
  temporalDeltas: number[];
}

export function assertConfinedPath(targetPath: string, baseDir: string): string {
  const resolvedBase = path.resolve(baseDir);
  const resolvedTarget = path.resolve(targetPath);
  if (
    resolvedTarget !== resolvedBase &&
    !resolvedTarget.startsWith(resolvedBase + path.sep)
  ) {
    throw new Error('Access denied: path outside confined workspace directory');
  }
  return resolvedTarget;
}

export async function probeMedia(filePath: string): Promise<MediaProbeInfo> {
  const safePath = assertConfinedPath(filePath, DATA_DIR);
  const { stdout } = await execFileAsync('ffprobe', [
    '-v',
    'quiet',
    '-print_format',
    'json',
    '-show_format',
    '-show_streams',
    safePath,
  ]);

  const parsed = JSON.parse(stdout) as {
    format?: { duration?: string };
    streams?: Array<{
      codec_type?: string;
      codec_name?: string;
      width?: number;
      height?: number;
      r_frame_rate?: string;
      sample_rate?: string;
    }>;
  };

  const streams = parsed.streams ?? [];
  const videoStream = streams.find((s) => s.codec_type === 'video');
  const audioStream = streams.find((s) => s.codec_type === 'audio');

  let fps = 0;
  if (videoStream?.r_frame_rate) {
    const [numStr, denStr] = videoStream.r_frame_rate.split('/');
    const num = Number(numStr);
    const den = Number(denStr || 1);
    if (den > 0 && Number.isFinite(num / den)) {
      fps = Number((num / den).toFixed(2));
    }
  }

  return {
    durationSec: Number(Number(parsed.format?.duration || 0).toFixed(2)),
    width: videoStream?.width ?? 0,
    height: videoStream?.height ?? 0,
    fps,
    videoCodec: videoStream?.codec_name ?? null,
    audioCodec: audioStream?.codec_name ?? null,
    sampleRate: audioStream?.sample_rate ? Number(audioStream.sample_rate) : null,
    hasVideo: Boolean(videoStream),
    hasAudio: Boolean(audioStream),
  };
}

export async function extractMediaArtifacts(
  uploadId: string,
  filePath: string
): Promise<ExtractedMediaArtifacts> {
  const safeInput = assertConfinedPath(filePath, DATA_DIR);
  const probe = await probeMedia(safeInput);

  const frameOutDir = assertConfinedPath(path.join(FRAMES_DIR, uploadId), FRAMES_DIR);
  fs.mkdirSync(frameOutDir, { recursive: true });

  const framePaths: string[] = [];
  const frameUrls: string[] = [];
  const frameBuffers: Buffer[] = [];

  if (probe.hasVideo) {
    const duration = Math.max(1, probe.durationSec || 4);
    const targetFrames = 6;
    const fpsFilter = `fps=${Math.max(0.25, targetFrames / duration).toFixed(4)}`;

    await execFileAsync('ffmpeg', [
      '-y',
      '-i',
      safeInput,
      '-vf',
      `${fpsFilter},scale=640:-2`,
      '-frames:v',
      String(targetFrames),
      path.join(frameOutDir, 'frame_%03d.png'),
    ]);

    const files = fs
      .readdirSync(frameOutDir)
      .filter((f) => f.endsWith('.png'))
      .sort();

    for (const f of files) {
      const full = path.join(frameOutDir, f);
      framePaths.push(full);
      frameUrls.push(`/api/media/frames/${encodeURIComponent(uploadId)}/${encodeURIComponent(f)}`);
      frameBuffers.push(fs.readFileSync(full));
    }
  }

  let wavPath: string | null = null;
  let signalSummary = analyzePcmSignal(Buffer.alloc(0));

  if (probe.hasAudio) {
    wavPath = assertConfinedPath(path.join(AUDIO_DIR, `${uploadId}.wav`), AUDIO_DIR);
    const pcmPath = assertConfinedPath(path.join(AUDIO_DIR, `${uploadId}.pcm`), AUDIO_DIR);

    await execFileAsync('ffmpeg', [
      '-y',
      '-i',
      safeInput,
      '-vn',
      '-ac',
      '1',
      '-ar',
      '16000',
      '-c:a',
      'pcm_s16le',
      wavPath,
    ]);

    await execFileAsync('ffmpeg', [
      '-y',
      '-i',
      wavPath,
      '-f',
      's16le',
      '-acodec',
      'pcm_s16le',
      pcmPath,
    ]);

    if (fs.existsSync(pcmPath)) {
      const pcmBuf = fs.readFileSync(pcmPath);
      signalSummary = analyzePcmSignal(pcmBuf, 16000, 64);
      fs.unlinkSync(pcmPath);
    }
  }

  const { sharpnessScores, temporalDeltas } = analyzeFrameSignals(frameBuffers);

  return {
    probe,
    framePaths,
    frameUrls,
    frameBuffers,
    wavPath,
    signalSummary,
    sharpnessScores,
    temporalDeltas,
  };
}

/**
 * Generates real, playable sample MP4 & WAV media files using FFmpeg and generated studio portraits
 * so users can test the complete real media processing pipeline in one click.
 */
export async function ensureSampleMedia(): Promise<void> {
  const existing = db.prepare(`SELECT COUNT(*) as cnt FROM media_uploads WHERE is_sample = 1`).get() as {
    cnt: number;
  };
  if (existing.cnt >= 3) return;

  const imagesDir = path.resolve(process.cwd(), 'src/assets/images');
  const imageFiles = fs.existsSync(imagesDir)
    ? fs.readdirSync(imagesDir).filter((f) => f.endsWith('.jpg') || f.endsWith('.png'))
    : [];

  const anchorImg = imageFiles.find((f) => f.includes('sample_broadcast_anchor'));
  const interviewImg = imageFiles.find((f) => f.includes('sample_interview_subject'));

  const now = new Date().toISOString();

  // Sample 1: Broadcast Anchor Video (6s MP4 with subtle motion + modulated speech-range audio)
  const s1Id = 'sample-broadcast-anchor';
  const s1Path = path.join(SAMPLES_DIR, 'broadcast_news_briefing.mp4');
  if (!fs.existsSync(s1Path) && anchorImg) {
    const imgPath = path.join(imagesDir, anchorImg);
    await execFileAsync('ffmpeg', [
      '-y',
      '-loop',
      '1',
      '-i',
      imgPath,
      '-f',
      'lavfi',
      '-i',
      'sine=frequency=185:sample_rate=16000:duration=6',
      '-vf',
      "scale=960:540,zoompan=z='min(zoom+0.0008,1.06)':d=150:s=960x540:fps=25",
      '-af',
      'tremolo=f=4.2:d=0.65,highpass=f=110,lowpass=f=3400',
      '-c:v',
      'libx264',
      '-pix_fmt',
      'yuv420p',
      '-c:a',
      'aac',
      '-shortest',
      '-t',
      '6',
      s1Path,
    ]);
  }

  if (fs.existsSync(s1Path)) {
    const existingRow = db.prepare(`SELECT id FROM media_uploads WHERE id = ?`).get(s1Id);
    if (!existingRow) {
      const probe = await probeMedia(s1Path);
      const stats = fs.statSync(s1Path);
      db.prepare(
        `INSERT INTO media_uploads (id, filename, original_name, mime_type, size_bytes, media_type, file_path, duration_sec, width, height, fps, video_codec, audio_codec, sample_rate, has_video, has_audio, is_sample, preview_url, created_at)
         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`
      ).run(
        s1Id,
        'broadcast_news_briefing.mp4',
        'broadcast_news_briefing.mp4',
        'video/mp4',
        stats.size,
        'video',
        s1Path,
        probe.durationSec,
        probe.width,
        probe.height,
        probe.fps,
        probe.videoCodec,
        probe.audioCodec,
        probe.sampleRate,
        1,
        1,
        1,
        `/api/media/file/${s1Id}`,
        now
      );
    }
  }

  // Sample 2: Research Interview Video (6s MP4 with high-frequency acoustic anomaly segment)
  const s2Id = 'sample-interview-probe';
  const s2Path = path.join(SAMPLES_DIR, 'research_interview_segment.mp4');
  if (!fs.existsSync(s2Path) && interviewImg) {
    const imgPath = path.join(imagesDir, interviewImg);
    await execFileAsync('ffmpeg', [
      '-y',
      '-loop',
      '1',
      '-i',
      imgPath,
      '-f',
      'lavfi',
      '-i',
      "aevalsrc='0.35*sin(2*PI*160*t)*(1+0.4*sin(2*PI*3.5*t))+0.12*sin(2*PI*320*t)*gt(t,2.5)*lt(t,4.2)':s=16000:d=6",
      '-vf',
      "scale=960:540,zoompan=z='1.03-0.0005*on':d=150:s=960x540:fps=25",
      '-c:v',
      'libx264',
      '-pix_fmt',
      'yuv420p',
      '-c:a',
      'aac',
      '-shortest',
      '-t',
      '6',
      s2Path,
    ]);
  }

  if (fs.existsSync(s2Path)) {
    const existingRow = db.prepare(`SELECT id FROM media_uploads WHERE id = ?`).get(s2Id);
    if (!existingRow) {
      const probe = await probeMedia(s2Path);
      const stats = fs.statSync(s2Path);
      db.prepare(
        `INSERT INTO media_uploads (id, filename, original_name, mime_type, size_bytes, media_type, file_path, duration_sec, width, height, fps, video_codec, audio_codec, sample_rate, has_video, has_audio, is_sample, preview_url, created_at)
         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`
      ).run(
        s2Id,
        'research_interview_segment.mp4',
        'research_interview_segment.mp4',
        'video/mp4',
        stats.size,
        'video',
        s2Path,
        probe.durationSec,
        probe.width,
        probe.height,
        probe.fps,
        probe.videoCodec,
        probe.audioCodec,
        probe.sampleRate,
        1,
        1,
        1,
        `/api/media/file/${s2Id}`,
        now
      );
    }
  }

  // Sample 3: Audio-only WAV probe (verifies missing-video graceful handling!)
  const s3Id = 'sample-audio-probe';
  const s3Path = path.join(SAMPLES_DIR, 'executive_voicemail_probe.wav');
  if (!fs.existsSync(s3Path)) {
    await execFileAsync('ffmpeg', [
      '-y',
      '-f',
      'lavfi',
      '-i',
      "aevalsrc='0.4*sin(2*PI*145*t)*(0.6+0.4*sin(2*PI*2.8*t))+0.18*sin(2*PI*290*t)*gt(t,1.8)*lt(t,3.4)':s=16000:d=5",
      '-c:a',
      'pcm_s16le',
      s3Path,
    ]);
  }

  if (fs.existsSync(s3Path)) {
    const existingRow = db.prepare(`SELECT id FROM media_uploads WHERE id = ?`).get(s3Id);
    if (!existingRow) {
      const probe = await probeMedia(s3Path);
      const stats = fs.statSync(s3Path);
      db.prepare(
        `INSERT INTO media_uploads (id, filename, original_name, mime_type, size_bytes, media_type, file_path, duration_sec, width, height, fps, video_codec, audio_codec, sample_rate, has_video, has_audio, is_sample, preview_url, created_at)
         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`
      ).run(
        s3Id,
        'executive_voicemail_probe.wav',
        'executive_voicemail_probe.wav',
        'audio/wav',
        stats.size,
        'audio',
        s3Path,
        probe.durationSec,
        0,
        0,
        0,
        null,
        probe.audioCodec,
        probe.sampleRate,
        0,
        1,
        1,
        `/api/media/file/${s3Id}`,
        now
      );
    }
  }

  logActivity(
    'system',
    'Sample Media Ready',
    'Verified playable MP4 and WAV sample probes in local workspace via FFmpeg.',
    'info'
  );
}

export async function inspectHardware(): Promise<HardwareProfile> {
  let pythonVersion: string | null = null;
  let cudaAvailable = false;
  let gpuName: string | null = null;
  let ffmpegAvailable = false;
  let diskFreeSummary = 'Unknown';

  try {
    const { stdout } = await execFileAsync('python3', ['--version']);
    pythonVersion = stdout.trim();
  } catch {
    pythonVersion = null;
  }

  try {
    await execFileAsync('ffmpeg', ['-version']);
    ffmpegAvailable = true;
  } catch {
    ffmpegAvailable = false;
  }

  try {
    const { stdout } = await execFileAsync('nvidia-smi', [
      '--query-gpu=name',
      '--format=csv,noheader',
    ]);
    if (stdout.trim()) {
      cudaAvailable = true;
      gpuName = stdout.trim().split('\n')[0];
    }
  } catch {
    cudaAvailable = false;
  }

  try {
    const { stdout } = await execFileAsync('df', ['-h', '.']);
    const lines = stdout.trim().split('\n');
    if (lines.length > 1) {
      const parts = lines[1].trim().split(/\s+/);
      diskFreeSummary = `${parts[3]} free of ${parts[1]}`;
    }
  } catch {
    diskFreeSummary = 'Unavailable';
  }

  const cpus = os.cpus();
  const totalMemoryMb = Math.round(os.totalmem() / (1024 * 1024));
  const freeMemoryMb = Math.round(os.freemem() / (1024 * 1024));
  const geminiConfigured = Boolean(
    process.env.GEMINI_API_KEY && process.env.GEMINI_API_KEY !== 'MY_GEMINI_API_KEY'
  );
  const externalInferenceUrl = getSetting('external_inference_url', '');

  return {
    osPlatform: os.platform(),
    osRelease: os.release(),
    arch: os.arch(),
    cpuModel: cpus[0]?.model ?? 'Unknown CPU',
    cpuCores: cpus.length,
    totalMemoryMb,
    freeMemoryMb,
    nodeVersion: process.version,
    pythonVersion,
    cudaAvailable,
    gpuName,
    ffmpegAvailable,
    geminiConfigured,
    externalInferenceUrl,
    executionMode: cudaAvailable
      ? 'GPU Accelerated'
      : geminiConfigured
        ? 'CPU Signal Pipeline + Server-Side Gemini Multimodal'
        : 'CPU Signal Pipeline Only (Gemini Key Missing)',
    recommendedBatchSize: totalMemoryMb >= 8192 ? 16 : 8,
    mixedPrecisionSupported: cudaAvailable,
    diskFreeSummary,
  };
}

export function createUploadId(): string {
  return `upl-${crypto.randomUUID().slice(0, 8)}`;
}
