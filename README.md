# ECHO-MIND — Adaptive Multimodal Intelligence Platform

ECHO-MIND is an integrated multimodal intelligence workstation unifying deepfake forensics, conversational emotion intelligence, continual learning, and selective machine unlearning within a single cohesive architecture.

## Core Modules

1. **Module 01 — Multimodal Deepfake Forensics**
   - Supports MP4, MOV, AVI, WEBM, WAV, MP3, M4A, and OGG uploads.
   - Uses `ffprobe` and `ffmpeg` for stream inspection, video frame sampling, and 16 kHz mono PCM/WAV extraction.
   - Computes acoustic RMS waveforms, Goertzel spectral band energy, frame sharpness proxies, and temporal frame-to-frame deltas, paired with server-side `@google/genai` (`gemini-4-hydrogen`) multimodal inspection and normalized saliency bounding boxes.
   - Outputs `LIKELY AUTHENTIC`, `POTENTIALLY MANIPULATED`, or `INCONCLUSIVE` with explicit non-proof disclaimers and downloadable JSON reports.

2. **Module 02 — Conversational Emotion Intelligence**
   - Analyzes audio prosody (RMS energy, autocorrelation pitch, zero-crossing rate), facial frames, and optional transcripts across 7 categories (`Neutral`, `Joy`, `Sadness`, `Anger`, `Fear`, `Surprise`, `Disgust`).
   - Displays Audio, Visual, and Text modality predictions separately and flags modality disagreements rather than hiding divergence.
   - Frames all results strictly as probabilistic affective observations.

3. **Module 03 — Continual Learning Engine**
   - Registers explicitly approved labelled datasets (private user media uploads never auto-enter training).
   - Implements **Sequential Fine-Tuning**, **Elastic Weight Consolidation (EWC)** via diagonal Fisher Information Matrix regularization, **Experience Replay** with capacity-bounded stratified sampling, and **EWC + Replay Hybrid**.
   - Persists versioned checkpoints to disk (`data/checkpoints/`), tracks knowledge retention and forgetting, and supports one-click rollback.

4. **Module 04 — Selective Machine Unlearning Engine**
   - Supports sample-level forget-set selection, replay-buffer/feature-cache provenance tracking, and confirmation before destructive operations.
   - Implements **Exact Retrain Baseline**, **SISA-Style Sharded Retraining**, and **Approximate Gradient Ascent + Retain Repair**.
   - Records immutable audit logs comparing forget-set accuracy, retain-set accuracy, execution time, and model utility before and after unlearning.

## Architecture & Directory Layout

- `server.ts` — Full-stack Express + Vite server (port `3000`) serving the React SPA and `/api/*` endpoints.
- `server/db.ts` — SQLite (`node:sqlite` WAL) persistence with deterministic seeding.
- `server/ml/` — Pure TypeScript signal processing, EWC/Replay continual learning, and SISA/Exact/Gradient unlearning engines (`server/ml/engines.test.ts` unit tests).
- `server/services/` — FFmpeg media extractor, hardware inspector, background job manager, and server-side Gemini multimodal inference.
- `database/schema.sql` — PostgreSQL / Supabase production schema with Row-Level Security (RLS) policies.
- `backend/` — Optional external FastAPI + PyTorch GPU worker reference service (`EXTERNAL_INFERENCE_URL` in Settings).
- `src/` — React 19, TypeScript, Tailwind CSS v4, Three.js interactive neural network canvas, and Recharts visualizations.

## Local Setup & Commands

```bash
npm install
cp .env.example .env
# Set GEMINI_API_KEY in .env (or AI Studio Secrets)
npm run dev
```

### Verification & Build

```bash
# Run ML & Signal Engine unit tests
npx --yes tsx --test server/ml/engines.test.ts

# Typecheck & production build
npm run lint
npm run build
npm run start
```
