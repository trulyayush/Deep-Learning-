import 'dotenv/config';
import express from 'express';
import path from 'node:path';
import fs from 'node:fs';
import { createServer as createViteServer } from 'vite';
import { seedInitialState } from './server/db';
import { ensureSampleMedia } from './server/services/mediaProcessor';
import { createApiRouter } from './server/routes/api';

const PORT = 3000;

async function bootstrap() {
  seedInitialState();
  await ensureSampleMedia().catch((err) => {
    console.warn('Sample media generation skipped:', err);
  });

  const app = express();
  app.use(express.json({ limit: '10mb' }));

  app.use('/api', createApiRouter());

  const isProd = process.env.NODE_ENV === 'production';
  const distDir = path.resolve(process.cwd(), 'dist');

  if (isProd && fs.existsSync(distDir)) {
    app.use(express.static(distDir));
    app.get('*', (_req, res) => {
      res.sendFile(path.join(distDir, 'index.html'));
    });
  } else {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  }

  app.listen(PORT, '0.0.0.0', () => {
    console.log(`ECHO-MIND workstation listening on http://0.0.0.0:${PORT}`);
  });
}

bootstrap().catch((err) => {
  console.error('Failed to start ECHO-MIND server:', err);
  process.exit(1);
});
