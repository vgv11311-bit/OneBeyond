import http from 'node:http';
import http2 from 'node:http2';
import https from 'node:https';
import { createReadStream } from 'node:fs';
import { stat } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const root = path.dirname(fileURLToPath(import.meta.url));
const dist = path.join(root, 'dist');
const gateway = process.env.ONE_BEYOND_API_GATEWAY?.replace(/\/$/, '');
const maxUpload = 100 * 1024 * 1024;
const excludedHeaders = new Set(['connection', 'keep-alive', 'proxy-authenticate', 'proxy-authorization', 'te', 'trailer', 'transfer-encoding', 'upgrade']);
const mime = {
  '.css': 'text/css; charset=utf-8', '.html': 'text/html; charset=utf-8',
  '.ico': 'image/x-icon', '.jpeg': 'image/jpeg', '.jpg': 'image/jpeg',
  '.js': 'text/javascript; charset=utf-8', '.json': 'application/json; charset=utf-8',
  '.png': 'image/png', '.svg': 'image/svg+xml', '.wasm': 'application/wasm',
  '.webp': 'image/webp', '.woff': 'font/woff', '.woff2': 'font/woff2',
};

function copyHeaders(headers) {
  return Object.fromEntries(Object.entries(headers).filter(([name]) =>
    !excludedHeaders.has(name.toLowerCase()) && name !== 'host' && !name.startsWith(':')));
}

export function createSiteServer(apiOrigin = gateway) {
  return http.createServer(async (req, res) => {
    const pathname = new URL(req.url ?? '/', 'http://localhost').pathname;
    if (pathname === '/api' || pathname.startsWith('/api/')) {
      if (!apiOrigin) {
        res.writeHead(503, { 'Content-Type': 'application/json', 'Cache-Control': 'no-store' });
        res.end(JSON.stringify({ error: 'Reconstruction server is offline.' }));
        return;
      }
      const target = new URL(req.url ?? '/', apiOrigin);
      if (target.protocol !== 'https:' && target.protocol !== 'http:') {
        res.writeHead(502).end();
        return;
      }
      if (pathname === '/api/upload' && Number(req.headers['content-length'] ?? 0) > maxUpload) {
        res.writeHead(413, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ error: 'Keep this upload below 100 MB.' }));
        return;
      }
      const transport = target.protocol === 'https:' ? https : http;
      if (target.protocol === 'https:') {
        // Cloud Run's HTTP/1 request limit is 32 MiB. Its h2c service accepts
        // streamed HTTP/2 requests up to the application's 100 MB limit.
        const session = http2.connect(target.origin);
        const upstream = session.request({
          ':method': req.method,
          ':path': target.pathname + target.search,
          ...copyHeaders(req.headers),
          'x-forwarded-host': req.headers.host ?? '',
        });
        let finished = false;
        const fail = () => {
          if (finished) return;
          finished = true;
          if (!res.headersSent) {
            res.writeHead(503, { 'Content-Type': 'application/json', 'Cache-Control': 'no-store' });
            res.end(JSON.stringify({ error: 'Reconstruction server is unavailable.' }));
          } else res.destroy();
          session.destroy();
        };
        session.setTimeout(300000, fail);
        session.on('error', fail);
        upstream.on('error', fail);
        upstream.on('response', headers => {
          if (finished) return;
          res.writeHead(Number(headers[':status'] ?? 502), copyHeaders(headers));
          upstream.pipe(res);
        });
        upstream.on('end', () => { finished = true; session.close(); });
        req.on('aborted', () => { upstream.close(); session.close(); });
        req.pipe(upstream);
        return;
      }
      const upstream = transport.request(target, {
        method: req.method,
        headers: { ...copyHeaders(req.headers), 'x-forwarded-host': req.headers.host ?? '' },
        timeout: 120000,
      }, upstreamResponse => {
        res.writeHead(upstreamResponse.statusCode ?? 502, copyHeaders(upstreamResponse.headers));
        upstreamResponse.pipe(res);
      });
      upstream.on('timeout', () => upstream.destroy(new Error('Gateway timed out')));
      upstream.on('error', () => {
        if (!res.headersSent) {
          res.writeHead(503, { 'Content-Type': 'application/json', 'Cache-Control': 'no-store' });
          res.end(JSON.stringify({ error: 'Reconstruction server is unavailable.' }));
        } else {
          res.destroy();
        }
      });
      req.pipe(upstream);
      return;
    }

    let requested;
    try { requested = decodeURIComponent(pathname); } catch { res.writeHead(400).end(); return; }
    const file = path.resolve(dist, `.${requested}`);
    if (file !== dist && !file.startsWith(`${dist}${path.sep}`)) { res.writeHead(400).end(); return; }
    let selected = file;
    try {
      if (!(await stat(selected)).isFile()) selected = path.join(dist, 'index.html');
    } catch {
      selected = path.join(dist, 'index.html');
    }
    try {
      const info = await stat(selected);
      if (!info.isFile()) throw new Error('Not a file');
      res.writeHead(200, {
        'Content-Type': mime[path.extname(selected)] ?? 'application/octet-stream',
        'Content-Length': info.size,
        'Cache-Control': selected.includes(`${path.sep}assets${path.sep}`) ? 'public, max-age=31536000, immutable' : 'no-cache',
      });
      if (req.method === 'HEAD') res.end(); else createReadStream(selected).pipe(res);
    } catch {
      res.writeHead(404).end();
    }
  });
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  createSiteServer().listen(Number(process.env.PORT || 3000), '0.0.0.0');
}
