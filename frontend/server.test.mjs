import assert from 'node:assert/strict';
import http from 'node:http';
import { once } from 'node:events';
import { test } from 'node:test';
import { createSiteServer } from './server.mjs';

test('site keeps API on its own origin and preserves cookies', async () => {
  const api = http.createServer((req, res) => {
    assert.equal(req.url, '/api/health');
    res.writeHead(200, { 'content-type': 'application/json', 'set-cookie': 'onebeyond_visitor=signed; Secure; HttpOnly' });
    res.end('{"status":"healthy"}');
  }).listen(0, '127.0.0.1');
  await once(api, 'listening');
  const site = createSiteServer(`http://127.0.0.1:${api.address().port}`).listen(0, '127.0.0.1');
  await once(site, 'listening');
  try {
    const response = await fetch(`http://127.0.0.1:${site.address().port}/api/health`);
    assert.equal(response.status, 200);
    assert.match(response.headers.get('set-cookie'), /onebeyond_visitor=signed/);
    assert.deepEqual(await response.json(), { status: 'healthy' });
  } finally {
    site.close(); api.close();
  }
});
