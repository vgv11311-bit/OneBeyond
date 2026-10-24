// Run with ONE_BEYOND_PLAYWRIGHT_MODULE pointing to an available Playwright
// package, or install Playwright in the calling environment. Uses saved rooms.
const { chromium } = require(process.env.ONE_BEYOND_PLAYWRIGHT_MODULE || 'playwright');
const fs = require('node:fs/promises');
const path = require('node:path');
const baseURL = process.env.ONE_BEYOND_URL || 'http://127.0.0.1:5000';
const reportRoot = path.resolve(__dirname, '../reports');
const label = (process.env.ONE_BEYOND_CHECK_LABEL || 'room').replace(/[^a-zA-Z0-9_-]/g, '');
(async () => {
  const browser = await chromium.launch({ channel: 'chrome', headless: true, timeout: 60000 });
  const context = await browser.newContext({ viewport: { width: 1440, height: 900 }, reducedMotion: 'reduce' });
  const page = await context.newPage();
  const errors = [];
  const logs = [];
  page.on('pageerror', error => errors.push(error.message));
  page.on('console', message => {
    logs.push({ type: message.type(), text: message.text().slice(0, 3000) });
    if (message.type() === 'error') errors.push(message.text());
  });
  await page.addInitScript(() => {
    window.__unityCheckEvents = [];
    document.addEventListener('one-beyond-unity', event => {
      try {
        const detail = JSON.parse(event.detail);
        if (detail.type === 'screenshot') detail.message = `[PNG base64: ${detail.message.length} chars]`;
        window.__unityCheckEvents.push(detail);
      } catch {}
    }, true);
  });
  try {
    const response = await context.request.get(`${baseURL}/api/spaces`, { timeout: 30000 });
    const { spaces = [] } = await response.json();
    const selected = process.env.ONE_BEYOND_CHECK_JOB
      ? spaces.find(room => room.job_id === process.env.ONE_BEYOND_CHECK_JOB)
      : spaces.find(room => !room.job_id.startsWith('sample-') && room.status === 'completed');
    if (!selected) throw new Error('No completed saved reconstruction is available.');
    await page.goto(`${baseURL}/spaces`, { waitUntil: 'domcontentloaded', timeout: 60000 });
    const openSelected = async () => {
      const card = page.getByTestId(`space-${selected.job_id}`);
      for (let batch = 0; batch < 30 && !await card.count(); batch++) {
        const more = page.getByRole('button', { name: 'Show more spaces', exact: true });
        await page.locator('[data-testid^="space-"]').first().waitFor({ timeout: 30000 });
        if (await card.count()) break;
        if (!await more.count()) break;
        await more.click();
      }
      await card.click({ timeout: 30000 });
    };
    await openSelected();
    await page.waitForFunction(() => window.__unityCheckEvents.some(event => event.type === 'loaded' || event.type === 'error') || document.querySelector('[role="alert"]'), undefined, { timeout: 180000 });
    const events = await page.evaluate(() => window.__unityCheckEvents);
    const failure = events.find(event => event.type === 'error');
    if (failure) throw new Error(`Unity room loading failed: ${failure.message}`);
    if (!events.some(event => event.type === 'loaded')) throw new Error(await page.getByRole('alert').innerText());
    const checkedQualities = [];
    for (const [setting, resolved] of [['high', 'high'], ['compatibility', 'compatibility'], ['balanced', 'balanced'], ['auto', 'balanced']]) {
      await page.getByRole('combobox', { name: 'Graphics quality', exact: true }).selectOption(setting);
      await page.waitForFunction(expected => window.__unityCheckEvents.filter(event => event.type === 'quality').at(-1)?.quality === expected, resolved, { timeout: 15000 });
      checkedQualities.push(setting);
    }
    await page.getByTitle('Studio Clean Light', { exact: true }).click();
    await page.getByTitle('Reset Camera Position', { exact: true }).click();
    const downloadedImage = page.waitForEvent('download', { timeout: 30000 });
    await page.getByTitle('Capture Spatial Screenshot', { exact: true }).click();
    const download = await downloadedImage;
    const capturePath = path.join(reportRoot, `unity-${label}-capture.png`);
    await download.saveAs(capturePath);
    const capture = await fs.readFile(capturePath);
    if (!capture.subarray(0, 8).equals(Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]))) throw new Error('Screenshot is not a PNG.');
    await page.getByTitle('Cinematic Lighting', { exact: true }).click();
    await page.waitForTimeout(1500);
    await page.screenshot({ path: path.join(reportRoot, `unity-browser-${label}.png`), fullPage: false });
    // Fixed quality and a UI-free crop make the movement comparison meaningful.
    await page.getByRole('combobox', { name: 'Graphics quality', exact: true }).selectOption('balanced');
    await page.getByRole('button', { name: 'Clear view', exact: true }).click();
    await page.locator('canvas').focus();
    const crop = { x: 300, y: 180, width: 700, height: 500 };
    const initialView = await page.screenshot({ clip: crop });
    await page.keyboard.down('d');
    await page.waitForTimeout(300);
    await page.keyboard.up('d');
    const movedView = await page.screenshot({ clip: crop });
    if (initialView.equals(movedView)) throw new Error('Keyboard movement did not change the room view.');
    await page.getByRole('button', { name: 'Show controls', exact: true }).click();
    await page.getByTitle('Reset Camera Position', { exact: true }).click();
    await page.getByRole('button', { name: 'Clear view', exact: true }).click();
    await page.waitForTimeout(300);
    const resetView = await page.screenshot({ clip: crop });
    if (!initialView.equals(resetView)) throw new Error('Camera reset did not restore the initial room view.');
    await page.getByRole('button', { name: 'Show controls', exact: true }).click();
    await page.getByTitle('Exit to Dashboard', { exact: true }).click();
    await page.getByRole('combobox', { name: 'Graphics quality', exact: true }).waitFor({ state: 'detached' });
    await openSelected();
    await page.waitForFunction(() => window.__unityCheckEvents.filter(event => event.type === 'loaded').length === 2 || document.querySelector('[role="alert"]'), undefined, { timeout: 180000 });
    const finalEvents = await page.evaluate(() => window.__unityCheckEvents);
    if (finalEvents.filter(event => event.type === 'loaded').length !== 2) throw new Error('Reopening the saved room failed.');
    await page.getByTitle('Exit to Dashboard', { exact: true }).click();
    await page.getByRole('combobox', { name: 'Graphics quality', exact: true }).waitFor({ state: 'detached' });
    const result = { room: { id: selected.job_id, name: selected.name }, events: finalEvents, checkedQualities, keyboardMovementChangedView: true, cameraResetRestoredView: true, loadedCycles: 2, screenshotBytes: capture.length, exitReturnedToGallery: true, errors, logs, screenshot: `unity-browser-${label}.png` };
    await fs.writeFile(path.join(reportRoot, `unity-browser-${label}-check.json`), JSON.stringify(result, null, 2));
    console.log(JSON.stringify({ room: result.room, checkedQualities, keyboardMovementChangedView: true,
      cameraResetRestoredView: true, loadedCycles: 2, screenshotBytes: capture.length,
      errors, report: `unity-browser-${label}-check.json` }, null, 2));
    if (errors.length) process.exitCode = 1;
  } catch (error) {
    await page.screenshot({ path: path.join(reportRoot, `unity-browser-${label}-failure.png`) }).catch(() => {});
    console.error(error.stack || String(error));
    const result = { errors, logs, events: await page.evaluate(() => window.__unityCheckEvents || []).catch(() => []) };
    await fs.writeFile(path.join(reportRoot, `unity-browser-${label}-check.json`), JSON.stringify(result, null, 2));
    console.error(JSON.stringify(result));
    process.exitCode = 1;
  } finally {
    await browser.close();
  }
})().catch(error => { console.error(error); process.exitCode = 1; });
