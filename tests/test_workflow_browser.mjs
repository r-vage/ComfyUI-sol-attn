// Optional live regression: use an isolated ComfyUI backend and Vite frontend.
// PLAYWRIGHT_MODULE points to an existing playwright or playwright-core module.
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { pathToFileURL } from 'node:url';
const { chromium } = await import(pathToFileURL(process.env.PLAYWRIGHT_MODULE).href);
const url = process.env.COMFY_TEST_URL || 'http://127.0.0.1:5173';
const fixtures = JSON.parse(readFileSync(new URL('./fixtures/node_interfaces.json', import.meta.url)));
// Preserve the original schema fixture while allowing documentation-only edits.
function workflowContract(value) {
  if (Array.isArray(value)) return value.map(workflowContract);
  if (value && typeof value === 'object') {
    return Object.fromEntries(Object.entries(value)
      .filter(([key]) => key !== 'tooltip')
      .map(([key, item]) => [key, workflowContract(item)]));
  }
  return value;
}
const browser = await chromium.launch({headless: true, executablePath: process.env.CHROMIUM_EXECUTABLE});
try {
  const page = await browser.newPage({viewport: {width: 1600, height: 1000}});
  const users = await (await page.request.get(`${url}/users`)).json();
  let userId = Object.keys(users.users || {})[0];
  if (!userId) {
    const response = await page.request.post(`${url}/users`, {data: {username: 'sol-validation'}});
    assert.equal(response.ok(), true);
    userId = await response.json();
  }
  await page.addInitScript(({userId}) => {
    localStorage.setItem('Comfy.userId', userId);
    localStorage.setItem('Comfy.userName', 'sol-validation');
  }, {userId});
  await page.goto(url, {waitUntil: 'domcontentloaded'});
  await page.waitForFunction(() => window.app?.extensionManager, null, {timeout: 60000});
  const schemas = await (await page.request.get(`${url}/object_info`)).json();
  for (const [name, fixture] of Object.entries(fixtures)) {
    assert.deepEqual(workflowContract(schemas[name].input), workflowContract(fixture.inputs), `${name} backend input schema`);
    assert.deepEqual(schemas[name].output, fixture.outputs, `${name} backend outputs`);
  }
  for (const vue of [false, true]) {
    await page.evaluate(async value => {
      await window.app.ui.settings.setSettingValueAsync('Comfy.VueNodes.Enabled', value);
    }, vue);
    const result = await page.evaluate(async fixtures => {
      const app = window.app;
      const LiteGraph = window.LiteGraph;
      app.graph.clear();
      for (const name of Object.keys(fixtures)) {
        const node = LiteGraph.createNode(name);
        if (!node) throw new Error(`Missing node: ${name}`);
        app.graph.add(node);
        node.pos = [100 + app.graph._nodes.length * 35, 100];
      }
      const before = app.graph.serialize();
      const snapshot = graph => graph.nodes.map(n => ({type: n.type, widgets: n.widgets_values,
        inputs: n.inputs.map(x => [x.name, x.type]), outputs: n.outputs.map(x => [x.name, x.type])}));
      await app.loadGraphData(structuredClone(before), true, false);
      const after = app.graph.serialize();
      const reloaded = snapshot(after);
      // Reload once more to catch changes in defaults or widget ordering.
      await app.loadGraphData(structuredClone(after), true, false);
      const again = snapshot(app.graph.serialize());
      const count = app.graph._nodes.length;
      app.graph.clear();
      return {before: snapshot(before), after: reloaded, again, count, remaining: app.graph._nodes.length};
    }, fixtures);
    assert.equal(result.count, 5);
    assert.deepEqual(result.before, result.after);
    assert.deepEqual(result.after, result.again);
    assert.equal(result.remaining, 0);
    console.log(`PASS five node schemas, creation, save/reload and removal (${vue ? 'Nodes 2.0' : 'classic'})`);
  }
} finally {
  await browser.close();
}
