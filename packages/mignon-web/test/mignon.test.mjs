// Run with: npm test   (builds first)   or   node --test test/mignon.test.mjs
import test from "node:test";
import assert from "node:assert";
import { createRequire } from "node:module";
import { existsSync, readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import * as Mignon from "../dist/mignon.mjs";

const modelPath = fileURLToPath(new URL("../../../examples/order-search/orders.mgn", import.meta.url));
const haveModel = existsSync(modelPath);
const opts = { skip: haveModel ? false : "examples/order-search/orders.mgn is not there" };

test("loads a model from a path and answers", opts, async () => {
  const m = await Mignon.load({ model: modelPath });
  assert.strictEqual(m.info.quantized, true);
  assert.strictEqual(m.info.task, "chat");
  assert.ok(m.info.params > 100000 && m.info.vocab_size > 100);
  const r = m.generate("unpaid electronics orders over $500 from last month");
  assert.match(r.text, /^orders find /);
  assert.ok(r.text.includes("--status unpaid") && r.text.includes("--min-amount 500"));
  assert.ok(r.tokens.length > 3 && r.minProb > 0 && r.minProb <= 1);
  assert.strictEqual(r.tokens.map(t => t.text).join("").trim(), r.text);
  m.free();
});

test("loads from bytes, from a URL and from a Response; several models at once", opts, async () => {
  const bytes = readFileSync(modelPath);
  const a = await Mignon.load({ model: bytes });
  const b = await Mignon.load({ model: new URL(`file://${modelPath}`) });
  const c = await Mignon.load({ model: new Response(bytes) });
  const want = a.generate("unpaid orders").text;
  assert.strictEqual(b.generate("unpaid orders").text, want);
  assert.strictEqual(c.generate("unpaid orders").text, want);
  a.free();
  assert.throws(() => a.generate("x"), /freed/);
  assert.strictEqual(b.generate("unpaid orders").text, want); // freeing one leaves the others
  b.free(); c.free();
});

test("maxTokens limits the reply", opts, async () => {
  const m = await Mignon.load({ model: modelPath });
  assert.ok(m.generate("unpaid orders", { maxTokens: 2 }).tokens.length <= 2);
  m.free();
});

test("reports a bad model file and a missing option", async () => {
  await assert.rejects(Mignon.load({ model: new Uint8Array([1, 2, 3]) }), /./);
  await assert.rejects(Mignon.load({}), /model is required/);
  await assert.rejects(Mignon.load({ model: "/no/such/model.mgn" }), /ENOENT|no such/);
});

test("the engine version is the package version", async () => {
  const pkg = JSON.parse(readFileSync(new URL("../package.json", import.meta.url), "utf8"));
  assert.strictEqual(await Mignon.version(), pkg.version);
});

test("the plain-script build works through require() as well", opts, async () => {
  const M = createRequire(import.meta.url)("../dist/mignon.js");
  const m = await M.load({ model: modelPath });
  assert.match(m.generate("unpaid orders").text, /^orders find /);
  m.free();
});
