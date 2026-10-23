// Run with: node --test examples/order-search
const test = require("node:test");
const assert = require("node:assert");
const Orders = require("./orders.js");

test("parseReply accepts the trained shapes", () => {
  const { query } = Orders.parseReply('orders find --status unpaid --category electronics --min-amount 500 --customer "Acme Corp" --sort newest --limit 5');
  assert.deepStrictEqual(query, { command: "find", status: "unpaid", category: "electronics", minAmount: 500,
    customer: "Acme Corp", sort: "newest", limit: 5 });
  assert.ok(Orders.parseReply("orders count --month march --year 2024").query);
  assert.ok(Orders.parseReply("orders find --month october --day 15").query);
});

test("parseReply accepts unknown only without options", () => {
  assert.strictEqual(Orders.parseReply("orders unknown").query.command, "unknown");
  assert.ok(Orders.parseReply("orders unknown --status paid").error);
});

test("parseReply rejects bad replies", () => {
  for (const reply of [
    "hello there", "orders delete --status paid", "orders find --status refundd", "orders find --min-amount abc",
    "orders find --min-amount 5.5", "orders find --min-amount 5 --max-amount 2", "orders find --when today --last-days 3",
    "orders find --status paid --status unpaid", "orders count --sort newest", 'orders find --customer "Acme',
    "orders find --status", "orders find --colour red", "orders find --day 5", "orders find --month april --day 31",
    "orders find --month february --day 29 --year 2025",
  ]) assert.ok(Orders.parseReply(reply).error, `accepted: ${reply}`);
});

const SUNDAY = "2026-10-18";
test("resolve", () => {
  const cases = [
    [{ when: "today" }, "2026-10-18", "2026-10-18"],
    [{ when: "yesterday" }, "2026-10-17", "2026-10-17"],
    [{ when: "this-week" }, "2026-10-12", "2026-10-18"],
    [{ when: "last-week" }, "2026-10-05", "2026-10-11"],
    [{ when: "this-month" }, "2026-10-01", "2026-10-31"],
    [{ when: "last-month" }, "2026-09-01", "2026-09-30"],
    [{ when: "last-year" }, "2025-01-01", "2025-12-31"],
    [{ lastDays: 30 }, "2026-09-18", "2026-10-18"],
    [{ month: "march" }, "2026-03-01", "2026-03-31"],
    [{ month: "december" }, "2025-12-01", "2025-12-31"],
    [{ month: "february", year: 2024 }, "2024-02-01", "2024-02-29"],
    [{ year: 2023 }, "2023-01-01", "2023-12-31"],
    [{ month: "october", day: 15 }, "2026-10-15", "2026-10-15"],
    [{ month: "october", day: 19 }, "2025-10-19", "2025-10-19"], // after today: last year's
    [{ month: "march", day: 3, year: 2025 }, "2025-03-03", "2025-03-03"],
    [{ month: "february", day: 29 }, "2024-02-29", "2024-02-29"],
  ];
  for (const [q, from, to] of cases) assert.deepStrictEqual(Orders.resolve(q, SUNDAY), { from, to }, JSON.stringify(q));
  assert.strictEqual(Orders.resolve({ status: "paid" }, SUNDAY), null);
});

test("last week on a Monday ends yesterday", () => {
  assert.deepStrictEqual(Orders.resolve({ when: "last-week" }, "2026-10-19"), { from: "2026-10-12", to: "2026-10-18" });
});

test("confidence reports the weakest and the mean token", () => {
  const c = Orders.confidence([{ prob: 1 }, { prob: 0.5 }, { prob: 0.9 }]);
  assert.strictEqual(c.min, 0.5);
  assert.ok(Math.abs(c.mean - 0.8) < 1e-12);
  assert.deepStrictEqual(Orders.confidence([]), { min: 1, mean: 1 });
});

test("answer combines the model's reply, validation, dates and confidence", () => {
  const generate = () => ({ text: "orders find --status unpaid --when last-month",
    tokens: [{ text: "orders", prob: 1 }, { text: " find", prob: 0.7 }] });
  const r = Orders.answer(generate, "unpaid orders from last month", SUNDAY);
  assert.strictEqual(r.query.status, "unpaid");
  assert.deepStrictEqual(r.range, { from: "2026-09-01", to: "2026-09-30" });
  assert.strictEqual(r.min, 0.7);
  const bad = Orders.answer(() => ({ text: "orders find --status nope", tokens: [] }), "x", SUNDAY);
  assert.ok(bad.error && !bad.query);
  const off = Orders.answer(() => ({ text: "orders unknown", tokens: [] }), "weather", SUNDAY);
  assert.strictEqual(off.query.command, "unknown");
  assert.strictEqual(off.range, undefined);
});

test("order ids and before/after dates", () => {
  assert.deepStrictEqual(Orders.parseReply("orders find --id 14299").query, { command: "find", id: 14299 });
  assert.deepStrictEqual(Orders.parseReply("orders find --before october-15").query, { command: "find", before: { month: "october", day: 15 } });
  assert.deepStrictEqual(Orders.parseReply("orders find --after march-2024 --before 2025").query,
    { command: "find", after: { month: "march", year: 2024 }, before: { year: 2025 } });
  for (const reply of [
    "orders find --id 5 --status paid", "orders count --id 5", "orders find --before 15-october", "orders find --after smarch",
    "orders find --before february-30", "orders find --before march --month may", "orders find --id abc",
  ]) assert.ok(Orders.parseReply(reply).error, `accepted: ${reply}`);
});

test("resolve before and after", () => {
  const cases = [
    [{ before: { month: "october", day: 15 } }, null, "2026-10-14"],
    [{ after: { month: "march" } }, "2026-04-01", null],
    [{ after: { month: "march", day: 3, year: 2025 } }, "2025-03-04", null],
    [{ before: { year: 2025 } }, null, "2024-12-31"],
    [{ after: { year: 2023 }, before: { month: "june", year: 2025 } }, "2024-01-01", "2025-05-31"],
  ];
  for (const [q, from, to] of cases) assert.deepStrictEqual(Orders.resolve(q, SUNDAY), { from, to }, JSON.stringify(q));
});

test("snapName copies the customer from the request", () => {
  const cases = [
    ["Perphlen", "orders for customer Periklis", "Periklis"],
    ["Perphyr", "orders from periklis", "periklis"],
    ["Pied Smith", "orders by Peri Smith", "Peri Smith"],
    ["Pernim", "Periklis orders sorted by amount", "Periklis"],
    ["Hooli", "Hooli's refunded purchases", "Hooli"],
    ["Acme Corp", "how many orders did acme corp place?", "Acme Corp"],
    ["Theodoros Kyri", "Theodoros Kyriakopoulos orders", "Theodoros Kyriakopoulos"], // cut off
    ["Yopenim", "how many orders did Yiannoula place", "Yiannoula"], // the only unfamiliar word
  ];
  for (const [name, request, want] of cases) {
    const s = Orders.snapName(name, request);
    assert.strictEqual(s.name, want, request);
  }
  assert.strictEqual(Orders.snapName("Zzyzx", "unpaid orders from last month").missing, true);
});

test("answer corrects the customer and stops counting the name's weak pieces", () => {
  const tokens = [{ text: "orders", prob: 1 }, { text: " find", prob: 1 }, { text: " --customer", prob: 1 },
    { text: " Per", prob: 0.9 }, { text: "phlen", prob: 0.3 }];
  const r = Orders.answer(() => ({ text: "orders find --customer Perphlen", tokens }), "orders for customer Periklis", SUNDAY);
  assert.strictEqual(r.query.customer, "Periklis");
  assert.strictEqual(r.fixes.length, 1);
  assert.strictEqual(r.min, 1);
});
