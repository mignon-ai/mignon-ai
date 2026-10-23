// Order-search logic for the demo page: everything that knows what an order query is. The model only
// writes a command-style reply such as
//   orders find --status unpaid --category electronics --min-amount 500 --when last-month
// This file validates that reply, turns its symbolic dates into days, and decides how sure the model
// was. It works in the page (global `Orders`) and in Node (`require("./orders.js")`) for the tests.
(function (root, factory) {
  if (typeof module === "object" && module.exports) module.exports = factory();
  else root.Orders = factory();
})(typeof self !== "undefined" ? self : this, function () {
  "use strict";

  const COMMANDS = ["find", "count", "total"];
  const STATUSES = ["paid", "unpaid", "overdue", "refunded", "cancelled", "shipped", "pending"];
  const CATEGORIES = ["electronics", "books", "clothing", "home", "toys", "sports", "beauty", "groceries"];
  const SORTS = ["newest", "oldest", "highest", "lowest"];
  const WHENS = ["today", "yesterday", "this-week", "last-week", "this-month", "last-month", "this-year", "last-year"];
  const MONTHS = ["january", "february", "march", "april", "may", "june", "july", "august", "september",
    "october", "november", "december"];
  const UNKNOWN = "unknown"; // the reply to a request that is not an order search
  const MAX_REPLY_TOKENS = 80;

  // option -> [query field, kind, allowed values or [min, max]]
  const OPTIONS = {
    status: ["status", "word", STATUSES],
    category: ["category", "word", CATEGORIES],
    sort: ["sort", "word", SORTS],
    when: ["when", "word", WHENS],
    month: ["month", "word", MONTHS],
    "min-amount": ["minAmount", "number", [1, 1e9]],
    "max-amount": ["maxAmount", "number", [1, 1e9]],
    "last-days": ["lastDays", "number", [1, 3650]],
    day: ["day", "number", [1, 31]],
    id: ["id", "number", [1, 1e9]],
    before: ["before", "edge", null],
    after: ["after", "edge", null],
    year: ["year", "number", [1990, 2100]],
    limit: ["limit", "number", [1, 1000]],
    customer: ["customer", "text", null],
  };

  // Splits on spaces and keeps "double quoted" text as one word.
  function splitWords(s) {
    const words = [];
    let cur = "", inQuote = false, has = false;
    for (const ch of s) {
      if (ch === '"') { inQuote = !inQuote; has = true; }
      else if (ch === " " && !inQuote) { if (has) { words.push(cur); cur = ""; has = false; } }
      else { cur += ch; has = true; }
    }
    if (inQuote) throw new Error("unterminated quote");
    if (has) words.push(cur);
    return words;
  }

  function daysIn(month, year) { // month 1..12
    return new Date(Date.UTC(year, month, 0)).getUTCDate();
  }

  // Validates a model reply. Returns {query} or {error}.
  function parseReply(reply) {
    try {
      return { query: parse(reply) };
    } catch (e) {
      return { error: e.message };
    }
  }

  function parse(reply) {
    const words = splitWords(reply);
    if (words.length < 2 || words[0] !== "orders") throw new Error('reply does not start with "orders <command>"');
    const q = { command: words[1] };
    if (q.command === UNKNOWN) {
      if (words.length !== 2) throw new Error('"unknown" takes no options');
      return q;
    }
    if (!COMMANDS.includes(q.command)) throw new Error(`unknown command "${q.command}"`);
    const seen = new Set();
    for (let i = 2; i < words.length; i += 2) {
      const key = words[i], value = words[i + 1];
      if (!key.startsWith("--") || value === undefined) throw new Error(`expected an option with a value at "${key}"`);
      if (seen.has(key)) throw new Error(`option ${key} given twice`);
      seen.add(key);
      const spec = OPTIONS[key.slice(2)];
      if (!spec) throw new Error(`unknown option ${key}`);
      const [field, kind, allowed] = spec;
      if (kind === "word") {
        if (!allowed.includes(value)) throw new Error(`${key}: "${value}" is not one of ${allowed.join(", ")}`);
        q[field] = value;
      } else if (kind === "number") {
        const n = Number(value);
        if (!/^\d+$/.test(value) || n < allowed[0] || n > allowed[1]) {
          throw new Error(`${key}: "${value}" is not a number from ${allowed[0]} to ${allowed[1]}`);
        }
        q[field] = n;
      } else if (kind === "edge") {
        q[field] = parseEdge(key, value);
      } else {
        if (!value.trim()) throw new Error(`${key} is empty`);
        q[field] = value;
      }
    }
    check(q);
    return q;
  }

  // The moment after --before / --after: "october-15", "october-15-2025", "march", "march-2024" or "2024".
  // Returns {month?, day?, year?} in the shape resolve() takes.
  function parseEdge(key, value) {
    if (/^\d{4}$/.test(value)) return { year: Number(value) };
    const m = /^([a-z]+)(?:-(\d{1,2}))?(?:-(\d{4}))?$/.exec(value);
    if (!m || !MONTHS.includes(m[1])) throw new Error(`${key}: "${value}" is not a date like october-15, march-2024 or 2024`);
    const edge = { month: m[1] };
    if (m[2]) edge.day = Number(m[2]);
    if (m[3]) edge.year = Number(m[3]);
    if (edge.day && (edge.day < 1 || edge.day > daysIn(MONTHS.indexOf(edge.month) + 1, edge.year || 2024))) {
      throw new Error(`${key}: ${edge.month} has no day ${edge.day}`);
    }
    return edge;
  }

  // Rejects combinations that contradict each other.
  function check(q) {
    if (q.minAmount && q.maxAmount && q.minAmount > q.maxAmount) {
      throw new Error(`--min-amount ${q.minAmount} is above --max-amount ${q.maxAmount}`);
    }
    if (q.day) {
      if (!q.month) throw new Error("--day needs a --month");
      // Without a year, February may have 29 days: the most recent leap year is used.
      if (q.day > daysIn(MONTHS.indexOf(q.month) + 1, q.year || 2024)) throw new Error(`${q.month} has no day ${q.day}`);
    }
    if ([q.when, q.lastDays, q.month || q.year, q.before || q.after].filter(Boolean).length > 1) throw new Error("more than one kind of date range");
    if (q.id && Object.keys(q).length > 2) throw new Error("--id stands alone");
    if (q.command !== "find" && (q.sort || q.limit || q.id)) throw new Error("--sort, --limit and --id only apply to find");
  }

  // ---- dates: UTC calendar days as "YYYY-MM-DD" strings ----
  const dayStart = (y, m, d) => new Date(Date.UTC(y, m, d)); // m is 0-based; overflow rolls over
  const iso = d => d.toISOString().slice(0, 10);

  // Turns the query's symbolic date into an inclusive {from, to} range relative to today (a
  // "YYYY-MM-DD" string or a Date), or null when the query has no date. Weeks run Monday to Sunday.
  // A month without a year means its most recent occurrence up to today.
  function resolve(q, today) {
    const t = typeof today === "string" ? new Date(today + "T00:00:00Z") : today;
    if (q.before || q.after) {
      // Open ranges: "before X" ends the day before X starts, "after X" starts the day after X ends.
      const add = (day, n) => { const d = new Date(day + "T00:00:00Z"); d.setUTCDate(d.getUTCDate() + n); return iso(d); };
      const range = { from: null, to: null };
      if (q.after) range.from = add(resolve(q.after, t).to, 1);
      if (q.before) range.to = add(resolve(q.before, t).from, -1);
      return range;
    }
    const y = t.getUTCFullYear(), m = t.getUTCMonth(), d = t.getUTCDate();
    const range = (from, to) => ({ from: iso(from), to: iso(to) });
    const monday = d - ((t.getUTCDay() + 6) % 7);
    if (q.when) {
      switch (q.when) {
        case "today": return range(t, t);
        case "yesterday": return range(dayStart(y, m, d - 1), dayStart(y, m, d - 1));
        case "this-week": return range(dayStart(y, m, monday), dayStart(y, m, monday + 6));
        case "last-week": return range(dayStart(y, m, monday - 7), dayStart(y, m, monday - 1));
        case "this-month": return range(dayStart(y, m, 1), dayStart(y, m + 1, 0));
        case "last-month": return range(dayStart(y, m - 1, 1), dayStart(y, m, 0));
        case "this-year": return range(dayStart(y, 0, 1), dayStart(y, 11, 31));
        case "last-year": return range(dayStart(y - 1, 0, 1), dayStart(y - 1, 11, 31));
      }
      return null;
    }
    if (q.lastDays) return range(dayStart(y, m, d - q.lastDays), t);
    if (q.month) {
      const mi = MONTHS.indexOf(q.month);
      let year = q.year;
      if (q.day) {
        if (!year) { // the most recent occurrence that is not in the future (back to a leap year for 29 February)
          year = y;
          while (dayStart(year, mi, q.day).getUTCMonth() !== mi || dayStart(year, mi, q.day) > t) year--;
        }
        const day = dayStart(year, mi, q.day);
        return range(day, day);
      }
      if (!year) year = mi > m ? y - 1 : y;
      return range(dayStart(year, mi, 1), dayStart(year, mi + 1, 0));
    }
    if (q.year) return range(dayStart(q.year, 0, 1), dayStart(q.year, 11, 31));
    return null;
  }

  // ---- customer names ----
  // The model writes a name by predicting it piece by piece, and long unusual names drift ("Periklis" ->
  // "Perphlen"). Code copies exactly, so the name is checked against the request and, when it is not there,
  // replaced by the closest run of words that is.
  const NOT_NAMES = new Set(("a an the and or of to in on at by for from with without that which who whom whose is are was were be " +
    "been have has had do does did it its my our we us me you your i show find list get give need pull up display can could would " +
    "please search look lookup fetch what whats let see any how many much count number total sum revenue all every each everything " +
    "place places placed made bought ordered order orders invoice invoices purchase purchases transaction transactions sale sales customer " +
    "customers client clients over under above below less more than least most at between up down today yesterday tomorrow week " +
    "weeks month months year years day days last past this next first top bottom only limit newest oldest latest earliest recent " +
    "biggest largest smallest cheapest priciest highest lowest expensive priced value amount amounts sorted sort by descending " +
    "ascending before after prior earlier later since until during about around bucks dollars dollar usd still not yet been " +
    "waiting payment processed their them they those these there here are").split(/\s+/));
  [...STATUSES, ...CATEGORIES, ...MONTHS, "settled", "outstanding", "open", "late", "past", "due", "returned", "canceled",
    "dispatched", "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec"].forEach(w => NOT_NAMES.add(w));

  function editDistance(a, b) {
    let prev = Array.from({ length: b.length + 1 }, (_, j) => j);
    for (let i = 1; i <= a.length; i++) {
      const cur = [i];
      for (let j = 1; j <= b.length; j++) {
        cur[j] = Math.min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (a[i - 1] === b[j - 1] ? 0 : 1));
      }
      prev = cur;
    }
    return prev[b.length];
  }

  // Returns {name, changed} with the name as it appears in the request, or {name, missing: true} when
  // nothing in the request resembles it.
  function snapName(name, request) {
    const lower = name.toLowerCase();
    const whole = new RegExp("(^|[^\\p{L}\\p{N}])" + lower.replace(/[.*+?^${}()|[\]\\]/g, "\\$&") + "($|[^\\p{L}\\p{N}])", "u");
    if (whole.test(request.toLowerCase())) return { name }; // already there as whole words (not a cut-off "Kyri")
    const words = request.split(/\s+/).map(w => w.replace(/^[^\p{L}\p{N}]+|[^\p{L}\p{N}]+$/gu, "").replace(/['’]s$/i, "")).filter(Boolean);
    const n = name.split(/\s+/).length;
    let best = null;
    for (let len = Math.max(1, n - 1); len <= n + 1; len++) {
      for (let i = 0; i + len <= words.length; i++) {
        const span = words.slice(i, i + len);
        if (span.some(w => NOT_NAMES.has(w.toLowerCase()) || /^[\d$#<>=+.,-]+$/.test(w))) continue;
        const cand = span.join(" ");
        const dist = cand.toLowerCase().startsWith(lower) ? 0 : editDistance(cand.toLowerCase(), lower) / Math.max(cand.length, lower.length);
        const prefix = cand.toLowerCase()[0] === lower[0];
        if (dist <= 0.75 && (prefix || dist <= 0.4) && (!best || dist < best.dist)) best = { name: cand, dist };
      }
    }
    if (!best) {
      // When the request has one run of unfamiliar words, that run is the name.
      const runs = [];
      let run = [];
      for (const w of [...words, ""]) {
        if (w && !NOT_NAMES.has(w.toLowerCase()) && !/^[\d$#<>=+.,-]+$/.test(w)) run.push(w);
        else if (run.length) { runs.push(run); run = []; }
      }
      if (runs.length === 1 && runs[0].length <= 3 && runs[0][0][0].toLowerCase() === lower[0]) best = { name: runs[0].join(" ") };
    }
    return best ? { name: best.name, changed: true } : { name, missing: true };
  }

  // The weakest and the mean probability of the tokens the model chose.
  function confidence(tokens) {
    if (!tokens.length) return { min: 1, mean: 1 };
    const probs = tokens.map(t => t.prob);
    return { min: Math.min(...probs), mean: probs.reduce((a, b) => a + b, 0) / probs.length };
  }

  // Asks the model (generate(prompt, maxTokens) -> {text, tokens}) and returns everything the page shows.
  function answer(generate, request, today) {
    const started = Date.now();
    const out = generate(request.trim(), MAX_REPLY_TOKENS);
    const reply = out.text.trim();
    const tokens = out.tokens || [];
    const res = { request, reply, tokens, ...confidence(tokens), fixes: [], warnings: [] };
    const parsed = parseReply(reply);
    if (parsed.error) res.error = parsed.error;
    else {
      res.query = parsed.query;
      if (res.query.customer) {
        const snap = snapName(res.query.customer, request);
        if (snap.changed) {
          res.customerFrom = res.query.customer;
          res.fixes.push(`customer "${res.query.customer}" corrected to "${snap.name}", as written in your request`);
          // the weak pieces were the name's: they no longer count against the reply
          const pieces = tokens.map(t => t.text), text = pieces.join(""), at = text.indexOf(res.query.customer);
          let pos = 0;
          const rest = tokens.filter((t, i) => { const from = pos; pos += pieces[i].length; return at < 0 || pos <= at || from >= at + res.query.customer.length; });
          Object.assign(res, confidence(rest));
          res.query.customer = snap.name;
        } else if (snap.missing) res.warnings.push(`the customer "${snap.name}" does not appear in your request`);
      }
      if (parsed.query.command !== UNKNOWN) res.range = resolve(parsed.query, today);
    }
    res.decodeMs = Date.now() - started;
    return res;
  }

  return { snapName, COMMANDS, STATUSES, CATEGORIES, SORTS, WHENS, MONTHS, UNKNOWN, parseReply, resolve, confidence, answer };
});
