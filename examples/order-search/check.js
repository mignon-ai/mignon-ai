// Runs the page's pipeline (wasm model + orders.js) in Node and scores it on a file of request/reply pairs.
//   node examples/order-search/check.js [pairs.json]        score a file (default: handwritten.json)
//   node examples/order-search/check.js "unpaid orders"     ask one or more requests
//   MODEL=path/to/int8.json node examples/order-search/check.js ...     try another model
// Needs the package built by build.sh (packages/mignon-web/dist). A reply counts as correct when it matches after the
// customer name has been corrected from the request, exactly as the page does it.
const fs = require("fs");
const { pathToFileURL } = require("url");
const here = __dirname + "/";
const Orders = require(here + "orders.js");

(async () => {
  const dist = here + "../../packages/mignon-web/dist/mignon.mjs";
  if (!fs.existsSync(dist)) { console.error("build the package first: examples/order-search/build.sh"); process.exit(1); }
  const Mignon = await import(pathToFileURL(dist).href);
  const model = await Mignon.load({ model: process.env.MODEL || here + "orders.mgn" });
  const today = new Date().toISOString().slice(0, 10);
  const ask = q => Orders.answer((p, n) => model.generate(p, { maxTokens: n }), q, today);
  // the reply with the corrected customer name in place of the model's
  const corrected = r => {
    if (!r.customerFrom) return r.reply;
    const quote = /\s/.test(r.query.customer) ? '"' : "";
    const from = r.customerFrom.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
    return r.reply.replace(new RegExp('"?' + from + '"?'), () => quote + r.query.customer + quote);
  };

  const args = process.argv.slice(2);
  if (args.length === 0 || (args.length === 1 && args[0].endsWith(".json"))) {
    const pairs = JSON.parse(fs.readFileSync(args[0] || here + "handwritten.json"));
    let ok = 0;
    for (const p of pairs) {
      const r = ask(p.instruction), got = corrected(r);
      if (got.toLowerCase() === p.output.toLowerCase()) ok++;
      else console.log(`MISS ${JSON.stringify(p.instruction)}\n  got  ${got} (weakest piece ${r.min.toFixed(2)})\n  want ${p.output}`);
    }
    console.log(`${ok}/${pairs.length} correct (${(100 * ok / pairs.length).toFixed(1)}%)`);
  } else {
    for (const q of args) {
      const r = ask(q);
      console.log(q, "=>", corrected(r), `| weakest piece ${r.min.toFixed(2)}`, r.range ? `| ${r.range.from}..${r.range.to}` : "", r.error || "");
    }
  }
  process.exit(0);
})();
