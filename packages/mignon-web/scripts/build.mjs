// Builds the package into dist/: the wasm program, Go's wasm_exec.js folded into two JavaScript files (an ES
// module and a plain script that defines `Mignon`), the types, the licence notice and the SRI hashes.
//   node scripts/build.mjs
import { execFileSync } from "node:child_process";
import { createHash } from "node:crypto";
import { copyFileSync, mkdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const pkgDir = resolve(here, "..");
const repo = resolve(pkgDir, "../..");
const dist = join(pkgDir, "dist");
const version = JSON.parse(readFileSync(join(pkgDir, "package.json"), "utf8")).version;
const run = (cmd, args, opts = {}) => execFileSync(cmd, args, { stdio: ["ignore", "pipe", "inherit"], ...opts }).toString().trim();

rmSync(dist, { recursive: true, force: true });
mkdirSync(dist, { recursive: true });

// 1. the wasm program, with the package version inside
run("go", ["build", "-trimpath", "-ldflags", `-s -w -X main.version=${version}`, "-o", join(dist, "mignon-web.wasm"), "./cmd/mignon-web"],
  { cwd: repo, env: { ...process.env, GOOS: "js", GOARCH: "wasm" } });

// 2. Go's loader for it: it has to be the one that belongs to the Go version that built the wasm
const goroot = run("go", ["env", "GOROOT"]);
const wasmExec = readFileSync(join(goroot, "lib", "wasm", "wasm_exec.js"), "utf8");
const goVersion = run("go", ["env", "GOVERSION"]);
const wrapper = readFileSync(join(pkgDir, "src", "wrapper.js"), "utf8");
const banner = `/*! @mignon-ai/mignon-web ${version} (engine built with ${goVersion}). Includes wasm_exec.js from the Go project (BSD-3-Clause), see NOTICE. */\n`;

const esm = banner + wasmExec + "\n" +
  wrapper.replaceAll("DEFAULT_WASM", "new URL(\"./mignon-web.wasm\", import.meta.url)") +
  "\nexport { load, version, Model };\n";
// as a classic script the default wasm location is the folder of the file (CommonJS) or of the script tag (page)
const script = banner + "var Mignon = (function () {\n" + wasmExec + "\n" +
  "const scriptSrc = typeof document !== \"undefined\" && document.currentScript ? document.currentScript.src : \"\";\n" +
  wrapper.replaceAll("DEFAULT_WASM", "(typeof __dirname === \"string\" ? __dirname + \"/mignon-web.wasm\" : scriptSrc ? new URL(\"mignon-web.wasm\", scriptSrc).href : \"mignon-web.wasm\")") +
  "\nreturn { load, version, Model };\n})();\n" +
  "if (typeof module === \"object\" && module.exports) module.exports = Mignon;\n";
writeFileSync(join(dist, "mignon.mjs"), esm);
writeFileSync(join(dist, "mignon.js"), script);

copyFileSync(join(pkgDir, "src", "index.d.ts"), join(dist, "index.d.ts"));
copyFileSync(join(pkgDir, "NOTICE"), join(dist, "NOTICE"));

// 3. subresource-integrity hashes for <script integrity="..."> and <link rel="preload">
const integrity = {};
for (const file of ["mignon.js", "mignon.mjs", "mignon-web.wasm"]) {
  integrity[file] = "sha384-" + createHash("sha384").update(readFileSync(join(dist, file))).digest("base64");
}
writeFileSync(join(dist, "integrity.json"), JSON.stringify({ version, ...integrity }, null, 2) + "\n");
console.log(`built ${version} with ${goVersion} into ${dist}`);
