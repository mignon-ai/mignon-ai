// Makes the tarball `npm pack` would make (name-version.tgz, files under package/), for installing with
// `npm install ./mignon-ai-mignon-web-0.1.0.tgz` or uploading, without needing npm here.
import { execFileSync } from "node:child_process";
import { cpSync, mkdirSync, readFileSync, rmSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const pkgDir = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const pkg = JSON.parse(readFileSync(join(pkgDir, "package.json"), "utf8"));
const out = join(pkgDir, "pack");
rmSync(out, { recursive: true, force: true });
mkdirSync(join(out, "package"), { recursive: true });
for (const f of ["package.json", ...pkg.files]) cpSync(join(pkgDir, f), join(out, "package", f), { recursive: true });
const name = `${pkg.name.replace("@", "").replace("/", "-")}-${pkg.version}.tgz`;
execFileSync("tar", ["-czf", join(pkgDir, name), "-C", out, "package"]);
rmSync(out, { recursive: true, force: true });
console.log(name);
