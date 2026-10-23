#!/usr/bin/env python3
"""
build_git_sft.py

Generates a natural, tag-free, high-entropy supervised fine-tuning (SFT) dataset for Git
with open-vocabulary entity synthesis and explicit induction copy-task training:
[
  {
    "instruction": "how to push main",
    "output": "git push origin main"
  },
  {
    "instruction": "stage changes in mignon",
    "output": "git add mignon"
  },
  {
    "instruction": "stage changes in models/user.env.example2",
    "output": "git add models/user.env.example2"
  }
]

Key Principles:
1. ZERO artificial slot tags (<val>, <branch>, etc.) in instructions or outputs.
2. Full phrasal coverage for all base and shorthand git queries:
   - "how to push {b}", "push {b}", "how to push", "push"
   - "how to pull {b}", "pull {b}", "pull", "pull with rebase"
   - "checkout {b}", "switch {b}", "branch {b}"
   - "stage {f}", "add {f}", "stage changes in {f}"
3. Open-vocabulary entity synthesis:
   - Filepaths include single-word identifiers ('mignon', 'foo', 'data', 'temp'),
     numbered/versioned extensions ('models/user.env.example2', 'test1', 'app2', 'v2'),
     and arbitrary developer identifiers.
   - Usernames include dotted ('p.stinis'), single names ('mignon'), underscored, numbered.
   - Branches include bare branches ('main', 'master', 'mignon', 'dev') and prefixed ones.
4. Induction / Copy-Task reinforcement:
   - Thousands of randomized entity pairs ensuring the CA associative memory learns
     in-context copying rather than memorizing a closed vocabulary list.
"""

import json
import os
import random
import string

from git_sft_render import pick, rephrase

random.seed(42)

OUTPUT_FILE = "data/git_sft.json"
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

# ==============================================================================
# Dynamic High-Entropy Entity Generators
# ==============================================================================

FIRST_NAMES = [
    "alex", "alice", "bob", "carol", "dan", "elena", "frank", "grace", "heidi",
    "ivan", "judy", "kevin", "laura", "mike", "nina", "oliver", "mignon", "quinn",
    "rachel", "sam", "tina", "victor", "wendy", "xander", "yara", "zack",
    "john", "jane", "peter", "david", "sarah", "chris", "emma", "lucas", "mark"
]

LAST_NAMES = [
    "stinis", "smith", "jones", "williams", "brown", "taylor", "davies", "wilson",
    "evans", "thomas", "johnson", "roberts", "walker", "wright", "robinson", "thompson",
    "white", "hughes", "edwards", "green", "hall", "wood", "harris", "martin",
    "clark", "cooper", "king", "baker", "harrison", "morgan", "chen", "patel"
]

COMMON_IDENTIFIERS = [
    "mignon", "test", "foo", "bar", "data", "temp", "build", "dist", "assets", "out",
    "demo", "example", "worker", "runner", "server", "client", "handler", "router",
    "model", "schema", "store", "cache", "token", "auth", "user", "payment",
    "billing", "report", "analytics", "logger", "config", "settings", "backup"
]

FILE_DIRS = [
    "src", "pkg", "internal", "components", "lib", "tests", "config", "scripts",
    "assets", "docs", "api", "cmd", "server", "client", "ui", "models", "utils",
    "internal/auth", "src/controllers", "pkg/database", "components/ui", "lib/crypto",
    "api/v1", "tests/unit", "tests/e2e", "config/env", "scripts/deploy", "models/user"
]

BASE_FILENAMES = [
    "main", "app", "index", "server", "router", "handler", "middleware", "auth",
    "user", "account", "database", "config", "settings", "schema", "queries",
    "helper", "utils", "types", "constants", "client", "worker", "queue",
    "button", "navbar", "modal", "card", "table", "dashboard", "docker-compose",
    "Makefile", "Dockerfile", "package", "Cargo", "requirements", "README", "env"
]

EXTENSIONS = [
    ".go", ".py", ".ts", ".tsx", ".js", ".jsx", ".rs", ".json", ".yaml", ".yml",
    ".md", ".sql", ".sh", ".html", ".css", ".toml", ".env", ".env.example",
    ".env.example2", ".example2", ".v2", ".bak", ".tmp", ".log", ".tar.gz"
]

BRANCH_PREFIXES = [
    "feature/", "feat/", "fix/", "bugfix/", "hotfix/", "chore/", "refactor/",
    "test/", "docs/", "perf/", "exmignonment/", "release/", "staging/", "jira/", "issue-"
]

BRANCH_WORDS = [
    "login", "user-auth", "jwt-rotation", "db-pool", "oauth2-flow", "navbar-ui",
    "checkout-cart", "speedup-simd", "payment-stripe", "webhook-listener",
    "profile-page", "dark-mode", "rate-limiter", "session-store", "email-notifications",
    "mobile-responsive", "search-indexing", "cache-redis", "api-gateway", "logging-v2"
]

ROLES = ["dev", "admin", "lead", "ops", "eng", "bot", "team", "core", "coder", "ninja"]
DOMAINS = ["gmail.com", "github.com", "corp.net", "company.org", "dev.io", "startup.ai", "acme.com", "cloud.tech"]
REMOTES = ["origin", "upstream", "backup", "staging", "production", "gh", "gitlab"]

def random_synthetic_word():
    # Produces random pronounceable developer identifiers (e.g. klor, mignon, zent, valox)
    consonants = "bcdfghjklmnpqrstvwxyz"
    vowels = "aeiou"
    length = random.randint(3, 6)
    w = ""
    for i in range(length):
        w += random.choice(consonants if i % 2 == 0 else vowels)
    return w

def random_username():
    style = random.randint(1, 6)
    first = random.choice(FIRST_NAMES)
    last = random.choice(LAST_NAMES)
    num = random.randint(1, 99)
    if style == 1:
        return f"{first[0]}.{last}"      # e.g. p.stinis
    elif style == 2:
        return f"{first}_{last}"         # e.g. alex_smith
    elif style == 3:
        return f"{first}.{last}"         # e.g. john.doe
    elif style == 4:
        return f"{first}{num}"           # e.g. mignon92
    elif style == 5:
        role = random.choice(ROLES)
        return f"{first}-{role}"         # e.g. alex-dev
    else:
        return first                     # e.g. mignon

def random_email(user=None):
    if not user:
        user = random_username()
    domain = random.choice(DOMAINS)
    clean_user = user.replace(" ", "").replace("'", "").lower()
    return f"{clean_user}@{domain}"

def random_filepath():
    style = random.randint(1, 6)
    if style == 1:
        # Bare single identifier: e.g. 'mignon', 'test', 'foo', 'data', 'temp'
        return random.choice(COMMON_IDENTIFIERS)
    elif style == 2:
        # Numbered identifier: e.g. 'test2', 'mignon2', 'app2', 'v2', 'example2'
        base = random.choice(COMMON_IDENTIFIERS)
        num = random.randint(1, 9)
        return f"{base}{num}"
    elif style == 3:
        # Standard root file
        name = random.choice(BASE_FILENAMES)
        ext = random.choice(EXTENSIONS)
        return f"{name}{ext}"
    elif style == 4:
        # Nested standard file
        d = random.choice(FILE_DIRS)
        name = random.choice(BASE_FILENAMES)
        ext = random.choice(EXTENSIONS)
        return f"{d}/{name}{ext}"
    elif style == 5:
        # File with versioned / numbered suffix: e.g. 'models/user.env.example2', 'config_v2.json'
        d = random.choice(["models", "config", "src", "tests", "data"])
        name = random.choice(["user", "app", "env", "settings", "backup"])
        suffix = random.choice([".env.example2", ".example2", "_v2.json", "2.go", "_test.py", ".bak"])
        return f"{d}/{name}{suffix}"
    else:
        # Synthetic project file
        w = random_synthetic_word()
        ext = random.choice([".go", ".py", ".ts", ".rs", ".json", ""])
        return f"{w}{ext}"

def random_branch():
    style = random.randint(1, 5)
    if style == 1:
        # Standard common branches
        return random.choice(["main", "master", "develop", "staging", "production", "dev", "test", "mignon"])
    elif style == 2:
        # Prefixed feature branch
        prefix = random.choice(BRANCH_PREFIXES)
        word = random.choice(BRANCH_WORDS)
        return f"{prefix}{word}"
    elif style == 3:
        # Ticket branch
        prefix = random.choice(["issue-", "JIRA-", "ticket-"])
        num = random.randint(100, 9999)
        word = random.choice(BRANCH_WORDS)
        return f"{prefix}{num}-{word}"
    elif style == 4:
        # Numbered / versioned branch
        base = random.choice(["release", "v", "test", "dev", "sprint"])
        num = random.randint(1, 5)
        return f"{base}{num}"
    else:
        # Bare custom word
        return random.choice(BRANCH_WORDS)

def random_commit_hash():
    return "".join(random.choices("0123456789abcdef", k=7))

def random_commit_message():
    actions = [
        "implement JWT refresh token rotation",
        "fix null pointer dereference on user logout",
        "accelerate CA forward pass with AVX2 SIMD",
        "add unit tests for payment webhook endpoints",
        "refactor database connection pool singleton",
        "update project dependencies to latest versions",
        "resolve layout overflow on mobile navigation bar",
        "add rate limiting middleware to public api",
        "initial commit of project structure and build files",
        "cleanup unused imports and deprecated functions",
        "handle connection timeout gracefully during sync",
        "support custom themes and dark mode preferences"
    ]
    style = random.randint(1, 3)
    if style == 1:
        t = random.choice(["feat", "fix", "perf", "refactor", "docs", "test", "chore"])
        s = random.choice(["auth", "ui", "db", "api", "core", "cli", "deps", "config"])
        act = random.choice(actions)
        return f"{t}({s}): {act}"
    else:
        act = random.choice(actions)
        return act

def random_tag():
    major = random.randint(0, 3)
    minor = random.randint(0, 15)
    patch = random.randint(0, 20)
    suffix = random.choice(["", "", "", "-rc.1", "-beta", "-alpha"])
    return f"v{major}.{minor}.{patch}{suffix}"

def random_url():
    user = random.choice(["org", "company", "team", "octocat", "devgroup"])
    repo = random.choice(["core", "mignon-ai", "api-service", "frontend", "utils", "tools"])
    if random.random() < 0.5:
        return f"https://github.com/{user}/{repo}.git"
    else:
        return f"git@github.com:{user}/{repo}.git"

# ==============================================================================
# Phrasal Intent Generators (Tag-Free, Natural, Comprehensive Coverage)
# ==============================================================================

def build_branch_samples(count=6000):
    samples = []
    for _ in range(count):
        b = random_branch()
        intent = random.randint(1, 9)
        if intent == 1:
            # Switch / Checkout existing branch
            prompts = [
                f"switch to branch {b}",
                f"checkout branch {b}",
                f"switch to {b}",
                f"checkout {b}",
                f"change to branch {b}",
                f"how do I switch to {b}?",
                f"how to checkout branch {b}?",
                f"move to branch {b}",
                f"please checkout {b}",
                f"git switch to {b}",
                f"git checkout {b}",
                f"switch {b}",
                f"goto {b}",
            ]
            samples.append((random.choice(prompts), f"git checkout {b}"))
        elif intent == 2:
            # Create and switch to new branch
            prompts = [
                f"create and switch to new branch {b}",
                f"create new branch {b} and checkout",
                f"make a new branch called {b} and switch to it",
                f"create and checkout branch {b}",
                f"how do I create a new branch {b} and switch to it?",
                f"branch off to {b}",
                f"start a new branch {b}",
                f"create branch {b} and move to it",
                f"new branch {b}",
                f"create branch {b}",
            ]
            samples.append((random.choice(prompts), f"git checkout -b {b}"))
        elif intent == 3:
            # Delete branch safely
            prompts = [
                f"delete local branch {b}",
                f"safely delete branch {b}",
                f"remove local branch {b}",
                f"how do I delete branch {b}?",
                f"delete the {b} branch locally",
                f"delete merged branch {b}",
                f"delete branch {b}",
            ]
            samples.append((random.choice(prompts), f"git branch -d {b}"))
        elif intent == 4:
            # Force delete branch
            prompts = [
                f"force delete unmerged branch {b}",
                f"force delete branch {b}",
                f"delete branch {b} forcefully",
                f"force remove local branch {b}",
                f"how to force delete branch {b}?",
                f"force delete {b}",
            ]
            samples.append((random.choice(prompts), f"git branch -D {b}"))
        elif intent == 5:
            # Rename current branch
            prompts = [
                f"rename current branch to {b}",
                f"change current branch name to {b}",
                f"how do I rename the current branch to {b}?",
                f"rename branch to {b}",
            ]
            samples.append((random.choice(prompts), f"git branch -m {b}"))
        elif intent == 6:
            # List local branches
            prompts = [
                "list all local branches",
                "show all local git branches",
                "view existing branches",
                "how do I list local branches?",
                "display branches",
                "list branches",
                "show branches",
            ]
            samples.append((random.choice(prompts), "git branch"))
        elif intent == 7:
            # List all branches including remote
            prompts = [
                "list all branches including remotes",
                "show both local and remote branches",
                "view all branches in repo",
                "how do I see remote branches too?",
                "list all branches",
            ]
            samples.append((random.choice(prompts), "git branch -a"))
        elif intent == 8:
            # Create branch without switching
            prompts = [
                f"create branch {b} without checking it out",
                f"make branch {b}",
                f"add branch {b}",
            ]
            samples.append((random.choice(prompts), f"git branch {b}"))
        else:
            # Delete remote branch
            prompts = [
                "delete remote branch {branch} on {remote}",
                "remove branch {branch} from remote {remote}",
                "how do I delete remote branch {branch}?",
                "delete {branch} from {remote}",
            ]
            samples.append(pick(prompts, "git push {remote} --delete {branch}", branch=b, remote=random.choice(REMOTES)))
    return samples

def build_staging_commit_samples(count=6500):
    samples = []
    for _ in range(count):
        f = random_filepath()
        m = random_commit_message()
        intent = random.randint(1, 8)
        if intent == 1:
            # Stage specific file / identifier (Supports single word, numbered, nested)
            prompts = [
                f"stage changes in {f}",
                f"stage {f}",
                f"add {f}",
                f"add file {f} to staging area",
                f"stage file {f}",
                f"how do I stage {f}?",
                f"how to stage changes in {f}",
                f"how do I stage file {f}?",
                f"add {f} to git index",
                f"prepare {f} for commit",
                f"git add {f}",
            ]
            samples.append((random.choice(prompts), f"git add {f}"))
        elif intent == 2:
            # Stage all files
            prompts = [
                "stage all modified and new files",
                "stage all changes",
                "add all files to staging",
                "how do I stage everything?",
                "git add all",
                "stage all tracked and untracked changes",
                "stage everything",
                "stage all",
                "add all",
            ]
            samples.append((random.choice(prompts), "git add ."))
        elif intent == 3:
            # Unstage specific file
            prompts = [
                f"unstage file {f}",
                f"remove {f} from staging area without deleting changes",
                f"how do I unstage {f}?",
                f"take {f} out of stage",
                f"unstage changes in {f}",
                f"unstage {f}",
            ]
            cmd = f"git restore --staged {f}" if random.random() < 0.8 else f"git reset HEAD {f}"
            samples.append((random.choice(prompts), cmd))
        elif intent == 4:
            # Commit staged changes with message
            prompts = [
                f"commit staged changes with message '{m}'",
                f"commit with message '{m}'",
                f"create commit saying '{m}'",
                f"how do I commit with comment '{m}'?",
                f"save staged changes with message '{m}'",
                f"git commit message '{m}'",
                f"commit '{m}'",
            ]
            samples.append((random.choice(prompts), f"git commit -m '{m}'"))
        elif intent == 5:
            # Add and commit all tracked
            prompts = [
                f"add and commit all tracked changes with message '{m}'",
                f"stage tracked files and commit with '{m}'",
                f"commit all modified files directly with message '{m}'",
                f"how do I commit all tracked files at once with message '{m}'?",
                f"commit all with message '{m}'",
            ]
            samples.append((random.choice(prompts), f"git commit -am '{m}'"))
        elif intent == 6:
            # Amend last commit message
            prompts = [
                f"amend the last commit message to '{m}'",
                f"change the previous commit message to '{m}'",
                f"how do I update last commit message to '{m}'?",
                f"edit last commit message to '{m}'",
                f"amend commit message '{m}'",
            ]
            samples.append((random.choice(prompts), f"git commit --amend -m '{m}'"))
        elif intent == 7:
            # Amend commit keeping message
            prompts = [
                "add staged changes into the last commit without changing message",
                "amend last commit without editing commit message",
                "combine staged changes with previous commit keeping same message",
                "how do I amend commit without editing message?",
                "amend without edit",
            ]
            samples.append((random.choice(prompts), "git commit --amend --no-edit"))
        else:
            # Unstage all
            prompts = [
                "unstage all staged files",
                "remove everything from staging area",
                "how do I unstage all files?",
                "clear the staging area",
                "unstage all",
            ]
            samples.append((random.choice(prompts), "git restore --staged ."))
    return samples

def build_remote_sync_samples(count=7000):
    samples = []
    for _ in range(count):
        b = random_branch()
        r = random.choice(REMOTES)
        url = random_url()
        intent = random.randint(1, 10)
        if intent == 1:
            # Bare push
            prompts = [
                "push committed changes to remote repository",
                "push changes to origin",
                "how do I push commits to remote?",
                "upload commits to remote",
                "how to push",
                "how do I push?",
                "push",
                "git push",
            ]
            samples.append((random.choice(prompts), "git push"))
        elif intent == 2:
            # Push specific branch (e.g. 'how to push main')
            prompts = [
                f"how to push {b}",
                f"how do I push {b}?",
                f"push {b}",
                f"push branch {b}",
                f"push {b} branch",
                f"how to push branch {b}",
                f"how do I push branch {b}?",
                f"upload branch {b}",
                f"git push {b}",
            ]
            samples.append((random.choice(prompts), f"git push origin {b}"))
        elif intent == 3:
            # Push branch to specific remote
            prompts = [
                "push {branch} to {remote}",
                "push branch {branch} to remote {remote}",
                "how to push {branch} to {remote}",
                "how do I push branch {branch} to {remote}?",
                "upload {branch} to {remote}",
            ]
            samples.append(pick(prompts, "git push {remote} {branch}", branch=b, remote=r))
        elif intent == 4:
            # Push and set upstream
            prompts = [
                "push branch {branch} and set upstream to {remote}",
                "push new branch {branch} tracking {remote}",
                "how do I push {branch} with upstream tracking?",
                "set upstream and push branch {branch} to {remote}",
                "push {branch} set upstream {remote}",
            ]
            samples.append(pick(prompts, "git push -u {remote} {branch}", branch=b, remote=r))
        elif intent == 5:
            # Force push with lease
            prompts = [
                "safely force push branch {branch} to {remote}",
                "force push with lease to {remote} {branch}",
                "how to force push safely without overwriting other's work?",
                "push force with lease {branch}",
                "force push {branch} to {remote}",
            ]
            samples.append(pick(prompts, "git push --force-with-lease {remote} {branch}", branch=b, remote=r))
        elif intent == 6:
            # Bare pull
            prompts = [
                "pull latest changes from remote",
                "how to pull",
                "how do I pull?",
                "pull",
                "pull updates",
                "git pull",
            ]
            samples.append((random.choice(prompts), "git pull"))
        elif intent == 7:
            # Pull specific branch
            prompts = [
                f"pull latest updates for {b}",
                f"pull branch {b}",
                f"how to pull {b}",
                f"how do I pull branch {b}?",
                f"fetch and merge branch {b}",
            ]
            samples.append((random.choice(prompts), f"git pull origin {b}"))
        elif intent == 8:
            # Pull with rebase
            prompts = [
                "pull latest changes from {remote} {branch} using rebase",
                "rebase pull from remote {remote} {branch}",
                "how do I pull with rebase to keep linear history from {remote}?",
                "git pull rebase from {remote} {branch}",
                "pull with rebase",
                "how to pull with rebase",
            ]
            samples.append(pick(prompts, "git pull --rebase {remote} {branch}", branch=b, remote=r))
        elif intent == 9:
            # Fetch all remotes
            prompts = [
                "fetch all branches and tags from all remotes",
                "fetch all remote changes and prune stale branches",
                "how do I fetch everything from remotes?",
                "update remote refs and prune deleted branches",
                "fetch all",
                "fetch prune",
            ]
            samples.append((random.choice(prompts), "git fetch --all --prune"))
        else:
            # Clone repo
            prompts = [
                f"clone repository from {url}",
                f"how do I clone git repository {url}?",
                f"download repository from {url}",
                f"clone {url}",
            ]
            samples.append((random.choice(prompts), f"git clone {url}"))
    return samples

def build_config_samples(count=5500):
    samples = []
    for _ in range(count):
        u = random_username()
        e = random_email(u)
        intent = random.randint(1, 8)
        if intent == 1:
            # Set global username (Supports dotted 'p.stinis', single 'mignon', etc.)
            prompts = [
                f"set global git username to {u}",
                f"set global git user name to {u}",
                f"configure git global username as {u}",
                f"change my git user name to {u} globally",
                f"how do I set my global git username to {u}?",
                f"set git user.name to {u} globally",
                f"update global username to {u}",
                f"git config global user.name {u}",
                f"make {u} my global git author name",
                f"change global username to {u}",
            ]
            samples.append((random.choice(prompts), f"git config --global user.name '{u}'"))
        elif intent == 2:
            # Set local username
            prompts = [
                f"set local git username to {u}",
                f"set git username for this repository to {u}",
                f"change user.name to {u} in current repo",
                f"how do I set local username to {u}?",
                f"set local user.name {u}",
            ]
            samples.append((random.choice(prompts), f"git config user.name '{u}'"))
        elif intent == 3:
            # Set global email
            prompts = [
                f"set global git email to {e}",
                f"configure my global git email as {e}",
                f"change git email to {e} globally",
                f"how do I set my global email to {e}?",
                f"set user.email to {e} in global config",
                f"update global git email to {e}",
                f"set global email {e}",
            ]
            samples.append((random.choice(prompts), f"git config --global user.email '{e}'"))
        elif intent == 4:
            # Set local email
            prompts = [
                f"set local git email to {e}",
                f"set email for current repo to {e}",
                f"configure local user.email as {e}",
            ]
            samples.append((random.choice(prompts), f"git config user.email '{e}'"))
        elif intent == 5:
            # List config
            prompts = [
                "list all git configuration settings",
                "show current git config",
                "view global and local git config",
                "how do I see my git settings?",
                "display git config list",
                "list git config",
            ]
            samples.append((random.choice(prompts), "git config --list"))
        elif intent == 6:
            # Set default editor
            ed = random.choice(["vim", "nano", "nvim", "code --wait", "emacs"])
            prompts = [
                f"set the default git editor to {ed}",
                f"change default text editor to {ed} for git",
                f"how do I configure {ed} as my git editor?",
                f"set git editor {ed}",
            ]
            samples.append((random.choice(prompts), f"git config --global core.editor '{ed}'"))
        elif intent == 7:
            # Set default initial branch
            b = random.choice(["main", "master", "develop"])
            prompts = [
                f"change default branch name to {b} globally",
                f"set init default branch to {b}",
                f"configure git to use {b} as the default initial branch",
            ]
            samples.append((random.choice(prompts), f"git config --global init.defaultBranch {b}"))
        else:
            # Unset config
            target = random.choice(["user.name", "user.email", "core.editor"])
            prompts = [
                f"unset global configuration for {target}",
                f"remove global config {target}",
                f"how do I delete global setting {target}?",
                f"unset global {target}",
            ]
            samples.append((random.choice(prompts), f"git config --global --unset {target}"))
    return samples

def build_undo_reset_samples(count=5000):
    samples = []
    for _ in range(count):
        f = random_filepath()
        c = random_commit_hash()
        intent = random.randint(1, 7)
        if intent == 1:
            # Discard changes in specific file
            prompts = [
                f"discard uncommitted changes in file {f}",
                f"revert uncommitted changes in {f}",
                f"discard changes in {f}",
                f"how do I undo changes in file {f}?",
                f"restore {f} to last committed state",
                f"throw away modifications in {f}",
                f"restore {f}",
                f"discard {f}",
            ]
            samples.append((random.choice(prompts), f"git restore {f}"))
        elif intent == 2:
            # Discard all local uncommitted changes
            prompts = [
                "discard all local uncommitted changes",
                "throw away all local modifications and reset",
                "revert all tracked files to HEAD",
                "how do I completely wipe all uncommitted changes?",
                "reset working directory to clean state",
                "hard reset working tree",
                "discard all changes",
            ]
            samples.append((random.choice(prompts), "git reset --hard HEAD"))
        elif intent == 3:
            # Soft reset last commit
            prompts = [
                "undo the last commit but keep changes staged",
                "how do I undo last commit keeping staged files?",
                "soft reset last commit",
                "rollback 1 commit but keep files staged",
                "soft reset HEAD~1",
            ]
            samples.append((random.choice(prompts), "git reset --soft HEAD~1"))
        elif intent == 4:
            # Mixed reset last commit
            prompts = [
                "undo the last commit and unstage changes",
                "how do I roll back 1 commit keeping code uncommitted?",
                "undo last commit but keep changes in working directory",
                "reset previous commit keeping code as unstaged edits",
                "undo last commit",
            ]
            samples.append((random.choice(prompts), "git reset HEAD~1"))
        elif intent == 5:
            # Hard reset last commit
            prompts = [
                "permanently delete the last commit and all its changes",
                "wipe last commit and discard its modifications completely",
                "hard reset 1 commit back",
                "delete previous commit permanently",
                "delete last commit",
            ]
            samples.append((random.choice(prompts), "git reset --hard HEAD~1"))
        elif intent == 6:
            # Revert commit
            prompts = [
                f"revert commit {c}",
                f"safely undo commit {c} with a new revert commit",
                f"how do I revert commit {c} in history?",
                f"create a revert commit for {c}",
            ]
            samples.append((random.choice(prompts), f"git revert {c}"))
        else:
            # Clean untracked files
            prompts = [
                "remove all untracked files and directories",
                "clean untracked files forcefully",
                "how do I delete all untracked files from repo?",
                "clean untracked files",
            ]
            samples.append((random.choice(prompts), "git clean -fd"))
    return samples

def build_stash_samples(count=4000):
    samples = []
    for _ in range(count):
        m = random_commit_message()
        idx = random.randint(0, 3)
        intent = random.randint(1, 6)
        if intent == 1:
            # Stash changes
            prompts = [
                "save temporary uncommitted changes to stash",
                "stash all current changes",
                "how do I stash working directory changes?",
                "temporarily put aside changes in git",
                "stash unstaged and staged files",
                "stash changes",
                "stash",
                "git stash",
            ]
            samples.append((random.choice(prompts), "git stash"))
        elif intent == 2:
            # Stash with message
            prompts = [
                f"stash current changes with description '{m}'",
                f"save changes to stash with label '{m}'",
                f"how do I stash changes with message '{m}'?",
                f"stash with message '{m}'",
            ]
            samples.append((random.choice(prompts), f"git stash push -m '{m}'"))
        elif intent == 3:
            # Stash untracked files
            prompts = [
                "stash all changes including untracked files",
                "stash everything including new untracked files",
                "how to stash untracked files too?",
                "stash untracked",
            ]
            samples.append((random.choice(prompts), "git stash -u"))
        elif intent == 4:
            # Pop stash
            prompts = [
                "apply the latest stashed changes and remove them from stash",
                "restore stashed changes and drop from stash list",
                "how do I pop the most recent stash?",
                "pop last git stash",
                "stash pop",
            ]
            samples.append((random.choice(prompts), "git stash pop"))
        elif intent == 5:
            # Apply stash
            prompts = [
                f"apply stash at index {idx} without removing it",
                f"restore changes from stash@{idx}",
                f"how to apply stash number {idx}?",
                f"apply stash {idx}",
            ]
            samples.append((random.choice(prompts), f"git stash apply stash@{{{idx}}}"))
        else:
            # List stashes
            prompts = [
                "list all saved stashes",
                "show stash list",
                "view all stashed changes in repo",
                "show stash",
            ]
            samples.append((random.choice(prompts), "git stash list"))
    return samples

def build_rebase_merge_samples(count=4000):
    samples = []
    for _ in range(count):
        b = random_branch()
        intent = random.randint(1, 7)
        if intent == 1:
            # Continue rebase
            prompts = [
                "continue rebase after resolving merge conflicts",
                "how do I continue rebase after resolving conflicts?",
                "resume rebase process",
                "continue git rebase",
                "continue rebase",
            ]
            samples.append((random.choice(prompts), "git rebase --continue"))
        elif intent == 2:
            # Abort rebase
            prompts = [
                "cancel ongoing rebase and return to original state",
                "abort current rebase operation",
                "how do I abort a rebase with conflicts?",
                "stop rebase and revert",
                "abort rebase",
            ]
            samples.append((random.choice(prompts), "git rebase --abort"))
        elif intent == 3:
            # Rebase onto branch
            prompts = [
                f"rebase current branch onto {b}",
                f"rebase commits on top of {b}",
                f"how do I rebase my work onto branch {b}?",
                f"rebase {b}",
            ]
            samples.append((random.choice(prompts), f"git rebase {b}"))
        elif intent == 4:
            # Merge branch
            prompts = [
                f"merge branch {b} into current branch",
                f"combine changes from branch {b}",
                f"how do I merge {b}?",
                f"merge {b}",
            ]
            samples.append((random.choice(prompts), f"git merge {b}"))
        elif intent == 5:
            # Abort merge
            prompts = [
                "abort merge in progress and reset to pre-merge state",
                "cancel current merge with conflicts",
                "how to abort git merge?",
                "abort merge",
            ]
            samples.append((random.choice(prompts), "git merge --abort"))
        elif intent == 6:
            # No fast forward merge
            prompts = [
                f"merge branch {b} creating a merge commit without fast-forward",
                f"no fast-forward merge of branch {b}",
                f"force merge commit when merging {b}",
            ]
            samples.append((random.choice(prompts), f"git merge --no-ff {b}"))
        else:
            # Squash merge
            prompts = [
                f"squash merge all commits from branch {b}",
                f"merge branch {b} as a single squashed commit",
                f"how to do a squash merge of {b}?",
                f"squash merge {b}",
            ]
            samples.append((random.choice(prompts), f"git merge --squash {b}"))
    return samples

def build_log_diff_status_samples(count=4000):
    samples = []
    for _ in range(count):
        f = random_filepath()
        c = random_commit_hash()
        intent = random.randint(1, 6)
        if intent == 1:
            # Status
            prompts = [
                "check working directory and staging status",
                "view current git status",
                "see modified, staged, and untracked files",
                "how do I check git status?",
                "git status",
                "status",
            ]
            samples.append((random.choice(prompts), "git status"))
        elif intent == 2:
            # Short status
            prompts = [
                "check git status in compact format",
                "show short format status",
                "view terse git status",
                "short status",
            ]
            samples.append((random.choice(prompts), "git status -s"))
        elif intent == 3:
            # Diff unstaged
            prompts = [
                "view unstaged diff of modifications",
                "show changes in working directory that are not staged",
                "how do I see what changed before staging?",
                "diff changes",
                "git diff",
                "diff",
            ]
            samples.append((random.choice(prompts), "git diff"))
        elif intent == 4:
            # Diff staged
            prompts = [
                "view diff of staged changes ready for commit",
                "show what changes are currently staged",
                "how do I see staged diff?",
                "git diff cached changes",
                "diff staged",
            ]
            samples.append((random.choice(prompts), "git diff --staged"))
        elif intent == 5:
            # Diff specific file
            prompts = [
                f"view diff for file {f}",
                f"show changes made to {f}",
                f"how do I check diff of {f}?",
                f"diff {f}",
            ]
            samples.append((random.choice(prompts), f"git diff {f}"))
        else:
            # Git log oneline
            n = random.choice([5, 10, 20])
            prompts = [
                f"show git commit history in compact one-line format limited to {n} commits",
                f"view last {n} commits on one line each",
                f"show concise commit log for last {n} commits",
                f"log last {n} commits",
            ]
            samples.append((random.choice(prompts), f"git log --oneline -n {n}"))
    return samples

def build_tag_samples(count=2000):
    samples = []
    for _ in range(count):
        t = random_tag()
        m = random_commit_message()
        r = random.choice(REMOTES)
        intent = random.randint(1, 4)
        if intent == 1:
            # Lightweight tag
            prompts = [
                f"create lightweight tag {t}",
                f"tag current commit as {t}",
                f"how do I add tag {t}?",
                f"tag {t}",
            ]
            samples.append((random.choice(prompts), f"git tag {t}"))
        elif intent == 2:
            # Annotated tag with message
            prompts = [
                f"create annotated release tag {t} with message '{m}'",
                f"tag release {t} saying '{m}'",
                f"how do I create signed/annotated tag {t} with comment '{m}'?",
            ]
            samples.append((random.choice(prompts), f"git tag -a {t} -m '{m}'"))
        elif intent == 3:
            # Push single tag
            prompts = [
                "push tag {tag} to remote {remote}",
                "publish tag {tag} to {remote}",
                "how to push tag {tag} to remote?",
                "push tag {tag}",
            ]
            samples.append(pick(prompts, "git push {remote} {tag}", tag=t, remote=r))
        else:
            # Push all tags
            prompts = [
                "push all local tags to remote {remote}",
                "upload all tags to {remote}",
                "how do I push all tags?",
                "push all tags",
            ]
            samples.append(pick(prompts, "git push {remote} --tags", remote=r))
    return samples

# ==============================================================================
# Advanced Subcommands (Cherry-pick, Worktree, Blame, Submodule, Bisect)
# ==============================================================================

def build_advanced_tools_samples(count=12000):
    samples = []
    for _ in range(count):
        commit_hash = "".join(random.choices("0123456789abcdef", k=7))
        b = random_branch()
        f = random_filepath()
        op = random.randint(1, 6)
        if op == 1:
            # Cherry pick
            action = random.choice(["apply", "cherry-pick", "bring in", "port"])
            prompts = [
                f"{action} commit {commit_hash} into current branch",
                f"how do I cherry pick commit {commit_hash}?",
                f"cherry-pick {commit_hash}",
                f"how to apply specific commit {commit_hash}",
            ]
            samples.append((random.choice(prompts), f"git cherry-pick {commit_hash}"))
            if random.random() < 0.2:
                samples.append((random.choice(["continue cherry-pick after resolving conflicts", "continue cherry-pick"]), "git cherry-pick --continue"))
                samples.append((random.choice(["abort cherry-pick", "cancel cherry-pick"]), "git cherry-pick --abort"))
        elif op == 2:
            # Worktree
            path = f"../{b.replace('/', '-')}"
            prompts = [
                f"create new worktree for branch {b} at path {path}",
                f"add git worktree at {path} for {b}",
                f"how to checkout branch {b} in another directory {path} without switching here",
            ]
            samples.append((random.choice(prompts), f"git worktree add {path} {b}"))
            if random.random() < 0.3:
                samples.append(("list active git worktrees", "git worktree list"))
                samples.append(("show git worktrees", "git worktree list"))
        elif op == 3:
            # Blame
            prompts = [
                f"see who wrote each line in {f}",
                f"who modified {f}?",
                f"show git blame for {f}",
                f"inspect line by line authorship in {f}",
                f"blame {f}",
            ]
            samples.append((random.choice(prompts), f"git blame {f}"))
        elif op == 4:
            # Submodules
            prompts = [
                "initialize and clone all git submodules in project",
                "update submodules recursively",
                "how do I fetch git submodules after clone?",
                "sync and update submodules",
            ]
            samples.append((random.choice(prompts), "git submodule update --init --recursive"))
        elif op == 5:
            # Bisect
            samples.append((random.choice(["start git binary search for bad commit", "start bisect session", "find regression with bisect"]), "git bisect start"))
            samples.append((random.choice(["mark current commit as bad in bisect", "bisect bad"]), "git bisect bad"))
            samples.append((random.choice(["mark commit as good in bisect", "bisect good"]), "git bisect good"))
            samples.append((random.choice(["exit git bisect and return to original HEAD", "reset bisect"]), "git bisect reset"))
        else:
            # Log graph
            prompts = [
                "show visual branch commit tree with graph and one line",
                "display ASCII commit history tree",
                "git log with ascii branch graph",
                "compact git tree view",
            ]
            samples.append((random.choice(prompts), "git log --graph --oneline --decorate --all"))
    return samples

# ==============================================================================
# Compound Multi-Step Real-World Workflows
# ==============================================================================

def build_compound_workflow_samples(count=15000):
    samples = []
    for _ in range(count):
        b = random_branch()
        f = random_filepath()
        msg = random_commit_message()
        r = random.choice(REMOTES)
        op = random.randint(1, 5)
        if op == 1:
            # Stash -> switch branch -> pop
            prompts = [
                f"I made changes by mistake, how to move my uncommitted work to a new branch {b}?",
                f"save current uncommitted work and switch to new branch {b}",
                f"move my dirty working tree changes to {b}",
            ]
            samples.append((random.choice(prompts), f"git stash && git checkout -b {b} && git stash pop"))
        elif op == 2:
            # Add all -> commit -> push
            prompts = [
                f"stage all changes, commit as '{msg}', and push to {b}",
                f"add everything, commit '{msg}' and push origin {b}",
                f"one-line command to stage all, commit '{msg}' and push {b}",
            ]
            samples.append((random.choice(prompts), f"git add -A && git commit -m '{msg}' && git push origin {b}"))
        elif op == 3:
            # Create branch -> push upstream setup
            prompts = [
                f"create branch {b} and immediately publish it tracking on {r}",
                f"checkout new branch {b} and push it to {r} setting upstream",
                f"create and push new branch {b} to {r}",
            ]
            samples.append((random.choice(prompts), f"git checkout -b {b} && git push -u {r} {b}"))
        elif op == 4:
            # Sync fork with upstream
            prompts = [
                f"fetch upstream and merge upstream/{b} into current branch",
                f"sync my fork with upstream {b}",
                f"update my local {b} with changes from upstream remote",
            ]
            samples.append((random.choice(prompts), f"git fetch upstream && git merge upstream/{b}"))
        else:
            # Stage file and amend commit without editing message
            prompts = [
                f"add {f} to the previous commit without changing the commit message",
                f"include {f} in the last commit silently",
                f"amend last commit with {f} keeping original message",
            ]
            samples.append((random.choice(prompts), f"git add {f} && git commit --amend --no-edit"))
    return samples

# ==============================================================================
# Open-Vocabulary Copy Induction Training Generator
# Forces neural CA associative weights to learn literal argument copying
# ==============================================================================

def build_copy_induction_samples(count=6000):
    samples = []
    for _ in range(count):
        # Generate an open-ended identifier that changes every time
        token = random_filepath() if random.random() < 0.5 else random_synthetic_word()
        branch = random_branch()
        user = random_username()
        op = random.randint(1, 5)
        if op == 1:
            # Staging copying: 'stage changes in <TOKEN>' -> 'git add <TOKEN>'
            tmpl = random.choice([
                f"stage changes in {token}",
                f"stage {token}",
                f"add {token}",
                f"how to stage changes in {token}",
                f"stage file {token}",
            ])
            samples.append((tmpl, f"git add {token}"))
        elif op == 2:
            # Branch switching copying: 'checkout <BRANCH>' -> 'git checkout <BRANCH>'
            tmpl = random.choice([
                f"checkout {branch}",
                f"switch to branch {branch}",
                f"switch to {branch}",
                f"how to checkout branch {branch}?",
                f"move to branch {branch}",
            ])
            samples.append((tmpl, f"git checkout {branch}"))
        elif op == 3:
            # Push copying: 'how to push <BRANCH>' -> 'git push origin <BRANCH>'
            tmpl = random.choice([
                f"how to push {branch}",
                f"push {branch}",
                f"how do I push {branch}?",
                f"push branch {branch}",
            ])
            samples.append((tmpl, f"git push origin {branch}"))
        elif op == 4:
            # Config copying: 'set global git username to <USER>' -> "git config --global user.name '<USER>'"
            tmpl = random.choice([
                f"set global git username to {user}",
                f"set global git user name to {user}",
                f"configure git global username as {user}",
                f"change global username to {user}",
            ])
            samples.append((tmpl, f"git config --global user.name '{user}'"))
        elif op == 5:
            # Commit message copying: 'commit changes with message "..."' -> 'git commit -m "..."'
            msg = random_commit_message()
            tmpl = random.choice([
                f"commit with message '{msg}'",
                f"commit changes with message '{msg}'",
                f"commit using comment '{msg}'",
                f"make a commit saying '{msg}'",
            ])
            samples.append((tmpl, f"git commit -m '{msg}'"))
        elif op == 6:
            # Tag copying: 'create tag <TAG>' -> 'git tag <TAG>'
            ver = f"v{random.randint(0, 3)}.{random.randint(0, 15)}.{random.randint(0, 9)}"
            if random.random() < 0.3:
                ver += random.choice(["-rc1", "-beta", "-alpha", "-hotfix"])
            tmpl = random.choice([
                f"create tag {ver}",
                f"tag current commit with {ver}",
                f"make a new release tag {ver}",
                f"tag {ver}",
            ])
            samples.append((tmpl, f"git tag {ver}"))
        elif op == 7:
            # Remote URL copying: 'add remote <REMOTE> <URL>' -> 'git remote add <REMOTE> <URL>'
            org = random_synthetic_word()
            repo = random_synthetic_word()
            url = random.choice([
                f"git@github.com:{org}/{repo}.git",
                f"https://github.com/{org}/{repo}.git",
                f"git@gitlab.com:{org}/{repo}.git",
            ])
            remote = random.choice(["origin", "upstream", "mirror", "backup"])
            tmpl = random.choice([
                f"add remote {remote} pointing to {url}",
                f"add new remote {remote} with url {url}",
                f"set up remote {remote} at {url}",
                f"git remote add {remote} {url}",
            ])
            samples.append((tmpl, f"git remote add {remote} {url}"))
        else:
            # Restore copying: 'discard changes in <TOKEN>' -> 'git restore <TOKEN>'
            tmpl = random.choice([
                f"discard changes in {token}",
                f"discard uncommitted changes in file {token}",
                f"restore {token}",
                f"revert uncommitted changes in {token}",
            ])
            samples.append((tmpl, f"git restore {token}"))
    return samples

# ==============================================================================
# Conceptual & Explanatory Micro-QA Pairs
# Teaches the model concise conceptual knowledge without exceeding attention window
# ==============================================================================

EXPLANATORY_PAIRS = [
    ("what is the difference between git fetch and git pull?",
     "git fetch downloads remote commits without modifying your working branch, whereas git pull fetches and immediately merges them into your current branch."),
    ("difference between git fetch and pull",
     "git fetch only downloads changes from the remote repository; git pull downloads and merges them into your current branch."),
    ("what does git status do?",
     "git status displays paths that have differences between the index file and the current HEAD commit, along with untracked files."),
    ("what is the difference between git reset and git revert?",
     "git reset moves HEAD backward to undo commits by rewriting history, while git revert creates a new commit that inverts the changes of a previous commit."),
    ("difference between git merge and git rebase",
     "git merge combines branches with a merge commit preserving exact history, while git rebase reapplies commits on top of another base branch for a linear history."),
    ("what is the git staging area?",
     "The staging area (index) is an intermediate workspace where changes are collected before being recorded into a commit via git add."),
    ("what is a detached HEAD in git?",
     "A detached HEAD state means your working directory points directly to a specific commit rather than to a named branch reference."),
    ("how to check git commit history?",
     "git log --oneline"),
    ("how do I undo the last commit but keep my changes?",
     "git reset --soft HEAD~1"),
    ("how to undo commit and keep files",
     "git reset --soft HEAD~1"),
    ("how to discard all local uncommitted changes?",
     "git restore ."),
    ("how to delete a remote branch?",
     "git push origin --delete <branch>"),
    ("how do I see differences between commits?",
     "git diff <commit1> <commit2>"),
    ("how to view commits on one line?",
     "git log --oneline"),
    ("what command shows changed files before committing?",
     "git status"),
    ("how to rename the current branch?",
     "git branch -m <new-name>"),
    ("how to temporarily shelve changes?",
     "git stash"),
    ("how to apply stashed changes?",
     "git stash pop"),
]

# ==============================================================================
# Generic (Under-specified) Queries, Canonical Placeholders, and Typo Robustness
# Covers queries where the user didn't name a branch/file, or misspelled commands
# ==============================================================================

GENERIC_AND_TYPO_PAIRS = [
    # Generic commands without arguments
    ("how to delete branch", "git branch -d <branch>"),
    ("delete branch", "git branch -d <branch>"),
    ("how to delete a branch", "git branch -d <branch>"),
    ("how do I delete branch?", "git branch -d <branch>"),
    ("delete branch locally", "git branch -d <branch>"),
    ("force delete branch", "git branch -D <branch>"),
    ("delete remote branch", "git push origin --delete <branch>"),
    ("how to delete remote branch", "git push origin --delete <branch>"),
    ("how to switch branch", "git checkout <branch>"),
    ("switch branch", "git checkout <branch>"),
    ("change branch", "git checkout <branch>"),
    ("checkout branch", "git checkout <branch>"),
    ("create new branch", "git checkout -b <branch>"),
    ("create and switch branch", "git checkout -b <branch>"),
    ("make new branch", "git checkout -b <branch>"),
    ("how to push", "git push"),
    ("push", "git push"),
    ("how to push changes", "git push"),
    ("how to pull", "git pull"),
    ("pull", "git pull"),
    ("pull changes", "git pull"),
    ("pull with rebase", "git pull --rebase"),
    ("how to stage changes", "git add ."),
    ("stage changes", "git add ."),
    ("stage all", "git add -A"),
    ("add all files", "git add ."),
    ("unstage changes", "git restore --staged ."),
    ("how to unstage", "git restore --staged <file>"),
    ("unstage file", "git restore --staged <file>"),
    ("view diff", "git diff"),
    ("show diff", "git diff"),
    ("check diff", "git diff"),
    ("view diff for file", "git diff <file>"),
    ("diff file", "git diff <file>"),
    ("show file changes", "git diff <file>"),
    ("view staged diff", "git diff --staged"),
    ("show staged changes", "git diff --cached"),
    ("status", "git status"),
    ("check status", "git status"),
    ("show status", "git status"),
    ("git status", "git status"),
    ("log", "git log --oneline"),
    ("show log", "git log --oneline"),
    ("view commits", "git log --oneline"),
    ("commit history", "git log --oneline"),
    ("stash changes", "git stash"),
    ("save stash", "git stash"),
    ("pop stash", "git stash pop"),
    ("apply stash", "git stash pop"),
    ("list stash", "git stash list"),
    ("clear stash", "git stash clear"),

    # Amend commit variants and typos
    ("amend commit", "git commit --amend"),
    ("ammend commit", "git commit --amend"),
    ("amend", "git commit --amend"),
    ("ammend", "git commit --amend"),
    ("amend last commit", "git commit --amend"),
    ("ammend last commit", "git commit --amend"),
    ("amend without edit", "git commit --amend --no-edit"),
    ("ammend without edit", "git commit --amend --no-edit"),
    ("amend commit without changing message", "git commit --amend --no-edit"),
    ("ammend commit without changing message", "git commit --amend --no-edit"),
    ("fix last commit message", "git commit --amend"),
    ("change last commit message", "git commit --amend"),

    # Commit typos
    ("comit changes", "git commit -m '<message>'"),
    ("comit with message", "git commit -m '<message>'"),
    ("comit", "git commit"),

    # Checkout / check out typos
    ("check out branch", "git checkout <branch>"),
    ("check out", "git checkout"),
    ("chekout branch", "git checkout <branch>"),
    ("chekout", "git checkout <branch>"),

    # Rebase typos & shortcuts
    ("rebase branch", "git rebase <branch>"),
    ("rebase", "git rebase"),
    ("continue rebase", "git rebase --continue"),
    ("abort rebase", "git rebase --abort"),

    # Cherry-pick typos
    ("cherry pick", "git cherry-pick <commit>"),
    ("cherrypick", "git cherry-pick <commit>"),
    ("cherry-pic", "git cherry-pick <commit>"),

    # Output for every commit / logs
    ("output for every commit", "git log --oneline"),
    ("is there a git command that can output for every commit", "git log --oneline"),
    ("print every commit", "git log --oneline"),
    ("list each commit on one line", "git log --oneline"),
]

def build_generic_and_typo_samples(repeat=150):
    samples = []
    for _ in range(repeat):
        for q, a in GENERIC_AND_TYPO_PAIRS:
            samples.append((q, a))
    return samples

def build_explanatory_samples(repeat=100):
    samples = []
    for _ in range(repeat):
        for q, a in EXPLANATORY_PAIRS:
            samples.append((q, a))
    return samples

# ==============================================================================
# Main Generation Runner
# ==============================================================================

def main():
    print("=== Generating Natural High-Entropy SFT Dataset (Tag-Free + Induction) ===")
    
    all_pairs = []
    all_pairs.extend(build_branch_samples(18000))
    all_pairs.extend(build_staging_commit_samples(20000))
    all_pairs.extend(build_remote_sync_samples(22000))
    all_pairs.extend(build_config_samples(16000))
    all_pairs.extend(build_undo_reset_samples(15000))
    all_pairs.extend(build_stash_samples(12000))
    all_pairs.extend(build_rebase_merge_samples(12000))
    all_pairs.extend(build_log_diff_status_samples(12000))
    all_pairs.extend(build_tag_samples(6000))
    all_pairs.extend(build_advanced_tools_samples(12000))
    all_pairs.extend(build_compound_workflow_samples(15000))
    all_pairs.extend(build_copy_induction_samples(22000))
    all_pairs.extend(build_generic_and_typo_samples(250))
    all_pairs.extend(build_explanatory_samples(300))

    random.shuffle(all_pairs)

    samples = [
        {"instruction": rephrase(inst), "output": out}
        for inst, out in all_pairs
    ]

    print(f"Total SFT samples generated: {len(samples)}")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(samples, f, indent=2)

    size_mb = os.path.getsize(OUTPUT_FILE) / (1024 * 1024)
    print(f"Saved {OUTPUT_FILE} ({size_mb:.2f} MB)")

    # Print specific target sanity checks
    print("\nSanity Checks on Previous Failing Targets:")
    for query in ["how to push main", "how to push", "stage changes in mignon", "stage changes in models/user.env.example2"]:
        matches = [s for s in samples if query in s["instruction"]]
        if matches:
            print(f"  Matched query '{query}':")
            for m in matches[:2]:
                print(f"    Q: {m['instruction']}")
                print(f"    A: {m['output']}")

if __name__ == "__main__":
    main()
