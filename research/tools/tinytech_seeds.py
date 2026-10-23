"""
tinytech_seeds.py

Seed lists for generate_tinytech.py. Every request combines one topic, one form, one reader and a few
required words, so tens of thousands of requests still produce varied text (the TinyStories recipe).
Kept as Python because the repository ignores *.json.
"""

GENERAL_TOPICS = [
    # files and the operating system
    "what a file is", "folders and paths", "file names and extensions", "hidden files",
    "copying and moving files", "deleting files safely", "file permissions", "the home directory",
    "absolute and relative paths", "text files versus binary files", "disk space", "backups",
    # terminal and shell
    "what a terminal is", "typing a command", "command options and flags", "arguments to a command",
    "the current directory", "listing files", "reading command help", "environment variables",
    "pipes between commands", "redirecting output to a file", "exit codes", "shell scripts",
    "installing a program with a package manager", "running a program in the background",
    # programming basics
    "what source code is", "variables", "functions", "loops", "conditions", "comments in code",
    "libraries and dependencies", "compiling a program", "running tests", "reading an error message",
    "a bug", "debugging step by step", "logging", "configuration files", "code style and formatting",
    "refactoring", "a code editor", "plain text formats like JSON and YAML",
    # teamwork and workflow
    "code review", "writing a good commit message", "a pull request", "an issue tracker",
    "releasing a new version", "version numbers", "a changelog", "continuous integration",
    "working on a team project", "open source projects", "licenses for code", "documentation",
    # networks and servers
    "what a server is", "a URL", "downloading a file", "SSH keys", "passwords and tokens",
    "a remote machine", "the cloud", "a website and its server",
]

GIT_TOPICS = [
    "what version control is", "what git is", "a repository", "the working tree", "the staging area",
    "staging a file", "making a commit", "the commit history", "a commit hash", "HEAD",
    "branches", "creating a branch", "switching branches", "merging branches", "a merge conflict",
    "resolving a merge conflict", "rebasing", "fast-forward merges", "a remote repository", "origin",
    "cloning a repository", "fetching changes", "pulling changes", "pushing commits",
    "upstream branches", "force pushing and why it is risky", "tags", "the stash",
    "undoing changes in a file", "unstaging a file", "resetting to an older commit", "reverting a commit",
    "amending the last commit", "the reflog", "cherry-picking a commit", "the .gitignore file",
    "viewing differences with diff", "reading the log", "blame and finding who changed a line",
    "bisect to find a bug", "git config and your name and email", "aliases", "submodules", "worktrees",
    "a detached HEAD", "cleaning untracked files", "forks", "main and master branches",
    "feature branches", "keeping a branch up to date", "deleting a branch", "renaming a branch",
]

FORMS = [
    "a short, clear explanation",
    "a tiny story about a developer who runs into this and learns what to do",
    "a question a beginner might ask, followed by a clear answer",
    "short step-by-step instructions written as sentences",
    "a comparison between this and something it is often confused with",
    "a common mistake and how to fix it",
    "a short conversation between two coworkers",
    "a definition followed by one simple example",
    "a note a team lead might write for a new team member",
    "an analogy from everyday life, then the real meaning",
]

READERS = [
    "a complete beginner", "a student in their first programming class", "a new team member",
    "a designer who is learning to code", "someone who only used a computer for email",
    "a junior developer", "a curious teenager",
]

# Plain words that are not too narrow; three are required per request to push lexical variety.
REQUIRED_WORDS = [
    "save", "change", "share", "check", "find", "fix", "remember", "careful", "quick", "safe",
    "old", "new", "list", "name", "copy", "move", "keep", "lose", "mistake", "team", "project",
    "folder", "message", "history", "version", "update", "download", "upload", "test", "break",
    "work", "try", "undo", "clean", "notice", "compare", "small", "large", "first", "last",
    "before", "after", "again", "every", "never", "always", "simple", "problem", "answer", "step",
]
