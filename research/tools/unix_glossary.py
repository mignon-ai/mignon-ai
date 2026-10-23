#!/usr/bin/env python3
"""
unix_glossary.py

A hand-written glossary of everyday Unix commands: for every command form a few ways a person would
ask for it. tldr-pages (build_shell_sft.py) gives one phrasing per command; this file gives the
variety of phrasing and the meaning of common flags.

Placeholders in a command are filled with varied values, and the same value appears in the request:
  {f} {f2} file   {d} {d2} directory   {a} archive   {p} search pattern   {n} number   {u} user
  {h} host   {k} port   {x} extension   {s} size   {c} command name

Each form is rendered VARIANTS_PER_PHRASING times per phrasing. One in HELDOUT_EVERY forms (by hash
of the command skeleton) is held out whole ("unseen_intent"); for the rest, the last rendering of one
in SEEN_HELDOUT_EVERY forms is held out ("seen_intent": same command, new values and wording).
"""

import re

from techcorpus_common import stable_hash

HELDOUT_EVERY = 6
SEEN_HELDOUT_EVERY = 5
VARIANTS_PER_PHRASING = 12

POOLS = {
    "f": ["report.txt", "notes.md", "server.log", "data.csv", "main.py", "config.yml", "script.sh", "app.js",
          "output.json", "todo.txt", "backup.sql", "index.html", "error.log", "users.csv", "readme.md"],
    "d": ["src", "docs", "build", "projects", "/var/log", "~/downloads", "/etc", "data", "/tmp/work", "photos",
          "backups", "/home/alice", "logs", "assets", "tests"],
    "a": ["backup.tar.gz", "site.tar.gz", "photos.zip", "release.tar", "logs.tar.bz2", "project.zip", "data.tgz"],
    "p": ["error", "TODO", "warning", "main", "password", "timeout", "failed", "localhost", "foo", "user"],
    "n": ["2", "3", "5", "10", "20", "50", "100"],
    "u": ["alice", "bob", "deploy", "www-data", "carol", "admin", "backup"],
    "h": ["example.com", "server1", "db.internal", "10.0.0.5", "web-01", "192.168.1.20", "prod.example.org"],
    "k": ["22", "80", "443", "3000", "5432", "8080", "9000"],
    "x": ["txt", "log", "py", "json", "csv", "md", "sh", "js"],
    "s": ["1M", "10M", "100M", "1G", "500k"],
    "c": ["python", "nginx", "node", "java", "firefox", "sshd"],
}
POOLS["f2"], POOLS["d2"] = POOLS["f"], POOLS["d"]

# (command, [requests...]). A request may use the same placeholders as the command.
G = [
    # --- listing and navigation
    ("ls -l", ["list files with details", "show the files in long format", "give me a detailed file listing"]),
    ("ls -a", ["list all files including hidden ones", "show hidden files too", "list everything, dotfiles included"]),
    ("ls -la", ["list all files with details, including hidden ones", "long listing with hidden files"]),
    ("ls -lh", ["list files with human readable sizes", "show file sizes in KB and MB", "long listing with readable sizes"]),
    ("ls -lt", ["list files sorted by modification time", "show the newest files first"]),
    ("ls -ltr", ["list files with the oldest first", "show files sorted by time, newest last"]),
    ("ls -lS", ["list files sorted by size", "show the biggest files first"]),
    ("ls -R", ["list files recursively", "show all files in subdirectories too"]),
    ("ls -d */", ["list only the directories", "show just the subfolders"]),
    ("ls -1", ["list files one per line", "print one file name per line"]),
    ("ls {d}", ["list the files in {d}", "what is inside {d}?", "show the contents of the folder {d}"]),
    ("ls *.{x}", ["list all .{x} files", "show the files ending in .{x}"]),
    ("cd {d}", ["go to the directory {d}", "change directory to {d}", "move into {d}"]),
    ("cd ..", ["go up one directory", "move to the parent folder"]),
    ("cd -", ["go back to the previous directory", "switch to the last directory I was in"]),
    ("cd ~", ["go to my home directory", "change to the home folder"]),
    ("pwd", ["print the current directory", "where am I?", "show the path of the working directory"]),
    ("tree -L {n}", ["show the directory tree {n} levels deep", "print a tree of the folder, depth {n}"]),
    ("tree -a", ["show the directory tree including hidden files", "print a tree with dotfiles"]),
    # --- copy, move, delete
    ("cp {f} {f2}", ["copy {f} to {f2}", "make a copy of {f} called {f2}", "duplicate {f} as {f2}"]),
    ("cp -r {d} {d2}", ["copy the directory {d} to {d2}", "recursively copy {d} into {d2}", "duplicate the folder {d} as {d2}"]),
    ("cp -i {f} {f2}", ["copy {f} to {f2} but ask before overwriting", "copy {f} to {f2} with a prompt if it exists"]),
    ("cp -v {f} {d}", ["copy {f} into {d} and show what is done", "verbosely copy {f} to {d}"]),
    ("cp -a {d} {d2}", ["copy {d} to {d2} preserving permissions and timestamps", "archive copy of {d} to {d2}"]),
    ("cp -u {f} {d}", ["copy {f} to {d} only if it is newer", "update {d} with {f} only when changed"]),
    ("cp -n {f} {d}", ["copy {f} to {d} without overwriting existing files", "copy {f} into {d} but never overwrite"]),
    ("mv {f} {f2}", ["rename {f} to {f2}", "move {f} to {f2}", "change the name of {f} to {f2}"]),
    ("mv {f} {d}", ["move {f} into {d}", "put {f} in the folder {d}"]),
    ("mv -i {f} {f2}", ["rename {f} to {f2}, asking before overwriting", "move {f} to {f2} with confirmation"]),
    ("mv -n {f} {d}", ["move {f} to {d} without overwriting", "move {f} into {d} but do not replace existing files"]),
    ("mv -v {f} {d}", ["move {f} to {d} and print what happens", "verbosely move {f} into {d}"]),
    ("rm {f}", ["delete {f}", "remove the file {f}", "get rid of {f}"]),
    ("rm -i {f}", ["delete {f} after asking for confirmation", "remove {f} but prompt first"]),
    ("rm -r {d}", ["delete the directory {d} and its contents", "remove the folder {d} recursively"]),
    ("rm -rf {d}", ["force delete the directory {d} without prompts", "remove {d} and everything in it, no questions", "wipe the folder {d} completely"]),
    ("rm -v {f}", ["delete {f} and say what was removed", "verbosely remove {f}"]),
    ("rm *.{x}", ["delete all .{x} files", "remove every file ending in .{x}"]),
    ("mkdir {d}", ["create a directory called {d}", "make a new folder named {d}"]),
    ("mkdir -p {d}", ["create the directory {d} and any missing parents", "make nested folders {d}", "create {d} including parent directories"]),
    ("mkdir -v {d}", ["create the directory {d} and print a message", "make a folder {d} verbosely"]),
    ("rmdir {d}", ["remove the empty directory {d}", "delete the folder {d} if it is empty"]),
    ("touch {f}", ["create an empty file called {f}", "make a new file {f}", "update the timestamp of {f}"]),
    ("ln -s {f} {f2}", ["create a symbolic link {f2} pointing to {f}", "make a symlink called {f2} to {f}"]),
    ("ln {f} {f2}", ["create a hard link {f2} to {f}", "make a hard link of {f} named {f2}"]),
    # --- viewing files
    ("cat {f}", ["show the contents of {f}", "print the file {f}", "display {f}"]),
    ("cat -n {f}", ["show {f} with line numbers", "print {f} and number every line"]),
    ("cat {f} {f2}", ["print {f} and {f2} one after the other", "show the contents of {f} followed by {f2}"]),
    ("cat {f} {f2} > {a}", ["combine {f} and {f2} into one file", "merge {f} and {f2} into a new file"]),
    ("tac {f}", ["print {f} in reverse line order", "show the lines of {f} from last to first"]),
    ("head {f}", ["show the first 10 lines of {f}", "print the beginning of {f}"]),
    ("head -n {n} {f}", ["show the first {n} lines of {f}", "print the top {n} lines of {f}"]),
    ("head -c {n} {f}", ["show the first {n} bytes of {f}", "print the first {n} characters of {f}"]),
    ("tail {f}", ["show the last 10 lines of {f}", "print the end of {f}"]),
    ("tail -n {n} {f}", ["show the last {n} lines of {f}", "print the final {n} lines of {f}"]),
    ("tail -f {f}", ["follow {f} as it grows", "watch new lines being added to {f}", "stream the log file {f} live"]),
    ("tail -n +{n} {f}", ["print {f} starting from line {n}", "show {f} from line {n} to the end"]),
    ("less {f}", ["page through {f}", "open {f} in a pager", "scroll through the file {f}"]),
    ("less -N {f}", ["page through {f} with line numbers", "view {f} in a pager showing line numbers"]),
    ("wc -l {f}", ["count the lines in {f}", "how many lines does {f} have?"]),
    ("wc -w {f}", ["count the words in {f}", "how many words are in {f}?"]),
    ("wc -c {f}", ["count the bytes in {f}", "how many characters does {f} contain?"]),
    ("file {f}", ["what type of file is {f}?", "show the file type of {f}"]),
    ("stat {f}", ["show detailed information about {f}", "display size, permissions and timestamps of {f}"]),
    # --- searching
    ("grep {p} {f}", ["search {f} for {p}", "find lines containing {p} in {f}", "show lines with {p} from {f}"]),
    ("grep -i {p} {f}", ["search {f} for {p} ignoring case", "case-insensitive search for {p} in {f}"]),
    ("grep -n {p} {f}", ["search {f} for {p} and show line numbers", "find {p} in {f} with line numbers"]),
    ("grep -v {p} {f}", ["show lines of {f} that do not contain {p}", "exclude lines with {p} from {f}"]),
    ("grep -c {p} {f}", ["count the lines in {f} that match {p}", "how many lines of {f} contain {p}?"]),
    ("grep -l {p} {d}/*", ["list the files in {d} that contain {p}", "which files in {d} mention {p}?"]),
    ("grep -w {p} {f}", ["search {f} for the whole word {p}", "match {p} as a complete word in {f}"]),
    ("grep -o {p} {f}", ["print only the matching part of {p} in {f}", "show just the matches of {p} from {f}"]),
    ("grep -E '{p}|{p2}' {f}", ["search {f} for {p} or {p2}", "find lines with either {p} or {p2} in {f}"]),
    ("grep -r {p} {d}", ["search the directory {d} recursively for {p}", "find {p} in all files under {d}"]),
    ("grep -rn {p} {d}", ["search {d} recursively for {p} with line numbers", "find {p} under {d} and show where"]),
    ("grep -ri {p} {d}", ["search {d} recursively for {p} ignoring case", "case-insensitive recursive search for {p} in {d}"]),
    ("grep -A {n} {p} {f}", ["show {n} lines after each match of {p} in {f}", "find {p} in {f} with {n} lines of trailing context"]),
    ("grep -B {n} {p} {f}", ["show {n} lines before each match of {p} in {f}", "find {p} in {f} with {n} lines of leading context"]),
    ("grep -C {n} {p} {f}", ["show {n} lines around each match of {p} in {f}", "find {p} in {f} with {n} lines of context"]),
    ("grep --include=*.{x} -r {p} {d}", ["search only .{x} files under {d} for {p}", "recursively grep {p} in {d} limited to .{x} files"]),
    ("find . -name {f}", ["find files named {f} here", "locate {f} in the current directory tree", "search for the file {f} below this folder"]),
    ("find . -iname '*.{x}'", ["find all .{x} files ignoring case", "locate files ending in .{x}, any capitalisation"]),
    ("find {d} -name '*.{x}'", ["find all .{x} files in {d}", "locate .{x} files under {d}"]),
    ("find . -type f", ["list all regular files below here", "find only files, not directories"]),
    ("find . -type d", ["list all directories below here", "find only folders in this tree"]),
    ("find . -mtime -{n}", ["find files modified in the last {n} days", "which files changed within {n} days?"]),
    ("find . -mtime +{n}", ["find files older than {n} days", "locate files not modified for more than {n} days"]),
    ("find . -size +{s}", ["find files larger than {s}", "locate files bigger than {s}"]),
    ("find . -empty", ["find empty files and directories", "locate files with no content"]),
    ("find . -maxdepth 1 -type f", ["list files in this directory only, no subfolders", "find files at the top level only"]),
    ("find . -name '*.{x}' -delete", ["delete all .{x} files below here", "find and remove every .{x} file"]),
    ("find . -name '*.{x}' -exec wc -l {} +", ["count the lines of all .{x} files", "run wc -l on every .{x} file found"]),
    ("find . -user {u}", ["find files owned by {u}", "locate files that belong to the user {u}"]),
    ("find . -perm 777", ["find files with 777 permissions", "locate files that everyone can write"]),
    ("which {c}", ["where is the {c} command?", "show the path of the {c} executable"]),
    ("whereis {c}", ["locate the binary, source and manual for {c}", "where are the files for {c}?"]),
    ("locate {f}", ["quickly find {f} anywhere on the system", "search the file index for {f}"]),
    # --- text processing
    ("sort {f}", ["sort the lines of {f}", "alphabetically sort {f}"]),
    ("sort -n {f}", ["sort {f} numerically", "order the lines of {f} by number"]),
    ("sort -r {f}", ["sort {f} in reverse order", "sort the lines of {f} backwards"]),
    ("sort -u {f}", ["sort {f} and remove duplicate lines", "give me the unique sorted lines of {f}"]),
    ("sort -k {n} {f}", ["sort {f} by column {n}", "order the lines of {f} by field {n}"]),
    ("sort -t, -k {n} {f}", ["sort the csv {f} by column {n}", "order {f} by comma separated field {n}"]),
    ("sort {f} | uniq", ["remove duplicate lines from {f}", "show the unique lines of {f}"]),
    ("sort {f} | uniq -c", ["count how often each line appears in {f}", "count the occurrences of every line in {f}"]),
    ("sort {f} | uniq -c | sort -nr", ["show the most frequent lines in {f}", "rank the lines of {f} by how often they appear"]),
    ("uniq -d {f}", ["show only the duplicate lines of {f}", "print repeated lines from {f}"]),
    ("cut -d, -f {n} {f}", ["extract column {n} of the csv {f}", "get field {n} from {f}, comma separated"]),
    ("cut -d: -f1 {f}", ["print the first colon separated field of {f}", "extract the first field of {f} split on colons"]),
    ("cut -c1-{n} {f}", ["keep only the first {n} characters of each line in {f}", "cut every line of {f} to {n} characters"]),
    ("tr a-z A-Z < {f}", ["convert {f} to uppercase", "make all text in {f} uppercase"]),
    ("tr A-Z a-z < {f}", ["convert {f} to lowercase", "make all text in {f} lowercase"]),
    ("tr -d '\\r' < {f}", ["remove carriage returns from {f}", "strip windows line endings from {f}"]),
    ("tr -s ' ' < {f}", ["squeeze repeated spaces in {f}", "collapse multiple spaces into one in {f}"]),
    ("sed 's/{p}/{p2}/g' {f}", ["replace {p} with {p2} in {f}", "substitute every {p} by {p2} in {f} and print the result"]),
    ("sed -i 's/{p}/{p2}/g' {f}", ["replace {p} with {p2} in {f} in place", "edit {f} directly, changing {p} into {p2}"]),
    ("sed -n '{n}p' {f}", ["print line {n} of {f}", "show only line number {n} from {f}"]),
    ("sed '/{p}/d' {f}", ["delete lines containing {p} from {f}", "print {f} without the lines matching {p}"]),
    ("sed -n '1,{n}p' {f}", ["print the first {n} lines of {f} with sed", "use sed to show lines 1 to {n} of {f}"]),
    ("awk '{print $1}' {f}", ["print the first column of {f}", "extract the first field of every line in {f}"]),
    ("awk -F, '{print $2}' {f}", ["print the second column of the csv {f}", "get the second comma separated field from {f}"]),
    ("awk '{sum += $1} END {print sum}' {f}", ["add up the numbers in the first column of {f}", "sum the first column of {f}"]),
    ("awk 'NR>1' {f}", ["print {f} without its header line", "skip the first line of {f}"]),
    ("diff {f} {f2}", ["compare {f} and {f2}", "show the differences between {f} and {f2}"]),
    ("diff -u {f} {f2}", ["show a unified diff of {f} and {f2}", "compare {f} with {f2} in unified format"]),
    ("diff -r {d} {d2}", ["compare the directories {d} and {d2}", "recursively diff {d} against {d2}"]),
    ("diff -q {f} {f2}", ["tell me only whether {f} and {f2} differ", "check quickly if {f} and {f2} are different"]),
    ("cmp {f} {f2}", ["compare {f} and {f2} byte by byte", "find the first byte where {f} and {f2} differ"]),
    ("echo {p}", ["print {p} to the screen", "output the text {p}"]),
    ("echo {p} > {f}", ["write {p} into the file {f}", "save the text {p} to {f}, overwriting it"]),
    ("echo {p} >> {f}", ["append {p} to the end of {f}", "add the line {p} to {f}"]),
    ("tee {f}", ["write the output to {f} and also show it", "save the output in {f} while printing it"]),
    ("tee -a {f}", ["append the output to {f} and also show it", "add the output to {f} without overwriting it"]),
    ("xargs -n1", ["run the command once per input item", "pass the input items one at a time"]),
    ("basename {f}", ["strip the directory from the path {f}", "get just the file name of {f}"]),
    ("dirname {d}/{f}", ["get the directory part of the path {d}/{f}", "show the folder containing {d}/{f}"]),
    ("realpath {f}", ["show the absolute path of {f}", "resolve {f} to its full path"]),
    # --- archives
    ("tar -czf {a} {d}", ["create a gzipped tar archive {a} of {d}", "compress the folder {d} into {a}", "make a .tar.gz of {d} called {a}"]),
    ("tar -xzf {a}", ["extract the gzipped archive {a}", "unpack {a}", "untar the .tar.gz file {a}"]),
    ("tar -xzf {a} -C {d}", ["extract {a} into the directory {d}", "unpack {a} into {d}"]),
    ("tar -tf {a}", ["list the contents of the archive {a}", "show what is inside {a} without extracting"]),
    ("tar -xf {a}", ["extract the tar archive {a}", "unpack the archive {a}"]),
    ("tar -cf {a} {d}", ["create a tar archive {a} from {d}", "pack the folder {d} into {a} without compression"]),
    ("tar -cjf {a} {d}", ["create a bzip2 compressed archive {a} of {d}", "make a .tar.bz2 of {d} named {a}"]),
    ("tar -xjf {a}", ["extract the bzip2 archive {a}", "unpack the .tar.bz2 file {a}"]),
    ("gzip {f}", ["compress {f} with gzip", "gzip the file {f}"]),
    ("gzip -k {f}", ["compress {f} but keep the original", "gzip {f} and leave the original file"]),
    ("gunzip {a}", ["decompress the gzip file {a}", "unzip the .gz file {a}"]),
    ("zip -r {a} {d}", ["zip the directory {d} into {a}", "create the zip archive {a} from the folder {d}"]),
    ("zip {a} {f} {f2}", ["zip {f} and {f2} into {a}", "put {f} and {f2} in a zip file called {a}"]),
    ("unzip {a}", ["unzip {a}", "extract the zip file {a}"]),
    ("unzip {a} -d {d}", ["unzip {a} into the folder {d}", "extract {a} to {d}"]),
    ("unzip -l {a}", ["list the files inside the zip {a}", "show the contents of {a} without extracting"]),
    # --- permissions
    ("chmod +x {f}", ["make {f} executable", "give {f} execute permission", "allow {f} to be run as a program"]),
    ("chmod 755 {f}", ["set the permissions of {f} to 755", "make {f} readable and executable by everyone, writable by me"]),
    ("chmod 644 {f}", ["set the permissions of {f} to 644", "make {f} readable by everyone and writable only by me"]),
    ("chmod 600 {f}", ["make {f} private so only I can read and write it", "set {f} to 600"]),
    ("chmod -R 755 {d}", ["set the permissions of {d} and everything in it to 755", "recursively make {d} 755"]),
    ("chmod u+w {f}", ["give the owner write permission on {f}", "let me edit {f}"]),
    ("chmod go-w {f}", ["remove write permission for group and others on {f}", "stop others from modifying {f}"]),
    ("chmod a+r {f}", ["let everyone read {f}", "make {f} readable by all users"]),
    ("chown {u} {f}", ["make {u} the owner of {f}", "change the owner of {f} to {u}"]),
    ("chown {u}:{u} {f}", ["set the owner and group of {f} to {u}", "change ownership of {f} to {u}:{u}"]),
    ("chown -R {u} {d}", ["make {u} the owner of {d} and everything in it", "recursively change the owner of {d} to {u}"]),
    ("sudo {c}", ["run {c} as the administrator", "execute {c} with root privileges"]),
    # --- processes and system
    ("ps aux", ["list all running processes", "show every process on the system"]),
    ("ps aux | grep {c}", ["find the running {c} processes", "is {c} running?"]),
    ("ps -ef", ["show all processes in full format", "list every process with parent IDs"]),
    ("kill {n}", ["stop the process with ID {n}", "terminate process {n}"]),
    ("kill -9 {n}", ["force kill the process {n}", "kill process {n} immediately"]),
    ("killall {c}", ["kill all {c} processes", "stop every running {c}"]),
    ("pkill -f {c}", ["kill processes whose command line matches {c}", "stop anything running {c} by pattern"]),
    ("top", ["show running processes and resource use live", "monitor CPU and memory usage"]),
    ("df -h", ["show free disk space in readable units", "how much disk space is left?", "list filesystem usage in GB"]),
    ("du -sh {d}", ["how big is the directory {d}?", "show the total size of {d}", "get the disk usage of {d} in readable form"]),
    ("du -h --max-depth=1", ["show the size of each subfolder", "list the disk usage of the folders here, one level deep"]),
    ("du -sh *", ["show the size of every file and folder here", "which items in this directory take the most space?"]),
    ("free -h", ["show memory usage in readable units", "how much RAM is free?"]),
    ("uptime", ["how long has the system been running?", "show the uptime and load average"]),
    ("whoami", ["what user am I?", "print my username"]),
    ("id", ["show my user and group IDs", "print my uid and groups"]),
    ("uname -a", ["show all system information", "print the kernel name and version"]),
    ("date +%Y-%m-%d", ["print today's date in year-month-day format", "show the current date as YYYY-MM-DD"]),
    ("date", ["show the current date and time", "what time is it?"]),
    ("sleep {n}", ["wait for {n} seconds", "pause for {n} seconds"]),
    ("history | grep {p}", ["search my command history for {p}", "find earlier commands containing {p}"]),
    ("history", ["show my command history", "list the commands I ran before"]),
    ("man {c}", ["show the manual for {c}", "open the help page of {c}"]),
    ("alias ll='ls -la'", ["create a shortcut ll for ls -la", "define the alias ll for a long listing with hidden files"]),
    ("export PATH=$PATH:{d}", ["add {d} to my PATH", "put the directory {d} on the PATH"]),
    ("env", ["show all environment variables", "print the environment"]),
    ("crontab -l", ["list my cron jobs", "show the scheduled tasks for my user"]),
    ("crontab -e", ["edit my cron jobs", "open my crontab for editing"]),
    ("systemctl status {c}", ["check the status of the {c} service", "is the {c} service running?"]),
    ("systemctl restart {c}", ["restart the {c} service", "restart {c} with systemd"]),
    ("systemctl enable {c}", ["start {c} automatically at boot", "enable the {c} service on startup"]),
    ("journalctl -u {c}", ["show the logs of the {c} service", "read the journal entries for {c}"]),
    ("journalctl -u {c} -f", ["follow the logs of the {c} service live", "watch the journal for {c}"]),
    # --- network
    ("curl {h}", ["fetch the page at {h}", "download the content of {h} and print it"]),
    ("curl -O {h}/{f}", ["download {f} from {h}", "save {h}/{f} to a file with the same name"]),
    ("curl -o {f} {h}", ["download {h} and save it as {f}", "save the response from {h} into {f}"]),
    ("curl -L {h}", ["fetch {h} and follow redirects", "download {h}, following any redirect"]),
    ("curl -I {h}", ["show only the response headers from {h}", "get the HTTP headers of {h}"]),
    ("curl -s {h}", ["fetch {h} without the progress output", "quietly download the content of {h}"]),
    ("curl -X POST -d '{p}' {h}", ["send a POST request with {p} to {h}", "post the data {p} to {h}"]),
    ("curl -H 'Accept: application/json' {h}", ["request {h} asking for JSON", "fetch {h} with an Accept JSON header"]),
    ("wget {h}/{f}", ["download {f} from {h}", "fetch {h}/{f}"]),
    ("wget -c {h}/{f}", ["resume the download of {f} from {h}", "continue a partial download of {h}/{f}"]),
    ("wget -O {f} {h}", ["download {h} and save it as {f}", "fetch {h} into the file {f}"]),
    ("wget -q {h}/{f}", ["download {f} from {h} quietly", "fetch {h}/{f} without output"]),
    ("wget -r {h}", ["download {h} recursively", "mirror the site {h}"]),
    ("ssh {u}@{h}", ["connect to {h} as {u}", "log in to the server {h} as the user {u}", "open an ssh session to {h} as {u}"]),
    ("ssh -p {k} {u}@{h}", ["connect to {h} as {u} on port {k}", "ssh into {h} using port {k} as {u}"]),
    ("ssh -i {f} {u}@{h}", ["connect to {h} as {u} with the key {f}", "ssh to {h} as {u} using the identity file {f}"]),
    ("ssh -L {k}:localhost:{k} {u}@{h}", ["forward local port {k} to port {k} on {h}", "create an ssh tunnel for port {k} through {h}"]),
    ("ssh-keygen -t ed25519", ["generate a new ed25519 ssh key", "create an ssh key pair with ed25519"]),
    ("scp {f} {u}@{h}:{d}", ["copy {f} to {d} on {h} as {u}", "upload {f} to the server {h} in {d}"]),
    ("scp {u}@{h}:{f} {d}", ["download {f} from {h} into {d}", "copy {f} from the server {h} to {d} as {u}"]),
    ("scp -r {d} {u}@{h}:{d2}", ["copy the folder {d} to {d2} on {h} as {u}", "upload the directory {d} to {h}:{d2}"]),
    ("rsync -avz {d} {u}@{h}:{d2}", ["sync {d} to {d2} on {h} with compression", "rsync the folder {d} to {h}:{d2} as {u}"]),
    ("rsync -av {d}/ {d2}/", ["sync the directory {d} into {d2}", "mirror {d} to {d2} preserving attributes"]),
    ("rsync -av --delete {d}/ {d2}/", ["make {d2} an exact copy of {d}, deleting extra files", "sync {d} to {d2} and remove files that no longer exist"]),
    ("rsync -avn {d}/ {d2}/", ["show what rsync would copy from {d} to {d2} without doing it", "dry run a sync of {d} into {d2}"]),
    ("rsync -av --progress {d}/ {d2}/", ["sync {d} to {d2} and show progress", "copy {d} into {d2} with rsync, showing progress"]),
    ("ping -c {n} {h}", ["ping {h} {n} times", "send {n} ping packets to {h}"]),
    ("ping {h}", ["check if {h} is reachable", "ping the host {h}"]),
    ("ss -tulpn", ["show listening ports and the programs using them", "list open network ports"]),
    ("lsof -i :{k}", ["what is using port {k}?", "find the process listening on port {k}"]),
    ("netstat -tulpn", ["list listening tcp and udp ports", "show which programs listen on which ports"]),
]


def normalize(cmd):
    """Command skeleton with placeholders removed (used to pick held-out forms)."""
    return re.sub(r"\{\w+\}", "", cmd)


def render(text, values):
    return re.sub(r"\{(\w+)\}", lambda m: values.get(m.group(1), m.group(0)), text)


def build(rng):
    """(train, seen_heldout, unseen_heldout) lists of {"instruction", "output"} pairs."""
    train, seen, unseen = [], [], []
    for command, requests in G:
        key = stable_hash(normalize(command), "unix")
        whole_heldout = key % HELDOUT_EVERY == 0
        keep_last = key % SEEN_HELDOUT_EVERY == 1
        for request in requests:
            for v in range(VARIANTS_PER_PHRASING):
                values = {}
                for name in re.findall(r"\{(\w+)\}", command + request):
                    if name in POOLS and name not in values:
                        values[name] = rng.choice(POOLS[name])
                # {p} and {p2}, {f} and {f2}, ... must be distinct names, like git_glossary
                if "p2" in command + request:
                    values["p2"] = rng.choice([p for p in POOLS["p"] if p != values.get("p")])
                for a, b in (("f", "f2"), ("d", "d2")):
                    if b in values and values.get(a) == values[b]:
                        values[b] = rng.choice([x for x in POOLS[a] if x != values[a]])
                pair = {"instruction": render(request, values), "output": render(command, values)}
                if whole_heldout:
                    unseen.append(pair)
                elif keep_last and v == VARIANTS_PER_PHRASING - 1:
                    seen.append(pair)
                else:
                    train.append(pair)
    in_train = {(p["instruction"], p["output"]) for p in train}
    seen = [p for p in seen if (p["instruction"], p["output"]) not in in_train]
    return train, seen, unseen
