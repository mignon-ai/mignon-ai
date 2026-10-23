// Command mignon trains, evaluates, quantizes and runs cellular-automaton language models.
//
// Usage: mignon <command> [flags]. The commands are train, eval, chat, generate, quantize, convert and sweep;
// flags shared by several commands (-load, the sampling flags) are registered by common flag groups.
package main

import (
	"fmt"
	"os"

	"github.com/mignon-ai/mignon-ai/internal/sweep"
)

// exitUsage is the status for a missing or unknown command.
const exitUsage = 2

var commands = []command{
	{"train", "Pretrain on raw text (-text) or fine-tune on prompt/response pairs (-sft)", runTrain},
	{"eval", "Report held-out perplexity and/or run the capability benchmark", runEval},
	{"chat", "Talk to a model in an interactive terminal session", runChat},
	{"generate", "Answer or continue a single -prompt", runGenerate},
	{"quantize", "Convert a model to INT8", runQuantize},
	{"convert", "Rewrite a model file in the binary format (reads the older JSON files too)", runConvert},
	{"sweep", "Run a configuration sweep from a JSON spec: mignon sweep <spec.json>", runSweep},
}

func main() {
	if len(os.Args) < 2 {
		fmt.Fprint(os.Stderr, usage(commands))
		os.Exit(exitUsage)
	}
	name := os.Args[1]
	switch name {
	case "help", "-h", "-help", "--help":
		fmt.Print(usage(commands))
		return
	}
	for _, c := range commands {
		if c.name == name {
			c.run(os.Args[2:])
			return
		}
	}
	fmt.Fprintf(os.Stderr, "mignon: unknown command %q\n\n%s", name, usage(commands))
	os.Exit(exitUsage)
}

// runSweep implements 'mignon sweep': run the configuration sweep described by the JSON spec in args[0].
func runSweep(args []string) {
	fs := newFlagSet("sweep")
	if _, err := parse(fs, args); err != nil || fs.NArg() != 1 {
		fatalf("Usage: mignon sweep <spec.json>\n")
	}
	spec, err := sweep.LoadSpec(fs.Arg(0))
	if err == nil {
		_, err = sweep.Run(spec)
	}
	if err != nil {
		fatalf("Sweep failed: %v\n", err)
	}
}

// fatalf prints the message to stdout, where the CLI reports all errors, and exits with status 1.
func fatalf(format string, args ...any) {
	fmt.Printf(format, args...)
	os.Exit(1)
}
