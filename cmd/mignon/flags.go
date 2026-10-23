package main

import (
	"flag"
	"fmt"
	"slices"
	"strings"

	"github.com/mignon-ai/mignon-ai/internal/generate"
)

// CLI defaults that are not part of the training configuration.
const (
	defaultHistoryTurns = 1
	defaultGenTokens    = 100
	defaultTemp         = 0.7
	defaultPenalty      = 1.2
	defaultTopP         = 0.90
	defaultTasksPath    = "eval/tasks.json"
)

// Flag groups shared by several subcommands. Each registers its flags on a FlagSet.

// modelFlags selects the model checkpoint to operate on.
type modelFlags struct {
	load string
}

func (m *modelFlags) register(fs *flag.FlagSet, usage string) {
	fs.StringVar(&m.load, "load", "", usage)
}

// samplingFlags control how text is generated.
type samplingFlags struct {
	mode      string
	genTokens int
	temp      float64
	penalty   float64
	topK      int
	topP      float64
	constrain string
}

func (s *samplingFlags) register(fs *flag.FlagSet) {
	fs.StringVar(&s.mode, "mode", modeAuto, "How input is used: 'auto' (chat for -sft models, text otherwise), 'chat' (<USER>/<BOT> template reply) or 'text' (plain continuation)")
	fs.IntVar(&s.genTokens, "gen-tokens", defaultGenTokens, fmt.Sprintf("Maximum tokens to generate (chat mode defaults to %d)", generate.ChatMaxResponseTokens))
	fs.Float64Var(&s.temp, "temp", defaultTemp, fmt.Sprintf("Sampling temperature, 0 = greedy argmax (chat mode defaults to %g)", chatTemp))
	fs.Float64Var(&s.penalty, "penalty", defaultPenalty, fmt.Sprintf("Repetition penalty, > 1 reduces token loops (chat mode defaults to %g)", chatPenalty))
	fs.IntVar(&s.topK, "top-k", 0, "Top-K sampling (0 = disabled)")
	fs.Float64Var(&s.topP, "top-p", defaultTopP, "Top-P (Nucleus) sampling cutoff")
	fs.StringVar(&s.constrain, "constrain", "", "Prompt/response pairs JSON whose replies define the words a command may use; chat replies that start with a command are restricted to them and to words of the request")
}

func (s *samplingFlags) validate() error {
	if !slices.Contains(generationModes, s.mode) {
		return fmt.Errorf("unknown -mode %q (want one of %v)", s.mode, generationModes)
	}
	return nil
}

// parse parses args into fs and returns the names of the flags set on the command line.
func parse(fs *flag.FlagSet, args []string) (map[string]bool, error) {
	if err := fs.Parse(args); err != nil {
		return nil, err
	}
	explicit := make(map[string]bool)
	fs.Visit(func(f *flag.Flag) { explicit[f.Name] = true })
	return explicit, nil
}

// command is a subcommand: mignon <name> [flags].
type command struct {
	name, summary string
	run           func(args []string)
}

// newFlagSet returns the flag set of subcommand name, exiting on syntax errors.
func newFlagSet(name string) *flag.FlagSet {
	return flag.NewFlagSet("mignon "+name, flag.ExitOnError)
}

// usage lists the subcommands.
func usage(commands []command) string {
	var b strings.Builder
	b.WriteString("Usage: mignon <command> [flags]\n\nCommands:\n")
	for _, c := range commands {
		fmt.Fprintf(&b, "  %-10s %s\n", c.name, c.summary)
	}
	b.WriteString("\nRun 'mignon <command> -h' for a command's flags.\n")
	return b.String()
}
