package main

import (
	"flag"
	"fmt"
	"os"

	"github.com/mignon-ai/mignon-ai/internal/eval"
	"github.com/mignon-ai/mignon-ai/internal/infer"
)

// evalOptions holds the flags of 'mignon eval'.
type evalOptions struct {
	modelFlags
	heldout            string
	tasks, constrain   string
	benchmark, verbose bool
	perDoc             bool
	maxTokens, maxDocs int
}

func parseEvalFlags(fs *flag.FlagSet, args []string) (*evalOptions, error) {
	o := &evalOptions{}
	o.register(fs, "Saved model to evaluate (required)")
	fs.StringVar(&o.heldout, "heldout", "", "Evaluate model perplexity on this held-out text file")
	fs.IntVar(&o.maxTokens, "max-tokens", 0, "Maximum tokens to evaluate for held-out perplexity (0 = all)")
	fs.BoolVar(&o.perDoc, "per-doc", false, "Score each <EOS>-separated document of -heldout independently, with a fresh context")
	fs.IntVar(&o.maxDocs, "max-docs", 0, "Maximum documents to evaluate with -per-doc (0 = all)")
	fs.BoolVar(&o.benchmark, "benchmark", false, "Run standardized multi-domain capability probe suite")
	fs.StringVar(&o.tasks, "tasks", defaultTasksPath, "Path to benchmark probe tasks JSON file")
	fs.StringVar(&o.constrain, "constrain", "", "Pairs JSON whose replies define the command grammar; the greedy decoding of benchmark tasks that are commands is restricted to it")
	fs.BoolVar(&o.verbose, "verbose", false, "Print detailed per-task predictions in benchmark evaluation")
	if _, err := parse(fs, args); err != nil {
		return nil, err
	}
	if o.heldout == "" && !o.benchmark {
		return nil, fmt.Errorf("nothing to evaluate: give -heldout and/or -benchmark")
	}
	return o, nil
}

// runEval implements 'mignon eval': held-out perplexity (-heldout) and/or the benchmark suite (-benchmark).
func runEval(args []string) {
	o, err := parseEvalFlags(newFlagSet("eval"), args)
	if err != nil {
		fatalf("Error: %v\n", err)
	}
	s := openSession(o.load)
	var getLogits infer.LogitsFn
	if s.qmodel != nil {
		getLogits = s.qmodel.Forward
	} else {
		getLogits = s.model.Forward
	}

	var heldout eval.HeldOut
	if o.heldout != "" {
		fmt.Printf("Evaluating held-out perplexity on %s...\n", o.heldout)
		text, err := os.ReadFile(o.heldout)
		if err != nil {
			fatalf("Error reading evaluation file: %v\n", err)
		}
		if o.perDoc {
			heldout = eval.PerplexityPerDoc(getLogits, s.tok, string(text), s.contextSize, o.maxDocs)
		} else {
			heldout = eval.Perplexity(getLogits, s.tok, string(text), s.contextSize, o.maxTokens)
		}
	}

	report := &eval.Report{ModelName: o.load, CategoryScores: make(map[string]eval.CategoryMetric)}
	if o.benchmark {
		fmt.Printf("Running capability benchmark suite from %s...\n", o.tasks)
		tasks, err := eval.LoadTasks(o.tasks)
		if err != nil {
			fatalf("Error loading benchmark tasks: %v\n", err)
		}
		report = eval.RunSuite(getLogits, s.tok, tasks, s.contextSize, o.load, s.task == taskChat, loadGrammar(o.constrain, s))
	}
	eval.PrintReport(report, heldout, o.heldout, o.verbose)
}
