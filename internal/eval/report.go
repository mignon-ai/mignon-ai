package eval

import (
	"fmt"
	"sort"
)

// PrintReport renders a structured terminal scorecard
func PrintReport(report *Report, heldout HeldOut, heldoutPath string, verbose bool) {
	heldoutPPL := heldout.PPL
	fmt.Println("\n=================== MODEL EVALUATION SCORECARD ===================")
	fmt.Printf("Model Checkpoint: %s\n\n", report.ModelName)

	if heldoutPath != "" && heldout.Tokens > 0 {
		fmt.Println("[1. Held-Out Generalization Evaluation]")
		fmt.Printf("  Evaluation Corpus:  %s (%d tokens)\n", heldoutPath, heldout.Tokens)
		fmt.Printf("  Cross-Entropy Loss: %.4f\n", heldout.Loss)
		fmt.Printf("  Held-Out Perplexity: %.2f\n", heldoutPPL)
		fmt.Printf("  Bits per Byte:      %.4f (comparable across tokenizers)\n", heldout.BPB)

		if heldoutPPL < 35.0 {
			fmt.Println("  Verdict:            EXCELLENT GENERALIZATION (Natural prose modeling)")
		} else if heldoutPPL < 100.0 {
			fmt.Println("  Verdict:            MODERATE GENERALIZATION (Emerging subword patterns)")
		} else {
			fmt.Println("  Verdict:            HIGH PERPLEXITY / MEMORIZATION ALERT")
		}
		fmt.Println()
	}

	if report.Overall.Total == 0 {
		return
	}
	fmt.Printf("[2. Multi-Domain Capability Benchmark Suite] (%d Probe Tasks)\n", report.Overall.Total)
	fmt.Printf("  Exact: greedy output matches | Top-%d: first token in top %d | NLL: per-token loss of the answer (lower is better)\n", TopK, TopK)

	cats := make([]string, 0, len(report.CategoryScores))
	for cat := range report.CategoryScores {
		cats = append(cats, cat)
	}
	sort.Strings(cats)
	for _, cat := range cats {
		printMetric(cat, report.CategoryScores[cat])
	}
	fmt.Println("  ------------------------------------------------------------------------------")
	printMetric("OVERALL", report.Overall)
	printFailureBreakdown(report)
	fmt.Println("==================================================================")

	if verbose && len(report.Results) > 0 {
		fmt.Println("\n[Detailed Benchmark Task Breakdown]")
		classifier := newFailureClassifier(reportTasks(report))
		for i, res := range report.Results {
			status, why := "FAIL", ""
			if res.Exact {
				status = "PASS"
			} else {
				if res.TopK {
					status = fmt.Sprintf("TOP%d", TopK)
				}
				why = " [" + string(classifier.classify(res)) + "]"
			}
			fmt.Printf("  [%3d] [%-4s] %-30s %q %q -> %q (exp: %q, NLL %.2f)%s\n",
				i+1, status, res.Task.Category, res.Task.Instruction, res.Task.Prompt, res.Predicted, res.Task.Expected, res.NLL, why)
		}
		fmt.Println("==================================================================")
	}
}

func printMetric(name string, m CategoryMetric) {
	fmt.Printf("  • %-34s Exact %3d/%-3d (%5.1f%%) | Top-%d %5.1f%% | NLL %.3f\n",
		name+":", m.Exact, m.Total, percent(m.Exact, m.Total), TopK, percent(m.TopK, m.Total), m.MeanNLL())
}

func percent(n, total int) float64 {
	if total == 0 {
		return 0
	}
	return 100 * float64(n) / float64(total)
}
