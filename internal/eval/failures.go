package eval

import (
	"fmt"
	"strings"
)

// FailureKind says why a task's greedy answer was wrong, judged at its first wrong word.
type FailureKind string

const (
	// FailWrongSpan: the expected word is in the request, but a different request word came out.
	FailWrongSpan FailureKind = "copy: wrong word from the request"
	// FailPartialCopy: the expected word is in the request and was started, then broken.
	FailPartialCopy FailureKind = "copy: right word started, then broken"
	// FailNoCopy: the expected word is in the request, but the model wrote something else.
	FailNoCopy FailureKind = "copy: invented instead of copying"
	// FailKnowledge: the expected word is not in the request (subcommand, option, fact).
	FailKnowledge FailureKind = "knowledge: wrong word not in the request"
	// FailEnding: every expected word is right, but the stop or the spacing is not.
	FailEnding FailureKind = "ending: right words, wrong stop or spacing"
)

// failureKinds lists every kind in report order.
var failureKinds = []FailureKind{FailWrongSpan, FailPartialCopy, FailNoCopy, FailKnowledge, FailEnding}

// minSharedRequests is how many distinct requests must have a word in their expected answers
// for it to count as answer vocabulary (git, pull, origin) rather than request-specific content
// (a branch name, a path, a message) the model has to copy.
const minSharedRequests = 2

// minPartialCopyFraction is the share of the expected word a wrong word must exceed in its
// common prefix to count as a broken copy rather than an invented one.
const minPartialCopyFraction = 0.5

// wordTrimChars are stripped from both ends of a word, so quoting and punctuation around a
// name do not hide that it was copied.
const wordTrimChars = "'\"`,:;()[]"

// words splits text on whitespace and trims quoting and punctuation from each word.
func words(text string) []string {
	var out []string
	for _, w := range strings.Fields(text) {
		if w = strings.Trim(w, wordTrimChars); w != "" {
			out = append(out, w)
		}
	}
	return out
}

// firstMismatch returns the first word where predicted departs from expected, the matched
// word count, and false when every expected word was produced.
func firstMismatch(expected, predicted []string) (want, got string, matched int, ok bool) {
	for i, w := range expected {
		if i >= len(predicted) {
			return w, "", i, true
		}
		if predicted[i] != w {
			return w, predicted[i], i, true
		}
	}
	return "", "", len(expected), false
}

// commonPrefixLen is the length in bytes of the longest common prefix of a and b.
func commonPrefixLen(a, b string) int {
	n := 0
	for n < len(a) && n < len(b) && a[n] == b[n] {
		n++
	}
	return n
}

// failureClassifier judges failures against the vocabulary shared by a whole task suite.
type failureClassifier struct {
	vocabulary map[string]bool // words in the expected answers of minSharedRequests+ requests
}

func newFailureClassifier(tasks []Task) failureClassifier {
	requests := make(map[string]map[string]bool) // word -> requests whose answers contain it
	for _, task := range tasks {
		for _, exp := range task.Expected {
			for _, w := range words(task.Prompt + exp) {
				if requests[w] == nil {
					requests[w] = make(map[string]bool)
				}
				requests[w][task.Instruction] = true
			}
		}
	}
	vocabulary := make(map[string]bool)
	for w, reqs := range requests {
		if len(reqs) >= minSharedRequests {
			vocabulary[w] = true
		}
	}
	return failureClassifier{vocabulary: vocabulary}
}

// classify explains a failed task. It compares the greedy answer with the expected answer it
// follows longest and judges the first wrong word: a request-specific word is a copy failure.
func (c failureClassifier) classify(res TaskResult) FailureKind {
	request := make(map[string]bool)
	for _, w := range words(res.Task.Instruction) {
		request[w] = true
	}
	// Only the continuation is compared: text glued onto the prompt must not break its last word.
	predicted := words(res.Predicted)

	bestMatched := -1
	var want, got string
	var mismatch bool
	for _, exp := range res.Task.Expected {
		w, g, matched, ok := firstMismatch(words(exp), predicted)
		if matched > bestMatched {
			bestMatched, want, got, mismatch = matched, w, g, ok
		}
	}

	switch {
	case !mismatch:
		return FailEnding
	case !request[want] || c.vocabulary[want]:
		return FailKnowledge
	case request[got]:
		return FailWrongSpan
	case float64(commonPrefixLen(want, got)) > minPartialCopyFraction*float64(len(want)):
		return FailPartialCopy
	default:
		return FailNoCopy
	}
}

// printFailureBreakdown counts the failed tasks of report by FailureKind.
func printFailureBreakdown(report *Report) {
	classifier := newFailureClassifier(reportTasks(report))
	counts := make(map[FailureKind]int)
	failed := 0
	for _, res := range report.Results {
		if !res.Exact {
			counts[classifier.classify(res)]++
			failed++
		}
	}
	if failed == 0 {
		return
	}
	fmt.Printf("\n[Failure Breakdown] (%d failed tasks, judged at the first wrong word)\n", failed)
	for _, kind := range failureKinds {
		fmt.Printf("  • %-44s %3d (%5.1f%%)\n", string(kind)+":", counts[kind], percent(counts[kind], failed))
	}
}

// reportTasks returns the tasks behind report's results.
func reportTasks(report *Report) []Task {
	tasks := make([]Task, len(report.Results))
	for i, res := range report.Results {
		tasks[i] = res.Task
	}
	return tasks
}
