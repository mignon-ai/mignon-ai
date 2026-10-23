package eval

import (
	"slices"
	"testing"

	"github.com/mignon-ai/mignon-ai/internal/tokenizer"
)

const suiteTestContext = 64

func suiteTokenizer() *tokenizer.Tokenizer {
	tok := tokenizer.New()
	tok.Train("stage file a\ngit add a\nstage file b\ngit add b\n", 60)
	return tok
}

// scriptedModel predicts, with high confidence, the token that follows the longest matching
// suffix of script; with no match it predicts token 0.
func scriptedModel(vocab int, script []int) func([]int) []float64 {
	return func(window []int) []float64 {
		logits := make([]float64, vocab)
		next := 0
		for n := len(window); n > 0 && next == 0; n-- {
			suffix := window[len(window)-n:]
			for i := 0; i+n < len(script); i++ {
				if slices.Equal(script[i:i+n], suffix) {
					next = script[i+n]
					break
				}
			}
		}
		logits[next] = 10
		return logits
	}
}

func TestRunSuiteUsesChatTemplateAndRequiresEOSForCompleteAnswers(t *testing.T) {
	tok := suiteTokenizer()
	// The model has learned exactly one chat turn.
	script := append(tok.EncodePrompt("stage file a"), tok.EncodeReply("git add a")...)
	model := scriptedModel(tok.VocabSize(), script)

	tasks := []Task{
		{Category: "full", Instruction: "stage file a", Expected: []string{"git add a"}, Complete: true},
		{Category: "next", Instruction: "stage file a", Prompt: "git", Expected: []string{" add"}},
		{Category: "wrong", Instruction: "stage file a", Prompt: "git", Expected: []string{" commit"}},
	}
	report := RunSuite(model, tok, tasks, suiteTestContext, "test", true, nil)
	exact := map[string]bool{}
	for _, r := range report.Results {
		exact[r.Task.Category] = r.Exact
	}
	if !exact["full"] || !exact["next"] || exact["wrong"] {
		t.Errorf("unexpected exact matches: %v", exact)
	}
	if report.CategoryScores["next"].MeanNLL() >= report.CategoryScores["wrong"].MeanNLL() {
		t.Errorf("the answer the model predicts must have the lower NLL")
	}
	if report.Overall.Total != len(tasks) || report.Overall.Exact != 2 {
		t.Errorf("overall = %+v, want 2/%d exact", report.Overall, len(tasks))
	}
}

func TestRunSuiteCompleteAnswerFailsWithoutEOS(t *testing.T) {
	tok := suiteTokenizer()
	// The model keeps talking after the command instead of ending its reply.
	script := append(tok.EncodePrompt("stage file a"), tok.Encode("git add a b")...)
	model := scriptedModel(tok.VocabSize(), script)
	task := Task{Category: "full", Instruction: "stage file a", Expected: []string{"git add a"}, Complete: true}
	if report := RunSuite(model, tok, []Task{task}, suiteTestContext, "test", true, nil); report.Results[0].Exact {
		t.Errorf("a reply that does not end after the expected command must not count as exact")
	}
}

func TestTaskContextFormats(t *testing.T) {
	tok := suiteTokenizer()
	task := Task{Instruction: "stage file a", Prompt: "git"}
	if got, want := task.context(tok, true), append(tok.EncodePrompt("stage file a"), tok.Encode("git")...); !slices.Equal(got, want) {
		t.Errorf("chat context = %v, want %v", got, want)
	}
	if got, want := task.context(tok, false), tok.Encode("stage file a\ngit"); !slices.Equal(got, want) {
		t.Errorf("text context = %v, want %v", got, want)
	}
	plain := Task{Prompt: "stage file"}
	if got, want := plain.context(tok, true), tok.Encode("stage file"); !slices.Equal(got, want) {
		t.Errorf("a task without instruction must be plain text even for chat models, got %v", got)
	}
}
