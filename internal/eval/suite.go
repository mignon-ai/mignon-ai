package eval

import (
	"encoding/json"
	"math"
	"os"
	"slices"
	"sort"

	"github.com/mignon-ai/mignon-ai/internal/constrain"
	"github.com/mignon-ai/mignon-ai/internal/infer"
	"github.com/mignon-ai/mignon-ai/internal/tokenizer"
)

// TopK is how many first-token candidates count as a near miss.
const TopK = 5

// Task is one benchmark probe: the model must continue a context with one of the expected texts.
type Task struct {
	Category string `json:"category"`
	// Instruction is the user request. Chat models get it through the chat template; raw-text
	// models get it as a line of text before Prompt.
	Instruction string `json:"instruction,omitempty"`
	// Prompt is the text the answer continues: the start of the reply for an instruction.
	Prompt string `json:"prompt"`
	// Expected lists accepted continuations exactly as they follow Prompt, leading space
	// included (" origin" after "git pull --rebase").
	Expected []string `json:"expected"`
	// Complete marks Expected as the whole reply, so a chat model must also end it with <EOS>.
	Complete bool `json:"complete,omitempty"`
}

// TaskResult records the outcome of a single benchmark task.
type TaskResult struct {
	Task      Task
	Predicted string  // greedy continuation, as long as the longest expected answer
	Exact     bool    // the greedy continuation is one of the expected answers
	TopK      bool    // the first token of an expected answer is among the TopK most likely
	NLL       float64 // mean negative log-likelihood per token of the likeliest expected answer
}

// CategoryMetric summarizes performance for a specific task category.
type CategoryMetric struct {
	Total  int
	Exact  int
	TopK   int
	SumNLL float64
}

// MeanNLL is the average per-token negative log-likelihood over the category's tasks.
func (m CategoryMetric) MeanNLL() float64 {
	if m.Total == 0 {
		return 0
	}
	return m.SumNLL / float64(m.Total)
}

// Report contains full benchmark suite evaluation results.
type Report struct {
	ModelName      string
	Overall        CategoryMetric
	CategoryScores map[string]CategoryMetric
	Results        []TaskResult
}

// LoadTasks parses a JSON benchmark suite file.
func LoadTasks(path string) ([]Task, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var tasks []Task
	if err := json.Unmarshal(raw, &tasks); err != nil {
		return nil, err
	}
	return tasks, nil
}

// context encodes what the model sees before its answer, in the format it was trained on.
func (task Task) context(tok *tokenizer.Tokenizer, chat bool) []int {
	switch {
	case task.Instruction == "":
		return tok.Encode(task.Prompt)
	case chat:
		return append(tok.EncodePrompt(task.Instruction), tok.Encode(task.Prompt)...)
	default:
		return tok.Encode(task.Instruction + "\n" + task.Prompt)
	}
}

// answer encodes one expected continuation, closed by <EOS> when it is a chat model's whole reply.
func (task Task) answer(tok *tokenizer.Tokenizer, expected string, chat bool) []int {
	if task.Complete && chat {
		return tok.EncodeReply(expected)
	}
	return tok.Encode(expected)
}

// RunSuite scores every task by teacher-forced log-likelihood of the expected answers and by
// whether greedy decoding reproduces one of them exactly. chat selects the chat template. A
// non-nil grammar constrains the greedy decoding of tasks whose reply is a command; the
// likelihood scores are not affected.
func RunSuite(getLogits infer.LogitsFn, tok *tokenizer.Tokenizer, tasks []Task, contextSize int, modelName string, chat bool, grammar *constrain.Grammar) *Report {
	report := &Report{ModelName: modelName, CategoryScores: make(map[string]CategoryMetric)}
	for _, task := range tasks {
		res, ok := runTask(getLogits, tok, task, contextSize, chat, grammar)
		if !ok {
			continue
		}
		cat := report.CategoryScores[task.Category]
		cat.add(res)
		report.CategoryScores[task.Category] = cat
		report.Overall.add(res)
		report.Results = append(report.Results, res)
	}
	return report
}

func (m *CategoryMetric) add(res TaskResult) {
	m.Total++
	m.SumNLL += res.NLL
	if res.Exact {
		m.Exact++
	}
	if res.TopK {
		m.TopK++
	}
}

func runTask(getLogits infer.LogitsFn, tok *tokenizer.Tokenizer, task Task, contextSize int, chat bool, grammar *constrain.Grammar) (TaskResult, bool) {
	ctx := task.context(tok, chat)
	answers := make([][]int, 0, len(task.Expected))
	longest := 0
	for _, exp := range task.Expected {
		if ids := task.answer(tok, exp, chat); len(ids) > 0 {
			answers = append(answers, ids)
			longest = max(longest, len(ids))
		}
	}
	if len(ctx) == 0 || len(answers) == 0 {
		return TaskResult{}, false
	}
	first := getLogits(window(ctx, contextSize))
	if len(first) == 0 {
		return TaskResult{}, false
	}

	// getLogits may reuse its output buffer, so the first-token ranking is taken before decoding.
	firstTop := topIDs(first, TopK)
	var state *constrain.State
	if grammar != nil && task.Instruction != "" {
		state = grammar.Start(task.Instruction, task.Prompt)
	}
	greedy := greedyContinuation(getLogits, ctx, longest, contextSize, state)
	res := TaskResult{Task: task, Predicted: tok.Decode(greedy), NLL: math.Inf(1)}
	for _, ids := range answers {
		res.Exact = res.Exact || (len(greedy) >= len(ids) && slices.Equal(greedy[:len(ids)], ids))
		res.TopK = res.TopK || slices.Contains(firstTop, ids[0])
		res.NLL = min(res.NLL, answerNLL(getLogits, ctx, ids, contextSize))
	}
	return res, true
}

// answerNLL is the mean negative log-likelihood per token of ids following ctx.
func answerNLL(getLogits infer.LogitsFn, ctx, ids []int, contextSize int) float64 {
	seq := slices.Clone(ctx)
	total := 0.0
	for _, id := range ids {
		total -= logProb(getLogits(window(seq, contextSize)), id)
		seq = append(seq, id)
	}
	return total / float64(len(ids))
}

// greedyContinuation returns the n most likely next tokens, each chosen given the previous ones.
// A non-nil state masks the tokens it does not allow.
func greedyContinuation(getLogits infer.LogitsFn, ctx []int, n, contextSize int, state *constrain.State) []int {
	seq := slices.Clone(ctx)
	for range n {
		logits := getLogits(window(seq, contextSize))
		next := 0
		if state != nil {
			logits = slices.Clone(logits)
			state.Mask(logits)
			next = argmax(logits)
			state.Advance(next)
		} else {
			next = argmax(logits)
		}
		seq = append(seq, next)
	}
	return seq[len(ctx):]
}

// window returns the last contextSize tokens.
func window(tokens []int, contextSize int) []int {
	return tokens[max(0, len(tokens)-contextSize):]
}

// logProb is the log-softmax probability of id under logits.
func logProb(logits []float64, id int) float64 {
	maxL := slices.Max(logits)
	sum := 0.0
	for _, l := range logits {
		sum += math.Exp(l - maxL)
	}
	return logits[id] - maxL - math.Log(sum)
}

func argmax(logits []float64) int {
	return slices.Index(logits, slices.Max(logits))
}

// topIDs returns the k token IDs with the highest logits, most likely first.
func topIDs(logits []float64, k int) []int {
	ids := make([]int, len(logits))
	for i := range ids {
		ids[i] = i
	}
	sort.SliceStable(ids, func(a, b int) bool { return logits[ids[a]] > logits[ids[b]] })
	return ids[:min(k, len(ids))]
}
