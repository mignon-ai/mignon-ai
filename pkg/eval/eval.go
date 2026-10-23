// Package eval measures a model on prompt/response pairs the way it is used: it decodes the reply greedily,
// compares it with the expected one, and relates each answer's correctness to how sure the model was. What
// counts as correct is up to the caller (a Grader); the default is an exact match.
package eval

import (
	"context"
	"sort"
	"strings"

	"github.com/mignon-ai/mignon-ai/pkg/data"
	"github.com/mignon-ai/mignon-ai/pkg/infer"
	"github.com/mignon-ai/mignon-ai/pkg/model"
)

// Grader decides whether got is a correct answer to prompt when want was expected. Use it to accept replies
// that differ in form from want, for example the same query with its options in another order.
type Grader func(prompt, want, got string) bool

// Exact accepts a reply equal to the expected one, ignoring surrounding white space.
func Exact(_, want, got string) bool { return strings.TrimSpace(want) == strings.TrimSpace(got) }

// Options controls Evaluate.
type Options struct {
	// MaxTokens bounds each reply; zero uses infer.DefaultMaxTokens.
	MaxTokens int
	// Grader judges each reply; nil means Exact.
	Grader Grader
	// Constraint, if set, creates the constraint for each reply (one per prompt).
	Constraint func(prompt string) infer.Constraint
	// Edges are the confidence bucket boundaries for the calibration table, ascending, covering 0 to 1.
	// Nil uses DefaultEdges.
	Edges []float64
}

// DefaultEdges split confidence into buckets that separate sure answers from guesses.
var DefaultEdges = []float64{0, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99, 1}

// Example is the outcome for one pair.
type Example struct {
	Prompt  string        `json:"prompt"`
	Want    string        `json:"want"`
	Got     string        `json:"got"`
	Correct bool          `json:"correct"`
	MinProb float64       `json:"min_prob"` // the weakest token of the reply: how sure the model was
	Tokens  []infer.Token `json:"tokens,omitempty"`
}

// Bucket counts the answers whose confidence fell in [Lo, Hi) (the last bucket includes Hi).
type Bucket struct {
	Lo       float64 `json:"lo"`
	Hi       float64 `json:"hi"`
	N        int     `json:"n"`
	Correct  int     `json:"correct"`
	Accuracy float64 `json:"accuracy"` // Correct/N, 0 when empty
}

// Report is the result of Evaluate.
type Report struct {
	Total    int       `json:"total"`
	Correct  int       `json:"correct"`
	Accuracy float64   `json:"accuracy"`
	Buckets  []Bucket  `json:"calibration"`
	Examples []Example `json:"examples"`
}

// Evaluate decodes every pair's prompt and grades the reply. It returns what it has measured so far with
// ctx.Err() when ctx is cancelled.
func Evaluate(ctx context.Context, m *model.Model, pairs []data.Pair, o Options) (*Report, error) {
	grade := o.Grader
	if grade == nil {
		grade = Exact
	}
	edges := o.Edges
	if len(edges) < 2 {
		edges = DefaultEdges
	}
	r := &Report{}
	for _, p := range pairs {
		if err := ctx.Err(); err != nil {
			r.finish(edges)
			return r, err
		}
		opts := infer.Options{MaxTokens: o.MaxTokens}
		if o.Constraint != nil {
			opts.Constraint = o.Constraint(p.Prompt)
		}
		out, err := infer.Generate(ctx, m, p.Prompt, opts)
		if err != nil {
			r.finish(edges)
			return r, err
		}
		r.Examples = append(r.Examples, Example{
			Prompt: p.Prompt, Want: p.Response, Got: out.Text,
			Correct: grade(p.Prompt, p.Response, out.Text), MinProb: out.MinProb(), Tokens: out.Tokens,
		})
	}
	r.finish(edges)
	return r, nil
}

func (r *Report) finish(edges []float64) {
	r.Total = len(r.Examples)
	r.Buckets = make([]Bucket, len(edges)-1)
	for i := range r.Buckets {
		r.Buckets[i].Lo, r.Buckets[i].Hi = edges[i], edges[i+1]
	}
	for _, e := range r.Examples {
		if e.Correct {
			r.Correct++
		}
		for i := range r.Buckets {
			b := &r.Buckets[i]
			if e.MinProb >= b.Lo && (e.MinProb < b.Hi || i == len(r.Buckets)-1) {
				b.N++
				if e.Correct {
					b.Correct++
				}
				break
			}
		}
	}
	if r.Total > 0 {
		r.Accuracy = float64(r.Correct) / float64(r.Total)
	}
	for i := range r.Buckets {
		if b := &r.Buckets[i]; b.N > 0 {
			b.Accuracy = float64(b.Correct) / float64(b.N)
		}
	}
}

// Threshold reports what happens if replies below minProb are rejected (shown to the user as unsure
// instead of used): the share of requests still answered and the accuracy among those.
func (r *Report) Threshold(minProb float64) (answered, accuracy float64) {
	n, ok := 0, 0
	for _, e := range r.Examples {
		if e.MinProb >= minProb {
			n++
			if e.Correct {
				ok++
			}
		}
	}
	if r.Total > 0 {
		answered = float64(n) / float64(r.Total)
	}
	if n > 0 {
		accuracy = float64(ok) / float64(n)
	}
	return answered, accuracy
}

// RecommendThreshold returns the lowest confidence cut-off at which the answers kept are at least
// targetAccuracy correct, with the share of requests still answered. ok is false when no cut-off reaches
// the target.
func (r *Report) RecommendThreshold(targetAccuracy float64) (minProb, answered float64, ok bool) {
	for _, cut := range append([]float64{0}, edgesOf(r.Examples)...) {
		a, acc := r.Threshold(cut)
		if a > 0 && acc >= targetAccuracy {
			return cut, a, true
		}
	}
	return 0, 0, false
}

// edgesOf returns the distinct confidence values of the examples in ascending order: every point where
// the kept set can change.
func edgesOf(examples []Example) []float64 {
	seen := map[float64]bool{}
	var out []float64
	for _, e := range examples {
		if !seen[e.MinProb] {
			seen[e.MinProb] = true
			out = append(out, e.MinProb)
		}
	}
	sort.Float64s(out)
	return out
}
