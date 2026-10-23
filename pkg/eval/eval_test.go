package eval_test

import (
	"context"
	"strings"
	"testing"

	"github.com/mignon-ai/mignon-ai/pkg/data"
	"github.com/mignon-ai/mignon-ai/pkg/eval"
	"github.com/mignon-ai/mignon-ai/pkg/infer"
	"github.com/mignon-ai/mignon-ai/pkg/model"
	"github.com/mignon-ai/mignon-ai/pkg/train"
)

var words = []string{"red", "green", "blue", "pink", "gray", "gold"}

func trained(t *testing.T) (*model.Model, []data.Pair) {
	t.Helper()
	cfg, err := train.PresetConfig("micro")
	if err != nil {
		t.Fatal(err)
	}
	cfg.Epochs, cfg.Threads = 150, 2
	var pairs []data.Pair
	for _, w := range words {
		pairs = append(pairs, data.Pair{Prompt: "paint it " + w, Response: "color --name " + w})
	}
	m, _, err := train.FineTune(context.Background(), pairs, train.Options{Config: cfg, Seed: 1, LR: 0.02})
	if err != nil {
		t.Fatal(err)
	}
	return m, pairs
}

func TestEvaluateScoresAndCalibrates(t *testing.T) {
	m, pairs := trained(t)
	// one pair with an answer the model will not give
	pairs = append(pairs, data.Pair{Prompt: "paint it red", Response: "something else entirely"})
	r, err := eval.Evaluate(context.Background(), m, pairs, eval.Options{})
	if err != nil {
		t.Fatal(err)
	}
	if r.Total != len(pairs) || r.Correct < 4 || r.Correct == r.Total || len(r.Examples) != r.Total {
		t.Fatalf("total %d correct %d", r.Total, r.Correct)
	}
	n, ok := 0, 0
	for _, b := range r.Buckets {
		n += b.N
		ok += b.Correct
	}
	if n != r.Total || ok != r.Correct {
		t.Fatalf("buckets hold %d/%d of %d/%d", ok, n, r.Correct, r.Total)
	}
	all, accAll := r.Threshold(0)
	if all != 1 || accAll != r.Accuracy {
		t.Fatalf("threshold 0: %v %v vs accuracy %v", all, accAll, r.Accuracy)
	}
	if answered, _ := r.Threshold(1.5); answered != 0 {
		t.Fatalf("nothing should pass a cut-off above 1, got %v", answered)
	}
}

func TestCustomGraderAndConstraintHook(t *testing.T) {
	m, pairs := trained(t)
	lenient := func(_, _, got string) bool { return strings.HasPrefix(got, "color") }
	called := 0
	r, err := eval.Evaluate(context.Background(), m, pairs, eval.Options{Grader: lenient,
		Constraint: func(string) infer.Constraint { called++; return nil }})
	if err != nil {
		t.Fatal(err)
	}
	if r.Correct != r.Total || called != len(pairs) {
		t.Fatalf("correct %d of %d, constraint created %d times", r.Correct, r.Total, called)
	}
}

func TestRecommendThreshold(t *testing.T) {
	m, pairs := trained(t)
	pairs = append(pairs, data.Pair{Prompt: "paint it red", Response: "never matches"})
	r, _ := eval.Evaluate(context.Background(), m, pairs, eval.Options{})
	cut, answered, ok := r.RecommendThreshold(1)
	if ok {
		if _, acc := r.Threshold(cut); acc != 1 || answered <= 0 {
			t.Fatalf("cut %v gives accuracy %v answering %v", cut, acc, answered)
		}
	}
	if _, _, ok := r.RecommendThreshold(1.1); ok {
		t.Fatal("an impossible target was met")
	}
}

func TestCancel(t *testing.T) {
	m, pairs := trained(t)
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	r, err := eval.Evaluate(ctx, m, pairs, eval.Options{})
	if err != context.Canceled || r.Total != 0 {
		t.Fatalf("%v %+v", err, r)
	}
}
