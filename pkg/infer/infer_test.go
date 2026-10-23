package infer_test

import (
	"context"
	"math"
	"strings"
	"testing"

	"github.com/mignon-ai/mignon-ai/pkg/data"
	"github.com/mignon-ai/mignon-ai/pkg/infer"
	"github.com/mignon-ai/mignon-ai/pkg/model"
	"github.com/mignon-ai/mignon-ai/pkg/train"
)

func tiny(t *testing.T) *model.Model {
	t.Helper()
	cfg, err := train.PresetConfig("micro")
	if err != nil {
		t.Fatal(err)
	}
	cfg.Epochs, cfg.Threads = 20, 2
	var pairs []data.Pair
	for _, w := range []string{"red", "green", "blue", "pink", "gray", "gold"} {
		pairs = append(pairs, data.Pair{Prompt: "paint it " + w, Response: "color --name " + w})
	}
	m, _, err := train.FineTune(context.Background(), pairs, train.Options{Config: cfg, Seed: 1, LR: 0.02})
	if err != nil {
		t.Fatal(err)
	}
	return m
}

// banAll allows no token but the ones in allowed.
type only struct{ allowed map[int]bool }

func (o only) Mask(logits []float64) {
	for i := range logits {
		if !o.allowed[i] {
			logits[i] = -math.MaxFloat64
		}
	}
}
func (only) Advance(int) {}

func TestGreedyIsDeterministicAndRecordsProbabilities(t *testing.T) {
	m := tiny(t)
	a, err := infer.Generate(context.Background(), m, "paint it red", infer.Options{})
	if err != nil {
		t.Fatal(err)
	}
	b, _ := infer.Generate(context.Background(), m, "paint it red", infer.Options{})
	if a.Text != b.Text || len(a.Tokens) == 0 {
		t.Fatalf("%+v vs %+v", a, b)
	}
	joined := ""
	for _, tk := range a.Tokens {
		if tk.Prob <= 0 || tk.Prob > 1 {
			t.Fatalf("probability %v", tk.Prob)
		}
		joined += tk.Text
	}
	if strings.TrimSpace(joined) != a.Text {
		t.Fatalf("tokens %q do not spell %q", joined, a.Text)
	}
}

func TestMaxTokensAndSampling(t *testing.T) {
	m := tiny(t)
	short, _ := infer.Generate(context.Background(), m, "paint it red", infer.Options{MaxTokens: 2})
	if len(short.Tokens) > 2 {
		t.Fatalf("%d tokens", len(short.Tokens))
	}
	sampled, err := infer.Generate(context.Background(), m, "paint it red", infer.Options{Temperature: 0.8, MaxTokens: 8})
	if err != nil || len(sampled.Tokens) != 0 {
		t.Fatalf("%+v %v", sampled, err)
	}
}

func TestConstraintRestrictsTheReply(t *testing.T) {
	m := tiny(t)
	ids := m.Encode("color")
	allowed := map[int]bool{}
	for _, id := range ids {
		allowed[id] = true
	}
	out, err := infer.Generate(context.Background(), m, "paint it red", infer.Options{MaxTokens: 6, Constraint: only{allowed}})
	if err != nil {
		t.Fatal(err)
	}
	for _, tk := range out.Tokens {
		if strings.Trim(tk.Text, " ") != "" && !strings.Contains("color", strings.TrimSpace(tk.Text)) {
			t.Fatalf("token %q outside the allowed set in %+v", tk.Text, out)
		}
	}
}

func TestCancelledContextStopsGeneration(t *testing.T) {
	m := tiny(t)
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	out, err := infer.Generate(ctx, m, "paint it red", infer.Options{})
	if err != context.Canceled || len(out.Tokens) != 0 {
		t.Fatalf("%+v %v", out, err)
	}
}

func TestNilModel(t *testing.T) {
	if _, err := infer.Generate(context.Background(), nil, "x", infer.Options{}); err == nil {
		t.Fatal("nil model accepted")
	}
}
