package generate

import (
	"reflect"
	"testing"

	"github.com/mignon-ai/mignon-ai/internal/ca"
	"github.com/mignon-ai/mignon-ai/internal/infer"
	"github.com/mignon-ai/mignon-ai/internal/quant"
	"github.com/mignon-ai/mignon-ai/internal/train"
)

const greedyTemperature = 0 // SampleRawLogitsAdvanced is argmax at this temperature

func TestSampleRawLogitsAdvanced(t *testing.T) {
	// Logits with clear distribution: token 2 is highest, token 0 is second, token 1 is lowest
	logits := []float64{2.0, 0.5, 5.0, -1.0}

	// 1. Greedy argmax (temp = 0.0) -> must pick index 2
	greedyChoice := SampleRawLogitsAdvanced(logits, nil, 0.0, 1.0, 0, 1.0)
	if greedyChoice != 2 {
		t.Errorf("Greedy sampling failed: got %d, want 2", greedyChoice)
	}

	// 2. Top-K = 1 with temperature -> must always pick index 2
	topKChoice := SampleRawLogitsAdvanced(logits, nil, 0.8, 1.0, 1, 1.0)
	if topKChoice != 2 {
		t.Errorf("Top-K=1 sampling failed: got %d, want 2", topKChoice)
	}

	// 3. Top-P (Nucleus) with high confidence on dominant token
	dominantLogits := []float64{-10.0, -10.0, 20.0, -10.0}
	topPChoice := SampleRawLogitsAdvanced(dominantLogits, nil, 1.0, 1.0, 0, 0.90)
	if topPChoice != 2 {
		t.Errorf("Top-P dominant token failed: got %d, want 2", topPChoice)
	}

	// 4. Repetition penalty: heavily penalize token 2 so token 0 is chosen
	penalizedChoice := SampleRawLogitsAdvanced(logits, []int{2}, 0.0, 10.0, 0, 1.0)
	if penalizedChoice == 2 {
		t.Errorf("Repetition penalty failed to suppress token 2: got %d", penalizedChoice)
	}
}

func BenchmarkSampleRawLogitsAdvanced(b *testing.B) {
	logits := make([]float64, 2048)
	for i := range logits {
		logits[i] = float64(i % 10)
	}
	recent := []int{5, 12, 88, 124, 2}

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_ = SampleRawLogitsAdvanced(logits, recent, 0.7, 1.2, 40, 0.90)
	}
}

// scriptedSession emits one-hot logits that make greedy sampling follow script.
type scriptedSession struct {
	script  []int
	vocab   int
	pos     int
	appends int
}

func (s *scriptedSession) logits() []float64 {
	out := make([]float64, s.vocab)
	out[s.script[s.pos]] = 1
	s.pos++
	return out
}
func (s *scriptedSession) Prefill([]int) []float64 { return s.logits() }
func (s *scriptedSession) Append(int) []float64    { s.appends++; return s.logits() }

// greedySampling draws maxTokens argmax tokens with no penalty or truncation.
func greedySampling(maxTokens int) Sampling {
	return Sampling{MaxTokens: maxTokens, Temperature: greedyTemperature, RepPenalty: 1, TopP: 1}
}

func TestSampleContinuationStopsAndLimits(t *testing.T) {
	const vocab, stop = 8, 7
	sess := &scriptedSession{script: []int{3, 4, stop, 5}, vocab: vocab}
	got := sampleContinuation(sess, []int{1}, nil, greedySampling(10), stop)
	if !reflect.DeepEqual(got, []int{3, 4}) {
		t.Fatalf("stop token: got %v, want [3 4]", got)
	}

	sess = &scriptedSession{script: []int{3, 4, 5, 6}, vocab: vocab}
	got = sampleContinuation(sess, []int{1}, nil, greedySampling(2), stop)
	if !reflect.DeepEqual(got, []int{3, 4}) || sess.appends != 1 {
		t.Fatalf("max tokens: got %v with %d appends, want [3 4] with 1 (no forward pass after the last token)", got, sess.appends)
	}
}

// banToken is a Constraint that forbids one token and records what it saw.
type banToken struct {
	banned   int
	advanced []int
}

func (b *banToken) Mask(logits []float64) { logits[b.banned] = -1e300 }
func (b *banToken) Advance(id int)        { b.advanced = append(b.advanced, id) }

func TestSampleContinuationAppliesConstraint(t *testing.T) {
	// The model's favourite token at every step is 3; the constraint forbids it, so the next best
	// (every other logit is 0, index 0 wins the argmax ties) is chosen instead.
	sess := &scriptedSession{script: []int{3, 3, 3}, vocab: 8}
	c := &banToken{banned: 3}
	sp := greedySampling(3)
	sp.Constraint = c
	got := sampleContinuation(sess, []int{1}, nil, sp)
	if !reflect.DeepEqual(got, []int{0, 0, 0}) || !reflect.DeepEqual(c.advanced, got) {
		t.Fatalf("got %v, constraint saw %v; want [0 0 0] for both", got, c.advanced)
	}
}

// BenchmarkGeneration compares cached INT8 generation with full recomputation per token on
// preset-sized models. Each iteration prefills a short prompt and generates past the window.
func BenchmarkGeneration(b *testing.B) {
	const (
		promptTokens = 16
		genFactor    = 2 // generate twice the context so the window overflows
	)
	for _, name := range []string{"small", "medium"} {
		p := train.Presets[name]
		model := ca.New(ca.Config{VocabSize: p.Vocab, StateSize: p.Dim, HiddenSize: p.Hidden, Blocks: ca.BuildBlockConfigs(p.Blocks, p.Steps, p.Dilation), TiedEmbed: p.Tied})
		model.InitWeights(1)
		q := quant.Quantize(model, p.Context)
		prompt := make([]int, promptTokens)
		for i := range prompt {
			prompt[i] = i + 1
		}
		genTokens := genFactor * p.Context
		modes := map[string]func() infer.Session{
			"cached":    func() infer.Session { return q.NewSession(p.Context) },
			"recompute": func() infer.Session { return infer.Recompute(q.Forward, p.Context) },
		}
		for _, mode := range []string{"cached", "recompute"} {
			b.Run(name+"/"+mode, func(b *testing.B) {
				for it := 0; it < b.N; it++ {
					sampleContinuation(modes[mode](), prompt, nil, greedySampling(genTokens))
				}
				b.ReportMetric(float64(b.N*genTokens)/b.Elapsed().Seconds(), "tok/s")
			})
		}
	}
}
