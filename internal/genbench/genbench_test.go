// Package genbench measures the per-token cost of incremental generation for the cellular
// automaton (INT8 streaming engine), the way a
// deployed model generates. Weights are untrained: generation cost does not depend on them.
//
//	go test ./internal/genbench -bench . -benchtime 3x
package genbench

import (
	"fmt"
	"math/rand"
	"testing"

	"github.com/mignon-ai/mignon-ai/internal/ca"
	"github.com/mignon-ai/mignon-ai/internal/infer"
	"github.com/mignon-ai/mignon-ai/internal/quant"
	"github.com/mignon-ai/mignon-ai/internal/train"
)

const (
	benchPreset = "medium"
	// appendTokens is the number of tokens generated per measurement once the window is nearly full.
	appendTokens = 64
	// hybridHeads and hybridWindow configure the CA+attention hybrid candidates.
	hybridHeads  = 4
	hybridWindow = 64
)

var benchContexts = []int{128, 512, 2048}

// stateSizer is implemented by sessions that can report their cache memory.
type stateSizer interface{ StateBytes() int }

// candidate is one architecture to benchmark at a context length.
type candidate struct {
	name       string
	build      func(context int) (session infer.Session, modelBytes int)
	macsPerTok func(context int) int // multiply-adds per generated token with `context` tokens attended
}

// presetConfig resolves the benchmark preset, overriding its steps per block when steps > 0 and
// adding attention sublayers with attnWindow when attnHeads > 0.
func presetConfig(b *testing.B, steps, attnHeads, attnWindow int) train.Config {
	layer := fmt.Sprintf(`{"preset":%q`, benchPreset)
	if steps > 0 {
		layer += fmt.Sprintf(`,"steps":%d`, steps)
	}
	if attnHeads > 0 {
		layer += fmt.Sprintf(`,"attn_heads":%d,"attn_window":%d`, attnHeads, attnWindow)
	}
	cfg, err := train.ResolveConfig([]byte(layer + "}"))
	if err != nil {
		b.Fatal(err)
	}
	return cfg
}

// caCandidate is the INT8 CA streaming engine with the given steps per block, plus attention
// sublayers when attnHeads > 0 (attnWindow 0 = full attention over the context).
func caCandidate(b *testing.B, steps, attnHeads, attnWindow int) candidate {
	cfg := presetConfig(b, steps, attnHeads, attnWindow)
	name := fmt.Sprintf("ca_int8_%dsteps", steps)
	if attnHeads > 0 {
		name += fmt.Sprintf("_attn_w%d", attnWindow)
	}
	return candidate{
		name: name,
		build: func(context int) (infer.Session, int) {
			q := quant.Quantize(cfg.BuildModel(cfg.Vocab, 1), context)
			return q.NewSession(context), cfg.BuildModel(cfg.Vocab, 1).NumParams() // ~1 byte per INT8 weight
		},
		macsPerTok: func(context int) int {
			D, H, V := cfg.StateDim, cfg.HiddenDim, cfg.Vocab
			inDim := ca.InputDim(D)
			tick := 2*H*inDim + 2*D*H + D*D // rule layers + selective latch
			macs := cfg.Blocks*cfg.Steps*tick + V*D
			if cfg.AttnHeads > 0 {
				span := context
				if cfg.AttnWindow > 0 {
					span = min(span, cfg.AttnWindow)
				}
				macs += cfg.Blocks * (4*D*D + 2*span*D) // projections + attention over the span
			}
			return macs
		},
	}
}

// BenchmarkAppend measures generating appendTokens tokens after prefilling the window up to
// appendTokens short of full, so no slide happens and every token attends to ~context tokens.
func BenchmarkAppend(b *testing.B) {
	cands := []candidate{
		caCandidate(b, 2, 0, 0), caCandidate(b, 1, 0, 0),
		caCandidate(b, 1, hybridHeads, 0), caCandidate(b, 1, hybridHeads, hybridWindow),
	}
	for _, ctxLen := range benchContexts {
		for _, c := range cands {
			b.Run(fmt.Sprintf("ctx=%d/%s", ctxLen, c.name), func(b *testing.B) {
				sess, modelBytes := c.build(ctxLen)
				rng := rand.New(rand.NewSource(1))
				prompt := make([]int, ctxLen-appendTokens)
				for i := range prompt {
					prompt[i] = rng.Intn(1000)
				}
				b.ResetTimer()
				for it := 0; it < b.N; it++ {
					b.StopTimer()
					sess.Prefill(prompt)
					b.StartTimer()
					for t := 0; t < appendTokens; t++ {
						sess.Append(rng.Intn(1000))
					}
				}
				b.StopTimer()
				b.ReportMetric(float64(b.Elapsed().Microseconds())/float64(b.N*appendTokens), "µs/token")
				b.ReportMetric(float64(c.macsPerTok(ctxLen))/1e6, "MMAC/token")
				b.ReportMetric(float64(modelBytes)/1024, "model_KB")
				if s, ok := sess.(stateSizer); ok {
					b.ReportMetric(float64(s.StateBytes())/1024, "state_KB")
				}
				b.ReportMetric(0, "ns/op") // per-op time spans prefill bookkeeping; µs/token is the metric
			})
		}
	}
}
