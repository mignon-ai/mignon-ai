package generate

import (
	"math"
	"math/rand"
	"slices"
	"sync"

	"github.com/mignon-ai/mignon-ai/internal/infer"
)

type tokenProb struct {
	idx  int
	prob float64
}

type samplerScratch struct {
	logits     []float64
	candidates []tokenProb
}

var samplerPool = sync.Pool{
	New: func() any {
		return &samplerScratch{
			logits:     make([]float64, 0, DefaultMaxVocabBuf),
			candidates: make([]tokenProb, 0, DefaultMaxVocabBuf),
		}
	},
}

// SampleRawLogitsAdvanced selects the next token index given raw float64 logits, temperature, repetition penalty, Top-K, and Top-P (Nucleus) sampling.
// Guaranteed 0 dynamic heap allocations per call via pooled scratch buffers and non-allocating in-place sorting.
func SampleRawLogitsAdvanced(scaledLogits []float64, recentTokens []int, temperature float64, repPenalty float64, topK int, topP float64) int {
	V := len(scaledLogits)
	if V == 0 {
		return 0
	}

	scratch := samplerPool.Get().(*samplerScratch)
	if cap(scratch.logits) < V {
		scratch.logits = make([]float64, V)
	} else {
		scratch.logits = scratch.logits[:V]
	}
	copy(scratch.logits, scaledLogits)
	logits := scratch.logits

	// Repetition penalty: reduce scores for recently generated tokens without map heap allocation
	if repPenalty > 1.0 && len(recentTokens) > 0 {
		for i, tok := range recentTokens {
			if tok < 0 || tok >= V {
				continue
			}
			alreadySeen := false
			for _, prev := range recentTokens[:i] {
				if prev == tok {
					alreadySeen = true
					break
				}
			}
			if !alreadySeen {
				if logits[tok] > 0 {
					logits[tok] /= repPenalty
				} else {
					logits[tok] *= repPenalty
				}
			}
		}
	}

	// Deterministic greedy argmax
	if temperature <= 0.001 {
		bestIdx := 0
		maxVal := logits[0]
		for i := 1; i < V; i++ {
			if logits[i] > maxVal {
				maxVal = logits[i]
				bestIdx = i
			}
		}
		samplerPool.Put(scratch)
		return bestIdx
	}

	// Temperature scaling & numerical stabilization
	invT := 1.0 / temperature
	maxLogit := -math.MaxFloat64
	for i := 0; i < V; i++ {
		scaled := logits[i] * invT
		logits[i] = scaled
		if scaled > maxLogit {
			maxLogit = scaled
		}
	}

	// 1. Standard Softmax & Top-K / Top-P filtering
	if topK <= 0 && topP >= 0.9999 {
		// Pure temperature sampling without candidate allocation
		sumExp := 0.0
		for i := 0; i < V; i++ {
			p := math.Exp(logits[i] - maxLogit)
			logits[i] = p
			sumExp += p
		}
		r := rand.Float64() * sumExp
		cum := 0.0
		chosen := 0
		for i := 0; i < V; i++ {
			cum += logits[i]
			if r <= cum {
				chosen = i
				break
			}
		}
		samplerPool.Put(scratch)
		return chosen
	}

	// 2. Truncated sampling (Top-K / Top-P)
	if cap(scratch.candidates) < V {
		scratch.candidates = make([]tokenProb, V)
	} else {
		scratch.candidates = scratch.candidates[:V]
	}
	candidates := scratch.candidates

	sumExp := 0.0
	for i := 0; i < V; i++ {
		p := math.Exp(logits[i] - maxLogit)
		candidates[i] = tokenProb{idx: i, prob: p}
		sumExp += p
	}
	invSum := 1.0 / sumExp
	for i := 0; i < V; i++ {
		candidates[i].prob *= invSum
	}

	// Sort candidates in descending probability
	slices.SortFunc(candidates, func(a, b tokenProb) int {
		if a.prob > b.prob {
			return -1
		}
		if a.prob < b.prob {
			return 1
		}
		return 0
	})

	// Apply Top-K truncation
	activeCount := V
	if topK > 0 && topK < activeCount {
		activeCount = topK
	}

	// Apply Top-P (Nucleus) truncation
	if topP > 0.0 && topP < 1.0 {
		cumP := 0.0
		cutoff := 0
		for i := 0; i < activeCount; i++ {
			cumP += candidates[i].prob
			cutoff = i + 1
			if cumP >= topP {
				break
			}
		}
		activeCount = cutoff
	}

	chosen := sampleFromCDF(candidates[:activeCount])
	samplerPool.Put(scratch)
	return chosen
}

func sampleFromCDF(candidates []tokenProb) int {
	if len(candidates) == 1 {
		return candidates[0].idx
	}
	sumP := 0.0
	for _, c := range candidates {
		sumP += c.prob
	}
	r := rand.Float64() * sumP
	cum := 0.0
	for _, c := range candidates {
		cum += c.prob
		if r <= cum {
			return c.idx
		}
	}
	return candidates[len(candidates)-1].idx
}

// sampleContinuation samples up to sp.MaxTokens tokens after context and stops before any stop
// token. The repetition penalty considers the last ChatRepetitionWindow tokens of
// penaltyPrefix followed by the generated tokens.
func sampleContinuation(sess infer.Session, context, penaltyPrefix []int, sp Sampling, stopTokens ...int) []int {
	if sess == nil {
		return nil
	}
	recent := append([]int(nil), penaltyPrefix...)
	var generated []int
	logits := sess.Prefill(context)
	for len(generated) < sp.MaxTokens && len(logits) > 0 {
		if sp.Cancel != nil && sp.Cancel() {
			break
		}
		window := recent
		if len(window) > ChatRepetitionWindow {
			window = window[len(window)-ChatRepetitionWindow:]
		}

		// Prevent repeating trigrams (3-gram repetition blocking):
		// If the trailing 2 tokens [t_{n-2}, t_{n-1}] already appeared earlier followed by token t_next,
		// suppress t_next so the generator cannot enter an autoregressive loop.
		if len(generated) >= 2 {
			w1 := generated[len(generated)-2]
			w2 := generated[len(generated)-1]
			for i := 0; i <= len(generated)-3; i++ {
				if generated[i] == w1 && generated[i+1] == w2 {
					banned := generated[i+2]
					if banned >= 0 && banned < len(logits) {
						logits[banned] = -math.MaxFloat64
					}
				}
			}
		}

		if sp.Constraint != nil {
			sp.Constraint.Mask(logits)
		}
		next := SampleRawLogitsAdvanced(logits, window, sp.Temperature, sp.RepPenalty, sp.TopK, sp.TopP)
		for _, stop := range stopTokens {
			if stop >= 0 && next == stop {
				return generated
			}
		}
		if sp.Constraint != nil {
			sp.Constraint.Advance(next)
		}
		generated = append(generated, next)
		recent = append(recent, next)
		if len(generated) < sp.MaxTokens {
			logits = sess.Append(next)
		}
	}
	return generated
}
