package quant

import (
	"math/rand"
	"reflect"
	"testing"
)

const (
	sessionTestWindow  = 12 // small window so a short sequence overflows it several times
	sessionTestTokens  = 40
	sessionTestPrefill = 5
)

// TestQuantSessionMatchesCall checks that cached generation gives bit-identical logits to a full
// INT8 forward pass, before and after the session's window overflows. Sessions stream past the
// window, so they must match a pass over the whole sequence from its start.
func TestQuantSessionMatchesCall(t *testing.T) {
	for _, tc := range quantEquivCases() {
		t.Run(tc.name, func(t *testing.T) {
			q := Quantize(buildQuantEquivModel(tc), tc.maxContext)
			rng := rand.New(rand.NewSource(9))
			tokens := make([]int, sessionTestTokens)
			for i := range tokens {
				tokens[i] = rng.Intn(quantEquivVocab)
			}

			check := func(got []float64, window []int, label string) {
				t.Helper()
				got = append([]float64(nil), got...) // Call reuses the logits buffer
				if want := q.Call(window); !reflect.DeepEqual(got, want) {
					t.Fatalf("%s: session logits differ from Call over window %v", label, window)
				}
			}

			if tc.maxContext < sessionTestTokens {
				t.Fatalf("maxContext %d cannot hold the %d-token reference pass", tc.maxContext, sessionTestTokens)
			}
			sess := q.NewSession(sessionTestWindow)
			window := append([]int(nil), tokens[:sessionTestPrefill]...)
			check(sess.Prefill(window), window, "prefill")
			for i := sessionTestPrefill; i < len(tokens); i++ {
				window = append(window, tokens[i])
				check(sess.Append(tokens[i]), window, "append")
			}

			// A prompt longer than the window keeps its most recent tokens.
			long := tokens[:2*sessionTestWindow]
			check(sess.Prefill(long), long[len(long)-sessionTestWindow:], "long prefill")
		})
	}
}

// TestQuantSessionStateIsBounded checks that a streaming session's memory does not grow with the
// sequence: its caches hold only the positions the longest dilation can reach.
func TestQuantSessionStateIsBounded(t *testing.T) {
	tc := quantEquivCaseNamed(t, "2blk")
	q := Quantize(buildQuantEquivModel(tc), tc.maxContext)
	sess := q.NewSession(sessionTestWindow).(*quantSession)
	sess.Prefill([]int{1, 2})
	before := sess.StateBytes()
	for i := 0; i < 10*tc.maxContext; i++ {
		sess.Append(i % quantEquivVocab)
	}
	if after := sess.StateBytes(); after != before {
		t.Fatalf("session state grew from %d to %d bytes", before, after)
	}
	if got, want := len(sess.st.norm[0]), q.normRing(); got != want {
		t.Fatalf("per-step cache holds %d positions, want %d", got, want)
	}
}

// TestQuantSessionWindowedAttentionIsBounded checks that a hybrid with an attention window keeps
// constant memory: its key/value rings hold only the window.
func TestQuantSessionWindowedAttentionIsBounded(t *testing.T) {
	tc := quantEquivCaseNamed(t, "2blk attn window")
	q := Quantize(buildQuantEquivModel(tc), tc.maxContext)
	sess := q.NewSession(sessionTestWindow).(*quantSession)
	sess.Prefill([]int{1, 2})
	before := sess.StateBytes()
	for i := 0; i < 10*tc.maxContext; i++ {
		sess.Append(i % quantEquivVocab)
	}
	if after := sess.StateBytes(); after != before {
		t.Fatalf("session state grew from %d to %d bytes", before, after)
	}
	if got := sess.st.attnCap; got != tc.attnWindow {
		t.Fatalf("attention ring holds %d positions, want the window %d", got, tc.attnWindow)
	}
}
