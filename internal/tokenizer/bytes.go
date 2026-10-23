package tokenizer

import "math"

// BitsPerByte converts a summed loss in nats over text of n bytes into bits per byte (0 when n is 0).
func BitsPerByte(nats float64, n int) float64 {
	if n <= 0 {
		return 0
	}
	return nats / (math.Ln2 * float64(n))
}

// TokenBytes returns, for every token id, the number of text bytes it decodes to (0 for control
// tokens), so a loss per token can be turned into bits per byte of text, which does not depend on
// the tokenizer.
func (t *Tokenizer) TokenBytes() []int {
	out := make([]int, len(t.Vocab))
	for id := range t.Vocab {
		out[id] = len(t.Decode([]int{id}))
	}
	return out
}
