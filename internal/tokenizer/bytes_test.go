package tokenizer

import (
	"math"
	"testing"
)

func TestTokenBytesSumToDecodedText(t *testing.T) {
	tok := New()
	text := "commit 1a2b fixes the ünïcode path\n"
	tok.Train(text, 60)
	ids := append(tok.Encode(text), tok.EOSTokenID())
	sizes, total := tok.TokenBytes(), 0
	for _, id := range ids {
		total += sizes[id]
	}
	if total != len(text) {
		t.Fatalf("token bytes sum to %d, decoded text has %d (control tokens must count 0)", total, len(text))
	}
	if got := BitsPerByte(math.Ln2*float64(total), total); math.Abs(got-1) > 1e-12 {
		t.Fatalf("ln 2 nats per byte must be 1 bit per byte, got %g", got)
	}
}
