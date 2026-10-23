package quant

import (
	"encoding/json"
	"testing"

	"github.com/mignon-ai/mignon-ai/internal/ca"
	"github.com/mignon-ai/mignon-ai/internal/checkpoint"
	"github.com/mignon-ai/mignon-ai/internal/tokenizer"
)

// The binary file must reload to exactly the model the JSON file reloads to, for every kind of tensor:
// attention sublayers, an untied head, several blocks.
func TestBinaryFileMatchesJSON(t *testing.T) {
	for name, tied := range map[string]bool{"tied": true, "untied": false} {
		t.Run(name, func(t *testing.T) {
			tok := tokenizer.New()
			tok.Train("binary model files hold weights as raw numbers", 80)
			net := ca.New(ca.Config{VocabSize: tok.VocabSize(), StateSize: 16, HiddenSize: 24,
				Blocks: ca.DefaultTwoBlockConfig(), TiedEmbed: tied, AttnHeads: 2, AttnWindow: 8})
			net.InitWeights(3)
			q := Quantize(net, 32)
			if q.NumParams() != net.NumParams() {
				t.Errorf("quantized model counts %d parameters, the full-precision one %d", q.NumParams(), net.NumParams())
			}
			meta := checkpoint.Metadata{ContextSize: 32, Task: "chat", Extra: json.RawMessage(`{"a":1}`)}

			bin, err := Encode(q, tok, meta)
			if err != nil {
				t.Fatal(err)
			}
			fromBin, tokBin, metaBin, err := Decode(bin)
			if err != nil {
				t.Fatal(err)
			}
			js, _ := json.Marshal(checkpointOf(q, tok, meta))
			fromJSON, _, _, err := Decode(js)
			if err != nil {
				t.Fatal(err)
			}
			if len(bin) >= len(js)/2 {
				t.Errorf("binary file is %d bytes, JSON %d", len(bin), len(js))
			}
			if tokBin.VocabSize() != tok.VocabSize() || metaBin.Task != "chat" || string(metaBin.Extra) != `{"a":1}` {
				t.Fatalf("tokenizer/metadata lost: %d %+v", tokBin.VocabSize(), metaBin)
			}
			ctx := []int{2, 7, 14, 3, 9}
			a, b := fromBin.Call(ctx), fromJSON.Call(ctx)
			for i := range a {
				if a[i] != b[i] {
					t.Fatalf("logit %d differs: binary %v, JSON %v", i, a[i], b[i])
				}
			}
		})
	}
}

func TestBinaryFileRejectsDamage(t *testing.T) {
	net := ca.New(ca.Config{VocabSize: 16, StateSize: 8, HiddenSize: 8, Blocks: ca.DefaultTwoBlockConfig(), TiedEmbed: true})
	good, err := Encode(Quantize(net, 16), nil, checkpoint.Metadata{ContextSize: 16})
	if err != nil {
		t.Fatal(err)
	}
	for name, data := range map[string][]byte{
		"cut in the arrays": good[:len(good)-5],
		"cut in the header": good[:20],
		"trailing bytes":    append(append([]byte{}, good...), 1, 2, 3),
		"wrong version":     append(append([]byte{}, good[:4]...), append([]byte{9, 0}, good[6:]...)...),
		"wrong kind":        append(append([]byte{}, good[:6]...), append([]byte{7}, good[7:]...)...),
	} {
		if _, _, _, err := Decode(data); err == nil {
			t.Errorf("%s: accepted", name)
		}
	}
}
