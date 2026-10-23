package quant

import (
	"encoding/json"
	"fmt"
	"os"
	"slices"

	"github.com/mignon-ai/mignon-ai/internal/binfmt"
	"github.com/mignon-ai/mignon-ai/internal/ca"
	"github.com/mignon-ai/mignon-ai/internal/checkpoint"
	"github.com/mignon-ai/mignon-ai/internal/tokenizer"
)

// CheckpointJSON represents the saved multi-block model
type CheckpointJSON struct {
	VocabSize  int                 `json:"vocab_size"`
	StateSize  int                 `json:"state_size"`
	HiddenSize int                 `json:"hidden_size"`
	TiedEmbed  bool                `json:"tied_embed"`
	Embed      Tensor              `json:"embed"`
	Blocks     []Block             `json:"blocks"`
	FinalNorm  []float32           `json:"final_norm,omitempty"`
	HeadW      Tensor              `json:"head_w,omitempty"`
	HeadB      []float32           `json:"head_b"`
	Vocab      []string            `json:"vocab"`
	Merges     []tokenizer.Merge   `json:"merges,omitempty"`
	Metadata   checkpoint.Metadata `json:"metadata"`
}

// checkpointOf describes q, its tokenizer and metadata as a checkpoint.
func checkpointOf(q *Model, tok *tokenizer.Tokenizer, meta checkpoint.Metadata) CheckpointJSON {
	meta.TiedEmbed = q.tiedEmbed
	meta.AttnHeads, meta.AttnWindow = q.attnHeads, q.attnWindow

	var headW Tensor
	if !q.tiedEmbed {
		headW = Tensor{Data: q.headW, Scale: q.headWScale}
	}

	cp := CheckpointJSON{
		VocabSize:  q.vocabSize,
		StateSize:  q.stateSize,
		HiddenSize: q.hiddenSize,
		TiedEmbed:  q.tiedEmbed,
		Embed:      Tensor{Data: q.embedWeights, Scale: q.embedScale},
		Blocks:     q.blocks,
		FinalNorm:  q.finalNorm,
		HeadW:      headW,
		HeadB:      q.headB,
		Metadata:   meta,
	}

	if tok != nil {
		cp.Vocab = tok.Vocab
		cp.Merges = tok.Merges
		cp.Metadata.TokenizerVersion = tokenizer.FormatVersion
	}
	return cp
}

// Save saves a quantized multi-block model to JSON
func Save(q *Model, tok *tokenizer.Tokenizer, meta checkpoint.Metadata, filename string) error {
	data, err := json.MarshalIndent(checkpointOf(q, tok, meta), "", "  ")
	if err != nil {
		return err
	}
	if err := checkpoint.BackupExisting(filename); err != nil {
		return err
	}
	return os.WriteFile(filename, data, 0644)
}

// arrays lists every weight array of cp in the order of the binary file: the embedding, then per block
// the rule weights and biases, the gate, the optional attention sublayer, then the final norm, the head
// weights and the head bias.
func (cp *CheckpointJSON) arrays() []binfmt.Array {
	i8 := func(p *[]int8) binfmt.Array { return binfmt.Array{I8: p} }
	f32 := func(p *[]float32) binfmt.Array { return binfmt.Array{F32: p} }
	out := []binfmt.Array{i8(&cp.Embed.Data)}
	for i := range cp.Blocks {
		b := &cp.Blocks[i]
		out = append(out, i8(&b.RuleW1.Data), f32(&b.RuleB1), i8(&b.RuleW2.Data), f32(&b.RuleB2), f32(&b.Gamma), i8(&b.RuleWSelect.Data))
		if b.Attn != nil {
			out = append(out, f32(&b.Attn.Norm), i8(&b.Attn.QKV.Data), i8(&b.Attn.Out.Data))
		}
	}
	return append(out, f32(&cp.FinalNorm), i8(&cp.HeadW.Data), f32(&cp.HeadB))
}

// NumParams counts the weights of the model: every element of every stored array. It equals the parameter
// count of the full-precision model it was quantized from.
func (q *Model) NumParams() int {
	cp := checkpointOf(q, nil, checkpoint.Metadata{})
	n := 0
	for _, a := range cp.arrays() {
		if a.I8 != nil {
			n += len(*a.I8)
		} else {
			n += len(*a.F32)
		}
	}
	return n
}

// withoutArrays returns a copy of cp whose weight arrays are empty: what goes into the binary header.
func (cp CheckpointJSON) withoutArrays() CheckpointJSON {
	cp.Blocks = slices.Clone(cp.Blocks)
	for i := range cp.Blocks {
		if a := cp.Blocks[i].Attn; a != nil {
			copied := *a
			cp.Blocks[i].Attn = &copied
		}
	}
	for _, a := range cp.arrays() {
		if a.I8 != nil {
			*a.I8 = nil
		} else {
			*a.F32 = nil
		}
	}
	return cp
}

// Encode returns the binary model file (mignon.model/1) of q.
func Encode(q *Model, tok *tokenizer.Tokenizer, meta checkpoint.Metadata) ([]byte, error) {
	cp := checkpointOf(q, tok, meta)
	header, err := json.Marshal(cp.withoutArrays())
	if err != nil {
		return nil, err
	}
	return binfmt.Encode(binfmt.Int8, header, cp.arrays()), nil
}

// SaveBinary writes the binary model file, keeping the previous file as .bak.
func SaveBinary(q *Model, tok *tokenizer.Tokenizer, meta checkpoint.Metadata, filename string) error {
	data, err := Encode(q, tok, meta)
	if err != nil {
		return err
	}
	if err := checkpoint.BackupExisting(filename); err != nil {
		return err
	}
	return os.WriteFile(filename, data, 0644)
}

// Load loads a quantized multi-block model from JSON
func Load(filename string) (*Model, *tokenizer.Tokenizer, *checkpoint.Metadata, error) {
	data, err := os.ReadFile(filename)
	if err != nil {
		return nil, nil, nil, err
	}
	q, tok, meta, err := Decode(data)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("%s: %w", filename, err)
	}
	return q, tok, meta, nil
}

// Decode builds a quantized model from the bytes of a saved checkpoint (what Load reads from a file;
// the browser build gets them from a download).
func Decode(data []byte) (*Model, *tokenizer.Tokenizer, *checkpoint.Metadata, error) {
	var cp CheckpointJSON
	if binfmt.IsBinary(data) {
		f, err := binfmt.Decode(data)
		if err != nil {
			return nil, nil, nil, err
		}
		if f.Kind != binfmt.Int8 {
			return nil, nil, nil, fmt.Errorf("not an INT8 model file")
		}
		if err := json.Unmarshal(f.Header, &cp); err != nil {
			return nil, nil, nil, err
		}
		if err := f.Fill(cp.arrays()); err != nil {
			return nil, nil, nil, err
		}
	} else if err := json.Unmarshal(data, &cp); err != nil {
		return nil, nil, nil, err
	}

	if err := checkBlockShapes(cp.Blocks, cp.StateSize, cp.HiddenSize); err != nil {
		return nil, nil, nil, err
	}
	q := NewModel(cp.VocabSize, cp.StateSize, cp.HiddenSize, cp.Metadata.ContextSize, len(cp.Blocks))
	q.tiedEmbed = cp.TiedEmbed
	q.embedWeights = cp.Embed.Data
	q.embedScale = cp.Embed.Scale
	q.blocks = cp.Blocks
	q.finalNorm = cp.FinalNorm
	q.headB = cp.HeadB

	if q.tiedEmbed {
		q.headW = q.embedWeights
		q.headWScale = q.embedScale
	} else {
		q.headW = cp.HeadW.Data
		q.headWScale = cp.HeadW.Scale
	}

	var tok *tokenizer.Tokenizer
	if len(cp.Vocab) > 0 {
		if err := tokenizer.CheckFormat(cp.Metadata.TokenizerVersion); err != nil {
			return nil, nil, nil, err
		}
		tok = tokenizer.FromVocabAndMerges(cp.Vocab, cp.Merges)
	}

	q.attnHeads, q.attnWindow = cp.Metadata.AttnHeads, cp.Metadata.AttnWindow

	return q, tok, &cp.Metadata, nil
}

// checkBlockShapes rejects checkpoints whose rule weights do not match the current neighbourhood
// width (ca.InputDim), which checkpoints trained with a different rule input do not.
func checkBlockShapes(blocks []Block, stateSize, hiddenSize int) error {
	want := 2 * hiddenSize * ca.InputDim(stateSize)
	for i := range blocks {
		if got := len(blocks[i].RuleW1.Data); got != want {
			return fmt.Errorf("block %d rule_w1 has %d weights, want %d (2*hidden*%d): trained with a different CA neighbourhood, retrain the model",
				i, got, want, ca.InputDim(stateSize))
		}
	}
	return nil
}
