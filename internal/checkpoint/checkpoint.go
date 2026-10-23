// Package checkpoint saves and loads full-precision model weights together with the BPE
// tokenizer and the hyperparameters needed to rebuild the model.
package checkpoint

import (
	"encoding/json"
	"fmt"
	"os"

	"github.com/mignon-ai/mignon-ai/internal/binfmt"
	"github.com/mignon-ai/mignon-ai/internal/tokenizer"
)

// Model is any neural model that exposes its trainable weights
type Model interface {
	Weights() []float32
}

// Metadata holds hyperparameter settings for model reconstruction
type Metadata struct {
	Blocks       int    `json:"blocks,omitempty"`
	StateSize    int    `json:"state_size,omitempty"`
	HiddenSize   int    `json:"hidden_size,omitempty"`
	Steps        int    `json:"steps,omitempty"`
	DilationBase int    `json:"dilation_base,omitempty"`
	ContextSize  int    `json:"context_size,omitempty"`
	Task         string `json:"task,omitempty"` // "chat" or "text"
	TiedEmbed    bool   `json:"tied_embed,omitempty"`
	AttnHeads    int    `json:"attn_heads,omitempty"`  // attention sublayer heads after each block; 0 = none
	AttnWindow   int    `json:"attn_window,omitempty"` // attention span in tokens; 0 = full causal
	// TokenizerVersion is the tokenizer.FormatVersion the model's tokenizer was trained under.
	TokenizerVersion int `json:"tokenizer_version,omitempty"`
	// Extra is opaque to the engine: whoever builds a model may store anything here (for example the
	// description of the task it was trained for) and gets it back unchanged from Load.
	Extra json.RawMessage `json:"extra,omitempty"`
}

// File holds all the data needed to recreate the trained model state
type File struct {
	Weights  []float32         `json:"weights"`
	Vocab    []string          `json:"vocab"`
	Merges   []tokenizer.Merge `json:"merges,omitempty"`
	Metadata Metadata          `json:"metadata,omitempty"`
}

// BackupExisting creates a .bak backup of filename if it already exists before overwriting.
func BackupExisting(filename string) error {
	if _, err := os.Stat(filename); err == nil {
		bak := filename + ".bak"
		data, err := os.ReadFile(filename)
		if err != nil {
			return fmt.Errorf("failed to read file for backup %s: %w", filename, err)
		}
		if err := os.WriteFile(bak, data, 0644); err != nil {
			return fmt.Errorf("failed to write backup file %s: %w", bak, err)
		}
		fmt.Printf("Created backup of existing checkpoint at %s\n", bak)
	}
	return nil
}

// Save exports model weights and the BPE tokenizer
func Save(model Model, tok *tokenizer.Tokenizer, meta Metadata, filename string) error {
	weights := model.Weights()

	meta.TokenizerVersion = tokenizer.FormatVersion
	cp := File{
		Weights:  weights,
		Vocab:    tok.Vocab,
		Merges:   tok.Merges,
		Metadata: meta,
	}

	data, err := json.MarshalIndent(cp, "", "  ")
	if err != nil {
		return err
	}

	if err := BackupExisting(filename); err != nil {
		return err
	}
	return os.WriteFile(filename, data, 0644)
}

// Encode returns the binary model file (mignon.model/1) of a full-precision model.
func Encode(model Model, tok *tokenizer.Tokenizer, meta Metadata) ([]byte, error) {
	meta.TokenizerVersion = tokenizer.FormatVersion
	header, err := json.Marshal(File{Vocab: tok.Vocab, Merges: tok.Merges, Metadata: meta})
	if err != nil {
		return nil, err
	}
	weights := model.Weights()
	return binfmt.Encode(binfmt.Full, header, []binfmt.Array{{F32: &weights}}), nil
}

// SaveBinary writes the binary model file, keeping the previous file as .bak.
func SaveBinary(model Model, tok *tokenizer.Tokenizer, meta Metadata, filename string) error {
	data, err := Encode(model, tok, meta)
	if err != nil {
		return err
	}
	if err := BackupExisting(filename); err != nil {
		return err
	}
	return os.WriteFile(filename, data, 0644)
}

// Load loads metadata, BPE tokenizer, and raw weights.
func Load(filename string) (*Metadata, *tokenizer.Tokenizer, []float32, error) {
	data, err := os.ReadFile(filename)
	if err != nil {
		return nil, nil, nil, err
	}
	return Decode(data)
}

// Decode reads a full-precision model from the bytes of a file, binary or JSON.
func Decode(data []byte) (*Metadata, *tokenizer.Tokenizer, []float32, error) {
	var cp File
	if binfmt.IsBinary(data) {
		f, err := binfmt.Decode(data)
		if err != nil {
			return nil, nil, nil, err
		}
		if f.Kind != binfmt.Full {
			return nil, nil, nil, fmt.Errorf("not a full-precision model file")
		}
		if err := json.Unmarshal(f.Header, &cp); err != nil {
			return nil, nil, nil, err
		}
		if err := f.Fill([]binfmt.Array{{F32: &cp.Weights}}); err != nil {
			return nil, nil, nil, err
		}
	} else if err := json.Unmarshal(data, &cp); err != nil {
		return nil, nil, nil, err
	}

	var tok *tokenizer.Tokenizer
	if len(cp.Vocab) > 0 {
		if err := tokenizer.CheckFormat(cp.Metadata.TokenizerVersion); err != nil {
			return nil, nil, nil, err
		}
		tok = tokenizer.FromVocabAndMerges(cp.Vocab, cp.Merges)
	}

	return &cp.Metadata, tok, cp.Weights, nil
}
