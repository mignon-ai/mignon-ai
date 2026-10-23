// Package model loads, saves and describes trained models. A Model is either full precision (it can be
// trained further) or INT8 (small and fast, for inference). Use pkg/train to make one, pkg/infer to run
// one and pkg/eval to measure one.
package model

import (
	"encoding/json"
	"fmt"
	"os"

	"github.com/mignon-ai/mignon-ai/internal/binfmt"
	"github.com/mignon-ai/mignon-ai/internal/ca"
	"github.com/mignon-ai/mignon-ai/internal/checkpoint"
	"github.com/mignon-ai/mignon-ai/internal/handle"
	"github.com/mignon-ai/mignon-ai/internal/quant"
)

// Task labels stored in a model file.
const (
	TaskText = "text" // continues raw text
	TaskChat = "chat" // answers prompts with the chat template
)

// defaultContext is the window of files that do not record one (written before it was stored).
const defaultContext = 64

// Model is a trained network with its tokenizer and metadata.
type Model struct {
	s handle.State
}

func init() {
	handle.Of = func(m any) *handle.State {
		if m, ok := m.(*Model); ok && m != nil {
			return &m.s
		}
		return nil
	}
	handle.Wrap = func(s handle.State) any {
		s.Meta = normalize(s.Meta)
		return &Model{s}
	}
}

// Info describes a model without exposing its internals.
type Info struct {
	Quantized   bool   `json:"quantized"`
	Task        string `json:"task"`
	Params      int    `json:"params"`
	VocabSize   int    `json:"vocab_size"`
	ContextSize int    `json:"context_size"`
	Blocks      int    `json:"blocks"`
	StateSize   int    `json:"state_size"`
	HiddenSize  int    `json:"hidden_size"`
}

// Load reads a model file: binary (what Save writes) or the older JSON form, INT8 or full precision, detected
// from its content.
func Load(path string) (*Model, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	m, err := Decode(data)
	if err != nil {
		return nil, fmt.Errorf("%s: %w", path, err)
	}
	return m, nil
}

// Decode builds a model from the bytes of a model file (what Load reads from disk; a browser gets them
// from a download).
func Decode(data []byte) (*Model, error) {
	if binfmt.IsBinary(data) {
		f, err := binfmt.Decode(data)
		if err != nil {
			return nil, err
		}
		if f.Kind == binfmt.Int8 {
			return decodeInt8(data)
		}
	} else if q, tok, meta, err := quant.Decode(data); err == nil && q.VocabSize() > 0 && q.NumBlocks() > 0 {
		return &Model{handle.State{Int8: q, Tok: tok, Meta: normalize(*meta)}}, nil
	}
	meta, tok, weights, err := checkpoint.Decode(data)
	if err != nil {
		return nil, err
	}
	if tok == nil {
		return nil, fmt.Errorf("model file holds no tokenizer")
	}
	if meta.StateSize < 1 || meta.HiddenSize < 1 || meta.Steps < 1 || meta.Blocks < 1 {
		return nil, fmt.Errorf("model file lacks architecture metadata (state_size, hidden_size, steps, blocks)")
	}
	net := ca.New(ca.Config{
		VocabSize:  tok.VocabSize(),
		StateSize:  meta.StateSize,
		HiddenSize: meta.HiddenSize,
		Blocks:     ca.BuildBlockConfigs(meta.Blocks, meta.Steps, meta.DilationBase),
		TiedEmbed:  meta.TiedEmbed,
		AttnHeads:  meta.AttnHeads,
		AttnWindow: meta.AttnWindow,
	})
	if len(weights) != net.NumParams() {
		return nil, fmt.Errorf("architecture mismatch: file has %d weights, the model needs %d", len(weights), net.NumParams())
	}
	net.SetWeights(weights)
	return &Model{handle.State{Full: net, Tok: tok, Meta: normalize(*meta)}}, nil
}

func decodeInt8(data []byte) (*Model, error) {
	q, tok, meta, err := quant.Decode(data)
	if err != nil {
		return nil, err
	}
	return &Model{handle.State{Int8: q, Tok: tok, Meta: normalize(*meta)}}, nil
}

// normalize fills the defaults of old files: no task means raw text, no context means the default window.
func normalize(m checkpoint.Metadata) checkpoint.Metadata {
	if m.Task == "" {
		m.Task = TaskText
	}
	if m.ContextSize <= 0 {
		m.ContextSize = defaultContext
	}
	return m
}

// Save writes the model to path, keeping the previous file as path+".bak" when there was one.
func (m *Model) Save(path string) error {
	if m.s.Int8 != nil {
		return quant.SaveBinary(m.s.Int8, m.s.Tok, m.s.Meta, path)
	}
	return checkpoint.SaveBinary(m.s.Full, m.s.Tok, m.s.Meta, path)
}

// Quantize returns an INT8 copy for inference. It fails on a model that is already INT8.
func (m *Model) Quantize() (*Model, error) {
	if m.s.Full == nil {
		return nil, fmt.Errorf("model is already quantized")
	}
	return &Model{handle.State{Int8: quant.Quantize(m.s.Full, m.s.Meta.ContextSize), Tok: m.s.Tok, Meta: m.s.Meta}}, nil
}

// Dequantize returns a full-precision copy that can be trained further. A full-precision model is
// returned as it is. Precision lost to quantization does not come back.
func (m *Model) Dequantize() *Model {
	if m.s.Full != nil {
		return m
	}
	return &Model{handle.State{Full: quant.Dequantize(m.s.Int8), Tok: m.s.Tok, Meta: m.s.Meta}}
}

// Quantized reports whether the model is INT8.
func (m *Model) Quantized() bool { return m.s.Int8 != nil }

// Task is TaskChat or TaskText.
func (m *Model) Task() string { return m.s.Meta.Task }

// ContextSize is the number of tokens the model reads at most.
func (m *Model) ContextSize() int { return m.s.Meta.ContextSize }

// Extra returns the opaque bytes stored with the model by SetExtra (nil when there are none).
func (m *Model) Extra() json.RawMessage { return m.s.Meta.Extra }

// SetExtra stores any JSON with the model, for example the description of its task. The engine never
// reads it; Save and Load keep the same JSON value (whitespace may change).
func (m *Model) SetExtra(extra json.RawMessage) error {
	if len(extra) > 0 && !json.Valid(extra) {
		return fmt.Errorf("extra is not valid JSON")
	}
	m.s.Meta.Extra = extra
	return nil
}

// Info describes the model.
func (m *Model) Info() Info {
	info := Info{Quantized: m.s.Int8 != nil, Task: m.s.Meta.Task, ContextSize: m.s.Meta.ContextSize}
	if m.s.Tok != nil {
		info.VocabSize = m.s.Tok.VocabSize()
	}
	if m.s.Int8 != nil {
		info.Params = m.s.Int8.NumParams()
		info.Blocks, info.StateSize, info.HiddenSize = m.s.Int8.NumBlocks(), m.s.Int8.StateSize(), m.s.Int8.HiddenSize()
	} else {
		info.Params = m.s.Full.NumParams()
		info.Blocks, info.StateSize, info.HiddenSize = m.s.Full.NumBlocks(), m.s.Full.StateSize(), m.s.Full.HiddenSize()
	}
	return info
}

// Encode turns text into token ids; Decode turns them back.
func (m *Model) Encode(text string) []int { return m.s.Tok.Encode(text) }

// Decode turns token ids into text.
func (m *Model) Decode(ids []int) string { return m.s.Tok.Decode(ids) }
