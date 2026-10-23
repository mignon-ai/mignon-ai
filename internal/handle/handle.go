// Package handle lets the public packages under pkg/ share a model's internals without exposing
// them: pkg/model registers Of at init, and pkg/infer, pkg/train and pkg/eval use it to reach the
// network, the tokenizer and the metadata behind a *model.Model.
package handle

import (
	"github.com/mignon-ai/mignon-ai/internal/ca"
	"github.com/mignon-ai/mignon-ai/internal/checkpoint"
	"github.com/mignon-ai/mignon-ai/internal/infer"
	"github.com/mignon-ai/mignon-ai/internal/quant"
	"github.com/mignon-ai/mignon-ai/internal/tokenizer"
)

// State is what a public model holds: a full-precision network or an INT8 one, never both.
type State struct {
	Full *ca.Model
	Int8 *quant.Model
	Tok  *tokenizer.Tokenizer
	Meta checkpoint.Metadata
}

// Of returns the state behind a *model.Model. It is set by pkg/model.
var Of func(model any) *State

// Wrap builds a *model.Model from a state. It is set by pkg/model; callers assert the type.
var Wrap func(State) any

// Provider returns the model as a source of generation sessions.
func (s *State) Provider() infer.SessionProvider {
	if s.Int8 != nil {
		return s.Int8
	}
	return infer.RecomputeProvider(s.Full.Forward)
}

// Logits returns the model's next-token function over a window of tokens.
func (s *State) Logits() infer.LogitsFn {
	if s.Int8 != nil {
		return s.Int8.Forward
	}
	return s.Full.Forward
}
