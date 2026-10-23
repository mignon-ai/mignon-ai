// Package infer runs a model: it answers a prompt (chat models) or continues it (text models).
package infer

import (
	"context"
	"fmt"
	"strings"

	"github.com/mignon-ai/mignon-ai/internal/generate"
	"github.com/mignon-ai/mignon-ai/internal/handle"
	"github.com/mignon-ai/mignon-ai/pkg/model"
)

// DefaultMaxTokens is the reply length limit when Options.MaxTokens is zero.
const DefaultMaxTokens = generate.ChatMaxResponseTokens

// Constraint restricts what a model may write. Before each token Mask receives the model's logits and sets
// those of tokens that are not allowed to a very negative number (-math.MaxFloat64); after the choice,
// Advance receives the token. A constraint is used for one reply: create a new one per call.
type Constraint interface {
	Mask(logits []float64)
	Advance(id int)
}

// Options controls Generate. The zero value is greedy decoding of up to DefaultMaxTokens tokens.
type Options struct {
	MaxTokens int
	// Temperature 0 decodes greedily: always the most likely token, with its probability recorded in
	// Result.Tokens. Above 0 the reply is sampled (Result.Tokens stays empty).
	Temperature float64
	// TopK, TopP and RepPenalty apply when sampling. Zero values disable them (TopP 0 means 1).
	TopK       int
	TopP       float64
	RepPenalty float64
	// Constraint, if set, restricts every token (see Constraint).
	Constraint Constraint
}

// Token is one generated piece of text and the probability the model gave it.
type Token struct {
	Text string  `json:"text"`
	Prob float64 `json:"prob"`
}

// Result is a generated reply.
type Result struct {
	Text string `json:"text"`
	// Tokens are the generated tokens with their probabilities. Greedy decoding only.
	Tokens []Token `json:"tokens,omitempty"`
}

// MinProb is the lowest token probability, 1 for an empty reply or one without recorded tokens. It is a
// quick measure of how sure the model was of the weakest piece of its answer.
func (r Result) MinProb() float64 {
	least := 1.0
	for _, t := range r.Tokens {
		least = min(least, t.Prob)
	}
	return least
}

// Generate answers prompt with a chat model, or continues it with a text model. It returns the text
// produced so far together with ctx.Err() when ctx is cancelled.
func Generate(ctx context.Context, m *model.Model, prompt string, opts Options) (Result, error) {
	s := handle.Of(m)
	if s == nil {
		return Result{}, fmt.Errorf("nil model")
	}
	maxTokens := opts.MaxTokens
	if maxTokens <= 0 {
		maxTokens = DefaultMaxTokens
	}
	canceled := func() bool { return ctx.Err() != nil }
	chat := m.Task() == model.TaskChat
	if opts.Temperature <= 0 {
		g := generate.DecodeGreedy(s.Provider(), s.Tok, prompt, generate.GreedyOptions{
			ContextSize: m.ContextSize(), MaxTokens: maxTokens, Chat: chat, Constraint: opts.Constraint, Cancel: canceled,
		})
		res := Result{Text: strings.TrimSpace(g.Text)}
		for _, p := range g.Pieces {
			res.Tokens = append(res.Tokens, Token{Text: p.Text, Prob: p.Prob})
		}
		return res, ctx.Err()
	}
	sp := generate.Sampling{
		MaxTokens: maxTokens, Temperature: opts.Temperature, TopK: opts.TopK, TopP: opts.TopP, RepPenalty: opts.RepPenalty,
		Constraint: opts.Constraint, Cancel: canceled,
	}
	if sp.TopP <= 0 {
		sp.TopP = 1
	}
	if chat {
		return Result{Text: generate.Chat(s.Provider(), s.Tok, prompt, m.ContextSize(), sp)}, ctx.Err()
	}
	return Result{Text: generate.Text(s.Provider(), s.Tok, prompt, m.ContextSize(), sp)}, ctx.Err()
}
