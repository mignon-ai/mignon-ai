// Package eval measures a trained model with held-out perplexity and a targeted benchmark suite.
package eval

import (
	"fmt"
	"math"

	"github.com/mignon-ai/mignon-ai/internal/infer"
	"github.com/mignon-ai/mignon-ai/internal/tokenizer"
)

// HeldOut is the result of Perplexity. BPB (bits per byte of text) compares models with
// different tokenizers; loss and perplexity are per token.
type HeldOut struct {
	Loss, PPL, BPB float64
	Tokens         int
}

// nll accumulates the negative log-likelihood of scored tokens and the text bytes they cover.
type nll struct {
	nats          float64
	bytes, tokens int
}

// score adds the loss of predicting tokens[from:to], each from the (at most contextSize) tokens before
// it. progress, when non-nil, is called with the number of tokens scored so far in this call.
func (n *nll) score(getLogits infer.LogitsFn, tokenBytes []int, tokens []int, from, to, contextSize int, progress func(done int)) {
	for t := from; t < to; t++ {
		logits := getLogits(window(tokens[:t], contextSize))
		target := tokens[t]
		if target >= 0 && target < len(logits) {
			n.nats -= logProb(logits, target)
			n.bytes += tokenBytes[target]
			n.tokens++
		}
		if progress != nil {
			progress(t - from + 1)
		}
	}
}

func (n *nll) result() HeldOut {
	if n.tokens == 0 {
		return HeldOut{}
	}
	avg := n.nats / float64(n.tokens)
	return HeldOut{Loss: avg, PPL: math.Exp(avg), BPB: tokenizer.BitsPerByte(n.nats, n.bytes), Tokens: n.tokens}
}

// Perplexity computes exact cross-entropy loss, perplexity and bits per byte on arbitrary text. The
// text is one continuous stream, so each token is conditioned on the contextSize tokens before it
// even across document boundaries; see PerplexityPerDoc for independent documents.
func Perplexity(getLogits infer.LogitsFn, tok *tokenizer.Tokenizer, text string, contextSize int, maxTokens int) HeldOut {
	if len(text) == 0 || tok == nil {
		return HeldOut{}
	}

	tokens := tok.Encode(text)
	if len(tokens) < 2 {
		return HeldOut{}
	}
	numTokens := len(tokens) - 1
	if maxTokens > 0 && numTokens > maxTokens {
		numTokens = maxTokens
	}

	var total nll
	total.score(getLogits, tok.TokenBytes(), tokens, 1, numTokens+1, contextSize, func(done int) {
		if done%100 == 0 || done == numTokens {
			fmt.Printf("\r  Evaluating held-out tokens: %d/%d (Current PPL: %.2f)...", done, numTokens, math.Exp(total.nats/float64(done)))
		}
	})
	fmt.Println()
	return total.result()
}
