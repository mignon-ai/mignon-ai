package generate

import (
	"math"
	"slices"

	"github.com/mignon-ai/mignon-ai/internal/infer"
	"github.com/mignon-ai/mignon-ai/internal/tokenizer"
)

// Piece is one generated token and the probability the model gave it.
type Piece struct {
	Text string  `json:"text"`
	Prob float64 `json:"prob"`
}

// Greedy is a reply decoded by always taking the most likely token.
type Greedy struct {
	Text   string  `json:"text"`
	Pieces []Piece `json:"tokens"`
}

// GreedyOptions configures DecodeGreedy.
type GreedyOptions struct {
	ContextSize int
	MaxTokens   int
	Chat        bool        // answer with the chat template; otherwise continue the prompt as raw text
	Constraint  Constraint  // restricts every step to the allowed tokens; nil = unrestricted
	Cancel      func() bool // checked before each token; true stops decoding with what exists so far
}

// ChatGreedy answers prompt with the chat template, taking the most likely token every step and
// recording its probability. Unlike Chat there is no sampling, repetition penalty or trigram
// blocking, which can change a number or name that has to be copied, so the reply is exactly what
// evaluation measures. Low probabilities mark the parts of the reply the model was unsure of.
func ChatGreedy(model infer.SessionProvider, tok *tokenizer.Tokenizer, prompt string, contextSize, maxTokens int) Greedy {
	return DecodeGreedy(model, tok, prompt, GreedyOptions{ContextSize: contextSize, MaxTokens: maxTokens, Chat: true})
}

// DecodeGreedy decodes the most likely reply (or continuation) to prompt, with the probability of each
// token. A constraint masks the logits before every choice, so the probabilities are over the allowed
// tokens.
func DecodeGreedy(model infer.SessionProvider, tok *tokenizer.Tokenizer, prompt string, o GreedyOptions) Greedy {
	context := tok.Encode(prompt)
	if o.Chat {
		context = tok.EncodePrompt(prompt)
	}
	ids, probs := greedyLoop(model.NewSession(o.ContextSize), context, o.MaxTokens, o.Constraint, o.Cancel,
		tok.EOSTokenID(), tok.UserTokenID(), tok.BotTokenID())
	out := Greedy{Text: tok.Decode(ids)}
	for i, id := range ids {
		out.Pieces = append(out.Pieces, Piece{Text: tok.Decode([]int{id}), Prob: probs[i]})
	}
	return out
}

// greedyContinuation decodes up to maxTokens most-likely tokens after context, stopping before any
// stop token, and returns them with their probabilities.
func greedyContinuation(sess infer.Session, context []int, maxTokens int, stops ...int) ([]int, []float64) {
	return greedyLoop(sess, context, maxTokens, nil, nil, stops...)
}

func greedyLoop(sess infer.Session, context []int, maxTokens int, c Constraint, cancel func() bool, stops ...int) ([]int, []float64) {
	var ids []int
	var probs []float64
	logits := sess.Prefill(context)
	for len(ids) < maxTokens && len(logits) > 0 {
		if cancel != nil && cancel() {
			break
		}
		if c != nil {
			c.Mask(logits)
		}
		best, prob := argmaxProb(logits)
		if best >= 0 && slices.Contains(stops, best) {
			break
		}
		if c != nil {
			c.Advance(best)
		}
		ids, probs = append(ids, best), append(probs, prob)
		if len(ids) < maxTokens {
			logits = sess.Append(best)
		}
	}
	return ids, probs
}

// argmaxProb returns the most likely token and its softmax probability.
func argmaxProb(logits []float64) (int, float64) {
	best := 0
	for i, v := range logits {
		if v > logits[best] {
			best = i
		}
	}
	sum := 0.0
	for _, v := range logits {
		sum += math.Exp(v - logits[best])
	}
	return best, 1 / sum
}
