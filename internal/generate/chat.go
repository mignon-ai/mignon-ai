// Package generate samples tokens from a trained model: text continuation, chat replies and the
// interactive chat loop.
package generate

import (
	"bufio"
	"fmt"
	"os"
	"slices"
	"strings"

	"github.com/mignon-ai/mignon-ai/internal/constrain"
	"github.com/mignon-ai/mignon-ai/internal/infer"
	"github.com/mignon-ai/mignon-ai/internal/tokenizer"
)

// RunInteractiveChat reads lines from the terminal and prints replyFn's answer to each.
func RunInteractiveChat(title string, replyFn func(input string) string) {
	fmt.Println("\n========================================")
	fmt.Printf("  %s\n", title)
	fmt.Println("  Type 'exit' or 'quit' to end the chat.")
	fmt.Println("========================================")

	scanner := bufio.NewScanner(os.Stdin)
	for {
		fmt.Print("\nYou > ")
		if !scanner.Scan() {
			break
		}
		input := strings.TrimSpace(scanner.Text())
		if input == "" {
			continue
		}
		if input == "exit" || input == "quit" {
			fmt.Println("Goodbye!")
			break
		}

		reply := replyFn(input)
		fmt.Printf("mignon-ai > %s\n", reply)
	}
}

// Sampling configures how a reply or continuation is drawn from the model's logits.
type Sampling struct {
	MaxTokens   int     // generated tokens at most
	Temperature float64 // 0 = greedy argmax
	RepPenalty  float64 // > 1 discourages recently used tokens
	TopK        int     // 0 = disabled
	TopP        float64 // nucleus cutoff; 1 = disabled
	// Grammar, when set, restricts chat replies that are commands to plausible words (see package
	// constrain). Constraint is the per-reply state it creates; callers leave it nil.
	Grammar    *constrain.Grammar
	Constraint Constraint
	Cancel     func() bool // checked before each token; true stops with what was generated so far
}

// Constraint masks the logits of tokens that are not allowed next and learns the sampled token.
type Constraint interface {
	Mask(logits []float64)
	Advance(id int)
}

// Text autoregressively samples a continuation of prompt using BPE decoding, stopping early at a
// document end or turn marker. The model generates incrementally from cached state.
func Text(model infer.SessionProvider, tok *tokenizer.Tokenizer, prompt string, contextSize int, sp Sampling) string {
	sess := model.NewSession(contextSize)
	if tok == nil {
		return ""
	}
	inputTokens := tok.Encode(prompt)
	genTokens := sampleContinuation(sess, inputTokens, inputTokens, sp, tok.EOSTokenID(), tok.UserTokenID(), tok.BotTokenID())
	return tok.Decode(genTokens)
}

// keepLastTurns drops the oldest turns (each ending in eosTokenID) until at most maxTurns remain.
func keepLastTurns(history []int, eosTokenID int, maxTurns int) []int {
	if maxTurns <= 0 {
		return nil
	}
	for turns := countTurns(history, eosTokenID); turns > maxTurns; turns-- {
		end := slices.Index(history, eosTokenID)
		if end < 0 {
			return nil
		}
		history = history[end+1:]
	}
	return history
}

func countTurns(history []int, eosTokenID int) int {
	n := 0
	for _, id := range history {
		if id == eosTokenID {
			n++
		}
	}
	return n
}

// ChatWithHistory generates a reply to prompt with the chat template while maintaining multi-turn
// history. maxHistoryTurns caps the previous completed turns kept (0 = stateless/single-turn).
// It returns the reply and the updated conversation history.
func ChatWithHistory(model infer.SessionProvider, tok *tokenizer.Tokenizer, history []int, prompt string, contextSize int, maxHistoryTurns int, sp Sampling) (string, []int) {
	promptText := strings.TrimSpace(prompt)
	if tok == nil || promptText == "" {
		return "", history
	}

	// Keep the newest turns that fit next to this prompt and room for the reply.
	turn := tok.EncodePrompt(promptText)
	history = keepLastTurns(history, tok.EOSTokenID(), maxHistoryTurns)
	history = tok.DropOldestTurns(history, contextSize-len(turn)-ChatReplyHeadroom)
	conversation := append(slices.Clone(history), turn...)

	if sp.Grammar != nil {
		if st := sp.Grammar.Start(promptText, ""); st != nil {
			sp.Constraint = st
		}
	}
	sess := model.NewSession(contextSize)
	responseTokens := sampleContinuation(sess, conversation, nil, sp, tok.EOSTokenID(), tok.UserTokenID(), tok.BotTokenID())

	// The finished turn joins the history exactly as training encodes it.
	updatedHistory := append(conversation, responseTokens...)
	updatedHistory = append(updatedHistory, tok.EOSTokenID())
	return strings.TrimSpace(tok.Decode(responseTokens)), updatedHistory
}

// Chat generates a single-turn reply with the chat template.
func Chat(model infer.SessionProvider, tok *tokenizer.Tokenizer, prompt string, contextSize int, sp Sampling) string {
	reply, _ := ChatWithHistory(model, tok, nil, prompt, contextSize, 0, sp)
	return reply
}
