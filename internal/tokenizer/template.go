package tokenizer

import "slices"

// Chat template. Every prompt/response pair, in training, generation and evaluation, is encoded as
//
//	<USER> prompt <BOT> response <EOS>
//
// and only through these two functions, so the model always sees turns in the format it learned.

// EncodePrompt encodes a user turn up to and including the marker after which the model answers.
func (t *Tokenizer) EncodePrompt(prompt string) []int {
	ids := []int{t.UserTokenID()}
	ids = append(ids, t.Encode(prompt)...)
	return append(ids, t.BotTokenID())
}

// EncodeReply encodes the model's answer to a turn, ending it with <EOS>.
func (t *Tokenizer) EncodeReply(reply string) []int {
	return append(t.Encode(reply), t.EOSTokenID())
}

// DropOldestTurns removes whole turns (each ending in <EOS>) from the front of history until it
// is at most maxLen tokens long. A history with no complete turn left is dropped entirely.
func (t *Tokenizer) DropOldestTurns(history []int, maxLen int) []int {
	eos := t.EOSTokenID()
	for len(history) > max(maxLen, 0) {
		end := slices.Index(history, eos)
		if end < 0 {
			return nil
		}
		history = history[end+1:]
	}
	return history
}
