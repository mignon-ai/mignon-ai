package tokenizer

import "unicode"

// maxWordBytes caps a pre-token so long runs without boundaries (hashes, minified code) cannot
// make BPE training quadratic.
const maxWordBytes = 128

// charClass groups characters that may share a pre-token.
type charClass int

const (
	classWord charClass = iota
	classSpace
	classPunct
	classDigit
)

func classOf(r rune) charClass {
	switch {
	case unicode.IsSpace(r):
		return classSpace
	case unicode.IsPunct(r) || unicode.IsSymbol(r):
		return classPunct
	case unicode.IsDigit(r):
		return classDigit
	default:
		return classWord
	}
}

// splitWords cuts text into the pre-tokens BPE merges within and calls emit for each in order;
// concatenating them gives back text. Pre-tokens are runs of letters/digits or of punctuation,
// each with at most one leading space, and runs of other whitespace. The space belongs to the
// next word ("git", " add"), as in GPT-2, Llama and SentencePiece, so a prompt that ends on a
// word ends on a token boundary and the answer starts with the space it had in training.
//
// Every digit is a pre-token of its own (as in Llama), so a number always encodes digit by digit
// instead of in chunks that depend on which digit strings were frequent.
func splitWords(text string, emit func(string)) {
	start := 0
	cur := classSpace
	for i, r := range text {
		c := classOf(r)
		if i == start {
			cur = c
			continue
		}
		if c == cur && c != classDigit && i-start < maxWordBytes {
			continue
		}
		if cur == classSpace && c != classSpace && text[i-1] == ' ' {
			// The space right before a word or punctuation run belongs to that run.
			if i-1 > start {
				emit(text[start : i-1])
			}
			start = i - 1
		} else {
			emit(text[start:i])
			start = i
		}
		cur = c
	}
	if start < len(text) {
		emit(text[start:])
	}
}
