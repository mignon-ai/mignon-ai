// Package tokenizer implements the byte-pair encoding (BPE) subword tokenizer and its special tokens.
package tokenizer

import (
	"encoding/json"
	"os"
	"slices"
	"strings"
)

// Special token constants
const (
	PadToken  = "<PAD>"
	UnkToken  = "<UNK>"
	UserToken = "<USER>"
	BotToken  = "<BOT>"
	EOSToken  = "<EOS>"
)

const (
	// MinCharsPerMillion is how often a character must appear in the tokenizer sample (per million
	// characters) to get its own vocab slot. Rarer characters (stray emoji, other scripts) encode as
	// <UNK> instead of each spending a slot. Samples under a million characters keep every character.
	MinCharsPerMillion = 1.0
	charsPerMillion    = 1e6
)

// DefaultSpecialTokens lists all base special tokens including slot boundaries
var DefaultSpecialTokens = []string{
	PadToken, UnkToken, UserToken, BotToken, EOSToken,
}

// Merge represents a learned BPE pair merge rule
type Merge struct {
	Pair1 string `json:"pair1"`
	Pair2 string `json:"pair2"`
}

// Tokenizer implements a Byte-Pair Encoding (BPE) subword tokenizer
type Tokenizer struct {
	Vocab      []string         `json:"vocab"`
	TokenToID  map[string]int   `json:"token_to_id"`
	Merges     []Merge          `json:"merges"`
	MergeRanks map[string]int   `json:"-"`
	WordCache  map[string][]int `json:"-"`
}

// New creates an empty tokenizer.
func New() *Tokenizer {
	return &Tokenizer{
		Vocab:      make([]string, 0),
		TokenToID:  make(map[string]int),
		Merges:     make([]Merge, 0),
		MergeRanks: make(map[string]int),
		WordCache:  make(map[string][]int),
	}
}

// FromVocabAndMerges reconstructs a trained tokenizer from its vocabulary and merge table.
func FromVocabAndMerges(vocab []string, merges []Merge) *Tokenizer {
	tok := &Tokenizer{
		Vocab:     vocab,
		Merges:    merges,
		TokenToID: make(map[string]int, len(vocab)),
		WordCache: make(map[string][]int),
	}
	for i, s := range vocab {
		tok.TokenToID[s] = i
	}
	tok.initRanks()
	return tok
}

// initRanks builds the lookup map for fast merge ranking
func (t *Tokenizer) initRanks() {
	t.MergeRanks = make(map[string]int, len(t.Merges))
	for rank, m := range t.Merges {
		key := m.Pair1 + "\x00" + m.Pair2
		t.MergeRanks[key] = rank
	}
	if t.WordCache == nil {
		t.WordCache = make(map[string][]int)
	}
}

// VocabSize returns the total number of tokens in vocabulary
func (t *Tokenizer) VocabSize() int {
	return len(t.Vocab)
}

// Train trains a BPE tokenizer on raw text up to targetVocabSize
func (t *Tokenizer) Train(text string, targetVocabSize int) {
	if len(text) == 0 {
		return
	}

	// 1. Initial base vocabulary with special tokens
	t.Vocab = make([]string, 0, len(DefaultSpecialTokens))
	t.TokenToID = make(map[string]int)
	for _, tok := range DefaultSpecialTokens {
		t.TokenToID[tok] = len(t.Vocab)
		t.Vocab = append(t.Vocab, tok)
	}

	// Collect characters frequent enough to deserve a vocab slot, in order of first appearance
	charFreq, totalChars := countChars(text)
	minFreq := minCharFreq(totalChars)
	for _, ch := range text {
		s := string(ch)
		if _, exists := t.TokenToID[s]; !exists && charFreq[ch] >= minFreq {
			t.TokenToID[s] = len(t.Vocab)
			t.Vocab = append(t.Vocab, s)
		}
	}

	// 2. Pre-tokenize into word-like units and count their frequencies. Decoding stays exact:
	// the pre-tokens concatenate back to text.
	wordFreq := make(map[string]int)
	splitWords(text, func(w string) { wordFreq[w]++ })

	type wordFreqEntry struct {
		word string
		freq int
	}
	allWords := make([]wordFreqEntry, 0, len(wordFreq))
	for w, freq := range wordFreq {
		// Words with rare characters would teach merges around tokens that encode as <UNK>
		if hasRareChar(w, charFreq, minFreq) {
			continue
		}
		allWords = append(allWords, wordFreqEntry{word: w, freq: freq})
	}
	// Sort by frequency, breaking ties lexicographically so the tokenizer is deterministic
	slices.SortFunc(allWords, func(a, b wordFreqEntry) int {
		if a.freq != b.freq {
			return b.freq - a.freq
		}
		return strings.Compare(a.word, b.word)
	})

	// 3. Iteratively learn BPE merges
	bt := newBPETrainer()
	for _, entry := range allWords {
		bt.addWord(entry.word, entry.freq)
	}
	t.Merges = make([]Merge, 0)
	t.learnMerges(bt, targetVocabSize)

	t.initRanks()
}

// countChars returns how often each character occurs in text and the total character count.
func countChars(text string) (map[rune]int, int) {
	freq := make(map[rune]int)
	total := 0
	for _, ch := range text {
		freq[ch]++
		total++
	}
	return freq, total
}

// minCharFreq is the occurrence count a character needs in a sample of totalChars characters
// to get a vocab slot (at least 1, so small samples keep every character).
func minCharFreq(totalChars int) int {
	return max(1, int(float64(totalChars)*MinCharsPerMillion/charsPerMillion))
}

func hasRareChar(word string, charFreq map[rune]int, minFreq int) bool {
	for _, ch := range word {
		if charFreq[ch] < minFreq {
			return true
		}
	}
	return false
}

// encodeWord encodes a single word (whitespace/punctuation bounded) using learned BPE ranks
func (t *Tokenizer) encodeWord(word string) []int {
	if len(word) == 0 {
		return nil
	}
	// 1. Initial characters
	tokens := make([]string, 0, len(word))
	for _, r := range word {
		tokens = append(tokens, string(r))
	}

	// 2. Iteratively merge adjacent pairs with best rank
	for len(tokens) > 1 {
		bestRank := 1<<31 - 1
		bestIdx := -1

		for i := 0; i < len(tokens)-1; i++ {
			key := tokens[i] + "\x00" + tokens[i+1]
			if rank, ok := t.MergeRanks[key]; ok {
				if rank < bestRank {
					bestRank = rank
					bestIdx = i
				}
			}
		}

		if bestIdx == -1 {
			break
		}

		merged := tokens[bestIdx] + tokens[bestIdx+1]
		newTokens := make([]string, 0, len(tokens)-1)
		newTokens = append(newTokens, tokens[:bestIdx]...)
		newTokens = append(newTokens, merged)
		newTokens = append(newTokens, tokens[bestIdx+2:]...)
		tokens = newTokens
	}

	ids := make([]int, len(tokens))
	unkID := t.TokenToID[UnkToken]
	for i, tok := range tokens {
		if id, ok := t.TokenToID[tok]; ok {
			ids[i] = id
		} else {
			ids[i] = unkID
		}
	}
	return ids
}

// AddSpecialToken registers a special token in vocabulary if not already present.
func (t *Tokenizer) AddSpecialToken(tok string) int {
	if t.TokenToID == nil {
		t.TokenToID = make(map[string]int)
	}
	if id, exists := t.TokenToID[tok]; exists {
		return id
	}
	id := len(t.Vocab)
	t.TokenToID[tok] = id
	t.Vocab = append(t.Vocab, tok)
	return id
}

// isSpecialToken reports whether a token string is a designated special/slot token
func (t *Tokenizer) isSpecialToken(s string) bool {
	if strings.HasPrefix(s, "<") && strings.HasSuffix(s, ">") {
		_, ok := t.TokenToID[s]
		return ok
	}
	return false
}

// encodePlain encodes text into BPE tokens without checking for special slot tokens.
func (t *Tokenizer) encodePlain(text string) []int {
	if len(text) == 0 {
		return nil
	}
	ids := make([]int, 0, len(text)/3)
	splitWords(text, func(w string) {
		encoded, ok := t.WordCache[w]
		if !ok {
			encoded = t.encodeWord(w)
			t.WordCache[w] = encoded
		}
		ids = append(ids, encoded...)
	})
	return ids
}

// Encode converts input text into a sequence of token IDs,
// extracting any registered special tokens atomically so they are never fragmented into subwords.
func (t *Tokenizer) Encode(text string) []int {
	if len(text) == 0 {
		return nil
	}

	if len(t.MergeRanks) == 0 && len(t.Merges) > 0 {
		t.initRanks()
	}
	if t.WordCache == nil {
		t.WordCache = make(map[string][]int)
	}

	// Fast path: if no '<' exists in text, encode directly as plain text
	if !strings.ContainsRune(text, '<') {
		return t.encodePlain(text)
	}

	var ids []int
	remaining := text
	for len(remaining) > 0 {
		startIdx := strings.IndexRune(remaining, '<')
		if startIdx == -1 {
			ids = append(ids, t.encodePlain(remaining)...)
			break
		}

		endRel := strings.IndexRune(remaining[startIdx:], '>')
		if endRel == -1 {
			// No matching '>', encode rest as plain text
			ids = append(ids, t.encodePlain(remaining)...)
			break
		}

		endIdx := startIdx + endRel + 1
		candidate := remaining[startIdx:endIdx]
		if specID, ok := t.TokenToID[candidate]; ok && t.isSpecialToken(candidate) {
			if startIdx > 0 {
				ids = append(ids, t.encodePlain(remaining[:startIdx])...)
			}
			ids = append(ids, specID)
			remaining = remaining[endIdx:]
		} else {
			// Not a registered special token; encode up to and including '<' and continue
			ids = append(ids, t.encodePlain(remaining[:startIdx+1])...)
			remaining = remaining[startIdx+1:]
		}
	}

	return ids
}

// PadTokenID returns the token ID used to pad sequences, or 0 when the vocabulary has no pad token.
func (t *Tokenizer) PadTokenID() int {
	if t == nil {
		return 0
	}
	return t.TokenToID[PadToken]
}

// EOSTokenID returns the token ID for the end-of-sequence marker, or -1 if the vocab has none.
func (t *Tokenizer) EOSTokenID() int { return t.specialID(EOSToken) }

// UserTokenID returns the token ID for the user turn marker, or -1 if the vocab has none.
func (t *Tokenizer) UserTokenID() int { return t.specialID(UserToken) }

// BotTokenID returns the token ID for the assistant turn marker, or -1 if the vocab has none.
func (t *Tokenizer) BotTokenID() int { return t.specialID(BotToken) }

func (t *Tokenizer) specialID(tok string) int {
	if t == nil {
		return -1
	}
	if id, ok := t.TokenToID[tok]; ok {
		return id
	}
	return -1
}

// Decode converts token IDs back into the exact original string
func (t *Tokenizer) Decode(ids []int) string {
	var sb strings.Builder
	for _, id := range ids {
		if id >= 0 && id < len(t.Vocab) {
			tok := t.Vocab[id]
			// Skip special non-printing markers
			if tok == PadToken || tok == EOSToken || tok == UserToken || tok == BotToken {
				continue
			}
			if tok == UnkToken {
				sb.WriteString("\uFFFD")
				continue
			}
			sb.WriteString(tok)
		}
	}
	return sb.String()
}

// Save exports tokenizer vocabulary and merge table to JSON
func (t *Tokenizer) Save(path string) error {
	data, err := json.MarshalIndent(t, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, data, 0644)
}

// Load loads a trained tokenizer from JSON
func Load(path string) (*Tokenizer, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}

	var tok Tokenizer
	if err := json.Unmarshal(data, &tok); err != nil {
		return nil, err
	}

	tok.TokenToID = make(map[string]int, len(tok.Vocab))
	for i, s := range tok.Vocab {
		tok.TokenToID[s] = i
	}
	tok.initRanks()

	return &tok, nil
}
