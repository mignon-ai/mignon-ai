package data

import (
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/mignon-ai/mignon-ai/internal/tokenizer"
)

func pairsTestTokenizer() *tokenizer.Tokenizer {
	tok := tokenizer.New()
	tok.Train("tell me a story boy named tim who is tim is a boy delete local branch git branch -d fix-auth", 80)
	return tok
}

// unmasked returns the target tokens the loss is computed on.
func unmasked(y []int) []int {
	var out []int
	for _, id := range y {
		if id != MaskedTarget {
			out = append(out, id)
		}
	}
	return out
}

func TestEncodePairsUsesChatTemplateAndMasksPrompt(t *testing.T) {
	tok := pairsTestTokenizer()
	pair := QAPair{Instruction: "delete local branch fix-auth", Output: "git branch -d fix-auth"}
	X, Y := EncodePairs([]QAPair{pair}, 64, tok)
	if len(X) != 1 {
		t.Fatalf("expected 1 sequence, got %d", len(X))
	}

	turn := tok.EncodePrompt(pair.GetPrompt())
	if !slices.Equal(X[0][:len(turn)], turn) {
		t.Errorf("sequence must start with the template prompt %v, got %v", turn, X[0][:len(turn)])
	}
	if got, want := unmasked(Y[0]), tok.EncodeReply(pair.GetResponse()); !slices.Equal(got, want) {
		t.Errorf("targets must be exactly the reply and <EOS>: got %v, want %v", got, want)
	}
}

func TestEncodePairsMasksHistory(t *testing.T) {
	tok := pairsTestTokenizer()
	pair := QAPair{
		Prompt:   "who is tim",
		Response: "tim is a boy",
		History:  []ChatTurn{{User: "tell me a story", Bot: "boy named tim"}},
	}
	X, Y := EncodePairs([]QAPair{pair}, 64, tok)
	if len(X) != 1 {
		t.Fatalf("expected 1 sequence, got %d", len(X))
	}
	history := append(tok.EncodePrompt("tell me a story"), tok.EncodeReply("boy named tim")...)
	if !slices.Equal(X[0][:len(history)], history) {
		t.Errorf("history turn missing from the start of the sequence")
	}
	if got, want := unmasked(Y[0]), tok.EncodeReply("tim is a boy"); !slices.Equal(got, want) {
		t.Errorf("only the last reply may be trained on: got %v, want %v", got, want)
	}
}

func TestEncodePairsTruncatesToContext(t *testing.T) {
	tok := pairsTestTokenizer()
	pair := QAPair{
		Prompt:   strings.Repeat("tell me a story ", 40),
		Response: strings.Repeat("tim is a boy ", 40),
		History:  []ChatTurn{{User: "who is tim", Bot: "a boy"}},
	}
	const contextSize = 32
	X, Y := EncodePairs([]QAPair{pair}, contextSize, tok)
	if len(X) != 1 || len(X[0]) != contextSize || len(Y[0]) != contextSize {
		t.Fatalf("expected one sequence of %d tokens", contextSize)
	}
	if X[0][0] != tok.UserTokenID() {
		t.Errorf("a truncated prompt must keep its <USER> marker, got %d", X[0][0])
	}
	if !slices.Contains(X[0], tok.BotTokenID()) || len(unmasked(Y[0])) == 0 {
		t.Errorf("a truncated pair must keep <BOT> and at least one reply target")
	}
}

func TestLoadPairsAcceptsBothFieldStylesAndDropsIncomplete(t *testing.T) {
	path := filepath.Join(t.TempDir(), "pairs.json")
	json := `[
		{"instruction": "undo commit", "output": "git reset HEAD~1"},
		{"prompt": "stash files", "response": "git stash"},
		{"instruction": "explain", "input": "git push", "output": "uploads commits"},
		{"instruction": "no output"}
	]`
	if err := os.WriteFile(path, []byte(json), 0644); err != nil {
		t.Fatal(err)
	}
	pairs, err := LoadPairs(path)
	if err != nil {
		t.Fatal(err)
	}
	want := [][2]string{{"undo commit", "git reset HEAD~1"}, {"stash files", "git stash"}, {"explain\ngit push", "uploads commits"}}
	if len(pairs) != len(want) {
		t.Fatalf("expected %d pairs, got %d", len(want), len(pairs))
	}
	for i, w := range want {
		if pairs[i].GetPrompt() != w[0] || pairs[i].GetResponse() != w[1] {
			t.Errorf("pair %d: got (%q, %q), want %q", i, pairs[i].GetPrompt(), pairs[i].GetResponse(), w)
		}
	}
}
