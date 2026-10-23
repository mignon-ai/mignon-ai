package constrain

import (
	"slices"
	"testing"

	"github.com/mignon-ai/mignon-ai/internal/tokenizer"
)

func testGrammar(t *testing.T) (*Grammar, *tokenizer.Tokenizer) {
	t.Helper()
	tok := tokenizer.New()
	tok.Train("git commit -m msg git push origin main git merge --squash x", 300)
	tok.AddSpecialToken(tokenizer.EOSToken)
	replies := []string{
		"git commit -m 'one'", "git commit -a -m 'two'", "git push origin main", "git push --force origin main",
		"git merge --squash topic", "git push origin dev", "git push origin x", "git push origin y", "git push origin z",
	}
	for range 5 {
		replies = append(replies, "git push HEAD~1")
	}
	requests := make([]string, len(replies))
	for i := range requests {
		requests[i] = "do it"
	}
	return NewGrammar(requests, replies, tok), tok
}

func allowedWord(t *testing.T, s *State, text string) bool {
	t.Helper()
	return s.tokenOK(text)
}

func TestLexicon(t *testing.T) {
	g, _ := testGrammar(t)
	if !slices.Equal(g.first, []string{"git"}) || !slices.Equal(g.second["git"], []string{"commit", "merge", "push"}) {
		t.Fatalf("first=%v second=%v", g.first, g.second)
	}
	if !slices.Contains(g.opts["git push"], "--force") || slices.Contains(g.opts["git push"], "--squash") {
		t.Fatalf("push options %v", g.opts["git push"])
	}
	if !slices.Contains(g.common, "origin") {
		t.Fatalf("common %v", g.common)
	}
}

func TestStartIgnoresNonCommands(t *testing.T) {
	g, _ := testGrammar(t)
	if g.Start("what is a branch", "The") != nil {
		t.Fatal("prefix outside the lexicon must not be constrained")
	}
	if g.Start("push it", "git push") == nil {
		t.Fatal("git prefix must be constrained")
	}
}

func TestPositions(t *testing.T) {
	g, _ := testGrammar(t)
	s := g.Start("push my-feature to origin", "git")
	if !allowedWord(t, s, " push") || allowedWord(t, s, " banana") {
		t.Fatal("subcommand must come from the lexicon")
	}
	if allowedWord(t, s, " push@x") || allowedWord(t, s, " push/origin") {
		t.Fatal("a subcommand must be a whole word")
	}
	s = g.Start("push my-feature to origin", "git push")
	for text, want := range map[string]bool{
		" --force": true, " --squash": false, " my-feature": true, " origin": true,
		" my-feat": true, " other": false, " --forc": true, " &&": true,
	} {
		if got := allowedWord(t, s, text); got != want {
			t.Errorf("%q after git push: %v, want %v", text, got, want)
		}
	}
	// A request word with punctuation, a number substituted by its pattern and a compound word.
	s = g.Start("reset to HEAD~3 on origin/dev", "git push")
	for text, want := range map[string]bool{" HEAD~2": true, " dev~3": false, " origin/dev": true, " origin/zzz": false} {
		if got := allowedWord(t, s, text); got != want {
			t.Errorf("%q: %v, want %v", text, got, want)
		}
	}
}

func TestMaskKeepsEOSForCompleteCommands(t *testing.T) {
	g, tok := testGrammar(t)
	s := g.Start("push", "git")
	logits := make([]float64, tok.VocabSize())
	s.Mask(logits)
	if logits[tok.EOSTokenID()] != negInf {
		t.Fatal("EOS must be blocked after only one word")
	}
	s = g.Start("push", "git push origin")
	logits = make([]float64, tok.VocabSize())
	s.Mask(logits)
	if logits[tok.EOSTokenID()] == negInf {
		t.Fatal("EOS must be allowed after a complete command")
	}
}
