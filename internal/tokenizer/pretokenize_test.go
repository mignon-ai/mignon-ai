package tokenizer

import (
	"slices"
	"strings"
	"testing"
)

func collectWords(text string) []string {
	var words []string
	splitWords(text, func(w string) { words = append(words, w) })
	return words
}

func TestSplitWordsGivesSpaceToNextWord(t *testing.T) {
	cases := map[string][]string{
		"git pull --rebase origin":  {"git", " pull", " --", "rebase", " origin"},
		"commit -m 'fix: bug'\ngit": {"commit", " -", "m", " '", "fix", ":", " bug", "'", "\n", "git"},
		"two  spaces\n\n tab\tend.": {"two", " ", " spaces", "\n\n", " tab", "\t", "end", "."},
		" leading":                  {" leading"},
		"trailing ":                 {"trailing", " "},
		"HEAD~1 && echo done":       {"HEAD", "~", "1", " &&", " echo", " done"},
	}
	for text, want := range cases {
		got := collectWords(text)
		if !slices.Equal(got, want) {
			t.Errorf("splitWords(%q) = %q, want %q", text, got, want)
		}
		if strings.Join(got, "") != text {
			t.Errorf("splitWords(%q) does not concatenate back to its input", text)
		}
	}
}

func TestSplitWordsCapsLongRuns(t *testing.T) {
	text := strings.Repeat("a", 3*maxWordBytes+5)
	words := collectWords(text)
	for _, w := range words {
		if len(w) > maxWordBytes {
			t.Fatalf("pre-token of %d bytes exceeds the %d byte cap", len(w), maxWordBytes)
		}
	}
	if strings.Join(words, "") != text {
		t.Fatalf("capped pre-tokens do not concatenate back to the input")
	}
}

// A prompt that ends on a word must encode the same alone as at the start of the full text, so
// the model continues it exactly as it would in training.
func TestPromptEndingOnWordIsTokenBoundary(t *testing.T) {
	tok := New()
	tok.Train(strings.Repeat("stage file package.yaml\ngit add package.yaml\ngit pull --rebase origin main\n", 50), 120)
	for _, c := range []struct{ prompt, answer string }{
		{"stage file package.yaml\ngit", " add package.yaml"},
		{"git pull --rebase", " origin main"},
		{"git pull", " --rebase"},
	} {
		joint := tok.Encode(c.prompt + c.answer)
		split := append(tok.Encode(c.prompt), tok.Encode(c.answer)...)
		if !slices.Equal(joint, split) {
			t.Errorf("%q + %q: joint encoding %v differs from prompt then answer %v", c.prompt, c.answer, joint, split)
		}
	}
}

func TestChatTemplate(t *testing.T) {
	tok := New()
	tok.Train("push to main git push origin main", 40)
	prompt := tok.EncodePrompt("push to main")
	if prompt[0] != tok.UserTokenID() || prompt[len(prompt)-1] != tok.BotTokenID() {
		t.Errorf("prompt must be wrapped in <USER> ... <BOT>, got %v", prompt)
	}
	reply := tok.EncodeReply("git push origin main")
	if reply[len(reply)-1] != tok.EOSTokenID() {
		t.Errorf("reply must end with <EOS>, got %v", reply)
	}
	if got := tok.Decode(append(prompt, reply...)); got != "push to maingit push origin main" {
		t.Errorf("markers must not decode to text, got %q", got)
	}
}

func TestDropOldestTurns(t *testing.T) {
	tok := New()
	tok.Train("hi hello bye", 30)
	turn := func(u, b string) []int { return append(tok.EncodePrompt(u), tok.EncodeReply(b)...) }
	first, second := turn("hi", "hello"), turn("bye", "bye")
	history := append(slices.Clone(first), second...)

	if got := tok.DropOldestTurns(history, len(history)); !slices.Equal(got, history) {
		t.Errorf("history that fits must be kept, got %v", got)
	}
	if got := tok.DropOldestTurns(history, len(second)); !slices.Equal(got, second) {
		t.Errorf("only the oldest turn should be dropped, got %v", got)
	}
	if got := tok.DropOldestTurns(history, 0); len(got) != 0 {
		t.Errorf("no room must drop all history, got %v", got)
	}
}
