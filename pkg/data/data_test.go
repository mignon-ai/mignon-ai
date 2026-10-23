package data_test

import (
	"strings"
	"testing"

	"github.com/mignon-ai/mignon-ai/pkg/data"
)

func TestReadPairsAcceptsBothSpellingsAndDropsIncomplete(t *testing.T) {
	in := `[{"prompt":"a","response":"b"},
	        {"instruction":"c","input":"d","output":"e"},
	        {"prompt":"only a prompt"},
	        {"prompt":"x","response":"y","history":[{"user":"u","bot":"v"}]}]`
	pairs, err := data.ReadPairs(strings.NewReader(in))
	if err != nil {
		t.Fatal(err)
	}
	if len(pairs) != 3 || pairs[0].Prompt != "a" || pairs[0].Response != "b" || pairs[1].Prompt != "c\nd" || pairs[1].Response != "e" {
		t.Fatalf("%+v", pairs)
	}
	if h := pairs[2].History; len(h) != 1 || h[0].User != "u" || h[0].Bot != "v" {
		t.Fatalf("history %+v", h)
	}
}

func TestReadPairsRejectsEmptyAndMalformed(t *testing.T) {
	for _, in := range []string{`[]`, `[{"prompt":"a"}]`, `{"not":"a list"}`, `nope`} {
		if _, err := data.ReadPairs(strings.NewReader(in)); err == nil {
			t.Errorf("accepted %q", in)
		}
	}
}
