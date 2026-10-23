package tokenizer

import (
	"math/rand"
	"slices"
	"strings"
	"testing"
)

// naiveMerges is the original BPE loop, which recounts every pair of every word for each merge.
// The incremental trainer must reproduce it merge for merge.
func naiveMerges(words map[string]int, vocab []string, targetVocabSize int) []Merge {
	type entry struct {
		toks []string
		freq int
	}
	var ws []entry
	for w, f := range words {
		var toks []string
		for _, r := range w {
			toks = append(toks, string(r))
		}
		ws = append(ws, entry{toks, f})
	}
	have := make(map[string]bool)
	for _, v := range vocab {
		have[v] = true
	}
	var merges []Merge
	for len(vocab) < targetVocabSize {
		pairFreq := make(map[string]int)
		for _, w := range ws {
			for i := 0; i+1 < len(w.toks); i++ {
				pairFreq[w.toks[i]+"\x00"+w.toks[i+1]] += w.freq
			}
		}
		bestKey, bestFreq := "", 0
		for k, f := range pairFreq {
			if f > bestFreq || (f == bestFreq && k < bestKey) {
				bestKey, bestFreq = k, f
			}
		}
		if bestFreq <= 1 {
			break
		}
		parts := strings.Split(bestKey, "\x00")
		p1, p2 := parts[0], parts[1]
		merges = append(merges, Merge{p1, p2})
		if !have[p1+p2] {
			have[p1+p2] = true
			vocab = append(vocab, p1+p2)
		}
		for i := range ws {
			var out []string
			toks := ws[i].toks
			for j := 0; j < len(toks); j++ {
				if j+1 < len(toks) && toks[j] == p1 && toks[j+1] == p2 {
					out = append(out, p1+p2)
					j++
				} else {
					out = append(out, toks[j])
				}
			}
			ws[i].toks = out
		}
	}
	return merges
}

// randomCorpus builds text from a small alphabet so merges overlap, repeat and tie often.
func randomCorpus(rng *rand.Rand, words int) string {
	const alphabet = "aab c1/-"
	var sb strings.Builder
	for i := 0; i < words; i++ {
		n := 1 + rng.Intn(9)
		for j := 0; j < n; j++ {
			sb.WriteByte(alphabet[rng.Intn(len(alphabet))])
		}
		sb.WriteByte(" \n"[rng.Intn(2)])
	}
	return sb.String()
}

func TestIncrementalBPEMatchesNaive(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	const targetVocab = 120
	for round := 0; round < 20; round++ {
		text := randomCorpus(rng, 400)
		tok := New()
		tok.Train(text, targetVocab)

		words := make(map[string]int)
		splitWords(text, func(w string) { words[w]++ })
		var base []string
		for _, v := range tok.Vocab {
			if len([]rune(v)) == 1 || slices.Contains(DefaultSpecialTokens, v) {
				base = append(base, v)
			}
		}
		want := naiveMerges(words, base, targetVocab)
		if !slices.Equal(tok.Merges, want) {
			t.Fatalf("round %d: incremental merges differ from naive BPE\n got %v\nwant %v", round, tok.Merges, want)
		}
	}
}

func TestSplitsDigitsAndLearnsUnspacedWords(t *testing.T) {
	text := strings.Repeat("src/page.go 2023\n page 1999\n", 30)
	tok := New()
	tok.Train(text, 80)
	for _, id := range tok.Encode("2023") {
		if len(tok.Vocab[id]) != 1 {
			t.Fatalf("numbers must encode digit by digit, got token %q", tok.Vocab[id])
		}
	}
	if got := tok.Encode("page"); len(got) != 1 {
		t.Fatalf("\"page\" after a slash should be one token, got %d", len(got))
	}
}

func TestCheckFormat(t *testing.T) {
	if err := CheckFormat(FormatVersion); err != nil {
		t.Fatal(err)
	}
	for _, v := range []int{0, 1, FormatVersion + 1} {
		if CheckFormat(v) == nil {
			t.Fatalf("format version %d must be rejected", v)
		}
	}
}
