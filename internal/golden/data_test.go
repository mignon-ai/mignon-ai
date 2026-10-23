package golden

// Data-pipeline part of the golden behavior snapshot (see golden_test.go). Encoders that shuffle
// with the global RNG are fingerprinted order-independently: the sorted multiset of (x, y) pairs.

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"testing"

	"github.com/mignon-ai/mignon-ai/internal/data"
)

const (
	goldenDataContext  = 16
	goldenDataVocab    = 120
	goldenDataStride   = 8
	goldenDataValFrac  = 0.2
	goldenDataMaxSeqs  = 40
	goldenDataManyDocs = 6  // >= MinDocsForDocSplit, so documents are split whole
	goldenDataExamples = 24 // >= MinValSamples, so pair sets get a validation split
)

// goldenCorpus mixes block-structured docs (Q&A style, "\n\n"-separated), long prose and short docs.
func goldenCorpus(n int) []string {
	docs := make([]string, 0, n)
	for i := 0; i < n; i++ {
		switch i % 3 {
		case 0:
			var blocks []string
			for b := 0; b < 5+i; b++ {
				blocks = append(blocks, fmt.Sprintf("Q: how do I run step %d?\nA: use git commit -m \"step %d\" then push.", b, b*i))
			}
			// A repeated block exercises the train/val dedup.
			blocks = append(blocks, "Q: how do I run step 0?\nA: use git commit -m \"step 0\" then push.")
			docs = append(docs, strings.Join(blocks, "\n\n"))
		case 1:
			docs = append(docs, strings.Repeat(fmt.Sprintf("The cellular automaton number %d updates every cell from its neighbours, ", i), 6))
		default:
			docs = append(docs, fmt.Sprintf("Short doc %d.", i))
		}
	}
	return docs
}

func fingerprintDataPipeline(t *testing.T) map[string]string {
	t.Helper()
	out := map[string]string{}
	few, many := goldenCorpus(2), goldenCorpus(goldenDataManyDocs)

	for name, docs := range map[string][]string{"few": few, "many": many} {
		trainDocs, valDocs := data.SplitForVal(docs, goldenDataValFrac)
		out["split:"+name] = hashJSON(t, [][]string{trainDocs, valDocs})

		trainSet, valSet, tok := data.PrepareTrainVal(docs, goldenDataContext, goldenDataVocab,
			goldenDataStride, goldenDataMaxSeqs, goldenDataValFrac)
		out["prepare:"+name+":tokenizer"] = fingerprintTokenizer(tok).Vocab
		out["prepare:"+name+":train"] = hashSeqSet(trainSet)
		out["prepare:"+name+":val"] = hashSeqSet(valSet)

		all, _, _ := data.PrepareTrainVal(docs, goldenDataContext, goldenDataVocab, 0, 0, 0)
		out["multifile:"+name] = hashSeqSet(all)
	}
	fp := fingerprintTokenizer(goldenTokenizer())
	out["tokenizer"] = fmt.Sprintf("%s|%s|roundtrip=%v", fp.Vocab, fp.Encoded, fp.Roundtrip)
	out["sample"] = hashBytes([]byte(data.SampleForTokenizer(many)))
	out["cachekey"] = data.CacheKey(many, goldenDataContext, goldenDataVocab, goldenDataStride, goldenDataMaxSeqs, goldenDataValFrac)

	// On-disk cache: first call encodes and stores, second call must return the identical set.
	t.Setenv("XDG_CACHE_HOME", t.TempDir())
	for _, pass := range []string{"miss", "hit"} {
		trainSet, valSet, tok, cached := data.PrepareTrainValCached(many, goldenDataContext, goldenDataVocab,
			goldenDataStride, goldenDataMaxSeqs, goldenDataValFrac)
		out["cache:"+pass] = fmt.Sprintf("cached=%v|%s|%s|%s", cached, fingerprintTokenizer(tok).Vocab,
			hashSeqSet(trainSet), hashSeqSet(valSet))
	}

	dir := t.TempDir()
	for i, doc := range few {
		writeGoldenFile(t, filepath.Join(dir, fmt.Sprintf("doc%d.txt", i)), doc)
	}
	writeGoldenFile(t, filepath.Join(dir, "empty.txt"), "")
	loaded, err := data.LoadTextCorpus(dir, nil)
	if err != nil {
		t.Fatal(err)
	}
	out["corpus"] = hashJSON(t, loaded)

	tok := goldenTokenizer()
	qa := goldenQAPairs()
	x, y := data.EncodePairs(qa, goldenDataContext, tok)
	out["pairs:encode"] = hashJSON(t, [][][]int{x, y})

	pairsPath := filepath.Join(dir, "pairs.json")
	writeGoldenFile(t, pairsPath, string(mustJSON(t, append(qa, goldenInstructionPairs()...))))
	pairs, err := data.LoadPairs(pairsPath)
	if err != nil {
		t.Fatal(err)
	}
	out["pairs:load"] = hashJSON(t, pairs)
	trainSet, valSet := data.EncodePairsTrainVal(pairs, goldenDataContext, tok, goldenDataValFrac)
	out["pairs:train"] = hashSeqSet(trainSet)
	out["pairs:val"] = hashSeqSet(valSet)
	out["pairs:tokenizer"] = fingerprintTokenizer(data.TrainPairTokenizer(pairs, goldenDataVocab)).Vocab
	return out
}

func goldenQAPairs() []data.QAPair {
	var pairs []data.QAPair
	for i := 0; i < goldenDataExamples; i++ {
		p := data.QAPair{Prompt: fmt.Sprintf("what does command %d do?", i), Response: fmt.Sprintf("it prints %d lines", i*3)}
		switch i % 4 {
		case 1:
			p = data.QAPair{Instruction: "explain git push", Input: fmt.Sprintf("branch %d", i), Output: "it uploads commits"}
		case 2:
			p.History = []data.ChatTurn{{User: "hi", Bot: "hello"}, {User: "who are you", Bot: "a cellular automaton"}}
		case 3:
			p.Response = strings.Repeat("very long answer ", 10) // exercises response truncation
		}
		pairs = append(pairs, p)
	}
	return append(pairs, data.QAPair{Prompt: "  ", Response: "skipped: empty prompt"})
}

func goldenInstructionPairs() []data.QAPair {
	var ex []data.QAPair
	for i := 0; i < goldenDataExamples; i++ {
		e := data.QAPair{Instruction: fmt.Sprintf("list files in dir %d", i), Output: fmt.Sprintf("ls dir%d", i)}
		switch i % 3 {
		case 1:
			e = data.QAPair{Prompt: "show the log", Input: fmt.Sprintf("last %d commits", i), Response: "git log -n " + fmt.Sprint(i)}
		case 2:
			e.Instruction = strings.Repeat("a very long instruction ", 8) // exercises prompt truncation
		}
		ex = append(ex, e)
	}
	return append(ex, data.QAPair{Instruction: "dropped: no output"})
}

// hashSeqSet fingerprints (x, y) sequence pairs independent of their order.
func hashSeqSet(set data.Sequences) string {
	pairs := make([]string, set.Len())
	for i := range set.X {
		pairs[i] = fmt.Sprint(set.X[i], set.Y[i])
	}
	sort.Strings(pairs)
	return fmt.Sprintf("n=%d:%s", len(pairs), hashBytes([]byte(strings.Join(pairs, "\n"))))
}

func mustJSON(t *testing.T, v any) []byte {
	t.Helper()
	b, err := json.Marshal(v)
	if err != nil {
		t.Fatal(err)
	}
	return b
}

func hashJSON(t *testing.T, v any) string {
	t.Helper()
	return hashBytes(mustJSON(t, v))
}

func writeGoldenFile(t *testing.T, path, content string) {
	t.Helper()
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}
}
