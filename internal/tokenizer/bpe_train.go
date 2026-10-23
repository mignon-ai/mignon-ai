package tokenizer

import (
	"container/heap"
	"strings"
)

// bpeWord is a distinct pre-token during BPE training: its current symbols and corpus frequency.
type bpeWord struct {
	syms []int32
	freq int
}

// pairKey packs two symbol ids into a map key.
type pairKey uint64

func makePair(a, b int32) pairKey       { return pairKey(uint64(uint32(a))<<32 | uint64(uint32(b))) }
func (p pairKey) parts() (int32, int32) { return int32(p >> 32), int32(uint32(p)) }

// bpeTrainer learns merges incrementally: it keeps every adjacent pair's frequency and the words
// containing it, so a merge only recounts the words it touches instead of the whole corpus.
type bpeTrainer struct {
	words   []bpeWord
	symbols []string         // symbol id -> text
	symID   map[string]int32 // text -> symbol id; equal strings share an id
	counts  map[pairKey]int
	where   map[pairKey][]int32 // words that contained the pair when it was counted (may be stale)
	queue   pairQueue
	seen    []int // per word: last merge step that recounted it
}

func newBPETrainer() *bpeTrainer {
	return &bpeTrainer{symID: make(map[string]int32), counts: make(map[pairKey]int), where: make(map[pairKey][]int32)}
}

func (bt *bpeTrainer) symbol(s string) int32 {
	if id, ok := bt.symID[s]; ok {
		return id
	}
	id := int32(len(bt.symbols))
	bt.symbols = append(bt.symbols, s)
	bt.symID[s] = id
	return id
}

// addWord registers a pre-token, split into characters, with its frequency.
func (bt *bpeTrainer) addWord(w string, freq int) {
	syms := make([]int32, 0, len(w))
	for _, r := range w {
		syms = append(syms, bt.symbol(string(r)))
	}
	bt.words = append(bt.words, bpeWord{syms: syms, freq: freq})
}

// countWord adds sign*freq for every adjacent pair of word wi, recording changed pairs.
func (bt *bpeTrainer) countWord(wi int32, sign int, changed map[pairKey]struct{}) {
	w := bt.words[wi]
	for i := 0; i+1 < len(w.syms); i++ {
		p := makePair(w.syms[i], w.syms[i+1])
		bt.counts[p] += sign * w.freq
		if sign > 0 {
			bt.where[p] = append(bt.where[p], wi)
		}
		changed[p] = struct{}{}
	}
}

// pairText is the tie-break order of a pair, matching the original "p1\x00p2" string keys.
func (bt *bpeTrainer) pairText(p pairKey) string {
	a, b := p.parts()
	return bt.symbols[a] + "\x00" + bt.symbols[b]
}

// best pops the most frequent pair (ties broken by pairText), skipping stale queue entries.
func (bt *bpeTrainer) best() (pairKey, int, bool) {
	for bt.queue.Len() > 0 {
		e := heap.Pop(&bt.queue).(pairEntry)
		if bt.counts[e.pair] == e.count && e.count > 0 {
			return e.pair, e.count, true
		}
	}
	return 0, 0, false
}

func (bt *bpeTrainer) push(changed map[pairKey]struct{}) {
	for p := range changed {
		if c := bt.counts[p]; c > 0 {
			heap.Push(&bt.queue, pairEntry{pair: p, count: c, text: bt.pairText(p)})
		} else {
			delete(bt.counts, p)
		}
	}
}

// merge replaces every non-overlapping left-to-right occurrence of p in the words containing it
// and returns the merged symbol.
func (bt *bpeTrainer) merge(p pairKey, step int) int32 {
	a, b := p.parts()
	merged := bt.symbol(bt.symbols[a] + bt.symbols[b])
	changed := make(map[pairKey]struct{})
	for _, wi := range bt.where[p] {
		if bt.seen[wi] == step {
			continue
		}
		bt.seen[wi] = step
		syms := bt.words[wi].syms
		if !containsPair(syms, a, b) {
			continue
		}
		bt.countWord(wi, -1, changed)
		out := syms[:0]
		for i := 0; i < len(syms); i++ {
			if i+1 < len(syms) && syms[i] == a && syms[i+1] == b {
				out = append(out, merged)
				i++
			} else {
				out = append(out, syms[i])
			}
		}
		bt.words[wi].syms = out
		bt.countWord(wi, +1, changed)
	}
	delete(bt.where, p)
	bt.push(changed)
	return merged
}

func containsPair(syms []int32, a, b int32) bool {
	for i := 0; i+1 < len(syms); i++ {
		if syms[i] == a && syms[i+1] == b {
			return true
		}
	}
	return false
}

// learnMerges merges the most frequent adjacent pair until the vocabulary holds targetVocabSize
// tokens or no pair occurs more than once, appending each merge to t.Merges and new tokens to
// t.Vocab.
func (t *Tokenizer) learnMerges(bt *bpeTrainer, targetVocabSize int) {
	bt.seen = make([]int, len(bt.words))
	for i := range bt.seen {
		bt.seen[i] = -1
	}
	changed := make(map[pairKey]struct{})
	for wi := range bt.words {
		bt.countWord(int32(wi), +1, changed)
	}
	bt.push(changed)
	for step := 0; len(t.Vocab) < targetVocabSize; step++ {
		p, count, ok := bt.best()
		if !ok || count <= 1 {
			break
		}
		a, b := p.parts()
		t.Merges = append(t.Merges, Merge{Pair1: bt.symbols[a], Pair2: bt.symbols[b]})
		merged := bt.symbols[bt.merge(p, step)]
		if _, exists := t.TokenToID[merged]; !exists {
			t.TokenToID[merged] = len(t.Vocab)
			t.Vocab = append(t.Vocab, merged)
		}
	}
}

// pairEntry is a queued pair frequency; entries go stale when the count changes after pushing.
type pairEntry struct {
	pair  pairKey
	count int
	text  string
}

// pairQueue is a max-heap by count, then by ascending pair text.
type pairQueue []pairEntry

func (q pairQueue) Len() int { return len(q) }
func (q pairQueue) Less(i, j int) bool {
	if q[i].count != q[j].count {
		return q[i].count > q[j].count
	}
	return strings.Compare(q[i].text, q[j].text) < 0
}
func (q pairQueue) Swap(i, j int) { q[i], q[j] = q[j], q[i] }
func (q *pairQueue) Push(x any)   { *q = append(*q, x.(pairEntry)) }
func (q *pairQueue) Pop() any {
	old := *q
	e := old[len(old)-1]
	*q = old[:len(old)-1]
	return e
}
