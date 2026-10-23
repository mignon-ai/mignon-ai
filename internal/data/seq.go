package data

import "math/rand"

// dataSeed fixes every permutation in the data pipeline (train/validation split and sequence
// shuffle), so an encoded dataset depends only on the corpus and encoding parameters. That keeps
// the on-disk cache, which is not keyed by run seed, reproducible; per-run order comes from the
// training loop's seeded epoch shuffle.
const dataSeed = 42

// causalSeq turns up to contextSize+1 tokens into a next-token pair: x[i] = tokens[i] and
// y[i] = tokens[i+1]. Positions past the tokens hold padding inputs and masked targets.
func causalSeq(tokens []int, contextSize, padID int) (x, y []int) {
	x = make([]int, contextSize)
	y = make([]int, contextSize)
	for i := range x {
		x[i] = padID
		y[i] = MaskedTarget
	}
	n := min(len(tokens), contextSize+1)
	copy(x, tokens[:n-1])
	copy(y, tokens[1:n])
	return x, y
}

// windowSeq returns the full context window of tokens starting at pos and its next-token targets.
func windowSeq(tokens []int, pos, contextSize int) (x, y []int) {
	x = make([]int, contextSize)
	y = make([]int, contextSize)
	copy(x, tokens[pos:pos+contextSize])
	copy(y, tokens[pos+1:pos+1+contextSize])
	return x, y
}

// maskTargets excludes the first n target positions (e.g. the prompt) from the loss.
func maskTargets(y []int, n int) {
	for i := 0; i < min(n, len(y)); i++ {
		y[i] = MaskedTarget
	}
}

// Sequences is a set of aligned input and target sequences: Y[i] holds the next-token targets of
// X[i] (MaskedTarget where excluded from the loss).
type Sequences struct {
	X, Y [][]int
}

// Len returns the number of sequences.
func (s Sequences) Len() int { return len(s.X) }

// Valid reports whether every input has a target sequence.
func (s Sequences) Valid() bool { return len(s.X) == len(s.Y) }

// Cap keeps at most limit sequences (0 = all).
func (s Sequences) Cap(limit int) Sequences {
	if limit > 0 && len(s.X) > limit {
		return Sequences{s.X[:limit], s.Y[:limit]}
	}
	return s
}

// shuffleSeqs deterministically shuffles input/target pairs together in place.
func shuffleSeqs(xs, ys [][]int) {
	rng := rand.New(rand.NewSource(dataSeed))
	for i := len(xs) - 1; i > 0; i-- {
		j := rng.Intn(i + 1)
		xs[i], xs[j] = xs[j], xs[i]
		ys[i], ys[j] = ys[j], ys[i]
	}
}

// splitTrainVal deterministically shuffles items and holds out valFraction of them (at least
// one, at most MaxValSequences) for validation. Splitting examples before encoding keeps every
// sequence of an example on the same side, so validation never sees training text.
func splitTrainVal[T any](items []T, valFraction float64) (train, val []T) {
	shuffled := make([]T, len(items))
	for i, idx := range rand.New(rand.NewSource(dataSeed)).Perm(len(items)) {
		shuffled[i] = items[idx]
	}
	valCount := min(max(int(float64(len(items))*valFraction), 1), MaxValSequences)
	trainCount := len(items) - valCount
	return shuffled[:trainCount], shuffled[trainCount:]
}
