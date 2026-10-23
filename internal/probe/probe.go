// Package probe provides synthetic capability probes: small models trained on generated token
// tasks that each isolate one ability (memory, long-range transport, in-context copying, local
// modelling). They train in seconds and give large, interpretable accuracy differences between
// architectures, which makes them a fast screen before confirming finalists on real text.
package probe

import (
	"fmt"
	"math/rand"
	"sort"
	"strings"
	"time"

	"github.com/mignon-ai/mignon-ai/internal/ca"
	"github.com/mignon-ai/mignon-ai/internal/data"
	"github.com/mignon-ai/mignon-ai/internal/train"
)

// Probe dataset sizes and shared sequence length.
const (
	SeqLen          = 64
	TrainSeqs       = 4000
	ValSeqs         = 200
	TestSeqs        = 500
	probeFirstToken = 1 // token 0 is reserved as an unused padding id in every probe vocabulary
)

// Associative recall: key/value pairs followed by queries (with replacement) for previously seen
// keys, filling the rest of the sequence. Eight pairs keeps the task inside the range that small
// models can partially solve, so it discriminates between architectures instead of sitting at chance.
const (
	assocKeys    = 32
	assocValues  = 32
	assocPairs   = 8
	assocQueries = SeqLen/2 - assocPairs
)

// Far recall: key/value pairs, a filler gap, then one query per key. The gap is longer than the
// reach of the presets' attention (two blocks of window 64 see ~126 tokens back) plus the CA
// dilation, so a binding can only cross it in the recurrent latch. Each key is queried once, so
// answers cannot be copied from earlier queries.
const (
	farRecallGap    = 192
	farRecallFiller = 16
	farRecallSeqLen = 4*assocPairs + farRecallGap
)

// Long-range copy: content tokens at the start must be reproduced at distinct recall markers late in the sequence.
const (
	copyContentTokens = 16
	copyFillerTokens  = 16
	copySlots         = 4
)

// Induction: an earlier span re-appears at a random later offset, so continuing it requires
// matching content rather than a fixed positional lag.
const (
	inductionVocab   = 32
	inductionMinSpan = 8
	inductionMaxSpan = 16
)

// Second-order Markov chain: each (prev2, prev1) context has one preferred successor.
const (
	markovVocab         = 16
	markovPreferredProb = 0.8
	markovTableSeed     = 20261003 // fixed so every run sees the same "language"
)

// Task is a generated sequence-prediction task with targets only where the answer is determined.
type Task struct {
	Name        string
	Description string
	VocabSize   int
	Chance      float64 // accuracy achievable without the ability the probe targets
	Ceiling     float64 // best achievable accuracy (below 1 for stochastic tasks)
	SeqLen      int     // length of every generated sequence
	generate    func(rng *rand.Rand) (x, y []int)
}

// Result is the outcome of training and evaluating one configuration on one probe.
type Result struct {
	Task         string
	Accuracy     float64
	Chance       float64
	Ceiling      float64
	ValLoss      float64
	TokensPerSec float64
	Duration     time.Duration
}

// maskedTargets returns a target slice of length n filled with MaskedTarget.
func maskedTargets(n int) []int {
	y := make([]int, n)
	for i := range y {
		y[i] = data.MaskedTarget
	}
	return y
}

// Tasks returns all probe tasks, keyed by name.
func Tasks() map[string]Task {
	tasks := []Task{assocRecallTask(), farRecallTask(), longCopyTask(), inductionTask(), markovTask()}
	m := make(map[string]Task, len(tasks))
	for _, t := range tasks {
		m[t.Name] = t
	}
	return m
}

// TaskNames returns the sorted names of all probe tasks.
func TaskNames() []string {
	var names []string
	for name := range Tasks() {
		names = append(names, name)
	}
	sort.Strings(names)
	return names
}

func assocRecallTask() Task {
	keyBase := probeFirstToken
	valBase := keyBase + assocKeys
	return Task{
		Name:        "assoc_recall",
		Description: "content-addressable memory: recall the value bound to a key seen earlier",
		VocabSize:   valBase + assocValues,
		// Guessing among the values present in the sequence needs no key binding, so that (not
		// uniform guessing over all values) is the floor that real associative recall must beat.
		Chance:  1.0 / assocPairs,
		Ceiling: 1.0,
		SeqLen:  SeqLen,
		generate: func(rng *rand.Rand) ([]int, []int) {
			n := 2 * (assocPairs + assocQueries)
			x, y := make([]int, 0, n), maskedTargets(n)
			keys := rng.Perm(assocKeys)[:assocPairs]
			values := make([]int, assocPairs)
			for i, k := range keys {
				values[i] = valBase + rng.Intn(assocValues)
				x = append(x, keyBase+k, values[i])
			}
			for i := 0; i < assocQueries; i++ {
				q := rng.Intn(assocPairs)
				y[len(x)] = values[q]
				x = append(x, keyBase+keys[q], values[q])
			}
			return x, y
		},
	}
}

func farRecallTask() Task {
	keyBase := probeFirstToken
	valBase := keyBase + assocKeys
	fillerBase := valBase + assocValues
	return Task{
		Name:        "far_recall",
		Description: "beyond-window memory: recall the value bound to a key seen before a long filler gap",
		VocabSize:   fillerBase + farRecallFiller,
		// As in assoc_recall, a bag of the values seen (without their keys) is the floor.
		Chance:  1.0 / assocPairs,
		Ceiling: 1.0,
		SeqLen:  farRecallSeqLen,
		generate: func(rng *rand.Rand) ([]int, []int) {
			x, y := make([]int, 0, farRecallSeqLen), maskedTargets(farRecallSeqLen)
			keys := rng.Perm(assocKeys)[:assocPairs]
			values := make([]int, assocPairs)
			for i, k := range keys {
				values[i] = valBase + rng.Intn(assocValues)
				x = append(x, keyBase+k, values[i])
			}
			for range farRecallGap {
				x = append(x, fillerBase+rng.Intn(farRecallFiller))
			}
			for _, q := range rng.Perm(assocPairs) {
				y[len(x)] = values[q]
				x = append(x, keyBase+keys[q], values[q])
			}
			return x, y
		},
	}
}

func longCopyTask() Task {
	contentBase := probeFirstToken
	fillerBase := contentBase + copyContentTokens
	recallBase := fillerBase + copyFillerTokens
	return Task{
		Name:        "long_copy",
		Description: "long-range transport: reproduce opening tokens after a long filler gap",
		VocabSize:   recallBase + copySlots,
		Chance:      1.0 / copyContentTokens,
		Ceiling:     1.0,
		SeqLen:      SeqLen,
		generate: func(rng *rand.Rand) ([]int, []int) {
			x, y := make([]int, SeqLen), maskedTargets(SeqLen)
			for i := range x {
				x[i] = fillerBase + rng.Intn(copyFillerTokens)
			}
			content := make([]int, copySlots)
			for i := range content {
				content[i] = contentBase + rng.Intn(copyContentTokens)
				x[i] = content[i]
			}
			// Recall block starts somewhere in the second half so every gap is long.
			start := SeqLen/2 + rng.Intn(SeqLen/2-copySlots+1)
			for i := 0; i < copySlots; i++ {
				x[start+i] = recallBase + i
				y[start+i] = content[i]
			}
			return x, y
		},
	}
}

func inductionTask() Task {
	return Task{
		Name:        "induction",
		Description: "in-context copying: continue an earlier span that starts repeating at a random offset",
		VocabSize:   probeFirstToken + inductionVocab,
		Chance:      1.0 / inductionVocab,
		Ceiling:     1.0,
		SeqLen:      SeqLen,
		generate: func(rng *rand.Rand) ([]int, []int) {
			x, y := make([]int, SeqLen), maskedTargets(SeqLen)
			for i := range x {
				x[i] = probeFirstToken + rng.Intn(inductionVocab)
			}
			span := inductionMinSpan + rng.Intn(inductionMaxSpan-inductionMinSpan+1)
			src := rng.Intn(SeqLen - 2*span + 1)
			dst := src + span + rng.Intn(SeqLen-src-2*span+1)
			copy(x[dst:dst+span], x[src:src+span])
			// From the first repeated token on, the rest of the span is determined by the earlier copy.
			for j := 0; j < span-1; j++ {
				y[dst+j] = x[dst+j+1]
			}
			return x, y
		},
	}
}

func markovTask() Task {
	tableRNG := rand.New(rand.NewSource(markovTableSeed))
	preferred := make([]int, markovVocab*markovVocab)
	for i := range preferred {
		preferred[i] = tableRNG.Intn(markovVocab)
	}
	// With probability p the preferred token is emitted, otherwise a uniform draw (which may also
	// hit the preferred token), so the Bayes-optimal accuracy is p + (1-p)/V.
	ceiling := markovPreferredProb + (1-markovPreferredProb)/markovVocab
	return Task{
		Name:        "markov2",
		Description: "local modelling: predict a second-order Markov chain from the two previous tokens",
		VocabSize:   probeFirstToken + markovVocab,
		Chance:      1.0 / markovVocab,
		Ceiling:     ceiling,
		SeqLen:      SeqLen,
		generate: func(rng *rand.Rand) ([]int, []int) {
			seq := make([]int, SeqLen+1)
			seq[0], seq[1] = rng.Intn(markovVocab), rng.Intn(markovVocab)
			for i := 2; i < len(seq); i++ {
				if rng.Float64() < markovPreferredProb {
					seq[i] = preferred[seq[i-2]*markovVocab+seq[i-1]]
				} else {
					seq[i] = rng.Intn(markovVocab)
				}
			}
			x, y := make([]int, SeqLen), maskedTargets(SeqLen)
			for i := 0; i < SeqLen; i++ {
				x[i] = probeFirstToken + seq[i]
				if i >= 1 { // position 0 lacks the second context token
					y[i] = probeFirstToken + seq[i+1]
				}
			}
			return x, y
		},
	}
}

// GenerateData builds n sequences of the task from rng.
func (t Task) GenerateData(rng *rand.Rand, n int) data.Sequences {
	set := data.Sequences{X: make([][]int, n), Y: make([][]int, n)}
	for i := 0; i < n; i++ {
		set.X[i], set.Y[i] = t.generate(rng)
	}
	return set
}

// longest returns the length of the longest sequence in any of the sets.
func longest(sets ...data.Sequences) int {
	n := 0
	for _, set := range sets {
		for _, x := range set.X {
			n = max(n, len(x))
		}
	}
	return n
}

// TargetAccuracy returns the argmax accuracy of model over all non-masked targets.
func TargetAccuracy(model *ca.Model, set data.Sequences) float64 {
	maxLen := longest(set)
	workers, refresh := model.NewWorkers(1, maxLen)
	ws := workers[0]
	weights := model.Weights()
	refresh(weights, 1)
	V := model.VocabSize()
	correct, total := 0, 0
	for s := range set.X {
		allProbs := ws.ForwardProbs(set.X[s], set.Y[s], weights)
		for i, tgt := range set.Y[s] {
			if tgt < 0 {
				continue
			}
			probs := allProbs[i*V : (i+1)*V]
			best := 0
			for v := 1; v < V; v++ {
				if probs[v] > probs[best] {
					best = v
				}
			}
			if best == tgt {
				correct++
			}
			total++
		}
	}
	if total == 0 {
		return 0
	}
	return float64(correct) / float64(total)
}

// Run trains a fresh model described by cfg on task and evaluates held-out accuracy.
// The probe data and model initialization both derive from seed, so two configurations run
// with the same seed see identical data and comparable random streams. cfg.Vocab and
// cfg.Context are ignored in favour of the task's own vocabulary and sequence length.
func Run(task Task, cfg train.Config, seed int64) Result {
	dataRNG := rand.New(rand.NewSource(seed))
	trainSet := task.GenerateData(dataRNG, TrainSeqs)
	valSet := task.GenerateData(dataRNG, ValSeqs)
	testSet := task.GenerateData(dataRNG, TestSeqs)

	cfg.Context = longest(trainSet, valSet, testSet) // the context must cover the task
	model := cfg.BuildModel(task.VocabSize, seed)
	tr := train.Run(model, trainSet, valSet, cfg.Options(seed, true))
	return Result{
		Task:         task.Name,
		Accuracy:     TargetAccuracy(model, testSet),
		Chance:       task.Chance,
		Ceiling:      task.Ceiling,
		ValLoss:      tr.BestLoss,
		TokensPerSec: tr.TokensPerSec,
		Duration:     tr.Duration,
	}
}

// ResolveTasks parses a comma-separated task list ("all" selects every task).
func ResolveTasks(spec string) ([]Task, error) {
	all := Tasks()
	if spec == "" || spec == "all" {
		var tasks []Task
		for _, name := range TaskNames() {
			tasks = append(tasks, all[name])
		}
		return tasks, nil
	}
	var tasks []Task
	for _, name := range strings.Split(spec, ",") {
		t, ok := all[strings.TrimSpace(name)]
		if !ok {
			return nil, fmt.Errorf("unknown probe %q (available: %s)", name, strings.Join(TaskNames(), ", "))
		}
		tasks = append(tasks, t)
	}
	return tasks, nil
}
