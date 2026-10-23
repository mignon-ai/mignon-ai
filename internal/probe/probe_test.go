package probe

import (
	"math/rand"
	"reflect"
	"testing"

	"github.com/mignon-ai/mignon-ai/internal/data"
	"github.com/mignon-ai/mignon-ai/internal/train"
)

func TestProbeTasksGenerateValidData(t *testing.T) {
	const samples = 50
	for _, name := range TaskNames() {
		task := Tasks()[name]
		rng := rand.New(rand.NewSource(1))
		set := task.GenerateData(rng, samples)
		xs, ys := set.X, set.Y
		for s := range xs {
			if len(xs[s]) != task.SeqLen || len(ys[s]) != task.SeqLen {
				t.Fatalf("%s: sequence length %d/%d, want %d", name, len(xs[s]), len(ys[s]), task.SeqLen)
			}
			targets := 0
			for i := range xs[s] {
				if xs[s][i] < probeFirstToken || xs[s][i] >= task.VocabSize {
					t.Fatalf("%s: input token %d outside [%d, %d)", name, xs[s][i], probeFirstToken, task.VocabSize)
				}
				if ys[s][i] != data.MaskedTarget {
					targets++
					if ys[s][i] < probeFirstToken || ys[s][i] >= task.VocabSize {
						t.Fatalf("%s: target %d outside vocabulary", name, ys[s][i])
					}
				}
			}
			if targets == 0 {
				t.Fatalf("%s: sequence without targets", name)
			}
		}
		if !(task.Chance > 0 && task.Chance < task.Ceiling && task.Ceiling <= 1) {
			t.Errorf("%s: inconsistent chance %.3f / ceiling %.3f", name, task.Chance, task.Ceiling)
		}
	}
}

func TestProbeDataIsSeeded(t *testing.T) {
	for _, name := range TaskNames() {
		task := Tasks()[name]
		a := task.GenerateData(rand.New(rand.NewSource(7)), 10)
		b := task.GenerateData(rand.New(rand.NewSource(7)), 10)
		if !reflect.DeepEqual(a, b) {
			t.Fatalf("%s: same seed produced different data", name)
		}
	}
}

// TestProbeTargetsAreDetermined checks each task's targets are answerable from the context alone.
func TestProbeTargetsAreDetermined(t *testing.T) {
	rng := rand.New(rand.NewSource(3))
	tasks := Tasks()

	set := tasks["assoc_recall"].GenerateData(rng, 20)
	xs, ys := set.X, set.Y
	for s := range xs {
		for i, tgt := range ys[s] {
			if tgt == data.MaskedTarget {
				continue
			}
			found := false
			for j := 0; j+1 < i; j += 2 {
				if xs[s][j] == xs[s][i] && xs[s][j+1] == tgt {
					found = true
				}
			}
			if !found {
				t.Fatalf("assoc_recall: query at %d has no matching earlier key/value pair", i)
			}
		}
	}

	set = tasks["induction"].GenerateData(rng, 200)
	xs, ys = set.X, set.Y
	offsets := map[int]bool{}
	for s := range xs {
		first := -1
		for i, tgt := range ys[s] {
			if tgt == data.MaskedTarget {
				continue
			}
			if tgt != xs[s][i+1] {
				t.Fatalf("induction: target at %d is not the next token", i)
			}
			if first < 0 {
				first = i
			}
		}
		// The target span must repeat an earlier span of the same tokens.
		n := 0
		for i := first; i < SeqLen && ys[s][i] != data.MaskedTarget; i++ {
			n++
		}
		found := false
		for src := 0; src+n+1 <= first; src++ {
			if reflect.DeepEqual(xs[s][src:src+n+1], xs[s][first:first+n+1]) {
				found = true
				offsets[first-src] = true
			}
		}
		if !found {
			t.Fatal("induction: repeated span has no earlier source")
		}
	}
	if len(offsets) < 10 {
		t.Fatalf("induction: only %d distinct repeat offsets; a fixed lag would make the task positional", len(offsets))
	}
}

// TestFarRecallAnswersLieBeyondTheGap checks that every far_recall answer is bound only in the
// opening pairs, at least farRecallGap tokens before its query, and that no key is queried twice.
func TestFarRecallAnswersLieBeyondTheGap(t *testing.T) {
	set := Tasks()["far_recall"].GenerateData(rand.New(rand.NewSource(5)), 20)
	pairsEnd := 2 * assocPairs
	for s, x := range set.X {
		queried := map[int]bool{}
		for i, tgt := range set.Y[s] {
			if tgt == data.MaskedTarget {
				continue
			}
			if queried[x[i]] {
				t.Fatalf("far_recall: key %d queried twice", x[i])
			}
			queried[x[i]] = true
			bound := -1
			for j := 0; j < pairsEnd; j += 2 {
				if x[j] == x[i] && x[j+1] == tgt {
					bound = j
				}
			}
			if bound < 0 || i-(bound+1) <= farRecallGap {
				t.Fatalf("far_recall: query at %d bound at %d, want a binding more than %d tokens back", i, bound, farRecallGap)
			}
		}
		if len(queried) != assocPairs {
			t.Fatalf("far_recall: %d queries, want %d", len(queried), assocPairs)
		}
	}
}

func TestResolveProbeTasks(t *testing.T) {
	all, err := ResolveTasks("all")
	if err != nil || len(all) != len(Tasks()) {
		t.Fatalf("all: got %d tasks, err %v", len(all), err)
	}
	two, err := ResolveTasks("induction, markov2")
	if err != nil || len(two) != 2 {
		t.Fatalf("list: got %d tasks, err %v", len(two), err)
	}
	if _, err := ResolveTasks("nope"); err == nil {
		t.Fatal("expected error for unknown probe")
	}
}

func TestTargetAccuracyUntrainedNearChance(t *testing.T) {
	task := Tasks()["induction"]
	cfg := train.DefaultConfig()
	cfg.StateDim, cfg.HiddenDim, cfg.Steps = 16, 24, 2
	m := cfg.BuildModel(task.VocabSize, 11)
	acc := TargetAccuracy(m, task.GenerateData(rand.New(rand.NewSource(11)), 100))
	// An untrained model must not score far above chance; a broken accuracy routine typically would.
	if acc > 4*task.Chance {
		t.Fatalf("untrained accuracy %.3f far above chance %.3f", acc, task.Chance)
	}
}

func TestSeededTrainingIsReproducible(t *testing.T) {
	task := Tasks()["markov2"]
	cfg := train.DefaultConfig()
	cfg.StateDim, cfg.HiddenDim, cfg.Steps, cfg.Epochs, cfg.Batch = 16, 24, 2, 2, train.DefaultMinBatchSize
	cfg.Threads = 1 // multi-threaded gradient reduction order is not deterministic

	run := func() []float32 {
		rng := rand.New(rand.NewSource(5))
		set := task.GenerateData(rng, 64)
		m := cfg.BuildModel(task.VocabSize, 5)
		train.Run(m, set, data.Sequences{}, cfg.Options(5, true))
		return append([]float32(nil), m.Weights()...)
	}
	if !reflect.DeepEqual(run(), run()) {
		t.Fatal("seeded single-threaded training must be bit-for-bit reproducible")
	}
}
