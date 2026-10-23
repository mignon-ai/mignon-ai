package sweep

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/mignon-ai/mignon-ai/internal/data"
)

func TestMeanStd(t *testing.T) {
	m, s := meanStd([]float64{1, 2, 3, 4})
	if m != 2.5 || math.Abs(s-math.Sqrt(5.0/3.0)) > 1e-12 {
		t.Fatalf("meanStd = %g, %g", m, s)
	}
	if _, s := meanStd([]float64{1}); !math.IsNaN(s) {
		t.Fatal("single sample std should be NaN")
	}
}

func TestPairedVerdict(t *testing.T) {
	cases := []struct {
		mean, std    float64
		n            int
		higherBetter bool
		want         string
	}{
		{0.1, 0.01, 3, true, "better"},
		{0.1, 0.01, 3, false, "worse"},
		{-0.1, 0.01, 3, false, "better"},
		{0.05, 0.1, 3, true, "≈ noise"},
		{0.005, 0.0001, 3, true, "≈ same"}, // consistent but below the minimum effect
		{0.1, 0.01, 1, true, "need ≥2 seeds"},
	}
	for _, c := range cases {
		if got := pairedVerdict(c.mean, c.std, c.n, c.higherBetter, sweepMinAccuracyEffect); got != c.want {
			t.Errorf("pairedVerdict(%+v) = %q, want %q", c, got, c.want)
		}
	}
}

func TestLoadSweepSpecValidation(t *testing.T) {
	dir := t.TempDir()
	write := func(body string) string {
		p := filepath.Join(dir, "spec.json")
		if err := os.WriteFile(p, []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
		return p
	}
	bad := []string{
		`{"seeds":[1],"probes":"all"}`,                          // no variants
		`{"seeds":[],"probes":"all","variants":[{"name":"a"}]}`, // no seeds
		`{"seeds":[0],"probes":"all","variants":[{"name":"a"}]}`,
		`{"seeds":[1],"variants":[{"name":"a"}]}`, // nothing to run
		`{"seeds":[1,2],"min_seeds":1,"probes":"all","variants":[{"name":"a"}]}`,
	}
	for _, body := range bad {
		if _, err := LoadSpec(write(body)); err == nil {
			t.Errorf("expected error for %s", body)
		}
	}
	if _, err := LoadSpec(write(`{"seeds":[1],"probes":"all","variants":[{"name":"a"}]}`)); err != nil {
		t.Errorf("valid spec rejected: %v", err)
	}
}

func TestRunSweepRejectsBadVariantBeforeTraining(t *testing.T) {
	spec := Spec{
		Seeds:    []int64{1},
		Probes:   "markov2",
		Variants: []json.RawMessage{[]byte(`{"name":"ok"}`), []byte(`{"name":"bad","opt":"nope"}`)},
	}
	if _, err := Run(spec); err == nil {
		t.Fatal("expected config validation error")
	}
}

func TestDatasetCacheRoundTrip(t *testing.T) {
	t.Setenv("XDG_CACHE_HOME", t.TempDir())
	t.Setenv("HOME", t.TempDir())
	docs := []string{
		"Once upon a time there was a cat. The cat liked to sit in the sun.\n\nOne day the cat met a dog.",
		"The dog and the cat became friends. They played in the park every day.\n\nThe end.",
	}
	const ctx, vocab, stride = 16, 300, 16
	train1, val1, tok1, cached1 := data.PrepareTrainValCached(docs, ctx, vocab, stride, 0, 0)
	train2, val2, tok2, cached2 := data.PrepareTrainValCached(docs, ctx, vocab, stride, 0, 0)
	if cached1 || !cached2 {
		t.Fatalf("expected miss then hit, got %v then %v", cached1, cached2)
	}
	if !reflect.DeepEqual(train1, train2) || !reflect.DeepEqual(val1, val2) {
		t.Fatal("cached sequences differ from freshly encoded ones")
	}
	if !reflect.DeepEqual(tok1.Merges, tok2.Merges) || tok1.VocabSize() != tok2.VocabSize() {
		t.Fatal("cached tokenizer differs")
	}
	if got, want := tok2.Encode("the cat and the dog"), tok1.Encode("the cat and the dog"); !reflect.DeepEqual(got, want) {
		t.Fatalf("cached tokenizer encodes differently: %v vs %v", got, want)
	}
	// Any parameter change must miss the cache.
	if _, _, _, cached := data.PrepareTrainValCached(docs, ctx+1, vocab, stride, 0, 0); cached {
		t.Fatal("changed context size must not hit the cache")
	}
}

// TestCommittedSweepSpecsResolve keeps the live research/eval/sweeps/*.json specs loadable (archive/ holds finished experiments): a spec using a removed or
// renamed config key must fail here, not hours into a run.
func TestCommittedSweepSpecsResolve(t *testing.T) {
	paths, err := filepath.Glob("../../research/eval/sweeps/*.json")
	if err != nil {
		t.Fatal(err)
	}
	for _, path := range paths {
		if strings.HasSuffix(path, ".results.json") {
			continue
		}
		t.Run(filepath.Base(path), func(t *testing.T) {
			spec, err := LoadSpec(path)
			if err != nil {
				t.Fatal(err)
			}
			if err := spec.checkConfigs(); err != nil {
				t.Fatal(err)
			}
		})
	}
}
