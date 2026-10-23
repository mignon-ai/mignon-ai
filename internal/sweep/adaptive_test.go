package sweep

import (
	"reflect"
	"testing"

	"github.com/mignon-ai/mignon-ai/internal/train"
)

func TestDiffSettled(t *testing.T) {
	cases := []struct {
		name   string
		diffs  []float64
		pooled float64
		want   bool
	}{
		{"one seed", []float64{0.5}, 0.001, false},
		{"clear effect", []float64{0.10, 0.11}, 0.005, true},
		{"clearly negligible", []float64{0.001, 0.002}, 0.001, true},
		{"near the threshold", []float64{0.004, 0.016}, 0.005, false},
		{"lucky agreement, noisy pool", []float64{0.02, 0.02}, 0.05, false},
	}
	for _, c := range cases {
		if got := diffSettled(c.diffs, c.pooled, sweepMinLossEffect); got != c.want {
			t.Errorf("%s: diffSettled = %v, want %v", c.name, got, c.want)
		}
	}
}

func TestPooledSD(t *testing.T) {
	// Variant spreads around their own means: {0,2} → var 2, {1,3} → var 2; pooled var 2.
	got := pooledSD([][]float64{nil, {0, 2}, {1, 3}, {5}})
	if want := 1.4142135623730951; got != want {
		t.Fatalf("pooledSD = %g, want %g", got, want)
	}
}

// TestRunBenchmarkSeedsSkipsSettledVariants runs a fake benchmark where one variant is clearly
// worse and another sits at the minimum-effect threshold: only the latter (and the baseline)
// should get the extra seeds.
func TestRunBenchmarkSeedsSkipsSettledVariants(t *testing.T) {
	seeds := []int64{1, 2, 3, 4}
	offsets := map[string][]float64{ // per-seed metric offsets from the baseline
		"base":  {0, 0, 0, 0},
		"clear": {0.20, 0.21, 0.20, 0.21},
		"close": {0.004, 0.016, 0.004, 0.016},
	}
	variants := []string{"base", "clear", "close"}
	cfgs := make([]train.Config, len(variants))
	for i := range cfgs {
		cfgs[i].Epochs = i // identifies the variant inside the fake run
	}
	b := benchmark{name: "fake", minEffect: sweepMinLossEffect, run: func(cfg train.Config, seed int64) Result {
		return Result{Metric: offsets[variants[cfg.Epochs]][seed-1]}
	}}

	count := map[string]int{}
	runBenchmarkSeeds(b, variants, cfgs, seeds, minAdaptiveSeeds, func(r Result) { count[r.Variant]++ })
	if want := map[string]int{"base": 4, "clear": 2, "close": 4}; !reflect.DeepEqual(count, want) {
		t.Fatalf("runs per variant %v, want %v", count, want)
	}

	count = map[string]int{}
	runBenchmarkSeeds(b, variants, cfgs, seeds, 0, func(r Result) { count[r.Variant]++ })
	if want := map[string]int{"base": 4, "clear": 4, "close": 4}; !reflect.DeepEqual(count, want) {
		t.Fatalf("non-adaptive runs per variant %v, want %v", count, want)
	}
}
