package sweep

import (
	"math"

	"github.com/mignon-ai/mignon-ai/internal/train"
)

// Adaptive seeding: every variant runs the first MinSeeds seeds; each further seed is only
// run for variants whose paired difference against the baseline is still undecided (plus the
// baseline itself, which every paired difference needs).

// minAdaptiveSeeds is the fewest seeds that give a paired-noise estimate.
const minAdaptiveSeeds = 2

// pairedDiffs returns, for every variant, its metric minus the baseline's (variant 0) on each
// seed where both ran on benchmark bench. The baseline's own entry is empty.
func pairedDiffs(bench string, variants []string, seeds []int64, runs []Result) [][]float64 {
	metric := map[string]map[int64]float64{}
	for _, r := range runs {
		if r.Benchmark != bench {
			continue
		}
		if metric[r.Variant] == nil {
			metric[r.Variant] = map[int64]float64{}
		}
		metric[r.Variant][r.Seed] = r.Metric
	}
	diffs := make([][]float64, len(variants))
	for i := 1; i < len(variants); i++ {
		for _, seed := range seeds {
			base, okB := metric[variants[0]][seed]
			v, okV := metric[variants[i]][seed]
			if okB && okV {
				diffs[i] = append(diffs[i], v-base)
			}
		}
	}
	return diffs
}

// pooledSD returns the standard deviation of paired differences pooled over all variants
// (each around its own mean), or NaN when there are no degrees of freedom.
func pooledSD(diffs [][]float64) float64 {
	ss, dof := 0.0, 0
	for _, d := range diffs {
		if len(d) < 2 {
			continue
		}
		_, sd := meanStd(d)
		ss += sd * sd * float64(len(d)-1)
		dof += len(d) - 1
	}
	if dof == 0 {
		return math.NaN()
	}
	return math.Sqrt(ss / float64(dof))
}

// diffSettled reports whether more seeds are unlikely to change the verdict: the mean paired
// difference is clearly beyond minEffect, or clearly within it. The noise estimate is the larger
// of the variant's own spread and the pooled spread, so two lucky seeds cannot settle it early.
func diffSettled(d []float64, pooled, minEffect float64) bool {
	if len(d) < minAdaptiveSeeds {
		return false
	}
	mean, sd := meanStd(d)
	if !math.IsNaN(pooled) && pooled > sd {
		sd = pooled
	}
	margin := sweepNoiseZ * sd / math.Sqrt(float64(len(d)))
	effect := math.Abs(mean)
	if effect >= minEffect && effect >= margin {
		return true
	}
	return effect+margin < minEffect
}

// unsettledVariants returns the indices (> 0) of variants whose verdict on bench is undecided.
func unsettledVariants(bench string, minEffect float64, variants []string, seeds []int64, runs []Result) []int {
	diffs := pairedDiffs(bench, variants, seeds, runs)
	pooled := pooledSD(diffs)
	var open []int
	for i := 1; i < len(variants); i++ {
		if !diffSettled(diffs[i], pooled, minEffect) {
			open = append(open, i)
		}
	}
	return open
}

// runBenchmarkSeeds runs variants on b seed by seed. Before each seed beyond the first
// minSeeds, variants that are already settled are skipped. minSeeds <= 0 runs every seed for
// every variant. onRun is called after each run with the result.
func runBenchmarkSeeds(b benchmark, variants []string, cfgs []train.Config, seeds []int64, minSeeds int, onRun func(Result)) []Result {
	var runs []Result
	active := make([]int, len(variants))
	for i := range active {
		active[i] = i
	}
	for si, seed := range seeds {
		if minSeeds > 0 && si >= minSeeds {
			open := unsettledVariants(b.name, b.minEffect, variants, seeds, runs)
			if len(open) == 0 {
				break
			}
			active = append([]int{0}, open...)
		}
		for _, i := range active {
			r := b.run(cfgs[i], seed)
			r.Benchmark, r.Variant, r.Seed = b.name, variants[i], seed
			runs = append(runs, r)
			onRun(r)
		}
	}
	return runs
}
