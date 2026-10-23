// Package sweep runs multi-seed configuration sweeps over probe tasks and text corpora and reports
// paired differences against the first variant.
package sweep

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"strings"
	"time"

	"github.com/mignon-ai/mignon-ai/internal/data"
	"github.com/mignon-ai/mignon-ai/internal/probe"
	"github.com/mignon-ai/mignon-ai/internal/train"
)

// A sweep compares configuration variants on probe tasks and/or a text corpus across several
// seeds. Every variant runs with the same seeds, so seed s gives every variant the same data
// order and probe data; differences are therefore measured pairwise against the first variant.
//
// Example spec:
//
//	{
//	  "seeds": [1, 2, 3],
//	  "probes": "all",
//	  "text": "data/stories",
//	  "base": {"preset": "micro", "opt": "adam", "lr": 0.016, "epochs": 6},
//	  "text_base": {"preset": "small", "epochs": 4, "max_seqs": 20000},
//	  "variants": [{"name": "base"}, {"name": "attn", "attn_heads": 4}]
//	}
type Spec struct {
	Seeds    []int64           `json:"seeds"`
	Probes   string            `json:"probes,omitempty"`    // comma-separated probe names, "all", or empty to skip
	Text     string            `json:"text,omitempty"`      // directory of .txt files, or empty to skip
	Base     json.RawMessage   `json:"base,omitempty"`      // config layer shared by all runs
	TextBase json.RawMessage   `json:"text_base,omitempty"` // extra layer for text runs only (typically a larger preset)
	Variants []json.RawMessage `json:"variants"`            // each must include "name"; other keys override base
	Output   string            `json:"output,omitempty"`    // optional path for machine-readable JSON results
	// MinSeeds, when set, runs only the first MinSeeds seeds for every variant and the remaining
	// seeds only for variants whose verdict is still undecided (see sweep_adaptive.go).
	MinSeeds int `json:"min_seeds,omitempty"`
}

const (
	// sweepNoiseZ is the paired-difference z-score below which a difference is reported as noise.
	sweepNoiseZ = 2.0
	// Differences smaller than these are reported as negligible even when consistent across seeds.
	sweepMinAccuracyEffect = 0.01
	sweepMinLossEffect     = 0.01
	// sweepMinBPBEffect is sweepMinLossEffect in bits per byte at the ~3 bytes per token of the
	// project's tokenizers (0.01 nats / ln 2 / 3 bytes).
	sweepMinBPBEffect = 0.005
	// sweepChanceMargin flags probe results below this multiple of chance accuracy.
	sweepChanceMargin = 1.5
)

// Result is one (benchmark, variant, seed) measurement.
type Result struct {
	Benchmark    string  `json:"benchmark"`
	Variant      string  `json:"variant"`
	Seed         int64   `json:"seed"`
	Metric       float64 `json:"metric"`
	TokensPerSec float64 `json:"tokens_per_sec"`
	Seconds      float64 `json:"seconds"`
}

// benchmark is one thing every variant is measured on.
type benchmark struct {
	name         string
	metricName   string
	higherBetter bool
	minEffect    float64
	chance       float64 // 0 when not applicable
	run          func(cfg train.Config, seed int64) Result
	textLayer    bool // whether the text_base layer applies
}

// LoadSpec reads a sweep spec from a JSON file.
func LoadSpec(path string) (Spec, error) {
	var spec Spec
	data, err := os.ReadFile(path)
	if err != nil {
		return spec, err
	}
	if err := json.Unmarshal(data, &spec); err != nil {
		return spec, fmt.Errorf("invalid sweep spec: %w", err)
	}
	if len(spec.Variants) == 0 {
		return spec, fmt.Errorf("sweep spec needs at least one variant")
	}
	if len(spec.Seeds) == 0 {
		return spec, fmt.Errorf("sweep spec needs at least one seed")
	}
	for _, s := range spec.Seeds {
		if s == 0 {
			return spec, fmt.Errorf("seed 0 means nondeterministic; use non-zero seeds")
		}
	}
	if spec.MinSeeds != 0 && spec.MinSeeds < minAdaptiveSeeds {
		return spec, fmt.Errorf("min_seeds must be at least %d to estimate noise", minAdaptiveSeeds)
	}
	if spec.Probes == "" && spec.Text == "" {
		return spec, fmt.Errorf("sweep spec needs \"probes\" and/or \"text\"")
	}
	return spec, nil
}

func variantName(raw json.RawMessage) string {
	var v struct {
		Name string `json:"name"`
	}
	_ = json.Unmarshal(raw, &v)
	return v.Name
}

// stripName removes the "name" key so variant layers contain only config keys.
func stripName(raw json.RawMessage) (json.RawMessage, error) {
	var m map[string]json.RawMessage
	if err := json.Unmarshal(raw, &m); err != nil {
		return nil, err
	}
	delete(m, "name")
	return json.Marshal(m)
}

// sweepBenchmarks builds the benchmark list described by spec.
func sweepBenchmarks(spec Spec) ([]benchmark, error) {
	var benches []benchmark
	if spec.Probes != "" {
		tasks, err := probe.ResolveTasks(spec.Probes)
		if err != nil {
			return nil, err
		}
		for _, task := range tasks {
			task := task
			benches = append(benches, benchmark{
				name:         "probe:" + task.Name,
				metricName:   "accuracy",
				higherBetter: true,
				minEffect:    sweepMinAccuracyEffect,
				chance:       task.Chance,
				run: func(cfg train.Config, seed int64) Result {
					r := probe.Run(task, cfg, seed)
					return Result{Metric: r.Accuracy, TokensPerSec: r.TokensPerSec, Seconds: r.Duration.Seconds()}
				},
			})
		}
	}
	if spec.Text != "" {
		docs, err := data.LoadTextCorpus(spec.Text, nil)
		if err != nil {
			return nil, err
		}
		benches = append(benches, benchmark{
			name:       "text:" + spec.Text,
			metricName: "val bits/byte",
			minEffect:  sweepMinBPBEffect,
			textLayer:  true,
			run: func(cfg train.Config, seed int64) Result {
				return runTextExperiment(docs, cfg, seed)
			},
		})
	}
	return benches, nil
}

// runTextExperiment trains cfg on the (cached) tokenized corpus and reports validation bits per
// byte at the best epoch, which stays comparable when variants use different tokenizers.
func runTextExperiment(docs []string, cfg train.Config, seed int64) Result {
	stride := cfg.Stride
	if stride <= 0 {
		stride = cfg.Context
	}
	trainSet, valSet, tok, _ := data.PrepareTrainValCached(docs, cfg.Context, cfg.Vocab, stride, cfg.MaxSeqs, data.DefaultValFraction)
	trainSet = trainSet.Cap(cfg.MaxSeqs)
	model := cfg.BuildModel(tok.VocabSize(), seed)
	opts := cfg.Options(seed, true)
	opts.TokenBytes = tok.TokenBytes()
	tr := train.Run(model, trainSet, valSet, opts)
	return Result{Metric: tr.BestBPB, TokensPerSec: tr.TokensPerSec, Seconds: tr.Duration.Seconds()}
}

// Run executes every benchmark x variant x seed and prints a comparison table.
// config resolves the layered config of one benchmark and variant layer.
func (spec Spec) config(textLayer bool, layer json.RawMessage) (train.Config, error) {
	if textLayer {
		return train.ResolveConfig(spec.Base, spec.TextBase, layer)
	}
	return train.ResolveConfig(spec.Base, layer)
}

// variantLayers returns each variant's name and its config layer (the variant without "name").
func (spec Spec) variantLayers() (names []string, layers []json.RawMessage, err error) {
	names = make([]string, len(spec.Variants))
	layers = make([]json.RawMessage, len(spec.Variants))
	for i, raw := range spec.Variants {
		names[i] = variantName(raw)
		if names[i] == "" {
			return nil, nil, fmt.Errorf("variant %d has no \"name\"", i)
		}
		if layers[i], err = stripName(raw); err != nil {
			return nil, nil, fmt.Errorf("variant %q: %w", names[i], err)
		}
	}
	return names, layers, nil
}

// checkConfigs resolves every variant's config for the probe and text layers the spec uses, so a
// typo fails before hours of training. It needs no corpus.
func (spec Spec) checkConfigs() error {
	names, layers, err := spec.variantLayers()
	if err != nil {
		return err
	}
	for _, textLayer := range []bool{false, true} {
		if (textLayer && spec.Text == "") || (!textLayer && spec.Probes == "") {
			continue
		}
		for i, layer := range layers {
			if _, err := spec.config(textLayer, layer); err != nil {
				return fmt.Errorf("%s / %s: %w", map[bool]string{false: "probes", true: "text"}[textLayer], names[i], err)
			}
		}
	}
	return nil
}

func Run(spec Spec) ([]Result, error) {
	if err := spec.checkConfigs(); err != nil {
		return nil, err
	}
	benches, err := sweepBenchmarks(spec)
	if err != nil {
		return nil, err
	}
	names, layers, _ := spec.variantLayers()

	total := len(benches) * len(layers) * len(spec.Seeds)
	adaptiveNote := ""
	if spec.MinSeeds > 0 {
		adaptiveNote = fmt.Sprintf(" at most (seeds after the first %d only for undecided variants)", spec.MinSeeds)
	}
	fmt.Printf("Sweep: %d benchmark(s) x %d variant(s) x %d seed(s) = %d runs%s\n", len(benches), len(layers), len(spec.Seeds), total, adaptiveNote)

	var runs []Result
	done := 0
	sweepStart := time.Now()
	for _, b := range benches {
		cfgs := make([]train.Config, len(layers))
		for i := range layers {
			cfgs[i], _ = spec.config(b.textLayer, layers[i])
		}
		runs = append(runs, runBenchmarkSeeds(b, names, cfgs, spec.Seeds, spec.MinSeeds, func(r Result) {
			done++
			fmt.Printf("  [%d/%d] %-20s %-16s seed=%-4d %s=%.4f  (%.0f tok/s, %.1fs)\n",
				done, total, b.name, r.Variant, r.Seed, b.metricName, r.Metric, r.TokensPerSec, r.Seconds)
		})...)
	}
	fmt.Printf("Sweep finished in %s\n\n", time.Since(sweepStart).Round(time.Second))

	for _, b := range benches {
		PrintTable(b, names, spec.Seeds, runs)
	}
	if spec.Output != "" {
		data, _ := json.MarshalIndent(runs, "", "  ")
		if err := os.WriteFile(spec.Output, data, 0o644); err != nil {
			return runs, err
		}
		fmt.Printf("Results written to %s\n", spec.Output)
	}
	return runs, nil
}

// meanStd returns the mean and sample standard deviation of xs.
func meanStd(xs []float64) (float64, float64) {
	if len(xs) == 0 {
		return math.NaN(), math.NaN()
	}
	mean := 0.0
	for _, x := range xs {
		mean += x
	}
	mean /= float64(len(xs))
	if len(xs) < 2 {
		return mean, math.NaN()
	}
	ss := 0.0
	for _, x := range xs {
		ss += (x - mean) * (x - mean)
	}
	return mean, math.Sqrt(ss / float64(len(xs)-1))
}

// PrintTable prints one benchmark's results with paired differences against the first variant.
func PrintTable(b benchmark, variants []string, seeds []int64, runs []Result) {
	bySeed := map[string]map[int64]Result{}
	for _, r := range runs {
		if r.Benchmark != b.name {
			continue
		}
		if bySeed[r.Variant] == nil {
			bySeed[r.Variant] = map[int64]Result{}
		}
		bySeed[r.Variant][r.Seed] = r
	}

	direction := "lower is better"
	if b.higherBetter {
		direction = "higher is better"
	}
	chanceNote := ""
	if b.chance > 0 {
		chanceNote = fmt.Sprintf(", chance %.3f", b.chance)
	}
	fmt.Printf("== %s (%s, %s%s) ==\n", b.name, b.metricName, direction, chanceNote)
	fmt.Printf("%-18s %-18s %-26s %-10s %s\n", "variant", "mean ± std", "paired Δ vs "+variants[0], "tok/s", "verdict")
	fmt.Println(strings.Repeat("-", 90))

	baseline := bySeed[variants[0]]
	for vi, v := range variants {
		var metrics, diffs, tps []float64
		for _, seed := range seeds {
			r, ok := bySeed[v][seed]
			if !ok {
				continue
			}
			metrics = append(metrics, r.Metric)
			tps = append(tps, r.TokensPerSec)
			if base, ok := baseline[seed]; ok {
				diffs = append(diffs, r.Metric-base.Metric)
			}
		}
		mean, std := meanStd(metrics)
		meanTPS, _ := meanStd(tps)
		deltaStr, verdict := "(baseline)", ""
		if vi > 0 {
			dMean, dStd := meanStd(diffs)
			deltaStr = fmt.Sprintf("%+.4f ± %.4f", dMean, dStd)
			verdict = pairedVerdict(dMean, dStd, len(diffs), b.higherBetter, b.minEffect)
		}
		if b.chance > 0 && mean < b.chance*sweepChanceMargin {
			verdict = strings.TrimSpace(verdict + " (~chance)")
		}
		fmt.Printf("%-18s %-18s %-26s %-10.0f %s\n", v, fmt.Sprintf("%.4f ± %.4f", mean, std), deltaStr, meanTPS, verdict)
	}
	fmt.Println()
}

// pairedVerdict classifies a paired difference as better, worse, or within noise.
func pairedVerdict(dMean, dStd float64, n int, higherBetter bool, minEffect float64) string {
	if math.Abs(dMean) < minEffect {
		return "≈ same"
	}
	if n < 2 || math.IsNaN(dStd) {
		return "need ≥2 seeds"
	}
	se := dStd / math.Sqrt(float64(n))
	if se == 0 || math.Abs(dMean) < sweepNoiseZ*se {
		return "≈ noise"
	}
	if (dMean > 0) == higherBetter {
		return "better"
	}
	return "worse"
}
