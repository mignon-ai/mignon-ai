package sweep

import (
	"encoding/json"
	"fmt"
	"os"
	"reflect"
	"sort"
	"testing"
)

// The probe regression guard enforces the AGENTS.md rule that a new generation of the model must
// not be significantly worse than the previous one. It trains the probe suite and compares mean
// accuracy per (variant, task) against eval/probe_baseline.json.
//
// It takes several minutes, so it only runs when PERI_PROBE_REGRESSION=1:
//
//	PERI_PROBE_REGRESSION=1 go test -run TestProbeRegression -v ./internal/sweep
//
// After an intentional improvement, refresh the baseline with:
//
//	PERI_PROBE_REGRESSION=1 PERI_UPDATE_PROBE_BASELINE=1 go test -run TestProbeRegression -v ./internal/sweep

const (
	probeBaselinePath = "../../eval/probe_baseline.json"
	// probeRegressionTolerance is the largest allowed drop in mean accuracy, chosen above the
	// seed-to-seed spread of the suite so only real regressions fail.
	probeRegressionTolerance = 0.05
)

// probeRegressionSpec is the fixed suite guarded by the baseline. Changing it invalidates the baseline.
func probeRegressionSpec() Spec {
	return Spec{
		Seeds: []int64{1, 2},
		// induction is excluded: every current architecture is at chance on it, so it adds runtime without signal.
		Probes:   "assoc_recall,long_copy,markov2",
		Base:     json.RawMessage(`{"preset":"micro","opt":"adam","lr":0.016,"epochs":12}`),
		Variants: []json.RawMessage{json.RawMessage(`{"name":"default"}`)},
	}
}

// probeBaseline is the committed reference result.
type probeBaseline struct {
	Spec     Spec               `json:"spec"`
	Accuracy map[string]float64 `json:"accuracy"` // "variant/benchmark" -> mean accuracy over seeds
}

func meanAccuracyByKey(runs []Result) map[string]float64 {
	sums, counts := map[string]float64{}, map[string]int{}
	for _, r := range runs {
		key := r.Variant + "/" + r.Benchmark
		sums[key] += r.Metric
		counts[key]++
	}
	for k := range sums {
		sums[k] /= float64(counts[k])
	}
	return sums
}

func TestProbeRegression(t *testing.T) {
	if os.Getenv("PERI_PROBE_REGRESSION") != "1" {
		t.Skip("set PERI_PROBE_REGRESSION=1 to run the probe regression suite")
	}
	spec := probeRegressionSpec()
	runs, err := Run(spec)
	if err != nil {
		t.Fatal(err)
	}
	current := meanAccuracyByKey(runs)

	if os.Getenv("PERI_UPDATE_PROBE_BASELINE") == "1" {
		data, _ := json.MarshalIndent(probeBaseline{Spec: spec, Accuracy: current}, "", "  ")
		if err := os.WriteFile(probeBaselinePath, append(data, '\n'), 0o644); err != nil {
			t.Fatal(err)
		}
		t.Logf("baseline written to %s", probeBaselinePath)
		return
	}

	data, err := os.ReadFile(probeBaselinePath)
	if err != nil {
		t.Fatalf("no baseline (%v); create one with PERI_UPDATE_PROBE_BASELINE=1", err)
	}
	var base probeBaseline
	if err := json.Unmarshal(data, &base); err != nil {
		t.Fatal(err)
	}
	specJSON, _ := json.Marshal(spec)
	baseSpecJSON, _ := json.Marshal(base.Spec)
	if !reflect.DeepEqual(specJSON, baseSpecJSON) {
		t.Fatalf("baseline was recorded with a different suite; regenerate it with PERI_UPDATE_PROBE_BASELINE=1")
	}

	keys := make([]string, 0, len(base.Accuracy))
	for k := range base.Accuracy {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	for _, k := range keys {
		cur, ok := current[k]
		if !ok {
			t.Errorf("%s: missing from current results", k)
			continue
		}
		delta := cur - base.Accuracy[k]
		msg := fmt.Sprintf("%-36s baseline %.3f  current %.3f  (%+.3f)", k, base.Accuracy[k], cur, delta)
		if delta < -probeRegressionTolerance {
			t.Errorf("REGRESSION %s", msg)
		} else {
			t.Log(msg)
		}
	}
}
