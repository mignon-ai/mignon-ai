package main

import (
	"flag"
	"testing"

	"github.com/mignon-ai/mignon-ai/internal/train"
)

func newTestFlagSet() *flag.FlagSet { return flag.NewFlagSet("mignon", flag.ContinueOnError) }

// parseTestFlags parses train flags, supplying the required -text unless args name a data source.
func parseTestFlags(t *testing.T, args ...string) *trainOptions {
	t.Helper()
	o, err := parseTrainFlags(newTestFlagSet(), append([]string{"-text", "data"}, args...))
	if err != nil {
		t.Fatalf("parseTrainFlags(%v): %v", args, err)
	}
	return o
}

func TestTrainRequiresExactlyOneDataSource(t *testing.T) {
	for _, args := range [][]string{{}, {"-text", "a", "-sft", "b.json"}} {
		if _, err := parseTrainFlags(newTestFlagSet(), args); err == nil {
			t.Errorf("args %v: expected an error", args)
		}
	}
}

func TestTokFromOnlyForNewTextModels(t *testing.T) {
	if o := parseTestFlags(t, "-tok-from", "a.json"); o.tokFrom != "a.json" {
		t.Fatalf("tokFrom = %q, want a.json", o.tokFrom)
	}
	for _, args := range [][]string{
		{"-text", "data", "-tok-from", "a.json", "-load", "b.json"},
		{"-sft", "p.json", "-tok-from", "a.json"},
	} {
		if _, err := parseTrainFlags(newTestFlagSet(), args); err == nil {
			t.Errorf("args %v: expected an error", args)
		}
	}
}

func TestApplyPresetWithExplicitOverrides(t *testing.T) {
	preset, ok := train.GetPreset("small")
	if !ok {
		t.Fatalf("small preset not found")
	}

	// Case 1: No overrides
	o := parseTestFlags(t)
	o.applyPreset(preset)
	if o.stateDim != preset.Dim || o.lr != preset.LR || o.epochs != preset.Epochs {
		t.Fatalf("Expected preset defaults, got dim=%d, lr=%f, epochs=%d", o.stateDim, o.lr, o.epochs)
	}

	// Case 2: User explicitly passed -epochs 25 -lr 0.001
	o = parseTestFlags(t, "-epochs", "25", "-lr", "0.001")
	o.applyPreset(preset)
	if o.epochs != 25 {
		t.Fatalf("Expected epochs override to 25, got %d", o.epochs)
	}
	if o.lr != 0.001 || o.baseLR() != 0.001 {
		t.Fatalf("Expected lr override to 0.001, got %f (base %f)", o.lr, o.baseLR())
	}
	if o.stateDim != preset.Dim {
		t.Fatalf("Expected dim to remain preset default %d, got %d", preset.Dim, o.stateDim)
	}

	// Case 3: User explicitly passed -state-dim 128 -hidden-dim 256
	o = parseTestFlags(t, "-state-dim", "128", "-hidden-dim", "256")
	o.applyPreset(preset)
	if o.stateDim != 128 || o.hidden != 256 {
		t.Fatalf("Expected state-dim=128, hidden-dim=256, got dim=%d, hidden=%d", o.stateDim, o.hidden)
	}
}

func TestLearningRateResolution(t *testing.T) {
	o := parseTestFlags(t, "-opt", "muon")
	if got := o.baseLR(); got != train.DefaultMuonLR {
		t.Fatalf("no preset, no -lr: got %g, want optimizer default %g", got, train.DefaultMuonLR)
	}
	if got, want := o.fineTuneLR(train.SFTLRFraction), train.DefaultMuonLR*train.SFTLRFraction; got != want {
		t.Fatalf("fine-tune LR: got %g, want %g", got, want)
	}
	o = parseTestFlags(t, "-lr", "0.5")
	if o.fineTuneLR(train.SFTLRFraction) != 0.5 {
		t.Fatalf("explicit -lr must not be scaled for fine-tuning")
	}
}
