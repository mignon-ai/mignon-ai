package train

import (
	"encoding/json"
	"math"
	"reflect"
	"testing"

	"github.com/mignon-ai/mignon-ai/internal/optim"
	"github.com/mignon-ai/mignon-ai/internal/tokenizer"
)

func TestResolveExperimentConfigLayering(t *testing.T) {
	cfg, err := ResolveConfig(nil)
	if err != nil {
		t.Fatal(err)
	}
	if want := DefaultConfig(); cfg != want {
		t.Fatalf("no layers should give defaults:\n got %+v\nwant %+v", cfg, want)
	}

	micro, _ := GetPreset("micro")
	cfg, err = ResolveConfig(json.RawMessage(`{"preset":"micro","attn_heads":2,"opt":"adam"}`), json.RawMessage(`{"state_dim":48}`))
	if err != nil {
		t.Fatal(err)
	}
	if cfg.HiddenDim != micro.Hidden || cfg.Steps != micro.Steps {
		t.Errorf("preset fields not applied: %+v", cfg)
	}
	if cfg.StateDim != 48 {
		t.Errorf("later layer should override preset: state_dim=%d", cfg.StateDim)
	}
	if cfg.AttnHeads != 2 {
		t.Errorf("explicit key lost: attn_heads=%d", cfg.AttnHeads)
	}
	if cfg.ResolvedLR() != micro.LR {
		t.Errorf("adam should use preset LR %g, got %g", micro.LR, cfg.ResolvedLR())
	}
}

func TestResolvedLRPerOptimizer(t *testing.T) {
	cases := []struct {
		layer string
		want  float64
	}{
		{`{}`, DefaultLR(DefaultConfig().Optimizer)},
		{`{"opt":"adam"}`, DefaultAdamLR},
		{`{"preset":"micro","opt":"adam"}`, Presets["micro"].LR},
		{`{"opt":"muon"}`, DefaultMuonLR},
		{`{"preset":"micro","opt":"muon"}`, DefaultMuonLR}, // no Muon LR measured for micro
		{`{"preset":"small","opt":"muon"}`, Presets["small"].MuonLR},
		{`{"opt":"muon","lr":0.04}`, 0.04},
	}
	for _, c := range cases {
		cfg, err := ResolveConfig(json.RawMessage(c.layer))
		if err != nil {
			t.Fatalf("%s: %v", c.layer, err)
		}
		if got := cfg.ResolvedLR(); got != c.want {
			t.Errorf("%s: ResolvedLR=%g want %g", c.layer, got, c.want)
		}
	}
}

func TestResolveExperimentConfigRejectsInvalid(t *testing.T) {
	for _, layer := range []string{
		`{"preset":"nope"}`,
		`{"latch":"nope"}`,
		`{"opt":"nope"}`,
		`{"steps":0}`,
		`{"steps":"five"}`,
		`{"lr_schedule":"step"}`,
		`{"lr_floor":1.5}`,
		`{"embed_lr_scale":0}`,
	} {
		if _, err := ResolveConfig(json.RawMessage(layer)); err == nil {
			t.Errorf("%s: expected error", layer)
		}
	}
}

func TestScheduleKeysReachTrainOptions(t *testing.T) {
	def, err := ResolveConfig()
	if err != nil {
		t.Fatal(err)
	}
	if opts := def.Options(1, true); opts.Schedule != optim.DefaultSchedule() || opts.EmbedLRScale != optim.DefaultMuonEmbedLRScale {
		t.Fatalf("defaults: got schedule %+v, embed scale %g", opts.Schedule, opts.EmbedLRScale)
	}
	cfg, err := ResolveConfig(json.RawMessage(`{"lr_schedule":"linear","lr_floor":0,"embed_lr_scale":0.25}`))
	if err != nil {
		t.Fatal(err)
	}
	opts := cfg.Options(1, true)
	if want := (optim.Schedule{Shape: optim.ScheduleLinear, Floor: 0}); opts.Schedule != want || opts.EmbedLRScale != 0.25 {
		t.Fatalf("got schedule %+v, embed scale %g", opts.Schedule, opts.EmbedLRScale)
	}
}

func TestSeededInitIsDeterministic(t *testing.T) {
	cfg := DefaultConfig()
	cfg.StateDim, cfg.HiddenDim, cfg.Steps = 16, 24, 2
	a := cfg.BuildModel(32, 99).Weights()
	b := cfg.BuildModel(32, 99).Weights()
	c := cfg.BuildModel(32, 100).Weights()
	if !reflect.DeepEqual(a, b) {
		t.Fatal("same seed must give identical weights")
	}
	if reflect.DeepEqual(a, c) {
		t.Fatal("different seeds should give different weights")
	}
}

func TestTokenizerTrainingIsDeterministic(t *testing.T) {
	text := "the cat sat on the mat. the dog sat on the log. a cat and a dog met on a mat by a log."
	var first []tokenizer.Merge
	for i := 0; i < 5; i++ {
		tok := tokenizer.New()
		tok.Train(text, 80)
		if i == 0 {
			first = tok.Merges
			continue
		}
		if !reflect.DeepEqual(first, tok.Merges) {
			t.Fatal("BPE merges differ between identical training runs")
		}
	}
}

// TestFineTuneLRScalesWithOptimizer keeps fine-tuning rates proportional to each optimizer's
// pretraining rate, and pins the Adam value.
func TestFineTuneLRScalesWithOptimizer(t *testing.T) {
	const tol = 1e-12
	if got := DefaultLR("adam") * SFTLRFraction; math.Abs(got-0.0032) > tol {
		t.Errorf("adam SFT LR %g, want 0.0032", got)
	}
	if DefaultLR("muon")*SFTLRFraction <= DefaultLR("adam")*SFTLRFraction {
		t.Error("muon fine-tune LR should follow its larger pretraining LR")
	}
}
