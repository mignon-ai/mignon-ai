package train

import "testing"

func TestPresetsLookup(t *testing.T) {
	// Exact parameter counts of the full model (attention included); they are the numbers in the
	// README, so a changed architecture must update both.
	cases := []struct {
		name   string
		params int
	}{
		{"micro", 30432},
		{"small", 218944},
		{"base", 538080},
		{"medium", 954496},
		{"large", 1946816},
		{"pro", 2572000},
		{"xl", 4080416},
	}
	for _, tc := range cases {
		preset, ok := GetPreset(tc.name)
		if !ok {
			t.Fatalf("Expected preset %q to be found", tc.name)
		}
		if preset.Dim <= 0 || preset.Hidden <= 0 || preset.Vocab <= 0 || preset.Context <= 0 {
			t.Fatalf("Preset %q has invalid dimensions: %+v", tc.name, preset)
		}
		cfg := DefaultConfig()
		cfg.applyPreset(preset)
		if err := cfg.Validate(); err != nil {
			t.Fatalf("Preset %q is invalid: %v", tc.name, err)
		}
		if params := cfg.BuildModel(preset.Vocab, 0).NumParams(); params != tc.params {
			t.Fatalf("Preset %q has %d parameters, want %d", tc.name, params, tc.params)
		}
	}

	if _, ok := GetPreset("nonexistent"); ok {
		t.Fatalf("Expected nonexistent preset to return false")
	}
}

// TestPresetLearningRates checks that a preset without a measured Muon LR falls back to the default.
func TestPresetLearningRates(t *testing.T) {
	if (Preset{}).LearningRate("muon") != DefaultMuonLR {
		t.Error("a preset without measurements must fall back to the optimizer default")
	}
}
