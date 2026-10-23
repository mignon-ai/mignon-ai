package ca

import (
	"reflect"
	"testing"
)

func TestConfigZeroValuesSelectDefaults(t *testing.T) {
	const vocab, state, hidden = 12, 8, 10
	implicit := NewLayout(Config{VocabSize: vocab, StateSize: state, HiddenSize: hidden})
	explicit := NewLayout(Config{VocabSize: vocab, StateSize: state, HiddenSize: hidden,
		Blocks: DefaultTwoBlockConfig()})
	if !reflect.DeepEqual(implicit, explicit) {
		t.Fatalf("zero-valued config fields should equal explicit defaults:\n%+v\n%+v", implicit, explicit)
	}
}

func TestModelConfigRoundTrip(t *testing.T) {
	cfg := Config{VocabSize: 12, StateSize: 8, HiddenSize: 10, Blocks: BuildBlockConfigs(2, 3, 2), TiedEmbed: true,
		AttnHeads: 2, AttnWindow: 4}
	m := New(cfg)
	if got := m.Config(); !reflect.DeepEqual(got, cfg) {
		t.Fatalf("Config() = %+v, want %+v", got, cfg)
	}
	if rebuilt := New(m.Config()); rebuilt.NumParams() != m.NumParams() {
		t.Fatalf("model rebuilt from Config() has %d params, want %d", rebuilt.NumParams(), m.NumParams())
	}
}
