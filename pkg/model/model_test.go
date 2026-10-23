package model_test

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"github.com/mignon-ai/mignon-ai/internal/handle"
	"github.com/mignon-ai/mignon-ai/internal/quant"
	"github.com/mignon-ai/mignon-ai/pkg/data"
	"github.com/mignon-ai/mignon-ai/pkg/model"
	"github.com/mignon-ai/mignon-ai/pkg/train"
)

func TestDecodeRejectsBadFiles(t *testing.T) {
	for name, in := range map[string]string{
		"not json":     `nope`,
		"no tokenizer": `{"weights":[1,2,3],"metadata":{"blocks":1,"state_size":4,"hidden_size":4,"steps":1}}`,
		"no metadata":  `{"weights":[1],"vocab":["a"]}`,
	} {
		if _, err := model.Decode([]byte(in)); err == nil {
			t.Errorf("%s: accepted", name)
		}
	}
}

func TestSetExtraValidatesJSON(t *testing.T) {
	var m model.Model
	if err := m.SetExtra(json.RawMessage(`{broken`)); err == nil {
		t.Fatal("invalid JSON accepted")
	}
}

func TestBinarySmallerThanJSONAndOldJSONStillLoads(t *testing.T) {
	cfg, err := train.PresetConfig("micro")
	if err != nil {
		t.Fatal(err)
	}
	cfg.Epochs, cfg.Threads = 2, 2
	pairs := []data.Pair{{Prompt: "a b", Response: "c"}, {Prompt: "d e", Response: "f"}, {Prompt: "g h", Response: "i"},
		{Prompt: "j k", Response: "l"}, {Prompt: "m n", Response: "o"}, {Prompt: "p q", Response: "r"},
		{Prompt: "s t", Response: "u"}, {Prompt: "v w", Response: "x"}, {Prompt: "y z", Response: "a"}, {Prompt: "b c", Response: "d"}}
	m, _, err := train.FineTune(context.Background(), pairs, train.Options{Config: cfg, Seed: 1})
	if err != nil {
		t.Fatal(err)
	}
	q, _ := m.Quantize()
	dir := t.TempDir()
	for name, mm := range map[string]*model.Model{"full": m, "int8": q} {
		bin := filepath.Join(dir, name+".bin")
		if err := mm.Save(bin); err != nil {
			t.Fatal(err)
		}
		loaded, err := model.Load(bin)
		if err != nil || loaded.Quantized() != mm.Quantized() || loaded.Info().VocabSize != mm.Info().VocabSize {
			t.Fatalf("%s: %v %+v", name, err, loaded)
		}
	}
	// a JSON file written by the previous version
	old := filepath.Join(dir, "old.json")
	if err := quant.Save(handle.Of(q).Int8, handle.Of(q).Tok, handle.Of(q).Meta, old); err != nil {
		t.Fatal(err)
	}
	if loaded, err := model.Load(old); err != nil || !loaded.Quantized() {
		t.Fatalf("old JSON model: %v", err)
	}
	st := func(p string) int64 { i, _ := os.Stat(p); return i.Size() }
	if st(filepath.Join(dir, "int8.bin")) >= st(old)/3 {
		t.Errorf("binary %d bytes vs JSON %d", st(filepath.Join(dir, "int8.bin")), st(old))
	}
}
