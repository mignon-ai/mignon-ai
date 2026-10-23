package train_test

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"path/filepath"
	"strings"
	"testing"

	"github.com/mignon-ai/mignon-ai/pkg/data"
	"github.com/mignon-ai/mignon-ai/pkg/infer"
	"github.com/mignon-ai/mignon-ai/pkg/model"
	"github.com/mignon-ai/mignon-ai/pkg/train"
)

func pairs() []data.Pair {
	var ps []data.Pair
	for _, w := range []string{"red", "green", "blue", "pink", "gray", "gold", "teal", "plum"} {
		for _, form := range []string{"paint it %s", "make it %s", "colour %s please", "%s"} {
			ps = append(ps, data.Pair{Prompt: fmt.Sprintf(form, w), Response: "color --name " + w})
		}
	}
	return ps
}

func microConfig(t *testing.T, epochs int) train.Config {
	t.Helper()
	cfg, err := train.PresetConfig("micro")
	if err != nil {
		t.Fatal(err)
	}
	cfg.Epochs, cfg.Threads = epochs, 2
	return cfg
}

func TestFineTuneSaveLoadGenerate(t *testing.T) {
	var epochs []train.Epoch
	best := 0
	m, res, err := train.FineTune(context.Background(), pairs(), train.Options{
		Config: microConfig(t, 30), Seed: 1, LR: 0.02,
		OnEpoch: func(e train.Epoch) { epochs = append(epochs, e) },
		OnBest:  func(*model.Model) { best++ },
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(epochs) != 30 || epochs[0].Epoch != 1 || best == 0 || res.BestEpoch == 0 {
		t.Fatalf("progress: %d epochs, %d best calls, result %+v", len(epochs), best, res)
	}
	if m.Task() != model.TaskChat {
		t.Fatalf("task %q", m.Task())
	}

	if err := m.SetExtra(json.RawMessage(`{"task":"colors"}`)); err != nil {
		t.Fatal(err)
	}
	q, err := m.Quantize()
	if err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(t.TempDir(), "m.json")
	if err := q.Save(path); err != nil {
		t.Fatal(err)
	}
	loaded, err := model.Load(path)
	if err != nil {
		t.Fatal(err)
	}
	var extra bytes.Buffer
	if err := json.Compact(&extra, loaded.Extra()); err != nil {
		t.Fatal(err)
	}
	if loaded.Info().Params != m.Info().Params || loaded.Info().Params == 0 {
		t.Fatalf("parameters: %d after quantizing, %d before", loaded.Info().Params, m.Info().Params)
	}
	if !loaded.Quantized() || extra.String() != `{"task":"colors"}` || loaded.ContextSize() != m.ContextSize() {
		t.Fatalf("loaded %+v extra %s", loaded.Info(), loaded.Extra())
	}
	out, err := infer.Generate(context.Background(), loaded, "paint it teal", infer.Options{})
	if err != nil {
		t.Fatal(err)
	}
	if !strings.HasPrefix(out.Text, "color --name") || len(out.Tokens) == 0 || out.MinProb() <= 0 || out.MinProb() > 1 {
		t.Fatalf("reply %+v", out)
	}
}

func TestFullPrecisionRoundTripKeepsWeights(t *testing.T) {
	m, _, err := train.FineTune(context.Background(), pairs(), train.Options{Config: microConfig(t, 2), Seed: 1})
	if err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(t.TempDir(), "full.json")
	if err := m.Save(path); err != nil {
		t.Fatal(err)
	}
	loaded, err := model.Load(path)
	if err != nil {
		t.Fatal(err)
	}
	if loaded.Quantized() || loaded.Info().Params != m.Info().Params {
		t.Fatalf("%+v vs %+v", loaded.Info(), m.Info())
	}
	a, _ := infer.Generate(context.Background(), m, "make it red", infer.Options{})
	b, _ := infer.Generate(context.Background(), loaded, "make it red", infer.Options{})
	if a.Text != b.Text {
		t.Fatalf("%q vs %q", a.Text, b.Text)
	}
}

func TestCancelStopsTrainingAndReturnsTheBestModel(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	m, res, err := train.FineTune(ctx, pairs(), train.Options{
		Config: microConfig(t, 50), Seed: 1,
		OnEpoch: func(e train.Epoch) {
			if e.Epoch == 2 {
				cancel()
			}
		},
	})
	if err != context.Canceled || !res.Canceled || m == nil {
		t.Fatalf("err %v result %+v model %v", err, res, m)
	}
}

func TestFineTuneContinuesABaseAndPretrainRunsOnText(t *testing.T) {
	cfg := microConfig(t, 2)
	docs := []string{strings.Repeat("the cat sat on the mat. a dog ran in the park. ", 80)}
	base, _, err := train.Pretrain(context.Background(), docs, train.Options{Config: cfg, Seed: 1})
	if err != nil {
		t.Fatal(err)
	}
	if base.Task() != model.TaskText {
		t.Fatalf("task %q", base.Task())
	}
	tuned, _, err := train.FineTune(context.Background(), pairs(), train.Options{Config: cfg, Base: base, Seed: 1})
	if err != nil {
		t.Fatal(err)
	}
	if tuned.Task() != model.TaskChat || tuned.Info().Blocks != base.Info().Blocks {
		t.Fatalf("%+v", tuned.Info())
	}
}

func TestBadInput(t *testing.T) {
	if _, _, err := train.FineTune(context.Background(), nil, train.Options{Config: microConfig(t, 1)}); err == nil {
		t.Fatal("no pairs accepted")
	}
	if _, err := train.PresetConfig("nope"); err == nil {
		t.Fatal("unknown preset accepted")
	}
}
