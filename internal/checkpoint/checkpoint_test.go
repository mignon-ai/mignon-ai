package checkpoint

import (
	"bytes"
	"encoding/json"
	"path/filepath"
	"reflect"
	"testing"

	"github.com/mignon-ai/mignon-ai/internal/tokenizer"
)

type fakeModel []float32

func (m fakeModel) Weights() []float32 { return m }

func TestSaveLoadRoundTripAndBackup(t *testing.T) {
	tok := tokenizer.New()
	tok.Train("checkpoint round trip text text text", 40)
	path := filepath.Join(t.TempDir(), "model.json")
	meta := Metadata{Blocks: 2, StateSize: 8, DilationBase: 3, Extra: json.RawMessage(`{"task":"x"}`)}
	first := fakeModel{0.5, -1.25, 3}

	if err := Save(first, tok, meta, path); err != nil {
		t.Fatal(err)
	}
	gotMeta, gotTok, gotWeights, err := Load(path)
	if err != nil {
		t.Fatal(err)
	}
	// Save always records the tokenizer version
	meta.TokenizerVersion = tokenizer.FormatVersion
	var compact bytes.Buffer
	if err := json.Compact(&compact, gotMeta.Extra); err != nil {
		t.Fatal(err)
	}
	gotMeta.Extra = json.RawMessage(compact.Bytes()) // Save indents the file, the value is the same
	if !reflect.DeepEqual(*gotMeta, meta) {
		t.Fatalf("metadata = %+v, want %+v", *gotMeta, meta)
	}
	if !reflect.DeepEqual(gotWeights, []float32(first)) {
		t.Fatalf("weights = %v, want %v", gotWeights, first)
	}
	if !reflect.DeepEqual(gotTok.Vocab, tok.Vocab) || !reflect.DeepEqual(gotTok.Merges, tok.Merges) {
		t.Fatal("tokenizer did not round-trip")
	}

	// Overwriting keeps the previous checkpoint as a .bak file.
	if err := Save(fakeModel{9}, tok, meta, path); err != nil {
		t.Fatal(err)
	}
	if _, _, bakWeights, err := Load(path + ".bak"); err != nil || !reflect.DeepEqual(bakWeights, []float32(first)) {
		t.Fatalf("backup weights = %v (err %v), want %v", bakWeights, err, first)
	}
}

func TestBinaryRoundTripAndJSONStillLoads(t *testing.T) {
	tok := tokenizer.New()
	tok.Train("checkpoint round trip text text text", 40)
	meta := Metadata{Blocks: 2, StateSize: 8, DilationBase: 3, Task: "chat", Extra: json.RawMessage(`{"task":"x"}`)}
	weights := fakeModel{0.5, -1.25, 3, 1e-30}
	dir := t.TempDir()

	bin := filepath.Join(dir, "m.bin")
	if err := SaveBinary(weights, tok, meta, bin); err != nil {
		t.Fatal(err)
	}
	gotMeta, gotTok, gotWeights, err := Load(bin)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(gotWeights, []float32(weights)) || gotMeta.Task != "chat" || gotMeta.Blocks != 2 ||
		string(gotMeta.Extra) != `{"task":"x"}` || gotMeta.TokenizerVersion != tokenizer.FormatVersion {
		t.Fatalf("binary round trip: %v %+v", gotWeights, gotMeta)
	}
	if !reflect.DeepEqual(gotTok.Vocab, tok.Vocab) || !reflect.DeepEqual(gotTok.Merges, tok.Merges) {
		t.Fatal("tokenizer did not round-trip")
	}

	js := filepath.Join(dir, "m.json")
	if err := Save(weights, tok, meta, js); err != nil {
		t.Fatal(err)
	}
	if _, _, w, err := Load(js); err != nil || !reflect.DeepEqual(w, []float32(weights)) {
		t.Fatalf("JSON file no longer loads: %v %v", w, err)
	}
}
