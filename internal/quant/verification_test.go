package quant

import (
	"os"
	"testing"

	"github.com/mignon-ai/mignon-ai/internal/ca"
	"github.com/mignon-ai/mignon-ai/internal/checkpoint"
	"github.com/mignon-ai/mignon-ai/internal/tokenizer"
)

// TestQuantizedMetadataPreservesStepsAndDilation verifies steps and dilationBase persist through checkpoint
func TestQuantizedMetadataPreservesStepsAndDilation(t *testing.T) {
	configs := []ca.BlockConfig{{Steps: 4, DilationBase: 2, InitialDilation: 1}}
	model := ca.New(ca.Config{VocabSize: 16, StateSize: 16, HiddenSize: 24, Blocks: configs, TiedEmbed: true})

	q := Quantize(model, 64)
	meta := checkpoint.Metadata{
		Blocks:       model.NumBlocks(),
		Steps:        configs[0].Steps,
		DilationBase: configs[0].DilationBase,
		ContextSize:  64,
	}

	tmpFile := "/tmp/peri_test_quant_meta.json"
	defer os.Remove(tmpFile)
	defer os.Remove(tmpFile + ".bak")

	tok := tokenizer.New()
	tok.Train("hello world test", 32)

	err := Save(q, tok, meta, tmpFile)
	if err != nil {
		t.Fatalf("Failed to save quantized model: %v", err)
	}

	loadedQ, _, loadedMeta, err := Load(tmpFile)
	if err != nil {
		t.Fatalf("Failed to load quantized model: %v", err)
	}

	if loadedMeta.Steps != 4 {
		t.Fatalf("Metadata Steps not preserved: got %d, want 4", loadedMeta.Steps)
	}
	if loadedMeta.DilationBase != 2 {
		t.Fatalf("Metadata DilationBase not preserved: got %d, want 2", loadedMeta.DilationBase)
	}
	if len(loadedQ.blocks) != 1 || loadedQ.blocks[0].Steps != 4 || loadedQ.blocks[0].DilationBase != 2 {
		t.Fatalf("Block fields not preserved: %+v", loadedQ.blocks[0])
	}
}
