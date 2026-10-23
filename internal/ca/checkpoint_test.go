package ca

import (
	"fmt"
	"math"
	"math/rand"
	"os"
	"testing"

	"github.com/mignon-ai/mignon-ai/internal/checkpoint"
	"github.com/mignon-ai/mignon-ai/internal/tokenizer"
)

func TestBuildBlockConfigsAndBackupProtection(t *testing.T) {
	// 1. Verify BuildBlockConfigs
	cfg1 := BuildBlockConfigs(1, 5, 2)
	if len(cfg1) != 1 || cfg1[0].Steps != 5 || cfg1[0].InitialDilation != 1 {
		t.Fatalf("Unexpected 1-block config: %+v", cfg1)
	}

	cfg2 := BuildBlockConfigs(2, 3, 2)
	if len(cfg2) != 2 || cfg2[0].InitialDilation != 1 || cfg2[1].InitialDilation != 4 {
		t.Fatalf("Unexpected 2-block config: %+v", cfg2)
	}

	cfg3 := BuildBlockConfigs(3, 3, 2)
	if len(cfg3) != 3 {
		t.Fatalf("Expected 3 blocks, got %d", len(cfg3))
	}

	// 2. Verify .bak overwrite protection
	tmpFile := "test_backup_protection.json"
	bakFile := tmpFile + ".bak"
	defer os.Remove(tmpFile)
	defer os.Remove(bakFile)

	initialContent := []byte(`{"initial": "checkpoint"}`)
	if err := os.WriteFile(tmpFile, initialContent, 0644); err != nil {
		t.Fatalf("Failed to write initial dummy file: %v", err)
	}

	// Test checkpoint.BackupExisting directly
	if err := checkpoint.BackupExisting(tmpFile); err != nil {
		t.Fatalf("checkpoint.BackupExisting failed: %v", err)
	}

	bakData, err := os.ReadFile(bakFile)
	if err != nil {
		t.Fatalf("Failed to read backup file %s: %v", bakFile, err)
	}
	if string(bakData) != string(initialContent) {
		t.Fatalf("Backup data mismatch: got %s, want %s", string(bakData), string(initialContent))
	}
}

func TestFullPrecisionMultiBlockSaveLoad(t *testing.T) {
	rand.Seed(42)
	vocabSize := 32
	stateSize := 16
	hiddenSize := 24
	configs := []BlockConfig{
		{Steps: 3, DilationBase: 2, InitialDilation: 1},
		{Steps: 2, DilationBase: 2, InitialDilation: 2},
	}

	model := New(Config{VocabSize: vocabSize, StateSize: stateSize, HiddenSize: hiddenSize, Blocks: configs, TiedEmbed: true})
	context := []int{1, 4, 9, 15}
	origOut := model.Forward(context)

	tok := tokenizer.New()
	tok.Vocab = make([]string, vocabSize)
	for i := 0; i < vocabSize; i++ {
		tok.Vocab[i] = fmt.Sprintf("t%d", i)
		tok.TokenToID[tok.Vocab[i]] = i
	}

	meta := checkpoint.Metadata{
		Blocks:       len(configs),
		StateSize:    stateSize,
		HiddenSize:   hiddenSize,
		Steps:        3,
		DilationBase: 2,
		ContextSize:  64,
		TiedEmbed:    true,
	}

	tmpFile := "test_full_precision_mb.json"
	defer os.Remove(tmpFile)
	defer os.Remove(tmpFile + ".bak")

	err := checkpoint.Save(model, tok, meta, tmpFile)
	if err != nil {
		t.Fatalf("Failed to save full-precision BPE checkpoint: %v", err)
	}

	loadedMeta, loadedTok, weights, err := checkpoint.Load(tmpFile)
	if err != nil {
		t.Fatalf("Failed to load checkpoint: %v", err)
	}
	if loadedMeta.Blocks != 2 {
		t.Fatalf("Expected Blocks=2 in metadata, got %d", loadedMeta.Blocks)
	}
	if loadedTok == nil || loadedTok.VocabSize() != vocabSize {
		t.Fatalf("Tokenizer vocab size mismatch")
	}

	// Reconstruct MultiBlock model using loaded metadata
	reloadedModel := New(Config{VocabSize: loadedTok.VocabSize(), StateSize: loadedMeta.StateSize, HiddenSize: loadedMeta.HiddenSize, Blocks: configs, TiedEmbed: loadedMeta.TiedEmbed})
	if reloadedModel.NumParams() != len(weights) {
		t.Fatalf("Weight count mismatch: params=%d vs weights=%d", reloadedModel.NumParams(), len(weights))
	}
	reloadedModel.SetWeights(weights)

	reloadedOut := reloadedModel.Forward(context)
	for i := range origOut {
		if math.Abs(origOut[i]-reloadedOut[i]) > 1e-6 {
			t.Fatalf("Mismatch at logit %d: orig=%f vs reloaded=%f", i, origOut[i], reloadedOut[i])
		}
	}
}

func TestAuditRobustness(t *testing.T) {

	// 2. Tokenizer streaming Encode with multi-byte runes and whitespace
	tok := tokenizer.New()
	corpus := "Hello, 世界! def calculate_sum(x): return x + 42"
	tok.Train(corpus, 64)

	enc := tok.Encode(corpus)
	dec := tok.Decode(enc)
	if dec != corpus {
		t.Fatalf("Round-trip mismatch with UTF-8 runes: got %q, want %q", dec, corpus)
	}

	// Empty string encode
	if len(tok.Encode("")) != 0 {
		t.Fatalf("Expected empty slice for empty string")
	}
}
