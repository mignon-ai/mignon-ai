package main

import (
	"fmt"

	"github.com/mignon-ai/mignon-ai/internal/ca"
	"github.com/mignon-ai/mignon-ai/internal/checkpoint"
	"github.com/mignon-ai/mignon-ai/pkg/model"
)

// modelMetadata describes m so the checkpoint can be rebuilt without command-line flags.
func modelMetadata(m *ca.Model, contextSize int, task string) checkpoint.Metadata {
	first := m.BlockConfigs()[0]
	return checkpoint.Metadata{
		Blocks:       m.NumBlocks(),
		StateSize:    m.StateSize(),
		HiddenSize:   m.HiddenSize(),
		Steps:        first.Steps,
		DilationBase: first.DilationBase,
		ContextSize:  contextSize,
		Task:         task,
		TiedEmbed:    m.TiedEmbed(),
		AttnHeads:    m.Config().AttnHeads,
		AttnWindow:   m.Config().AttnWindow,
	}
}

// quantizeOptions holds the flags of 'mignon quantize'.
type quantizeOptions struct {
	modelFlags
	save string
}

// runQuantize implements 'mignon quantize': convert a full-precision model to INT8 and save it.
func runQuantize(args []string) {
	fs := newFlagSet("quantize")
	o := &quantizeOptions{}
	o.register(fs, "Full-precision model to quantize (required)")
	fs.StringVar(&o.save, "save", "", "Path to save the INT8 model (required)")
	if _, err := parse(fs, args); err != nil || o.save == "" || o.load == "" {
		fatalf("Error: -load and -save are required.\n")
	}
	fmt.Printf("Loading model from %s...\n", o.load)
	m, err := model.Load(o.load)
	if err != nil {
		fatalf("Error reading checkpoint: %v\n", err)
	}
	if m.Quantized() {
		fatalf("Error: %s is not a full-precision model.\n", o.load)
	}
	fmt.Println("Quantizing model to INT8 (symmetric affine scaling)...")
	q, err := m.Quantize()
	if err != nil {
		fatalf("Error: %v\n", err)
	}
	fmt.Printf("Saving INT8 quantized model to %s...\n", o.save)
	if err := q.Save(o.save); err != nil {
		fatalf("Error saving quantized model: %v\n", err)
	}
	fmt.Println("Quantized model saved successfully.")
}

// runConvert implements 'mignon convert': load a model file in any supported format and save it in the
// binary one.
func runConvert(args []string) {
	fs := newFlagSet("convert")
	o := &quantizeOptions{}
	o.register(fs, "Model file to convert (required)")
	fs.StringVar(&o.save, "save", "", "Path of the binary model file (required)")
	if _, err := parse(fs, args); err != nil || o.save == "" || o.load == "" {
		fatalf("Error: -load and -save are required.\n")
	}
	m, err := model.Load(o.load)
	if err != nil {
		fatalf("Error reading %s: %v\n", o.load, err)
	}
	if err := m.Save(o.save); err != nil {
		fatalf("Error saving %s: %v\n", o.save, err)
	}
	fmt.Printf("Saved %s as a binary model file.\n", o.save)
}
