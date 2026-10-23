package main

import (
	"flag"
	"fmt"
	"slices"
	"strings"

	"github.com/mignon-ai/mignon-ai/internal/train"
)

// trainOptions holds the flags of 'mignon train'. Preset handling overwrites unset architecture fields.
type trainOptions struct {
	modelFlags // -load continues training from a checkpoint
	save       string

	// Data
	textDir, sftData string
	maxSeqs, stride  int
	noCache          bool
	tokFrom          string

	// Architecture
	preset           string
	steps, dilation  int
	context          int
	stateDim, hidden int
	vocab            int
	tied             bool
	blocks           int
	attnHeads        int
	attnWindow       int

	// Optimization
	threads       int
	seed          int64
	epochs, batch int
	lr, decay     float64
	optimizer     string

	explicit   map[string]bool // flags set on the command line
	presetUsed train.Preset    // zero when no -preset was given
}

// parseTrainFlags registers the train flags on fs and parses args.
func parseTrainFlags(fs *flag.FlagSet, args []string) (*trainOptions, error) {
	o := &trainOptions{}
	def := train.DefaultConfig()
	o.register(fs, "Saved model to continue training or fine-tune (e.g., model.json)")
	fs.StringVar(&o.save, "save", "", "Path to save the best model to during training")

	fs.StringVar(&o.textDir, "text", "", "Directory containing .txt files for raw text training")
	fs.StringVar(&o.sftData, "sft", "", "Path to a JSON file of prompt/response (or instruction/output) pairs to fine-tune on with the chat template")
	fs.IntVar(&o.maxSeqs, "max-seqs", 0, "Max training sequences to train on (0 = all)")
	fs.IntVar(&o.stride, "stride", 0, "Stride between consecutive context windows (0 = auto context)")
	fs.BoolVar(&o.noCache, "no-cache", false, "Disable the on-disk tokenizer/dataset cache for -text")
	fs.StringVar(&o.tokFrom, "tok-from", "", "Train a new -text model with the tokenizer of this checkpoint instead of learning one (keeps tokenizers equal across curriculum experiments)")

	fs.StringVar(&o.preset, "preset", "", "Model configuration preset: "+strings.Join(train.PresetNames(), ", "))
	fs.IntVar(&o.steps, "steps", def.Steps, "Number of CA communication steps")
	fs.IntVar(&o.dilation, "dilation", def.Dilation, "Cellular Automata dilation base (2 for geometric 1,2,4,8...)")
	fs.IntVar(&o.context, "context", def.Context, "Context window size in tokens")
	fs.IntVar(&o.stateDim, "state-dim", def.StateDim, "State vector dimension for CA cells")
	fs.IntVar(&o.hidden, "hidden-dim", def.HiddenDim, "Hidden layer size for CA MLP rule")
	fs.IntVar(&o.vocab, "vocab", def.Vocab, "Target vocabulary size for BPE tokenizer")
	fs.BoolVar(&o.tied, "tied", def.Tied, "Use weight-tied embeddings")
	fs.IntVar(&o.blocks, "blocks", def.Blocks, "Number of hierarchical CA blocks")
	fs.IntVar(&o.attnHeads, "attn-heads", def.AttnHeads, "Heads of a causal attention sublayer after each CA block for new models (0 = none; must divide -state-dim)")
	fs.IntVar(&o.attnWindow, "attn-window", def.AttnWindow, "Attention span in tokens for -attn-heads (0 = full causal)")

	fs.IntVar(&o.epochs, "epochs", def.Epochs, "Number of training epochs")
	fs.IntVar(&o.batch, "batch", def.Batch, "Mini-batch size for parallel training")
	fs.Float64Var(&o.lr, "lr", train.DefaultAdamLR, fmt.Sprintf("Learning rate (%g for Adam, %g for Muon)", train.DefaultAdamLR, train.DefaultMuonLR))
	fs.Float64Var(&o.decay, "decay", def.WeightDecay, "Weight decay (L2 regularization) for AdamW")
	fs.StringVar(&o.optimizer, "opt", def.Optimizer, "Optimizer: "+strings.Join(train.Optimizers, " or "))
	fs.IntVar(&o.threads, "threads", 0, "Number of CPU worker threads (0 = auto, leaves headroom for OS/UI)")
	fs.Int64Var(&o.seed, "seed", 0, "Random seed for weight init and data order (0 = nondeterministic)")

	var err error
	if o.explicit, err = parse(fs, args); err != nil {
		return nil, err
	}
	if (o.textDir == "") == (o.sftData == "") {
		return nil, fmt.Errorf("exactly one of -text or -sft is required")
	}
	if o.tokFrom != "" && (o.load != "" || o.textDir == "") {
		return nil, fmt.Errorf("-tok-from only applies to new -text models; -load already brings its tokenizer")
	}
	if !slices.Contains(train.Optimizers, o.optimizer) {
		return nil, fmt.Errorf("unknown -opt %q (want one of %v)", o.optimizer, train.Optimizers)
	}
	return o, nil
}

// applyPreset copies preset values into every architecture and training flag not set on the command line.
func (o *trainOptions) applyPreset(preset train.Preset) {
	fmt.Printf("Applying '%s' model preset...\n", preset.Name)
	o.presetUsed = preset
	setInt := func(name string, dst *int, v int) {
		if !o.explicit[name] {
			*dst = v
		}
	}
	setFloat := func(name string, dst *float64, v float64) {
		if !o.explicit[name] {
			*dst = v
		}
	}
	setInt("blocks", &o.blocks, preset.Blocks)
	setInt("state-dim", &o.stateDim, preset.Dim)
	setInt("hidden-dim", &o.hidden, preset.Hidden)
	setInt("steps", &o.steps, preset.Steps)
	setInt("dilation", &o.dilation, preset.Dilation)
	setInt("vocab", &o.vocab, preset.Vocab)
	setInt("context", &o.context, preset.Context)
	if !o.explicit["tied"] {
		o.tied = preset.Tied
	}
	setInt("batch", &o.batch, preset.Batch)
	setFloat("lr", &o.lr, preset.LR)
	setFloat("decay", &o.decay, preset.WeightDecay)
	setInt("epochs", &o.epochs, preset.Epochs)
	setInt("attn-heads", &o.attnHeads, preset.AttnHeads)
	setInt("attn-window", &o.attnWindow, preset.AttnWindow)
}

// baseLR is the learning rate before fine-tuning scaling: the explicit -lr, or the preset/optimizer default.
func (o *trainOptions) baseLR() float64 {
	if o.explicit["lr"] {
		return o.lr
	}
	return o.presetUsed.LearningRate(o.optimizer) // zero preset falls back to the optimizer default
}

// fineTuneLR scales the optimizer default by fraction unless -lr was given explicitly.
func (o *trainOptions) fineTuneLR(fraction float64) float64 {
	if o.explicit["lr"] {
		return o.lr
	}
	return train.DefaultLR(o.optimizer) * fraction
}

// architecture returns the architecture flags as a training config.
func (o *trainOptions) architecture() train.Config {
	return train.Config{
		Blocks:     max(o.blocks, minBlocks),
		StateDim:   o.stateDim,
		HiddenDim:  o.hidden,
		Steps:      o.steps,
		Dilation:   o.dilation,
		Tied:       o.tied,
		AttnHeads:  o.attnHeads,
		AttnWindow: o.attnWindow,
	}
}
