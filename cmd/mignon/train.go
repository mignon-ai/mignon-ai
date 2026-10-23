package main

import (
	"context"
	"fmt"
	"os"
	"runtime"
	"strings"

	"github.com/mignon-ai/mignon-ai/internal/ca"
	"github.com/mignon-ai/mignon-ai/internal/checkpoint"
	"github.com/mignon-ai/mignon-ai/internal/data"
	"github.com/mignon-ai/mignon-ai/internal/train"
	pkgdata "github.com/mignon-ai/mignon-ai/pkg/data"
	"github.com/mignon-ai/mignon-ai/pkg/model"
	pkgtrain "github.com/mignon-ai/mignon-ai/pkg/train"
)

// runTrain implements 'mignon train': pretrain, continue or fine-tune a model and save the best checkpoint.
func runTrain(args []string) {
	o, err := parseTrainFlags(newFlagSet("train"), args)
	if err != nil {
		fatalf("Error: %v\n", err)
	}
	if o.preset != "" {
		preset, ok := train.GetPreset(o.preset)
		if !ok {
			fatalf("Unknown preset '%s'. Available presets: %s\n", o.preset, strings.Join(train.PresetNames(), ", "))
		}
		o.applyPreset(preset)
	}

	workers := o.threads
	if workers <= 0 {
		workers = runtime.NumCPU()
	}
	workers = max(workers, 1)
	if o.sftData != "" {
		trainPairs(o, workers)
		return
	}
	s := &session{task: taskText, contextSize: o.context}
	if o.load != "" {
		loadSession(o.load, s)
	}
	trainText(o, s, workers)
}

// trainConfig is the training configuration the command-line flags describe.
func (o *trainOptions) trainConfig(threads int) train.Config {
	cfg := o.architecture()
	def := train.DefaultConfig()
	cfg.Vocab, cfg.Context = o.vocab, o.context
	cfg.Optimizer, cfg.Batch, cfg.WeightDecay, cfg.Epochs = o.optimizer, o.batch, o.decay, o.epochs
	cfg.Stride, cfg.MaxSeqs, cfg.Threads = o.stride, o.maxSeqs, threads
	cfg.LRSchedule, cfg.LRFloor, cfg.EmbedLRScale = def.LRSchedule, def.LRFloor, def.EmbedLRScale
	if o.explicit["lr"] {
		cfg.LR = o.lr
	}
	return cfg
}

// trainPairs fine-tunes (or trains from scratch) on the -sft file of prompt/response pairs, through the
// public training package.
func trainPairs(o *trainOptions, workers int) {
	fmt.Printf("Reading prompt/response pairs from: %s\n", o.sftData)
	pairs, err := pkgdata.LoadPairs(o.sftData)
	if err != nil {
		fatalf("Error reading pairs: %v\n", err)
	}
	fmt.Printf("Loaded %d prompt/response pairs.\n", len(pairs))
	var base *model.Model
	if o.load != "" {
		fmt.Printf("Loading model from %s...\n", o.load)
		if base, err = model.Load(o.load); err != nil {
			fatalf("Error reading checkpoint: %v\n", err)
		}
	}
	opts := pkgtrain.Options{Config: o.trainConfig(workers), Base: base, Seed: o.seed, Log: os.Stdout}
	if o.explicit["lr"] {
		opts.LR = o.lr
	}
	if o.save != "" {
		opts.OnBest = func(m *model.Model) {
			if err := m.Save(o.save); err == nil {
				fmt.Printf(" [Checkpoint: best model saved to %s]\n", o.save)
			}
		}
	}
	if _, _, err := pkgtrain.FineTune(context.Background(), pairs, opts); err != nil {
		fatalf("Error: %v\n", err)
	}
}

// trainText pretrains on a directory of raw text, or continues training a loaded model on it.
func trainText(o *trainOptions, s *session, workers int) {
	s.task = taskText
	fmt.Printf("Reading training data from directory: %s\n", o.textDir)
	docs, err := data.LoadTextCorpus(o.textDir, func(format string, args ...any) { fmt.Printf(format, args...) })
	if err != nil {
		fatalf("Error: %v\n", err)
	}

	stride := o.stride
	if stride <= 0 {
		stride = s.contextSize
		fmt.Printf("Using auto-stride: %d tokens (full non-overlapping context window for dense causal training)\n", stride)
	}

	var trainSet, valSet data.Sequences
	lr := o.baseLR()
	if s.model != nil && s.tok != nil {
		fmt.Printf("Continuing training on loaded model (vocab size: %d, params: %d)...\n", s.tok.VocabSize(), s.model.NumParams())
		trainSet, valSet = data.EncodeTrainVal(docs, s.contextSize, s.tok, stride, o.maxSeqs, data.DefaultValFraction)
		lr = o.fineTuneLR(train.SFTLRFraction)
	} else {
		if o.tokFrom != "" {
			s.tok = loadTokenizer(o.tokFrom)
			fmt.Printf("Using the tokenizer of %s (vocab size: %d).\n", o.tokFrom, s.tok.VocabSize())
			trainSet, valSet = data.EncodeTrainVal(docs, s.contextSize, s.tok, stride, o.maxSeqs, data.DefaultValFraction)
		} else if o.noCache {
			trainSet, valSet, s.tok = data.PrepareTrainVal(docs, s.contextSize, o.vocab, stride, o.maxSeqs, data.DefaultValFraction)
		} else {
			var cached bool
			trainSet, valSet, s.tok, cached = data.PrepareTrainValCached(docs, s.contextSize, o.vocab, stride, o.maxSeqs, data.DefaultValFraction)
			if cached {
				fmt.Println("Loaded tokenizer and encoded dataset from cache (use -no-cache to rebuild).")
			}
		}
		s.model = newModel(o, s.tok.VocabSize())
		fmt.Printf("Initialized Cellular Automata BPE model (blocks=%d, stateSize=%d, hiddenSize=%d, vocabSize=%d, tied=%v).\n",
			s.model.NumBlocks(), o.stateDim, o.hidden, s.tok.VocabSize(), o.tied)
	}
	runTraining(o, s, trainSet, valSet, lr, workers)
}

// newModel builds an untrained model from the architecture options, seeded by -seed.
func newModel(o *trainOptions, vocabSize int) *ca.Model {
	return o.architecture().BuildModel(vocabSize, o.seed)
}

// runTraining trains s.model, saving the best model to -save whenever validation improves.
func runTraining(o *trainOptions, s *session, trainSet, valSet data.Sequences, lr float64, workers int) {
	var onCheckpoint train.CheckpointFunc
	if o.save != "" && s.tok != nil {
		model, tok, meta := s.model, s.tok, modelMetadata(s.model, s.contextSize, s.task)
		onCheckpoint = func(epoch int, loss float64, weights []float32) {
			model.SetWeights(weights)
			if err := checkpoint.SaveBinary(model, tok, meta, o.save); err == nil {
				fmt.Printf(" [Checkpoint: best model saved to %s]\n", o.save)
			}
		}
	}
	train.Run(s.model, trainSet, valSet, train.Options{
		Epochs:       o.epochs,
		LR:           lr,
		Optimizer:    o.optimizer,
		BatchSize:    o.batch,
		WeightDecay:  o.decay,
		Threads:      workers,
		Seed:         o.seed,
		TokenBytes:   s.tok.TokenBytes(),
		OnCheckpoint: onCheckpoint,
	})
}
