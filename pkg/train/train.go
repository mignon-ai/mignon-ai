// Package train trains models: pretraining on raw text and fine-tuning on prompt/response pairs. It works on
// plain Go values (no files, no flags, no stdout unless asked), can be cancelled through a context, and
// reports progress through callbacks, so a program can run it as a job.
package train

import (
	"context"
	"encoding/json"
	"fmt"
	"io"

	"github.com/mignon-ai/mignon-ai/internal/ca"
	"github.com/mignon-ai/mignon-ai/internal/checkpoint"
	idata "github.com/mignon-ai/mignon-ai/internal/data"
	"github.com/mignon-ai/mignon-ai/internal/handle"
	"github.com/mignon-ai/mignon-ai/internal/tokenizer"
	itrain "github.com/mignon-ai/mignon-ai/internal/train"
	"github.com/mignon-ai/mignon-ai/pkg/data"
	"github.com/mignon-ai/mignon-ai/pkg/model"
)

// Config describes a model's architecture and training hyperparameters. Start from PresetConfig and change
// what you need.
type Config = itrain.Config

// Result summarizes a finished run. Losses are validation losses when the data was large enough to hold
// some back.
type Result = itrain.Result

// SFTLRFraction is the share of the optimizer's pretraining learning rate used for fine-tuning.
const SFTLRFraction = itrain.SFTLRFraction

// Epoch describes one completed epoch.
type Epoch = itrain.EpochStats

// PresetNames lists the model sizes: micro, small, base, medium, large, pro, xl.
func PresetNames() []string { return itrain.PresetNames() }

// PresetConfig returns the configuration of a model size with its tuned hyperparameters.
func PresetConfig(name string) (Config, error) {
	raw, _ := json.Marshal(map[string]string{"preset": name})
	return itrain.ResolveConfig(raw)
}

// Options controls Pretrain and FineTune.
type Options struct {
	// Config is the architecture of a new model and the hyperparameters of the run. Ignored for the
	// architecture when Base is set.
	Config Config
	// Base is the model to continue from. Nil trains a new model of Config's architecture. An INT8 base is
	// converted to full precision first, which does not restore what quantization lost.
	Base *model.Model
	// Seed makes weight initialization and data order reproducible. Zero is nondeterministic.
	Seed int64
	// LR overrides the learning rate. Zero uses the configured one for pretraining and SFTLRFraction of the
	// optimizer default for fine-tuning.
	LR float64
	// OnEpoch is called after every completed epoch.
	OnEpoch func(Epoch)
	// OnBest is called when validation loss improves, with the model holding the best weights so far. The
	// model is the live one being trained: save or copy it before returning (m.Save, m.Quantize).
	OnBest func(m *model.Model)
	// Log, if set, receives human-readable progress. Nil is silent.
	Log io.Writer
}

// Pretrain trains on raw text documents. With Base set it continues training that model on them.
func Pretrain(ctx context.Context, docs []string, o Options) (*model.Model, Result, error) {
	if len(docs) == 0 {
		return nil, Result{}, fmt.Errorf("no documents to train on")
	}
	if err := o.Config.Validate(); err != nil {
		return nil, Result{}, err
	}
	cfg := o.Config
	stride := cfg.Stride
	var net *ca.Model
	var tok *tokenizer.Tokenizer
	var trainSet, valSet idata.Sequences
	lr := cfg.ResolvedLR()
	contextSize := cfg.Context
	if o.Base != nil {
		s := handle.Of(o.Base.Dequantize())
		net, tok, contextSize = s.Full, s.Tok, o.Base.ContextSize()
		if stride <= 0 {
			stride = contextSize
		}
		trainSet, valSet = idata.EncodeTrainVal(docs, contextSize, tok, stride, cfg.MaxSeqs, idata.DefaultValFraction)
		lr = itrain.DefaultLR(cfg.Optimizer) * itrain.SFTLRFraction
	} else {
		if stride <= 0 {
			stride = contextSize
		}
		trainSet, valSet, tok = idata.PrepareTrainVal(docs, contextSize, cfg.Vocab, stride, cfg.MaxSeqs, idata.DefaultValFraction)
		net = cfg.BuildModel(tok.VocabSize(), o.Seed)
	}
	return run(ctx, o, net, tok, trainSet, valSet, lr, contextSize, model.TaskText)
}

// FineTune trains on prompt/response pairs with the chat template: the model learns to write the response
// to a prompt. Without Base it trains a new model whose tokenizer is learned from the pairs.
func FineTune(ctx context.Context, pairs []data.Pair, o Options) (*model.Model, Result, error) {
	if len(pairs) == 0 {
		return nil, Result{}, fmt.Errorf("no pairs to train on")
	}
	if err := o.Config.Validate(); err != nil {
		return nil, Result{}, err
	}
	cfg := o.Config
	qa := make([]idata.QAPair, len(pairs))
	for i, p := range pairs {
		qa[i] = idata.QAPair{Prompt: p.Prompt, Response: p.Response}
		for _, t := range p.History {
			qa[i].History = append(qa[i].History, idata.ChatTurn{User: t.User, Bot: t.Bot})
		}
	}
	var net *ca.Model
	var tok *tokenizer.Tokenizer
	contextSize := cfg.Context
	if o.Base != nil {
		s := handle.Of(o.Base.Dequantize())
		net, tok, contextSize = s.Full, s.Tok, o.Base.ContextSize()
		for _, special := range tokenizer.DefaultSpecialTokens {
			tok.AddSpecialToken(special)
		}
		if tok.VocabSize() > net.VocabSize() {
			net.ResizeVocab(tok.VocabSize()) // room for the special tokens the base never saw
		}
	} else {
		tok = idata.TrainPairTokenizer(qa, cfg.Vocab)
		net = cfg.BuildModel(tok.VocabSize(), o.Seed)
	}
	trainSet, valSet := idata.EncodePairsTrainVal(qa, contextSize, tok, idata.DefaultValFraction)
	if trainSet.Len() == 0 {
		return nil, Result{}, fmt.Errorf("no training sequences: every pair is too long for a context of %d tokens", contextSize)
	}
	return run(ctx, o, net, tok, trainSet.Cap(cfg.MaxSeqs), valSet, itrain.DefaultLR(cfg.Optimizer)*itrain.SFTLRFraction, contextSize, model.TaskChat)
}

func run(ctx context.Context, o Options, net *ca.Model, tok *tokenizer.Tokenizer, trainSet, valSet idata.Sequences,
	lr float64, contextSize int, task string) (*model.Model, Result, error) {
	if o.LR > 0 {
		lr = o.LR
	}
	meta := metadata(net, contextSize, task)
	if o.Base != nil {
		meta.Extra = o.Base.Extra()
	}
	m, _ := handle.Wrap(handle.State{Full: net, Tok: tok, Meta: meta}).(*model.Model)

	opts := o.Config.Options(o.Seed, o.Log == nil)
	opts.LR, opts.Log, opts.Context = lr, o.Log, ctx
	opts.TokenBytes = tok.TokenBytes()
	opts.OnEpoch = o.OnEpoch
	if o.OnBest != nil {
		opts.OnCheckpoint = func(epoch int, loss float64, weights []float32) {
			net.SetWeights(weights)
			o.OnBest(m)
		}
	}
	res := itrain.Run(net, trainSet, valSet, opts)
	if res.Canceled {
		return m, res, ctx.Err()
	}
	return m, res, nil
}

// metadata describes net so a saved file can be rebuilt without configuration.
func metadata(net *ca.Model, contextSize int, task string) checkpoint.Metadata {
	first := net.BlockConfigs()[0]
	return checkpoint.Metadata{
		Blocks:       net.NumBlocks(),
		StateSize:    net.StateSize(),
		HiddenSize:   net.HiddenSize(),
		Steps:        first.Steps,
		DilationBase: first.DilationBase,
		ContextSize:  contextSize,
		Task:         task,
		TiedEmbed:    net.TiedEmbed(),
		AttnHeads:    net.Config().AttnHeads,
		AttnWindow:   net.Config().AttnWindow,
	}
}
