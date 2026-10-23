package main

import (
	"fmt"
	"github.com/mignon-ai/mignon-ai/internal/infer"

	"github.com/mignon-ai/mignon-ai/internal/ca"
	"github.com/mignon-ai/mignon-ai/internal/handle"
	"github.com/mignon-ai/mignon-ai/internal/quant"
	"github.com/mignon-ai/mignon-ai/internal/tokenizer"
	"github.com/mignon-ai/mignon-ai/internal/train"
	"github.com/mignon-ai/mignon-ai/pkg/model"
)

// Task labels stored in checkpoint metadata.
const (
	taskText = "text"
	taskChat = "chat"
)

// minBlocks is the fewest blocks a model can have.
const minBlocks = 1

// session is the model state shared by the load, train, save and generate steps.
type session struct {
	model       *ca.Model    // full-precision model, nil until loaded or trained
	qmodel      *quant.Model // INT8 model when one was loaded or quantized
	tok         *tokenizer.Tokenizer
	task        string
	contextSize int
}

// activeModel is the model used for generation: the quantized one when present.
func (s *session) activeModel() infer.SessionProvider {
	if s.qmodel != nil {
		return s.qmodel
	}
	if s.model == nil {
		return nil
	}
	return infer.RecomputeProvider(s.model.Forward)
}

// modeLabel names the inference engine in generation banners.
func (s *session) modeLabel() string {
	if s.qmodel != nil {
		return "INT8"
	}
	return "BPE"
}

// openSession loads the model at path for evaluation, quantization or generation, exiting when
// no path was given.
func openSession(path string) *session {
	if path == "" {
		fatalf("Error: -load <model.json> is required.\n")
	}
	s := &session{task: taskText, contextSize: train.DefaultConfig().Context}
	loadSession(path, s)
	return s
}

// loadSession loads the model file at path (binary or JSON, INT8 or full precision) into s.
func loadSession(path string, s *session) {
	fmt.Printf("Loading model from %s...\n", path)
	m, err := model.Load(path)
	if err != nil {
		fatalf("Error reading checkpoint: %v\n", err)
	}
	st := handle.Of(m)
	s.tok, s.task, s.contextSize = st.Tok, st.Meta.Task, st.Meta.ContextSize
	if st.Int8 != nil {
		fmt.Printf("Detected INT8 Quantized Multi-Block model (vocab: %d, state: %d, hidden: %d, blocks: %d).\n",
			st.Int8.VocabSize(), st.Int8.StateSize(), st.Int8.HiddenSize(), st.Int8.NumBlocks())
		s.qmodel = st.Int8
		return
	}
	s.model = st.Full
	fmt.Printf("Detected BPE Tokenizer model (vocab size: %d, merges: %d, dilation: %d).\n",
		st.Tok.VocabSize(), len(st.Tok.Merges), st.Meta.DilationBase)
	fmt.Println("Model loaded successfully.")
}

// loadTokenizer returns the tokenizer stored in a model file.
func loadTokenizer(path string) *tokenizer.Tokenizer {
	m, err := model.Load(path)
	if err != nil {
		fatalf("Error reading tokenizer from %s: %v\n", path, err)
	}
	return handle.Of(m).Tok
}
