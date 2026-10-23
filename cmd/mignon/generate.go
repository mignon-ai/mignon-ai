package main

import (
	"flag"
	"fmt"
	"github.com/mignon-ai/mignon-ai/internal/infer"
	"strconv"
	"strings"

	"github.com/mignon-ai/mignon-ai/internal/constrain"
	"github.com/mignon-ai/mignon-ai/internal/data"
	"github.com/mignon-ai/mignon-ai/internal/generate"
)

// Generation modes selected by -mode: how chat and generate feed input to the model.
const (
	modeAuto = "auto" // chat for models fine-tuned on pairs, text for raw-text models
	modeChat = "chat" // wrap input in the <USER> prompt <BOT> template and stop at <EOS>
	modeText = "text" // continue the input as plain text
)

var generationModes = []string{modeAuto, modeChat, modeText}

// Chat replies sample more conservatively than free text unless -temp/-penalty are given.
const (
	chatTemp    = 0.35
	chatPenalty = 1.15
)

// Interactive chat commands.
const (
	cmdReset   = "/reset"
	cmdClear   = "/clear"
	cmdHistory = "/history"
)

// textReplyPrefix marks REPL output as a continuation of the typed text rather than an answer.
const textReplyPrefix = "..."

// genOptions holds the flags of 'mignon chat' and 'mignon generate'.
type genOptions struct {
	modelFlags
	samplingFlags
	prompt   string
	history  int
	explicit map[string]bool // flags set on the command line
}

func parseGenFlags(fs *flag.FlagSet, args []string, interactive bool) (*genOptions, error) {
	o := &genOptions{}
	o.modelFlags.register(fs, "Saved model to generate with (required)")
	o.samplingFlags.register(fs)
	if interactive {
		fs.IntVar(&o.history, "history", defaultHistoryTurns, "Number of past conversation turns to retain in chat mode (0 = stateless/single-turn)")
	} else {
		fs.StringVar(&o.prompt, "prompt", "", "Text to answer or continue (required)")
	}
	var err error
	if o.explicit, err = parse(fs, args); err != nil {
		return nil, err
	}
	if !interactive && o.prompt == "" {
		return nil, fmt.Errorf("-prompt is required")
	}
	if err := o.validate(); err != nil {
		return nil, err
	}
	return o, nil
}

// runChat implements 'mignon chat': an interactive terminal session.
func runChat(args []string) {
	o, err := parseGenFlags(newFlagSet("chat"), args, true)
	if err != nil {
		fatalf("Error: %v\n", err)
	}
	chatLoop(o, openSession(o.load))
}

// runGenerate implements 'mignon generate': a single answer or continuation of -prompt.
func runGenerate(args []string) {
	o, err := parseGenFlags(newFlagSet("generate"), args, false)
	if err != nil {
		fatalf("Error: %v\n", err)
	}
	runPrompt(o, openSession(o.load))
}

// generationMode resolves -mode against the task the model was trained for.
func generationMode(o *genOptions, s *session) string {
	if o.mode != modeAuto {
		return o.mode
	}
	if s.task == taskChat {
		return modeChat
	}
	return modeText
}

// sampling returns the -temp/-penalty/-top-k/-top-p/-gen-tokens settings for mode. Chat replies
// default to shorter, more conservative sampling unless those flags were given explicitly.
func sampling(o *genOptions, mode string) generate.Sampling {
	sp := generate.Sampling{MaxTokens: o.genTokens, Temperature: o.temp, RepPenalty: o.penalty, TopK: o.topK, TopP: o.topP}
	if mode != modeChat {
		return sp
	}
	if !o.explicit["gen-tokens"] {
		sp.MaxTokens = generate.ChatMaxResponseTokens
	}
	if !o.explicit["temp"] {
		sp.Temperature = chatTemp
	}
	if !o.explicit["penalty"] {
		sp.RepPenalty = chatPenalty
	}
	return sp
}

// generationSetup returns the model, mode and sampling for chat or generate, exiting when no
// model is loaded and warning when chat mode is forced on a raw-text model.
func generationSetup(o *genOptions, s *session) (infer.SessionProvider, string, generate.Sampling) {
	model := s.activeModel()
	if model == nil {
		fatalf("Error: No model loaded or trained. Cannot generate.\n")
	}
	mode := generationMode(o, s)
	if mode == modeChat && s.task != taskChat {
		fmt.Println("\n[Notice] This model was trained on raw text, not prompt/response pairs, so it has never seen the chat template.")
		fmt.Println("Use -mode text (the default for this model) to test continuation, or fine-tune it with -sft first.")
	}
	sp := sampling(o, mode)
	sp.Grammar = loadGrammar(o.constrain, s)
	return model, mode, sp
}

// loadGrammar builds the command grammar from the replies of the pairs file at path ("" = none).
func loadGrammar(path string, s *session) *constrain.Grammar {
	if path == "" {
		return nil
	}
	if s.tok == nil {
		fatalf("Error: -constrain needs a model with a tokenizer.\n")
	}
	pairs, err := data.LoadPairs(path)
	if err != nil {
		fatalf("Error loading -constrain pairs: %v\n", err)
	}
	requests := make([]string, len(pairs))
	replies := make([]string, len(pairs))
	for i := range pairs {
		requests[i], replies[i] = pairs[i].GetPrompt(), pairs[i].GetResponse()
	}
	return constrain.NewGrammar(requests, replies, s.tok)
}

// banner describes the engine and sampling settings above generated output.
func banner(s *session, mode string, sp generate.Sampling) string {
	return fmt.Sprintf("mignon-ai %s mode (%s, temp=%.2f, penalty=%.2f, top-k=%d, top-p=%.2f, max %d tokens)",
		mode, s.modeLabel(), sp.Temperature, sp.RepPenalty, sp.TopK, sp.TopP, sp.MaxTokens)
}

// chatLoop starts the interactive REPL: multi-turn chat in chat mode, one continuation per line
// in text mode.
func chatLoop(o *genOptions, s *session) {
	model, mode, sp := generationSetup(o, s)
	title := banner(s, mode, sp)
	if mode == modeText {
		generate.RunInteractiveChat(title, func(input string) string {
			return textReplyPrefix + generate.Text(model, s.tok, input, s.contextSize, sp)
		})
		return
	}
	var history []int
	maxTurns := o.history
	generate.RunInteractiveChat(title, func(input string) string {
		trimmed := strings.TrimSpace(input)
		if trimmed == cmdReset || trimmed == cmdClear {
			history = nil
			return "(Conversation history cleared.)"
		}
		if strings.HasPrefix(trimmed, cmdHistory) {
			if parts := strings.Fields(trimmed); len(parts) >= 2 {
				if n, err := strconv.Atoi(parts[1]); err == nil && n >= 0 {
					maxTurns = n
					if maxTurns == 0 {
						history = nil
					}
					return fmt.Sprintf("(Chat memory set to %d turn(s).)", maxTurns)
				}
			}
			return fmt.Sprintf("(Current chat memory: %d turn(s). Usage: /history <N>)", maxTurns)
		}
		var reply string
		reply, history = generate.ChatWithHistory(model, s.tok, history, input, s.contextSize, maxTurns, sp)
		return reply
	})
}

// runPrompt answers -prompt once: a chat reply in chat mode, a continuation in text mode.
func runPrompt(o *genOptions, s *session) {
	model, mode, sp := generationSetup(o, s)
	var out string
	if mode == modeChat {
		out = generate.Chat(model, s.tok, o.prompt, s.contextSize, sp)
	} else {
		out = generate.Text(model, s.tok, o.prompt, s.contextSize, sp)
	}
	title := "--- " + banner(s, mode, sp) + " ---"
	fmt.Printf("\n%s\nPrompt: %s\nOutput: %s\n%s\n", title, o.prompt, strings.TrimSpace(out), strings.Repeat("-", len(title)))
}
