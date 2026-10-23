package data

import (
	"encoding/json"
	"fmt"
	"os"
	"strings"

	"github.com/mignon-ai/mignon-ai/internal/tokenizer"
)

// ChatTurn is one earlier exchange in a multi-turn conversation.
type ChatTurn struct {
	User string `json:"user,omitempty"`
	Bot  string `json:"bot,omitempty"`
}

// QAPair is one prompt/response example for fine-tuning. It accepts both the chat field names
// (prompt/response, with optional history) and the instruction field names
// (instruction/input/output).
type QAPair struct {
	Prompt      string     `json:"prompt,omitempty"`
	Response    string     `json:"response,omitempty"`
	Instruction string     `json:"instruction,omitempty"`
	Input       string     `json:"input,omitempty"`
	Output      string     `json:"output,omitempty"`
	History     []ChatTurn `json:"history,omitempty"`
}

// GetPrompt returns the prompt or instruction, followed by the input on its own line if any.
func (q *QAPair) GetPrompt() string {
	p := q.Prompt
	if p == "" {
		p = q.Instruction
	}
	if q.Input != "" {
		p += "\n" + q.Input
	}
	return strings.TrimSpace(p)
}

// GetResponse returns the response or output.
func (q *QAPair) GetResponse() string {
	if q.Response != "" {
		return strings.TrimSpace(q.Response)
	}
	return strings.TrimSpace(q.Output)
}

// LoadPairs reads a JSON array of prompt/response pairs, dropping pairs without both.
func LoadPairs(path string) ([]QAPair, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("failed to read %s: %w", path, err)
	}
	var all []QAPair
	if err := json.Unmarshal(raw, &all); err != nil {
		return nil, fmt.Errorf("failed to parse %s: %w", path, err)
	}
	var valid []QAPair
	for _, p := range all {
		if p.GetPrompt() != "" && p.GetResponse() != "" {
			valid = append(valid, p)
		}
	}
	if len(valid) == 0 {
		return nil, fmt.Errorf("no prompt/response pairs found in %s", path)
	}
	return valid, nil
}

// EncodePairs encodes pairs with the chat template into causal sequences whose targets cover
// only the response and its <EOS>: history and the prompt are masked. Older history turns are
// dropped and over-long prompts keep their end so each pair fits in contextSize.
func EncodePairs(pairs []QAPair, contextSize int, tok *tokenizer.Tokenizer) (X, Y [][]int) {
	seqCap := contextSize + 1 // causalSeq turns contextSize+1 tokens into contextSize targets
	padID := tok.PadTokenID()
	markers := len(tok.EncodePrompt("")) // <USER> and <BOT>
	for _, pair := range pairs {
		p, r := pair.GetPrompt(), pair.GetResponse()
		if p == "" || r == "" {
			continue
		}
		var history []int
		for _, turn := range pair.History {
			u, b := strings.TrimSpace(turn.User), strings.TrimSpace(turn.Bot)
			if u != "" && b != "" {
				history = append(history, tok.EncodePrompt(u)...)
				history = append(history, tok.EncodeReply(b)...)
			}
		}
		turn, reply := tok.EncodePrompt(p), tok.EncodeReply(r)
		history = tok.DropOldestTurns(history, seqCap-len(turn)-len(reply))

		// Keep <USER> and the end of the prompt, leaving room for at least one reply token.
		maxTurn := seqCap - len(history) - 1
		if maxTurn < markers {
			continue
		}
		if len(turn) > maxTurn {
			turn = append([]int{turn[0]}, turn[len(turn)-maxTurn+1:]...)
		}
		reply = reply[:min(len(reply), seqCap-len(history)-len(turn))]

		seq := append(append(history, turn...), reply...)
		x, y := causalSeq(seq, contextSize, padID)
		// The <BOT> position predicts the first reply token; everything before it is masked.
		maskTargets(y, len(history)+len(turn)-1)
		X = append(X, x)
		Y = append(Y, y)
	}
	return X, Y
}

// EncodePairsTrainVal splits pairs into train and validation sets before encoding, so no
// validation pair shares a sequence with training.
func EncodePairsTrainVal(pairs []QAPair, contextSize int, tok *tokenizer.Tokenizer, valFraction float64) (train, val Sequences) {
	if valFraction <= 0 || len(pairs) < MinValSamples {
		train.X, train.Y = EncodePairs(pairs, contextSize, tok)
		return train, Sequences{}
	}
	trainPairs, valPairs := splitTrainVal(pairs, valFraction)
	train.X, train.Y = EncodePairs(trainPairs, contextSize, tok)
	val.X, val.Y = EncodePairs(valPairs, contextSize, tok)
	return train, val
}

// TrainPairTokenizer trains a BPE tokenizer on the prompts and responses of pairs.
func TrainPairTokenizer(pairs []QAPair, targetVocabSize int) *tokenizer.Tokenizer {
	var text strings.Builder
	for _, p := range pairs {
		text.WriteString(p.GetPrompt() + "\n" + p.GetResponse() + "\n\n")
	}
	tok := tokenizer.New()
	tok.Train(text.String(), targetVocabSize)
	return tok
}
