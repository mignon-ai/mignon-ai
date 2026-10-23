// Package data reads the training data: prompt/response pairs for fine-tuning and a directory of text
// files for pretraining.
package data

import (
	"encoding/json"
	"fmt"
	"io"
	"os"

	internal "github.com/mignon-ai/mignon-ai/internal/data"
)

// Pair is one example for fine-tuning: the model learns to answer Prompt with Response.
type Pair struct {
	Prompt   string `json:"prompt"`
	Response string `json:"response"`
	// History holds earlier turns of a conversation, oldest first. Most tasks leave it empty.
	History []Turn `json:"history,omitempty"`
}

// Turn is one earlier exchange.
type Turn struct {
	User string `json:"user,omitempty"`
	Bot  string `json:"bot,omitempty"`
}

// UnmarshalJSON accepts both field spellings: prompt/response and instruction/output (with an optional
// input line after the instruction).
func (p *Pair) UnmarshalJSON(raw []byte) error {
	var q internal.QAPair
	if err := json.Unmarshal(raw, &q); err != nil {
		return err
	}
	p.Prompt, p.Response = q.GetPrompt(), q.GetResponse()
	p.History = nil
	for _, t := range q.History {
		p.History = append(p.History, Turn{User: t.User, Bot: t.Bot})
	}
	return nil
}

// ReadPairs reads a JSON array of pairs and drops those without both a prompt and a response. It fails
// when none are left.
func ReadPairs(r io.Reader) ([]Pair, error) {
	var all []Pair
	if err := json.NewDecoder(r).Decode(&all); err != nil {
		return nil, fmt.Errorf("parsing pairs: %w", err)
	}
	var valid []Pair
	for _, p := range all {
		if p.Prompt != "" && p.Response != "" {
			valid = append(valid, p)
		}
	}
	if len(valid) == 0 {
		return nil, fmt.Errorf("no prompt/response pairs found")
	}
	return valid, nil
}

// LoadPairs reads the pairs file at path.
func LoadPairs(path string) ([]Pair, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	pairs, err := ReadPairs(f)
	if err != nil {
		return nil, fmt.Errorf("%s: %w", path, err)
	}
	return pairs, nil
}

// LoadCorpus reads every .txt file under dir as one document each, for pretraining.
func LoadCorpus(dir string) ([]string, error) {
	return internal.LoadTextCorpus(dir, nil)
}
