package eval

import (
	"fmt"
	"strings"

	"github.com/mignon-ai/mignon-ai/internal/data"
	"github.com/mignon-ai/mignon-ai/internal/infer"
	"github.com/mignon-ai/mignon-ai/internal/tokenizer"
)

// PerplexityPerDoc scores each document of text (split on data.DocSeparator) independently: the
// context starts fresh with EOS and every token of the document, including the first, is scored.
// This is how most per-document benchmarks (e.g. TinyStories validation stories) are scored, so
// the result is comparable with them. maxDocs limits the number of documents (0 = all).
func PerplexityPerDoc(getLogits infer.LogitsFn, tok *tokenizer.Tokenizer, text string, contextSize, maxDocs int) HeldOut {
	eos := tok.EOSTokenID()
	tokenBytes := tok.TokenBytes()
	docs := splitDocs(text)
	if maxDocs > 0 && len(docs) > maxDocs {
		docs = docs[:maxDocs]
	}

	var total nll
	for i, doc := range docs {
		tokens := append([]int{eos}, tok.Encode(doc)...)
		total.score(getLogits, tokenBytes, tokens, 1, len(tokens), contextSize, nil)
		fmt.Printf("\r  Evaluating documents: %d/%d...", i+1, len(docs))
	}
	fmt.Println()
	return total.result()
}

// docPadding is the blank line that surrounds data.DocSeparator; it belongs to the separator, so
// the document's own bytes (including its own whitespace) are scored unchanged.
const docPadding = "\n\n"

// splitDocs splits text on data.DocSeparator and drops empty documents.
func splitDocs(text string) []string {
	var docs []string
	for _, d := range strings.Split(text, data.DocSeparator) {
		if d = strings.TrimSuffix(strings.TrimPrefix(d, docPadding), docPadding); d != "" {
			docs = append(docs, d)
		}
	}
	return docs
}
