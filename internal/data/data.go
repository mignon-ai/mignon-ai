// Package data turns raw corpora, chat logs and SFT examples into fixed-length training
// sequences: inputs X and dense next-token targets Y, where MaskedTarget marks positions that
// must not contribute to the loss (prompts, padding).
package data

import "github.com/mignon-ai/mignon-ai/internal/tokenizer"

// DocSeparator, on a line of its own between blank lines ("\n\n<EOS>\n\n"), ends a document
// inside a corpus file. A file without separators is one document. Every document ends with the
// EOS token, so the model learns where unrelated texts packed into one context window begin.
const DocSeparator = tokenizer.EOSToken

// MaskedTarget is the ignore_index value for positions that should not contribute to loss or gradients
const MaskedTarget = -1

// Validation splits
const (
	// DefaultValFraction is the fraction of training sequences held out as the validation set.
	DefaultValFraction = 0.10
	// MinValSamples is the minimum total sequences required before a validation split is made.
	MinValSamples = 20
	// MaxValSequences caps the validation sample size so epoch validation remains sub-second.
	MaxValSequences = 1000
	// MinDocsForDocSplit is the minimum number of documents required to split by whole documents rather than contiguous intra-document regions.
	MinDocsForDocSplit = 5
)
