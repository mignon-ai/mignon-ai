package data

import (
	"fmt"
	"strings"

	"github.com/mignon-ai/mignon-ai/internal/tokenizer"
)

// Tokenizer training samples.
const (
	// tokenizerSampleBytes bounds the sample the tokenizer trains on. Each document contributes in
	// proportion to its size, as tokenizerSampleChunk-byte chunks spread evenly through it.
	tokenizerSampleBytes = 16 * 1024 * 1024
	tokenizerSampleChunk = 16 * 1024
	// tokenizerSampleMinChunks keeps a small per-document budget spread out too, in smaller chunks.
	tokenizerSampleMinChunks = 16
)

// SampleForTokenizer extracts the text the tokenizer is trained on. Document separators are
// removed: they become EOS tokens, so their spelling must not compete for BPE merges.
func SampleForTokenizer(docs []string) string {
	total := 0
	for _, doc := range docs {
		total += len(doc)
	}
	var sb strings.Builder
	for _, doc := range docs {
		doc = strings.ReplaceAll(doc, DocSeparator, "")
		budget := len(doc)
		if total > tokenizerSampleBytes {
			budget = int(float64(len(doc)) * tokenizerSampleBytes / float64(total))
		}
		writeSpreadSample(&sb, doc, budget)
	}
	return sb.String()
}

// writeSpreadSample writes about budget bytes of doc as evenly spaced chunks cut at line breaks,
// so the sample covers the whole document instead of its beginning.
func writeSpreadSample(sb *strings.Builder, doc string, budget int) {
	if budget >= len(doc) {
		sb.WriteString(doc)
		sb.WriteString("\n")
		return
	}
	chunks := max(tokenizerSampleMinChunks, budget/tokenizerSampleChunk)
	size, spacing := budget/chunks, len(doc)/chunks
	for i := 0; i < chunks; i++ {
		start := i * spacing
		if i > 0 {
			nl := strings.IndexByte(doc[start:], '\n')
			if nl < 0 {
				break
			}
			start += nl + 1
		}
		end := min(start+size, len(doc))
		if nl := strings.LastIndexByte(doc[start:end], '\n'); nl > 0 && end < len(doc) {
			end = start + nl + 1
		}
		sb.WriteString(doc[start:end])
		if !strings.HasSuffix(doc[start:end], "\n") {
			sb.WriteString("\n")
		}
	}
}

// findCleanSplitPoint finds a whitespace or newline near the target ratio to avoid slicing mid-word.
func findCleanSplitPoint(text string, ratio float64) int {
	if len(text) == 0 {
		return 0
	}
	target := int(float64(len(text)) * ratio)
	if target <= 0 {
		return 0
	}
	if target >= len(text) {
		return len(text)
	}

	const searchWindow = 256
	for delta := 0; delta < searchWindow && target-delta >= 0; delta++ {
		b := text[target-delta]
		if b == '\n' || b == ' ' {
			return target - delta + 1
		}
	}
	for delta := 1; delta < searchWindow && target+delta < len(text); delta++ {
		b := text[target+delta]
		if b == '\n' || b == ' ' {
			return target + delta + 1
		}
	}
	return target
}

// SplitForVal splits raw text documents into disjoint train and validation sets
// before windowing to prevent validation leakage across overlapping windows.
func SplitForVal(docs []string, valFraction float64) (trainDocs, valDocs []string) {
	if valFraction <= 0 || len(docs) == 0 {
		return docs, nil
	}

	if len(docs) >= MinDocsForDocSplit {
		valDocCount := int(float64(len(docs)) * valFraction)
		if valDocCount < 1 {
			valDocCount = 1
		}
		if valDocCount >= len(docs) {
			valDocCount = len(docs) - 1
		}
		trainDocs = docs[:len(docs)-valDocCount]
		valDocs = dropBlocksSeenInTrain(trainDocs, docs[len(docs)-valDocCount:])
		return trainDocs, valDocs
	}

	// For few or single documents, split by contiguous regions within each document
	for _, doc := range docs {
		validBlocks := splitBlocks(doc)

		if len(validBlocks) >= 2 {
			valBlockCount := int(float64(len(validBlocks)) * valFraction)
			if valBlockCount < 1 {
				valBlockCount = 1
			}
			if valBlockCount >= len(validBlocks) {
				valBlockCount = len(validBlocks) - 1
			}
			trainPart := strings.Join(validBlocks[:len(validBlocks)-valBlockCount], "\n\n")
			valPart := strings.Join(validBlocks[len(validBlocks)-valBlockCount:], "\n\n")
			if len(trainPart) > 0 {
				trainDocs = append(trainDocs, trainPart)
			}
			if len(valPart) > 0 {
				valDocs = append(valDocs, valPart)
			}
		} else {
			if len(doc) < 100 {
				trainDocs = append(trainDocs, doc)
			} else {
				splitPoint := findCleanSplitPoint(doc, 1.0-valFraction)
				trainPart := strings.TrimSpace(doc[:splitPoint])
				valPart := strings.TrimSpace(doc[splitPoint:])
				if len(trainPart) > 0 {
					trainDocs = append(trainDocs, trainPart)
				}
				if len(valPart) > 0 {
					valDocs = append(valDocs, valPart)
				}
			}
		}
	}
	return trainDocs, dropBlocksSeenInTrain(trainDocs, valDocs)
}

// splitBlocks returns the non-empty, trimmed "\n\n"-separated blocks of doc.
func splitBlocks(doc string) []string {
	var blocks []string
	for _, b := range strings.Split(doc, "\n\n") {
		if b = strings.TrimSpace(b); b != "" {
			blocks = append(blocks, b)
		}
	}
	return blocks
}

// isDocSeparator reports whether block marks the end of a document (see DocSeparator).
func isDocSeparator(block string) bool {
	return block == DocSeparator
}

// encodeBlocks encodes the blocks of a corpus file for atomic packing. Blocks inside a document
// keep their blank-line spacing, the last block of each document ends with a newline and EOS,
// and document separators produce no tokens of their own.
func encodeBlocks(blocks []string, tok *tokenizer.Tokenizer) [][]int {
	eosID := tok.EOSTokenID()
	var encoded [][]int
	for i, b := range blocks {
		if isDocSeparator(b) {
			continue
		}
		if i < len(blocks)-1 && !isDocSeparator(blocks[i+1]) {
			encoded = append(encoded, tok.Encode(b+"\n\n"))
			continue
		}
		ids := tok.Encode(b + "\n")
		if eosID >= 0 {
			ids = append(ids, eosID)
		}
		encoded = append(encoded, ids)
	}
	return encoded
}

// dropBlocksSeenInTrain removes validation blocks whose exact text also appears in training,
// so repeated text in a corpus cannot make validation loss look better than it is.
func dropBlocksSeenInTrain(trainDocs, valDocs []string) []string {
	trainBlocks := make(map[string]struct{})
	for _, doc := range trainDocs {
		for _, b := range splitBlocks(doc) {
			trainBlocks[b] = struct{}{}
		}
	}
	var kept []string
	dropped, total := 0, 0
	for _, doc := range valDocs {
		var unseen []string
		for _, b := range splitBlocks(doc) {
			if isDocSeparator(b) {
				unseen = append(unseen, b) // a boundary, not text that could leak
				continue
			}
			total++
			if _, seen := trainBlocks[b]; seen {
				dropped++
				continue
			}
			unseen = append(unseen, b)
		}
		if len(unseen) > 0 {
			kept = append(kept, strings.Join(unseen, "\n\n"))
		}
	}
	if dropped == 0 {
		return valDocs
	}
	fmt.Printf("[Data] dropped %d of %d validation blocks that also appear in training\n", dropped, total)
	return kept
}

// PrepareTrainVal prepares train and validation sequences for dense causal multi-token prediction,
// training a new BPE tokenizer on a representative sample of docs.
func PrepareTrainVal(docs []string, contextSize int, targetVocabSize int, stride int, maxSeqs int, valFraction float64) (train, val Sequences, tok *tokenizer.Tokenizer) {
	if len(docs) == 0 {
		return Sequences{}, Sequences{}, nil
	}
	tok = tokenizer.New()
	tok.Train(SampleForTokenizer(docs), targetVocabSize)
	train, val = EncodeTrainVal(docs, contextSize, tok, stride, maxSeqs, valFraction)
	return train, val, tok
}

// EncodeTrainVal encodes documents into train and validation sequences using an existing tokenizer.
// It splits documents or contiguous regions prior to windowing so validation windows never overlap with train windows.
func EncodeTrainVal(docs []string, contextSize int, tok *tokenizer.Tokenizer, stride int, maxSeqs int, valFraction float64) (train, val Sequences) {
	if len(docs) == 0 || tok == nil {
		return Sequences{}, Sequences{}
	}
	if valFraction <= 0 {
		train.X, train.Y = Encode(docs, contextSize, tok, stride, maxSeqs)
		return train, Sequences{}
	}

	trainDocs, valDocs := SplitForVal(docs, valFraction)
	train.X, train.Y = Encode(trainDocs, contextSize, tok, stride, maxSeqs)
	if len(valDocs) > 0 {
		val.X, val.Y = Encode(valDocs, contextSize, tok, stride, MaxValSequences)
	}
	return train, val
}

// Encode encodes documents into sequences using an existing tokenizer.
func Encode(docs []string, contextSize int, tok *tokenizer.Tokenizer, stride int, maxSeqs int) ([][]int, [][]int) {
	if len(docs) == 0 || tok == nil {
		return nil, nil
	}

	if stride < 1 {
		stride = contextSize / 2
		if stride < 1 {
			stride = 1
		}
	}

	// Pre-encode sequences per document with atomic block packing
	docSeqsX := make([][][]int, len(docs))
	docSeqsY := make([][][]int, len(docs))
	totalSeqs := 0
	maxSeqsPerDoc := 0

	eosID := tok.EOSTokenID()
	padID := tok.PadTokenID()

	for dIdx, doc := range docs {

		// Check if the file contains discrete blocks separated by double-newlines
		validBlocks := splitBlocks(doc)

		if len(validBlocks) > 1 {
			// Atomic block packing: never cut a Q&A or command pair across a context boundary
			var currTokens []int
			for _, bIDs := range encodeBlocks(validBlocks, tok) {
				if len(currTokens)+len(bIDs) <= contextSize+1 {
					currTokens = append(currTokens, bIDs...)
				} else {
					if len(currTokens) > 1 {
						xSeq, ySeq := causalSeq(currTokens, contextSize, padID)
						docSeqsX[dIdx] = append(docSeqsX[dIdx], xSeq)
						docSeqsY[dIdx] = append(docSeqsY[dIdx], ySeq)
					}
					if len(bIDs) > contextSize+1 {
						// Block itself is larger than contextSize, slice with stride
						for pos := 0; pos <= len(bIDs)-contextSize-1; pos += stride {
							xSeq, ySeq := windowSeq(bIDs, pos, contextSize)
							docSeqsX[dIdx] = append(docSeqsX[dIdx], xSeq)
							docSeqsY[dIdx] = append(docSeqsY[dIdx], ySeq)
						}
						currTokens = nil
					} else {
						currTokens = bIDs
					}
				}
			}
			if len(currTokens) > 1 {
				xSeq, ySeq := causalSeq(currTokens, contextSize, padID)
				docSeqsX[dIdx] = append(docSeqsX[dIdx], xSeq)
				docSeqsY[dIdx] = append(docSeqsY[dIdx], ySeq)
			}
		} else {
			// Continuous stream slicing for single long prose/stories
			tIDs := tok.Encode(doc)
			if len(tIDs) > 0 {
				if eosID >= 0 {
					tIDs = append(tIDs, eosID)
				}
				if len(tIDs) <= contextSize {
					xSeq, ySeq := causalSeq(tIDs, contextSize, padID)
					docSeqsX[dIdx] = append(docSeqsX[dIdx], xSeq)
					docSeqsY[dIdx] = append(docSeqsY[dIdx], ySeq)
				} else {
					for pos := 0; pos <= len(tIDs)-contextSize-1; pos += stride {
						xSeq, ySeq := windowSeq(tIDs, pos, contextSize)
						docSeqsX[dIdx] = append(docSeqsX[dIdx], xSeq)
						docSeqsY[dIdx] = append(docSeqsY[dIdx], ySeq)
					}
				}
			}
		}

		n := len(docSeqsX[dIdx])
		totalSeqs += n
		if n > maxSeqsPerDoc {
			maxSeqsPerDoc = n
		}
	}

	if totalSeqs == 0 {
		return nil, nil
	}
	if maxSeqs > 0 && totalSeqs > maxSeqs {
		totalSeqs = maxSeqs
	}

	// 3. Balanced round-robin interleaving across all documents
	X := make([][]int, 0, totalSeqs)
	YDense := make([][]int, 0, totalSeqs)

	for step := 0; step < maxSeqsPerDoc; step++ {
		for dIdx := range docs {
			if step < len(docSeqsX[dIdx]) {
				X = append(X, docSeqsX[dIdx][step])
				YDense = append(YDense, docSeqsY[dIdx][step])
				if len(X) >= totalSeqs {
					break
				}
			}
		}
		if len(X) >= totalSeqs {
			break
		}
	}

	// 4. In-place Fisher-Yates shuffle
	shuffleSeqs(X, YDense)

	return X, YDense
}
