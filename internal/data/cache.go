package data

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/gob"
	"encoding/hex"
	"fmt"
	"os"
	"path/filepath"

	"github.com/mignon-ai/mignon-ai/internal/tokenizer"
)

// dataCacheVersion must be bumped whenever tokenizer training or sequence encoding changes,
// so stale cached datasets are never reused.
const dataCacheVersion = 6

// dataCacheKeyHexLen is the number of hex characters of the content hash used in cache directory names.
const dataCacheKeyHexLen = 16

const (
	dataCacheTokenizerFile = "tokenizer.json"
	dataCacheSequencesFile = "sequences.gob"
)

// cachedSequences is the on-disk form of an encoded train/validation split.
type cachedSequences struct {
	TrainX, TrainY, ValX, ValY [][]int
}

// CacheKey hashes the corpus content and every parameter that affects tokenization and encoding.
func CacheKey(docs []string, contextSize, vocabSize, stride, maxSeqs int, valFraction float64) string {
	h := sha256.New()
	fmt.Fprintf(h, "v%d|ctx%d|vocab%d|tok%d|stride%d|max%d|val%g|docs%d|", dataCacheVersion, contextSize, vocabSize, tokenizer.FormatVersion, stride, maxSeqs, valFraction, len(docs))
	for _, doc := range docs {
		var n [8]byte
		binary.LittleEndian.PutUint64(n[:], uint64(len(doc)))
		h.Write(n[:]) // length prefix keeps document boundaries unambiguous
		h.Write([]byte(doc))
	}
	return hex.EncodeToString(h.Sum(nil))[:dataCacheKeyHexLen]
}

// dataCacheDir returns the cache directory for key, under the user cache directory.
func dataCacheDir(key string) (string, error) {
	base, err := os.UserCacheDir()
	if err != nil {
		return "", err
	}
	return filepath.Join(base, "mignon-ai", "data-"+key), nil
}

func loadCachedDataset(dir string) (*tokenizer.Tokenizer, cachedSequences, error) {
	var seqs cachedSequences
	tok, err := tokenizer.Load(filepath.Join(dir, dataCacheTokenizerFile))
	if err != nil {
		return nil, seqs, err
	}
	f, err := os.Open(filepath.Join(dir, dataCacheSequencesFile))
	if err != nil {
		return nil, seqs, err
	}
	defer f.Close()
	err = gob.NewDecoder(f).Decode(&seqs)
	return tok, seqs, err
}

// saveCachedDataset writes via a temporary directory and renames it, so a partially written
// cache is never visible to concurrent or later runs.
func saveCachedDataset(dir string, tok *tokenizer.Tokenizer, seqs cachedSequences) error {
	if err := os.MkdirAll(filepath.Dir(dir), 0o755); err != nil {
		return err
	}
	tmp, err := os.MkdirTemp(filepath.Dir(dir), filepath.Base(dir)+".tmp-")
	if err != nil {
		return err
	}
	defer os.RemoveAll(tmp)
	if err := tok.Save(filepath.Join(tmp, dataCacheTokenizerFile)); err != nil {
		return err
	}
	f, err := os.Create(filepath.Join(tmp, dataCacheSequencesFile))
	if err != nil {
		return err
	}
	if err := gob.NewEncoder(f).Encode(seqs); err != nil {
		f.Close()
		return err
	}
	if err := f.Close(); err != nil {
		return err
	}
	if _, err := os.Stat(dir); err == nil {
		return nil // another run already populated this cache entry
	}
	return os.Rename(tmp, dir)
}

// PrepareTrainValCached is PrepareTrainVal with an on-disk cache keyed by corpus
// content and encoding parameters. Cache failures fall back to (and never break) normal preparation.
// The returned bool reports whether the cache was used.
func PrepareTrainValCached(docs []string, contextSize, vocabSize, stride, maxSeqs int, valFraction float64) (train, val Sequences, tok *tokenizer.Tokenizer, cached bool) {
	dir, dirErr := dataCacheDir(CacheKey(docs, contextSize, vocabSize, stride, maxSeqs, valFraction))
	if dirErr == nil {
		if t, s, err := loadCachedDataset(dir); err == nil {
			return Sequences{s.TrainX, s.TrainY}, Sequences{s.ValX, s.ValY}, t, true
		}
	}
	train, val, tok = PrepareTrainVal(docs, contextSize, vocabSize, stride, maxSeqs, valFraction)
	if dirErr == nil && tok != nil {
		if err := saveCachedDataset(dir, tok, cachedSequences{train.X, train.Y, val.X, val.Y}); err != nil && !os.IsExist(err) {
			fmt.Printf("[Warning] could not write dataset cache: %v\n", err)
		}
	}
	return train, val, tok, false
}
