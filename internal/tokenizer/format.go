package tokenizer

import "fmt"

// FormatVersion identifies the pre-tokenization and merge-training rules of a tokenizer. A saved
// tokenizer only reproduces its training-time encoding under the rules that made it, so
// checkpoints record the version and loaders reject any other (see CheckFormat).
const FormatVersion = 2

// CheckFormat returns an error when a tokenizer saved under version v cannot be used with the
// current rules.
func CheckFormat(v int) error {
	if v != FormatVersion {
		return fmt.Errorf("tokenizer format version %d is not supported (want %d): retrain the model", v, FormatVersion)
	}
	return nil
}
