package generate

// Chat generation limits
const (
	// ChatMaxResponseTokens caps autoregressive generation per chat turn.
	ChatMaxResponseTokens = 60
	// ChatReplyHeadroom is the number of context tokens kept free for the reply when old
	// conversation turns are dropped to fit the window.
	ChatReplyHeadroom = 16
	// ChatRepetitionWindow is the look-back window of recent tokens checked for repetition.
	ChatRepetitionWindow = 6
)

// Sampling
const (
	// DefaultMaxVocabBuf is the initial vocab-sized scratch buffer for sampling.
	DefaultMaxVocabBuf = 256
)
