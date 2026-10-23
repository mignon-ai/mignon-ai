// Package infer defines how generation code talks to a model backend: next-token logits for a
// window (LogitsFn) and incremental sessions that keep state between tokens.
package infer

// LogitsFn returns next-token logits for a window of tokens.
type LogitsFn func(window []int) []float64

// Session produces next-token logits for a sequence that grows one token at a time.
// The returned slice is only valid until the next call on the session (or on its model).
type Session interface {
	// Prefill starts a new sequence and returns the logits after its last token (nil when empty).
	Prefill(tokens []int) []float64
	// Append extends the sequence by one token and returns the logits after it.
	Append(token int) []float64
}

// SessionProvider is implemented by models that support incremental (cached) generation.
type SessionProvider interface {
	NewSession(contextSize int) Session
}

// recomputeSession runs the full forward pass over the latest window for every token.
type recomputeSession struct {
	forward LogitsFn
	window  int
	tokens  []int
}

func (s *recomputeSession) Prefill(tokens []int) []float64 {
	s.tokens = append(s.tokens[:0], tokens...)
	return s.logits()
}

func (s *recomputeSession) Append(token int) []float64 {
	s.tokens = append(s.tokens, token)
	return s.logits()
}

func (s *recomputeSession) logits() []float64 {
	if len(s.tokens) == 0 {
		return nil
	}
	w := s.tokens
	if s.window > 0 && len(w) > s.window {
		w = w[len(w)-s.window:]
	}
	return s.forward(w)
}

// Recompute returns a session that runs forward over the latest window (at most window tokens)
// on every call. It is the uncached reference the cached sessions are checked against.
func Recompute(forward LogitsFn, window int) Session {
	return &recomputeSession{forward: forward, window: window}
}

// RecomputeProvider adapts a stateless forward function to a SessionProvider whose sessions
// recompute the latest window on every token.
type RecomputeProvider LogitsFn

// NewSession implements SessionProvider.
func (f RecomputeProvider) NewSession(contextSize int) Session {
	return Recompute(LogitsFn(f), contextSize)
}
