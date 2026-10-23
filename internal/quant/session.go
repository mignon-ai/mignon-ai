package quant

import "github.com/mignon-ai/mignon-ai/internal/infer"

// quantSession extends cached INT8 state by one position per token, at a cost and memory that do
// not depend on how long the sequence is.
//
// The selective latch forgets, so sessions stream without limit: the CA has no absolute positions
// and its caches only hold the last few positions, so a token far past the window costs the same
// as any other.
type quantSession struct {
	q      *Model
	st     *quantSeqState
	window int
	tokens []int // the most recent window of tokens, to rebuild the state when out of sync
	fed    int   // positions fed into st since the last prefill
}

// NewSession implements infer.SessionProvider for the INT8 model.
func (q *Model) NewSession(contextSize int) infer.Session {
	q.mu.Lock()
	defer q.mu.Unlock()
	window := q.maxContext
	if contextSize > 0 && contextSize < window {
		window = contextSize
	}
	return &quantSession{q: q, st: q.newQuantSeqState(), window: window}
}

func (s *quantSession) Prefill(tokens []int) []float64 {
	if len(tokens) > s.window {
		tokens = tokens[len(tokens)-s.window:]
	}
	s.tokens = append(s.tokens[:0], tokens...)
	if len(s.tokens) == 0 {
		return nil
	}
	return s.prefill()
}

func (s *quantSession) Append(token int) []float64 {
	s.tokens = append(s.tokens, token)
	if len(s.tokens) > s.window {
		s.tokens = append(s.tokens[:0], s.tokens[1:]...)
	}
	s.q.mu.Lock()
	defer s.q.mu.Unlock()
	if s.fed == 0 || s.st.n != s.fed {
		// The session was not prefilled (or is out of sync); rebuild from the tokens it holds.
		s.fed = len(s.tokens)
		return s.q.prefill(s.st, s.tokens)
	}
	s.q.appendPosition(s.st, token, true)
	s.fed++
	return s.q.headLogits(s.st.cur)
}

func (s *quantSession) prefill() []float64 {
	s.q.mu.Lock()
	defer s.q.mu.Unlock()
	s.fed = len(s.tokens)
	return s.q.prefill(s.st, s.tokens)
}

// StateBytes returns the memory held by the session's per-sequence caches.
func (s *quantSession) StateBytes() int {
	return s.st.bytes()
}
