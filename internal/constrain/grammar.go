// Package constrain restricts decoding of shell and git commands to words the training data and
// the request make plausible: the subcommand and options come from a lexicon learned from
// prompt/response pairs, and every other word has to be copied from the request or be common
// answer vocabulary (origin, HEAD, &&). The model still chooses among the allowed words.
package constrain

import (
	"slices"
	"strings"
	"unicode"

	"github.com/mignon-ai/mignon-ai/internal/tokenizer"
)

// commonShare is the share of replies that must contain a non-option word without their request
// containing it (origin, HEAD, user.name) for the word to be allowed anywhere. Names that are
// only ever copied (branches, paths) fall below it, and so do the rare names a dataset reuses
// without the request mentioning them: those must come from the request.
const commonShare = 0.005

// minCommonReplies is the floor of that share for small training sets.
const minCommonReplies = 5

// trimChars are stripped from both ends of a word, so quoting and punctuation around a name do
// not hide that it was copied (a request ends in "?" or "."; a message is quoted).
const trimChars = "'\"`,:;()[]?!."

// maxTrimmed bounds the quote and punctuation characters a word may carry around its core ('fix:
// or name'), so a name cannot be followed by an endless run of punctuation.
const maxTrimmed = 3

// partSeparators split a compound word (origin/main, --author=me) into independently checked parts.
const partSeparators = "/:=,@"

// glue words end a command segment: the next word starts a new command.
var glue = map[string]bool{"&&": true, "||": true, "|": true, ";": true}

// Grammar is the lexicon of a command language plus the token strings of one tokenizer.
type Grammar struct {
	first  []string            // words that start a command (git)
	second map[string][]string // first word -> words that may follow it (subcommands)
	opts   map[string][]string // "first second" -> options seen with that subcommand
	common []string            // non-option words shared by many replies
	text   []string            // token id -> decoded text; "" for control tokens
	eos    int
}

// NewGrammar learns the lexicon from the requests and replies of a training set (same length).
func NewGrammar(requests, replies []string, tok *tokenizer.Tokenizer) *Grammar {
	g := &Grammar{second: map[string][]string{}, opts: map[string][]string{}, eos: tok.EOSTokenID()}
	first := map[string]bool{}
	second := map[string]map[string]bool{}
	opts := map[string]map[string]bool{}
	pairs := map[string]map[int]bool{} // non-option word -> reply indexes where it was not asked for
	for i, reply := range replies {
		asked := map[string]bool{}
		for _, w := range strings.Fields(requests[i]) {
			asked[trim(w)] = true
		}
		for _, seg := range segments(strings.Fields(reply)) {
			if len(seg) == 0 {
				continue
			}
			first[seg[0]] = true
			if len(seg) > 1 {
				add(second, seg[0], seg[1])
			}
			for j, w := range seg {
				w = trim(w)
				switch {
				case w == "":
				case isOption(w) && j >= 2:
					add(opts, seg[0]+" "+seg[1], norm(w))
				case j >= 2:
					if asked[w] {
						continue
					}
					if pairs[norm(w)] == nil {
						pairs[norm(w)] = map[int]bool{}
					}
					pairs[norm(w)][i] = true
				}
			}
		}
	}
	minCommon := max(minCommonReplies, int(commonShare*float64(len(replies))))
	g.first = keys(first)
	for k, v := range second {
		g.second[k] = keys(v)
	}
	for k, v := range opts {
		g.opts[k] = keys(v)
	}
	for w, in := range pairs {
		if len(in) >= minCommon {
			g.common = append(g.common, w)
		}
	}
	slices.Sort(g.common)
	g.text = make([]string, tok.VocabSize())
	for id := range g.text {
		if id != g.eos {
			g.text[id] = tok.Decode([]int{id})
		}
	}
	return g
}

func add(m map[string]map[string]bool, k, v string) {
	if m[k] == nil {
		m[k] = map[string]bool{}
	}
	m[k][v] = true
}

func keys(m map[string]bool) []string {
	out := make([]string, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	slices.Sort(out)
	return out
}

// segments splits words at glue words.
func segments(words []string) [][]string {
	var out [][]string
	var cur []string
	for _, w := range words {
		if glue[w] {
			out = append(out, cur)
			cur = nil
			continue
		}
		cur = append(cur, w)
	}
	return append(out, cur)
}

func trim(w string) string { return strings.Trim(w, trimChars) }

func isOption(w string) bool { return len(w) > 1 && w[0] == '-' }

// norm maps digit runs to a single 0, so HEAD~3 matches HEAD~2 and -n 5 matches -n 10.
func norm(w string) string {
	var b strings.Builder
	digit := false
	for _, r := range w {
		if unicode.IsDigit(r) {
			if !digit {
				b.WriteByte('0')
			}
			digit = true
			continue
		}
		digit = false
		b.WriteRune(r)
	}
	return b.String()
}

// State tracks one reply being generated.
type State struct {
	g       *Grammar
	request []string // request words and their parts, exactly as written
	seg     []string // completed words of the current command segment
	cur     string   // the word being written
	off     bool     // the reply is not a command: nothing is constrained
}

// Start begins a reply to request that already begins with prefix (text the model must continue).
// It returns nil when the prefix does not start with a command word of the lexicon. Without a
// prefix the reply is only constrained once its first word is a command word, so ordinary answers
// are left alone.
func (g *Grammar) Start(request, prefix string) *State {
	s := &State{g: g}
	for _, w := range strings.Fields(request) {
		if w = trim(w); w != "" {
			s.request = append(s.request, w)
			if parts := splitParts(w); len(parts) > 1 {
				s.request = append(s.request, parts...)
			}
		}
	}
	if len(g.first) == 0 {
		return nil
	}
	if f := strings.Fields(prefix); len(f) > 0 && !slices.Contains(g.first, f[0]) {
		return nil
	}
	s.seg, s.cur = s.feed(nil, "", prefix)
	return s
}

// feed appends text to the (seg, cur) state and returns the new state.
func (s *State) feed(seg []string, cur, text string) ([]string, string) {
	for _, r := range text {
		if !unicode.IsSpace(r) {
			cur += string(r)
			continue
		}
		if cur == "" {
			continue
		}
		if glue[cur] {
			seg = nil
		} else {
			seg = append(seg, cur)
		}
		cur = ""
	}
	return seg, cur
}

// Advance records the sampled token.
func (s *State) Advance(id int) {
	if id >= 0 && id < len(s.g.text) {
		s.seg, s.cur = s.feed(s.seg, s.cur, s.g.text[id])
		if len(s.seg) > 0 && !slices.Contains(s.g.first, s.seg[0]) {
			s.off = true
		}
	}
}

// Mask sets the logits of tokens that cannot extend a plausible command to -Inf. It leaves the
// logits alone when nothing is allowed, so the model is never stuck.
func (s *State) Mask(logits []float64) {
	// Until the first word is a command word the reply may still be an ordinary answer.
	if s.off || (len(s.seg) == 0 && !slices.Contains(s.g.first, trim(s.cur))) {
		return
	}
	allowed := make([]bool, len(logits))
	n := 0
	for id := range logits {
		if id >= len(s.g.text) {
			continue
		}
		switch {
		case id == s.g.eos:
			allowed[id] = len(s.seg) >= 2 && (s.cur == "" || s.completeOK(s.seg, s.cur))
		case s.g.text[id] == "":
		default:
			allowed[id] = s.tokenOK(s.g.text[id])
		}
		if allowed[id] {
			n++
		}
	}
	if n == 0 {
		return
	}
	for id := range logits {
		if !allowed[id] {
			logits[id] = negInf
		}
	}
}

var negInf = -1e300

// tokenOK reports whether appending text keeps the command plausible: every word the token
// completes is allowed where it stands, and the word it leaves open can still become one.
func (s *State) tokenOK(text string) bool {
	if !strings.ContainsFunc(text, unicode.IsSpace) {
		return s.partialOK(s.seg, s.cur+text)
	}
	seg, cur := s.seg, s.cur
	for _, r := range text {
		if unicode.IsSpace(r) {
			if cur != "" {
				if !glue[cur] && !s.completeOK(seg, cur) {
					return false
				}
				if glue[cur] {
					seg = nil
				} else {
					seg = append(slices.Clone(seg), cur)
				}
				cur = ""
			}
			continue
		}
		cur += string(r)
	}
	return cur == "" || s.partialOK(seg, cur)
}

// candidates returns the words allowed at the first two positions of a command, nil elsewhere.
func (s *State) candidates(seg []string) []string {
	switch len(seg) {
	case 0:
		return s.g.first
	case 1:
		return s.g.second[seg[0]]
	}
	return nil
}

// wordOK reports whether w (not trimmed) may stand at the position after seg; partial words
// need only be a prefix of an allowed word.
func (s *State) wordOK(seg []string, w string, partial bool) bool {
	raw := w
	w = trim(w)
	if len(raw)-len(w) > maxTrimmed {
		return false
	}
	if w == "" {
		// A lone quote opens or closes a message; longer runs of punctuation are noise.
		return len(raw) <= 1
	}
	if glue[w] {
		return len(seg) >= 2
	}
	nw := norm(w)
	if cands := s.candidates(seg); len(cands) > 0 {
		return matches(cands, nw, partial)
	}
	if isOption(w) || (partial && strings.HasPrefix(w, "-")) {
		var optSet []string
		if len(seg) >= 2 {
			optSet = s.g.opts[seg[0]+" "+seg[1]]
		}
		return matches(optSet, nw, partial) || matches(s.request, w, partial)
	}
	return matches(s.g.common, nw, partial) || matches(s.request, w, partial)
}

func matches(set []string, w string, partial bool) bool {
	for _, c := range set {
		if c == w || (partial && strings.HasPrefix(c, w)) {
			return true
		}
	}
	return false
}

// completeOK checks a finished word, splitting compound words at partSeparators when the whole
// word is not allowed.
func (s *State) completeOK(seg []string, w string) bool {
	return s.compoundOK(seg, w, false)
}

func (s *State) partialOK(seg []string, w string) bool {
	return s.compoundOK(seg, w, true)
}

func (s *State) compoundOK(seg []string, w string, partial bool) bool {
	if s.wordOK(seg, w, partial) {
		return true
	}
	t := trim(w)
	// The command and subcommand are whole words of the lexicon, never compounds.
	if len(seg) < 2 || !strings.ContainsAny(t, partSeparators) {
		return false
	}
	parts := splitParts(t)
	if strings.ContainsRune(partSeparators, rune(t[len(t)-1])) {
		parts = append(parts, "") // open part after a trailing separator
	}
	for i, p := range parts {
		last := i == len(parts)-1
		if p == "" {
			continue
		}
		// Parts after the first are not option-checked against the subcommand position.
		probe := seg
		if i > 0 {
			probe = []string{"", "", ""}
		}
		if !s.wordOK(probe, p, partial && last) {
			return false
		}
	}
	return true
}

func splitParts(w string) []string {
	return strings.FieldsFunc(w, func(r rune) bool { return strings.ContainsRune(partSeparators, r) })
}
