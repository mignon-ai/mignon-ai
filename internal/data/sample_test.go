package data

import (
	"fmt"
	"strings"
	"testing"
)

// The tokenizer sample must cover a large document from start to end, in whole lines, and stay
// within budget.
func TestSpreadSampleCoversWholeDocument(t *testing.T) {
	const lines = 20000
	var doc strings.Builder
	for i := 0; i < lines; i++ {
		fmt.Fprintf(&doc, "line %d\n", i)
	}
	text := doc.String()
	budget := len(text) / 10

	var sb strings.Builder
	writeSpreadSample(&sb, text, budget)
	sample := sb.String()
	if len(sample) > budget {
		t.Fatalf("sample of %d bytes exceeds budget %d", len(sample), budget)
	}
	last := -1
	for _, line := range strings.Split(strings.TrimSuffix(sample, "\n"), "\n") {
		if _, err := fmt.Sscanf(line, "line %d", &last); err != nil {
			t.Fatalf("sample cut a line: %q", line)
		}
	}
	if last < lines*9/10 {
		t.Fatalf("sample ends at line %d of %d; it must reach the end of the document", last, lines)
	}
}
