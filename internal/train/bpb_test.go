package train

import "testing"

func TestTargetBytesSkipsMaskedAndPastValidLength(t *testing.T) {
	tokenBytes := []int{0, 1, 3}
	ys := [][]int{{1, 2, -1, 2}, {2, 2, 2, 2}}
	counts, total := targetBytes(ys, []int{3, 2}, tokenBytes)
	if counts[0] != 2 || counts[1] != 2 || total != 1+3+3+3 {
		t.Fatalf("counts %v total %d, want [2 2] and 10", counts, total)
	}
}
