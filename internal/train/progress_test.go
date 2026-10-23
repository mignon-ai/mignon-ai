package train

import (
	"strings"
	"testing"
)

func TestRenderProgressBar(t *testing.T) {
	bar0 := renderProgressBar(0, 10)
	if strings.Contains(bar0, "█") {
		t.Errorf("0%% progress bar should not have filled blocks: got %q", bar0)
	}

	bar100 := renderProgressBar(100, 10)
	if strings.Contains(bar100, "░") {
		t.Errorf("100%% progress bar should not have empty blocks: got %q", bar100)
	}

	bar50 := renderProgressBar(50, 10)
	if len([]rune(bar50)) != 10 {
		t.Errorf("Expected width 10, got %d", len([]rune(bar50)))
	}
}
