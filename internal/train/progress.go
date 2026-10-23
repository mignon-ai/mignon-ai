package train

import (
	"fmt"
	"strings"
	"time"
)

// renderProgressBar formats a visual terminal progress bar of given character width
func renderProgressBar(pct float64, width int) string {
	if width <= 0 {
		width = 20
	}
	filled := int(pct / 100.0 * float64(width))
	if filled > width {
		filled = width
	}
	if filled < 0 {
		filled = 0
	}
	var sb strings.Builder
	for i := 0; i < filled; i++ {
		sb.WriteString("█")
	}
	for i := filled; i < width; i++ {
		sb.WriteString("░")
	}
	return sb.String()
}

// epochProgress formats the in-place progress line of a running epoch: progress bar, throughput,
// running loss and the estimated time left, from the sequences and tokens done so far.
func epochProgress(epoch, epochs, seqsDone, seqs, tokensDone, tokens int, elapsed time.Duration, loss float64) string {
	pct := float64(seqsDone) / float64(seqs) * 100.0
	rate := float64(tokensDone) / elapsed.Seconds()
	eta := "0s"
	if rate > 0 && tokensDone < tokens {
		eta = (time.Duration(float64(tokens-tokensDone)/rate) * time.Second).Round(time.Second).String()
	}
	return fmt.Sprintf("\r  Epoch %d/%d [%-20s] %5.1f%% (%d/%d seqs, %.0f tok/s, loss: %.4f, ETA: %s)   ",
		epoch, epochs, renderProgressBar(pct, 20), pct, seqsDone, seqs, rate, loss, eta)
}
