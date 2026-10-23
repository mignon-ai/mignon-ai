package train

// targetBytes sums the text bytes the valid targets of every sequence decode to, and counts the
// targets, so a mean loss per sequence can be turned back into total nats per byte.
func targetBytes(ys [][]int, lens []int, tokenBytes []int) (counts []int, total int) {
	counts = make([]int, len(ys))
	for i, y := range ys {
		for _, t := range y[:lens[i]] {
			if t >= 0 && t < len(tokenBytes) {
				counts[i]++
				total += tokenBytes[t]
			}
		}
	}
	return counts, total
}
