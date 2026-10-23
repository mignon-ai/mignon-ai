// Package golden pins the observable behaviour of every package (weights, logits, checkpoints,
// tokenizer and data pipeline) to a committed snapshot, so refactors can prove they change nothing.
// It contains only tests.
package golden
