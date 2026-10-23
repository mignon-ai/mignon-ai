// Package train runs mini-batch training of a ca.Model on dense causal sequences (parallel
// workers, optimizer, LR schedule, validation and best-epoch restore) and describes experiments:
// Config layers architecture and hyperparameters over the tuned model-size presets.
package train

import (
	"cmp"
	"context"
	"fmt"
	"io"
	"math"
	"math/rand"
	"os"
	"slices"
	"time"

	"github.com/mignon-ai/mignon-ai/internal/ca"
	"github.com/mignon-ai/mignon-ai/internal/data"
	"github.com/mignon-ai/mignon-ai/internal/optim"
	"github.com/mignon-ai/mignon-ai/internal/tokenizer"
)

// CheckpointFunc is called whenever validation or training loss achieves a new best epoch
type CheckpointFunc func(epoch int, loss float64, weights []float32)

// Options configures Run.
type Options struct {
	Epochs       int
	LR           float64
	Optimizer    string // "adam" or "muon"
	BatchSize    int    // 0 = auto
	WeightDecay  float64
	Threads      int            // 0 = auto
	Seed         int64          // data-order seed; 0 = nondeterministic
	Schedule     optim.Schedule // zero value = optim.DefaultSchedule
	EmbedLRScale float64        // Muon only: embedding/head LR as a fraction of LR; 0 = DefaultMuonEmbedLRScale
	Quiet        bool           // suppress progress and per-epoch output
	Log          io.Writer      // where progress goes unless Quiet; nil = standard output
	TokenBytes   []int          // text bytes per token id (tokenizer.TokenBytes); enables bits per byte
	OnCheckpoint CheckpointFunc
	OnEpoch      func(EpochStats) // called after every completed epoch
	// Context, when cancelled, stops training before the next batch: Run restores the best weights seen so
	// far and returns with Result.Canceled set. Nil = never cancelled.
	Context context.Context
}

// EpochStats describes one completed epoch.
type EpochStats struct {
	Epoch, Epochs int // 1-based
	TrainLoss     float64
	ValLoss       float64 // 0 without validation data
	ValBPB        float64 // 0 without validation data or token bytes
	TokensPerSec  float64
	LR            float64
	Duration      time.Duration
	Best          bool // the epoch improved on all earlier ones
}

// Result summarizes a training run. Losses are validation losses when validation data
// was provided, otherwise training losses.
type Result struct {
	BestLoss       float64
	BestEpoch      int
	FinalLoss      float64
	FinalTrainLoss float64
	BestBPB        float64 // validation bits per byte at the best epoch (0 without TokenBytes or validation)
	FinalBPB       float64
	TokensPerSec   float64 // mean training throughput across epochs
	Duration       time.Duration
	Canceled       bool // Options.Context ended the run early
}

// newOptimizer returns the optimizer named by name ("muon", anything else is AdamW).
// embedLRScale > 0 overrides Muon's embedding LR fraction.
func newOptimizer(name string, params optim.ParamLayout, embedLRScale float64) optim.Optimizer {
	if name == "muon" {
		muon := optim.NewMuon(params, optim.DefaultMuonMomentum, optim.DefaultMuonNSSteps, optim.GradClipAdam)
		if embedLRScale > 0 {
			muon.EmbedLRScale = embedLRScale
		}
		return muon
	}
	return optim.NewAdam(params.TotalParams, optim.GradClipAdam)
}

// Run trains model on dense causal sequences and restores the best weights at the end.
func Run(model *ca.Model, trainSet, valSet data.Sequences, opts Options) Result {
	var result Result
	if trainSet.Len() == 0 || !trainSet.Valid() || !valSet.Valid() {
		return result
	}
	trainStart := time.Now()
	logOut := opts.Log
	if logOut == nil {
		logOut = os.Stdout
	}
	logf := func(format string, args ...any) {
		if !opts.Quiet {
			fmt.Fprintf(logOut, format, args...)
		}
	}
	shuffleRNG := rand.New(rand.NewSource(rand.Int63()))
	if opts.Seed != 0 {
		shuffleRNG = rand.New(rand.NewSource(opts.Seed))
	}

	numParams := model.NumParams()
	weights := slices.Clone(model.Weights())
	params := model.OptimParams()
	if params.TotalParams != numParams {
		panic(fmt.Sprintf("parameter layout mismatch: layout=%d vs model=%d", params.TotalParams, numParams))
	}

	maxContext := maxSeqLen(trainSet, valSet)
	numWorkers := resolveWorkers(opts.Threads, trainSet.Len())
	// Workers may share state derived from the weights (such as packed matrices); refresh
	// rebuilds it after every optimizer step.
	workers, refresh := model.NewWorkers(numWorkers, maxContext)
	refresh(weights, numWorkers)
	eng := &engine{workers: workers, weights: weights, batchGrads: make([]float64, numParams)}

	batchSize := max(opts.BatchSize, DefaultMinBatchSize)
	if opts.BatchSize <= 0 {
		batchSize = max(numWorkers*8, DefaultMinBatchSize)
	}

	trainLens, totalTokens := validLengths(trainSet, maxContext)
	valLens, _ := validLengths(valSet, maxContext)
	valTargets, valBytes := targetBytes(valSet.Y, valLens, opts.TokenBytes)

	valMsg := ""
	if valSet.Len() > 0 {
		valMsg = fmt.Sprintf(", val: %d seqs", valSet.Len())
	}
	logf("%s: %d CPU worker threads (batch size: %d, optimizer: %s, params: %d, tokens/epoch: %d%s)\n",
		model.Describe(), numWorkers, batchSize, opts.Optimizer, numParams, totalTokens, valMsg)

	opt := newOptimizer(opts.Optimizer, params, opts.EmbedLRScale)
	best := newBestTracker(weights, opts.OnCheckpoint)

	batchOrder := make([]int, batchSize)
	totalSteps := opts.Epochs * ((trainSet.Len() + batchSize - 1) / batchSize)
	warmupSteps := optim.WarmupSteps(totalSteps)
	currentStep := 0

	canceled := func() bool { return opts.Context != nil && opts.Context.Err() != nil }
	for epoch := 0; epoch < opts.Epochs && !result.Canceled; epoch++ {
		epochStart := time.Now()
		lastProgressPrint := time.Now()
		processedTokens := 0
		var currentLR float64

		indices := shuffleRNG.Perm(trainSet.Len())
		epochLoss := 0.0
		numBatches := 0

		for bStart := 0; bStart < trainSet.Len(); bStart += batchSize {
			if canceled() {
				result.Canceled = true
				break
			}
			currentLR = opts.Schedule.At(opts.LR, currentStep, totalSteps, warmupSteps)
			currentStep++

			bEnd := min(bStart+batchSize, trainSet.Len())
			batch := longestFirst(batchOrder, indices[bStart:bEnd], trainLens)

			batchLoss := eng.batchGradient(trainSet, trainLens, batch)
			invB := 1.0 / float64(len(batch))
			epochLoss += batchLoss * invB
			numBatches++

			opt.Step(weights, eng.batchGrads, currentLR, opts.WeightDecay, invB, numWorkers)
			refresh(weights, numWorkers)

			for _, idx := range batch {
				processedTokens += trainLens[idx]
			}

			if time.Since(lastProgressPrint) >= 1*time.Second || bEnd == trainSet.Len() {
				lastProgressPrint = time.Now()
				logf("%s", epochProgress(epoch+1, opts.Epochs, bEnd, trainSet.Len(), processedTokens, totalTokens,
					time.Since(epochStart), epochLoss/float64(numBatches)))
			}
		}

		if result.Canceled {
			break // a partial epoch is neither validated nor counted
		}
		meanLoss := epochLoss / float64(numBatches)
		duration := time.Since(epochStart)
		tokensPerSec := float64(totalTokens) / duration.Seconds()

		stats := EpochStats{Epoch: epoch + 1, Epochs: opts.Epochs, TrainLoss: meanLoss, TokensPerSec: tokensPerSec, LR: currentLR, Duration: duration}
		if valSet.Len() > 0 {
			lossSum, nats := eng.validate(valSet, valLens, valTargets)
			valLoss := lossSum / float64(valSet.Len())
			valBPB := tokenizer.BitsPerByte(nats, valBytes)

			if best.update(epoch+1, valLoss, weights) {
				result.BestBPB = valBPB
				stats.Best = true
			}
			stats.ValLoss, stats.ValBPB = valLoss, valBPB

			result.FinalLoss = valLoss
			result.FinalBPB = valBPB
			bpbMsg := ""
			if valBytes > 0 {
				bpbMsg = fmt.Sprintf(", BPB: %.4f", valBPB)
			}
			logf("\rEpoch %3d/%d [%v, %.0f tok/s] - Train Loss: %.4f | Val Loss: %.4f (Val PPL: %.2f%s, LR: %.4f)                     \n",
				epoch+1, opts.Epochs, duration.Round(time.Millisecond), tokensPerSec, meanLoss, valLoss, math.Exp(valLoss), bpbMsg, currentLR)
		} else {
			stats.Best = best.update(epoch+1, meanLoss, weights)
			result.FinalLoss = meanLoss
			logf("\rEpoch %3d/%d [%v, %.0f tok/s] - Cross-Entropy: %.4f (Perplexity: %.2f, LR: %.4f)                     \n",
				epoch+1, opts.Epochs, duration.Round(time.Millisecond), tokensPerSec, meanLoss, math.Exp(meanLoss), currentLR)
		}

		result.FinalTrainLoss = meanLoss
		result.TokensPerSec += tokensPerSec / float64(opts.Epochs)
		model.SetWeights(weights)
		if opts.OnEpoch != nil {
			opts.OnEpoch(stats)
		}
	}

	result.BestLoss = best.loss
	result.BestEpoch = best.epoch
	result.Duration = time.Since(trainStart)

	if result.Canceled {
		logf("Training cancelled.\n")
		if best.epoch >= 0 {
			model.SetWeights(best.weights)
		}
	} else if best.epoch >= 0 && best.epoch != opts.Epochs {
		label := "Loss"
		if valSet.Len() > 0 {
			label = "Val Loss"
		}
		logf("Training complete. Restoring best model from Epoch %d (%s: %.4f, PPL: %.2f)\n",
			best.epoch, label, best.loss, math.Exp(best.loss))
		model.SetWeights(best.weights)
	} else {
		logf("Training complete.\n")
	}
	return result
}

// initialBestLoss is the loss reported before any epoch has run (finite, so results stay JSON-encodable).
const initialBestLoss = 1e9

// bestTracker remembers the weights of the epoch with the lowest loss so far.
type bestTracker struct {
	loss         float64
	epoch        int // 1-based; -1 before the first update
	weights      []float32
	onCheckpoint CheckpointFunc
}

func newBestTracker(weights []float32, onCheckpoint CheckpointFunc) *bestTracker {
	return &bestTracker{loss: initialBestLoss, epoch: -1, weights: slices.Clone(weights), onCheckpoint: onCheckpoint}
}

// update records weights as the best if loss improves on it, reports the checkpoint and says
// whether it did.
func (b *bestTracker) update(epoch int, loss float64, weights []float32) bool {
	if loss >= b.loss {
		return false
	}
	b.loss, b.epoch = loss, epoch
	copy(b.weights, weights)
	if b.onCheckpoint != nil {
		b.onCheckpoint(epoch, loss, b.weights)
	}
	return true
}

// longestFirst copies batch into buf ordered by descending sequence length, so the workers
// pulling sequences in order finish the batch together instead of waiting on a long one
// picked up last. The batch's gradient is the same sum; work stealing already varies its order.
func longestFirst(buf, batch, lens []int) []int {
	out := buf[:len(batch)]
	copy(out, batch)
	slices.SortStableFunc(out, func(a, b int) int { return cmp.Compare(lens[b], lens[a]) })
	return out
}
