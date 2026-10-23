package optim

import (
	"fmt"
	"math"
)

// Learning-rate schedule: linear warmup from LRWarmupStartFraction of the peak, then a decay
// of the chosen shape down to Floor·peak at the last step.
const (
	ScheduleCosine = "cosine"
	ScheduleLinear = "linear"

	// LRWarmupFraction is the share of all optimizer steps spent warming up.
	LRWarmupFraction = 0.05
	// LRWarmupStartFraction is the LR at step 0 as a fraction of the peak.
	LRWarmupStartFraction = 0.1
	// DefaultLRFloorFraction is the final LR as a fraction of the peak.
	DefaultLRFloorFraction = 0.1
)

// Schedule describes the decay after warmup. The zero value means DefaultSchedule.
type Schedule struct {
	Shape string  // ScheduleCosine or ScheduleLinear
	Floor float64 // final LR as a fraction of the peak
}

// DefaultSchedule returns the schedule used when none is configured.
func DefaultSchedule() Schedule {
	return Schedule{Shape: ScheduleCosine, Floor: DefaultLRFloorFraction}
}

// orDefault returns DefaultSchedule for the zero value.
func (s Schedule) orDefault() Schedule {
	if s == (Schedule{}) {
		return DefaultSchedule()
	}
	return s
}

// Validate reports an unknown shape or a floor outside [0, 1].
func (s Schedule) Validate() error {
	s = s.orDefault()
	switch s.Shape {
	case ScheduleCosine, ScheduleLinear:
	default:
		return fmt.Errorf("unknown lr schedule %q", s.Shape)
	}
	if s.Floor < 0 || s.Floor > 1 {
		return fmt.Errorf("lr floor %g must be within [0, 1]", s.Floor)
	}
	return nil
}

// WarmupSteps returns the number of warmup steps for a run of totalSteps.
func WarmupSteps(totalSteps int) int {
	warmup := int(float64(totalSteps) * LRWarmupFraction)
	if warmup < 1 && totalSteps > 1 {
		warmup = 1
	}
	return warmup
}

// At returns the learning rate for an optimizer step.
func (s Schedule) At(peak float64, step, totalSteps, warmupSteps int) float64 {
	s = s.orDefault()
	if totalSteps <= 1 {
		return peak
	}
	if warmupSteps > 0 && step < warmupSteps {
		ratio := float64(step) / float64(warmupSteps)
		return peak * (LRWarmupStartFraction + (1-LRWarmupStartFraction)*ratio)
	}
	decaySteps := totalSteps - warmupSteps
	if decaySteps <= 0 {
		return peak
	}
	progress := math.Min(float64(step-warmupSteps)/float64(decaySteps), 1)
	remaining := 1 - progress // linear: 1 → 0
	if s.Shape == ScheduleCosine {
		remaining = 0.5 * (1 + math.Cos(math.Pi*progress))
	}
	return peak * (s.Floor + (1-s.Floor)*remaining)
}
