package optim

import (
	"math"
	"testing"
)

const scheduleTestTol = 1e-12

func TestDefaultScheduleMatchesCosineToFloor(t *testing.T) {
	const peak, total, warmup = 0.04, 200, 10
	for step := 0; step <= total; step++ {
		want := peak
		if step < warmup {
			want = peak * (LRWarmupStartFraction + (1-LRWarmupStartFraction)*float64(step)/warmup)
		} else {
			progress := float64(step-warmup) / float64(total-warmup)
			want = peak * (DefaultLRFloorFraction + (1-DefaultLRFloorFraction)*0.5*(1+math.Cos(math.Pi*progress)))
		}
		if got := (Schedule{}).At(peak, step, total, warmup); math.Abs(got-want) > scheduleTestTol {
			t.Fatalf("step %d: got %g, want %g", step, got, want)
		}
	}
}

func TestLinearScheduleDecaysToFloor(t *testing.T) {
	const peak, total, warmup = 1.0, 110, 10
	s := Schedule{Shape: ScheduleLinear, Floor: 0}
	for _, tc := range []struct {
		step int
		want float64
	}{{warmup, 1}, {60, 0.5}, {total, 0}, {total + 5, 0}} {
		if got := s.At(peak, tc.step, total, warmup); math.Abs(got-tc.want) > scheduleTestTol {
			t.Fatalf("step %d: got %g, want %g", tc.step, got, tc.want)
		}
	}
}

func TestWarmupStepsMatchesFraction(t *testing.T) {
	for _, total := range []int{0, 1, 2, 19, 20, 100, 1234} {
		want := total * 5 / 100 // LRWarmupFraction = 5%
		if want < 1 && total > 1 {
			want = 1
		}
		if got := WarmupSteps(total); got != want {
			t.Fatalf("WarmupSteps(%d) = %d, want %d", total, got, want)
		}
	}
}

func TestLRScheduleValidate(t *testing.T) {
	valid := []Schedule{{}, DefaultSchedule(), {Shape: ScheduleLinear, Floor: 0}}
	for _, s := range valid {
		if err := s.Validate(); err != nil {
			t.Fatalf("%+v: unexpected error %v", s, err)
		}
	}
	invalid := []Schedule{{Shape: "step", Floor: 0.1}, {Shape: ScheduleCosine, Floor: -0.1}, {Shape: ScheduleLinear, Floor: 2}}
	for _, s := range invalid {
		if s.Validate() == nil {
			t.Fatalf("%+v: expected an error", s)
		}
	}
}

// TestDefaultScheduleWarmupAndDecay verifies per-step linear warmup and cosine decay to 10%
func TestDefaultScheduleWarmupAndDecay(t *testing.T) {
	initialLR := 0.01
	totalSteps := 100
	warmupSteps := 10

	// Step 0: starts at 10%
	lr0 := DefaultSchedule().At(initialLR, 0, totalSteps, warmupSteps)
	expected0 := initialLR * 0.1
	if math.Abs(lr0-expected0) > 1e-6 {
		t.Fatalf("Step 0 LR mismatch: got %f, want %f", lr0, expected0)
	}

	// Warmup midpoint: step 5 -> 55%
	lr5 := DefaultSchedule().At(initialLR, 5, totalSteps, warmupSteps)
	expected5 := initialLR * (0.1 + 0.9*0.5)
	if math.Abs(lr5-expected5) > 1e-6 {
		t.Fatalf("Step 5 LR mismatch: got %f, want %f", lr5, expected5)
	}

	// End of warmup: step 10 -> 100%
	lr10 := DefaultSchedule().At(initialLR, 10, totalSteps, warmupSteps)
	if math.Abs(lr10-initialLR) > 1e-6 {
		t.Fatalf("Step 10 LR mismatch: got %f, want %f", lr10, initialLR)
	}

	// Final step: step 100 -> 10%
	lr100 := DefaultSchedule().At(initialLR, 100, totalSteps, warmupSteps)
	expected100 := initialLR * 0.1
	if math.Abs(lr100-expected100) > 1e-6 {
		t.Fatalf("Step 100 LR mismatch: got %f, want %f", lr100, expected100)
	}
}
