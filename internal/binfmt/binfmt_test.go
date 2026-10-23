package binfmt

import (
	"math"
	"slices"
	"testing"
)

func TestRoundTrip(t *testing.T) {
	i8 := []int8{-127, 0, 5, 127}
	f32 := []float32{0, -1.5, math.MaxFloat32, 1e-30}
	var empty []int8
	data := Encode(Int8, []byte(`{"k":1}`), []Array{{I8: &i8}, {F32: &f32}, {I8: &empty}})
	f, err := Decode(data)
	if err != nil || f.Kind != Int8 || string(f.Header) != `{"k":1}` {
		t.Fatalf("%v %+v", err, f)
	}
	var gotI8, gotEmpty []int8
	var gotF32 []float32
	if err := f.Fill([]Array{{I8: &gotI8}, {F32: &gotF32}, {I8: &gotEmpty}}); err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(gotI8, i8) || !slices.Equal(gotF32, f32) || len(gotEmpty) != 0 {
		t.Fatalf("%v %v %v", gotI8, gotF32, gotEmpty)
	}
}

func TestDecodeRejectsNonBinary(t *testing.T) {
	if IsBinary([]byte(`{"weights":[]}`)) {
		t.Fatal("JSON taken for binary")
	}
	if _, err := Decode([]byte("MGNN")); err == nil {
		t.Fatal("truncated file accepted")
	}
}
