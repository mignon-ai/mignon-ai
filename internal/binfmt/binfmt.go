// Package binfmt is the container of the binary model file (mignon.model/1). Layout, little-endian:
//
//	"MGNN"            4 bytes   magic
//	version           uint16    1
//	kind              uint8     Full or Int8
//	reserved          uint8     0
//	header length     uint32    n
//	header            n bytes   JSON: metadata, tokenizer and the model's shape (everything small)
//	arrays            ...       the weight arrays, in the order the kind defines
//
// An array is a uint32 element count followed by the elements: int8 or float32 (IEEE 754, little-endian).
// Weights make up almost all of a model, and as raw numbers they take 1 byte (INT8) or 4 bytes
// (float32) each, where JSON text needs about 5 to 12.
package binfmt

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"math"
)

// Magic starts every binary model file.
const Magic = "MGNN"

// Version is the container version this package reads and writes.
const Version = 1

// Kind says which model the arrays belong to.
type Kind uint8

const (
	Full Kind = 1 // full-precision model: one float32 array of all weights
	Int8 Kind = 2 // INT8 model: tensors and vectors in the order of quant.arrays
)

// Array is one weight array to write or fill: exactly one of the fields is set.
type Array struct {
	I8  *[]int8
	F32 *[]float32
}

// IsBinary reports whether data starts like a binary model file.
func IsBinary(data []byte) bool { return bytes.HasPrefix(data, []byte(Magic)) }

// Encode writes a model file.
func Encode(kind Kind, header []byte, arrays []Array) []byte {
	size := 12 + len(header)
	for _, a := range arrays {
		switch {
		case a.I8 != nil:
			size += 4 + len(*a.I8)
		case a.F32 != nil:
			size += 4 + 4*len(*a.F32)
		}
	}
	buf := make([]byte, 0, size)
	buf = append(buf, Magic...)
	buf = binary.LittleEndian.AppendUint16(buf, Version)
	buf = append(buf, byte(kind), 0)
	buf = binary.LittleEndian.AppendUint32(buf, uint32(len(header)))
	buf = append(buf, header...)
	for _, a := range arrays {
		switch {
		case a.I8 != nil:
			buf = binary.LittleEndian.AppendUint32(buf, uint32(len(*a.I8)))
			for _, v := range *a.I8 {
				buf = append(buf, byte(v))
			}
		case a.F32 != nil:
			buf = binary.LittleEndian.AppendUint32(buf, uint32(len(*a.F32)))
			for _, v := range *a.F32 {
				buf = binary.LittleEndian.AppendUint32(buf, math.Float32bits(v))
			}
		}
	}
	return buf
}

// File is a parsed model file whose arrays have not been read yet.
type File struct {
	Kind   Kind
	Header []byte
	rest   []byte
}

// Decode checks the container and splits off the header.
func Decode(data []byte) (*File, error) {
	if !IsBinary(data) {
		return nil, fmt.Errorf("not a binary model file")
	}
	if len(data) < 12 {
		return nil, fmt.Errorf("binary model file is truncated")
	}
	if v := binary.LittleEndian.Uint16(data[4:]); v != Version {
		return nil, fmt.Errorf("binary model file version %d is not supported (this build reads version %d)", v, Version)
	}
	kind := Kind(data[6])
	if kind != Full && kind != Int8 {
		return nil, fmt.Errorf("unknown model kind %d", kind)
	}
	n := int(binary.LittleEndian.Uint32(data[8:]))
	if n < 0 || 12+n > len(data) {
		return nil, fmt.Errorf("binary model file is truncated (header)")
	}
	return &File{Kind: kind, Header: data[12 : 12+n], rest: data[12+n:]}, nil
}

// Fill reads the arrays in order and stores them through the pointers of a. Every array must be present
// and nothing may follow the last one.
func (f *File) Fill(arrays []Array) error {
	rest := f.rest
	for i, a := range arrays {
		if len(rest) < 4 {
			return fmt.Errorf("binary model file is truncated (array %d)", i)
		}
		n := int(binary.LittleEndian.Uint32(rest))
		rest = rest[4:]
		switch {
		case a.I8 != nil:
			if n > len(rest) {
				return fmt.Errorf("binary model file is truncated (array %d)", i)
			}
			out := make([]int8, n)
			for j := range out {
				out[j] = int8(rest[j])
			}
			*a.I8, rest = out, rest[n:]
		case a.F32 != nil:
			if n > len(rest)/4 {
				return fmt.Errorf("binary model file is truncated (array %d)", i)
			}
			out := make([]float32, n)
			for j := range out {
				out[j] = math.Float32frombits(binary.LittleEndian.Uint32(rest[4*j:]))
			}
			*a.F32, rest = out, rest[4*n:]
		}
	}
	if len(rest) != 0 {
		return fmt.Errorf("binary model file has %d unexpected trailing bytes", len(rest))
	}
	return nil
}
