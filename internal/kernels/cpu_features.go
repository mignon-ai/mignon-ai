package kernels

import "golang.org/x/sys/cpu"

// HasAVX2 reports whether the CPU supports the AVX2 and FMA instructions the assembly kernels
// need. When false, every kernel uses its pure-Go implementation. Tests clear it to exercise the
// fallbacks.
var HasAVX2 = cpu.X86.HasAVX2 && cpu.X86.HasFMA
