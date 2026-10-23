//go:build !amd64

package kernels

// gemmKernel6x16ASM is never called off amd64 (HasAVX2 is false); it exists so gemm.go builds.
func gemmKernel6x16ASM(k int, a *float32, rsA, csA int, b *float32, ldb int, c *float32, ldc int) {
	panic("gemmKernel6x16ASM: AVX2 kernel is amd64-only")
}
