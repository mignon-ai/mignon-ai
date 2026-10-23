//go:build amd64

package kernels

// gemmKernel6x16ASM computes the 6×16 tile C += A·B over k steps (AVX2/FMA). Strides are in
// elements: A(i, p) at a[i*rsA + p*csA], B(p, j) at b[p*ldb + j], C(i, j) at c[i*ldc + j].
//
//go:noescape
func gemmKernel6x16ASM(k int, a *float32, rsA, csA int, b *float32, ldb int, c *float32, ldc int)
