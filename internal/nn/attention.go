package nn

import (
	"math"
	"math/rand"

	"github.com/mignon-ai/mignon-ai/internal/kernels"
	"github.com/mignon-ai/mignon-ai/internal/optim"
)

// Attention is a pre-norm causal multi-head softmax attention sublayer on a residual stream x
// (N x D): x += Wo · Attend(Wqkv · RMSNorm(x)). Position i attends to positions
// WindowStart(i)..i. Its parameters live in a flat weight vector at the offsets below.
type Attention struct {
	D, Heads, Window int
	Norm             int // D RMSNorm scales
	QKV              int // 3D x D query, key and value projections
	Out              int // D x D output projection
}

// AttentionParams returns the parameter count of an attention sublayer of width D.
func AttentionParams(D int) int {
	return D + 4*D*D
}

// NewAttention places an attention sublayer of width D at offset off.
func NewAttention(off, D, heads, window int) Attention {
	return Attention{D: D, Heads: heads, Window: window, Norm: off, QKV: off + D, Out: off + D + 3*D*D}
}

// Matrices lists the sublayer's projection matrices for Muon.
func (a Attention) Matrices() []optim.Matrix {
	D := a.D
	return []optim.Matrix{
		{Name: "attn_qkv", Offset: a.QKV, Rows: 3 * D, Cols: D, Stride: D},
		{Name: "attn_out", Offset: a.Out, Rows: D, Cols: D, Stride: D},
	}
}

// Init sets unit norm scales, uniform ±sqrt(gain/D) query/key/value projections and a zero output
// projection, so the sublayer starts as the identity on the residual stream.
func (a Attention) Init(w []float32, rng *rand.Rand, gain float64) {
	D := a.D
	for d := 0; d < D; d++ {
		w[a.Norm+d] = 1
	}
	scale := math.Sqrt(gain / float64(D))
	for i := a.QKV; i < a.QKV+3*D*D; i++ {
		w[i] = float32((rng.Float64()*2 - 1) * scale)
	}
	clear(w[a.Out : a.Out+D*D])
}

// WindowStart returns the first position query i attends to (0 for full causal attention).
func WindowStart(i, window int) int {
	if window > 0 && i-window+1 > 0 {
		return i - window + 1
	}
	return 0
}

// AttentionActs holds one sublayer's forward activations for the backward pass.
type AttentionActs struct {
	n, xHat []float32 // N x D norm output and normalized input
	rms     []float32 // N
	qkv     []float32 // N x 3D
	probs   []float32 // heads x N x N attention probabilities
	att     []float32 // N x D attention output before the output projection
}

// NewAttentionActs allocates activations for up to maxContext positions.
func (a Attention) NewAttentionActs(maxContext int) AttentionActs {
	N, D := maxContext, a.D
	return AttentionActs{
		n: make([]float32, N*D), xHat: make([]float32, N*D), rms: make([]float32, N),
		qkv: make([]float32, N*3*D), probs: make([]float32, a.Heads*N*N), att: make([]float32, N*D),
	}
}

// AttentionScratch holds per-worker backward buffers shared by every sublayer of a model.
type AttentionScratch struct {
	dAtt, dn []float32 // N x D
	dqkv     []float32 // N x 3D
	row      []float32 // N
	Pack     []float32 // packing scratch for the weight-gradient products
}

// NewAttentionScratch allocates scratch for up to maxContext positions of width D.
func NewAttentionScratch(maxContext, D int) AttentionScratch {
	N := maxContext
	return AttentionScratch{dAtt: make([]float32, N*D), dn: make([]float32, N*D), dqkv: make([]float32, N*3*D), row: make([]float32, N)}
}

// Forward adds the sublayer's output to x (N x D) in place, recording activations in acts. packs
// holds the projections of w packed for the forward products (MatrixPack.Fwd).
func (a Attention) Forward(N int, w, x []float32, acts *AttentionActs, s *AttentionScratch, packs *AttentionPacks) {
	D := a.D
	RMSNormRows(N, D, x, w[a.Norm:a.Norm+D], acts.n, acts.xHat, acts.rms)
	clear(acts.qkv[:N*3*D])
	Linear(N, D, 3*D, acts.n, &packs.QKV.Fwd, acts.qkv)
	a.attend(N, acts.qkv, acts.probs, acts.att, s.row)
	Linear(N, D, D, acts.att, &packs.Out.Fwd, x)
}

// Backward takes dx, the gradient wrt the sublayer's output, and adds the gradient through the
// attention branch so dx becomes the gradient wrt its input (the residual passes unchanged).
// Parameter gradients accumulate into grads. packs holds the projections of w packed for the
// input gradients (MatrixPack.Bwd).
func (a Attention) Backward(N int, w, grads, dx []float32, acts *AttentionActs, s *AttentionScratch, packs *AttentionPacks) {
	D := a.D
	clear(s.dAtt[:N*D])
	LinearBackward(&s.Pack, N, D, D, acts.att, dx, &packs.Out.Bwd, grads[a.Out:], D, s.dAtt)
	clear(s.dqkv[:N*3*D])
	a.attendBackward(N, acts.qkv, acts.probs, s.dAtt, s.dqkv, s.row)
	clear(s.dn[:N*D])
	LinearBackward(&s.Pack, N, D, 3*D, acts.n, s.dqkv, &packs.QKV.Bwd, grads[a.QKV:], D, s.dn)
	RMSNormRowsBackward(N, D, s.dn, acts.xHat, w[a.Norm:a.Norm+D], acts.rms, dx, grads[a.Norm:a.Norm+D])
}

// attend computes the attention output o (N x D) from packed qkv rows [q | k | v] (N x 3D),
// storing the probabilities per head as N x N row-major blocks (only the attended band of each
// row is written and read). scratch must hold at least N values.
func (a Attention) attend(N int, qkv, probs, o, scratch []float32) {
	D, dh := a.D, a.D/a.Heads
	scale := float32(1 / math.Sqrt(float64(dh)))
	clear(o[:N*D])
	for h := 0; h < a.Heads; h++ {
		hp := probs[h*N*N:]
		for i := 0; i < N; i++ {
			lo := WindowStart(i, a.Window)
			q := qkv[i*3*D+h*dh : i*3*D+(h+1)*dh]
			row := hp[i*N+lo : i*N+i+1]
			for j := lo; j <= i; j++ {
				row[j-lo] = kernels.DotF32(q, qkv[j*3*D+D+h*dh:j*3*D+D+(h+1)*dh]) * scale
			}
			SoftmaxInPlaceF32(row, scratch[:len(row)])
			out := o[i*D+h*dh : i*D+(h+1)*dh]
			for j := lo; j <= i; j++ {
				kernels.AddScaledF32(out, qkv[j*3*D+2*D+h*dh:j*3*D+2*D+(h+1)*dh], row[j-lo])
			}
		}
	}
}

// attendBackward accumulates into dqkv (N x 3D) the gradient of attend given dO (N x D).
func (a Attention) attendBackward(N int, qkv, probs, dO, dqkv, scratch []float32) {
	D, dh := a.D, a.D/a.Heads
	scale := float32(1 / math.Sqrt(float64(dh)))
	for h := 0; h < a.Heads; h++ {
		hp := probs[h*N*N:]
		for i := 0; i < N; i++ {
			lo := WindowStart(i, a.Window)
			row := hp[i*N+lo : i*N+i+1]
			dOi := dO[i*D+h*dh : i*D+(h+1)*dh]
			// dP_j = dO_i · v_j, then the softmax Jacobian: dS_j = P_j (dP_j - Σ_k P_k dP_k).
			dP := scratch[:len(row)]
			var sum float32
			for j := lo; j <= i; j++ {
				dP[j-lo] = kernels.DotF32(dOi, qkv[j*3*D+2*D+h*dh:j*3*D+2*D+(h+1)*dh])
				sum += row[j-lo] * dP[j-lo]
			}
			q := qkv[i*3*D+h*dh : i*3*D+(h+1)*dh]
			dq := dqkv[i*3*D+h*dh : i*3*D+(h+1)*dh]
			for j := lo; j <= i; j++ {
				p := row[j-lo]
				ds := p * (dP[j-lo] - sum) * scale
				kOff := j*3*D + D + h*dh
				vOff := j*3*D + 2*D + h*dh
				kernels.AddScaledF32(dq, qkv[kOff:kOff+dh], ds)
				kernels.AddScaledF32(dqkv[kOff:kOff+dh], q, ds)
				kernels.AddScaledF32(dqkv[vOff:vOff+dh], dOi, p)
			}
		}
	}
}
