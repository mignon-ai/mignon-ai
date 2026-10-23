package quant

import (
	"math"

	"github.com/mignon-ai/mignon-ai/internal/kernels"
	"github.com/mignon-ai/mignon-ai/internal/nn"
)

// INT8 attention sublayer of hybrid models: after each block, the block output x is updated to
// x += Wo · Attend(Wqkv · RMSNorm(x)) over the keys and values of earlier positions. Keys and
// values live in a per-block ring of attnCache() positions: the attention window when one is set
// (constant cost and memory however long the stream), otherwise the model's context.

// blockAttention holds one block's quantized attention parameters.
type blockAttention struct {
	Norm []float32 `json:"norm"`
	QKV  Tensor    `json:"qkv"` // 3D x D
	Out  Tensor    `json:"out"` // D x D
}

// quantizeAttention extracts the attention sublayer of a CA block.
func quantizeAttention(weights []float32, a nn.Attention) *blockAttention {
	D := a.D
	norm := make([]float32, D)
	for d := range norm {
		norm[d] = weights[a.Norm+d]
	}
	return &blockAttention{
		Norm: norm,
		QKV:  NewTensorF32(weights[a.QKV : a.QKV+3*D*D]),
		Out:  NewTensorF32(weights[a.Out : a.Out+D*D]),
	}
}

// dequantize writes the sublayer back into a full-precision weight vector.
func (ba *blockAttention) dequantize(weights []float32, a nn.Attention) {
	D := a.D
	for d, v := range ba.Norm {
		weights[a.Norm+d] = float32(v)
	}
	copy(weights[a.QKV:a.QKV+3*D*D], ba.QKV.DequantizeF32())
	copy(weights[a.Out:a.Out+D*D], ba.Out.DequantizeF32())
}

// attnCache returns how many positions each block's key/value ring holds.
func (q *Model) attnCache() int {
	if q.attnWindow > 0 {
		return q.attnWindow
	}
	return q.maxContext
}

// attnScratch holds the per-model working vectors of the attention step. Keys, values and the
// attention arithmetic are float32, like the INT8 products that feed them.
type attnScratch struct {
	norm, xHat []float32 // D
	qkv        []float32 // 3D
	att, proj  []float32 // D
}

func newAttnScratch(D int) attnScratch {
	return attnScratch{
		norm: make([]float32, D), xHat: make([]float32, D),
		qkv: make([]float32, 3*D), att: make([]float32, D), proj: make([]float32, D),
	}
}

// attendPosition applies block b's attention sublayer to st.cur for position pos, recording the
// position's key and value in the block's ring.
func (q *Model) attendPosition(st *quantSeqState, b int, ba *blockAttention, pos int) {
	D, heads := q.stateSize, q.attnHeads
	dh := D / heads
	s := &q.attn
	nn.RMSNormForwardF32(st.cur, ba.Norm, s.norm, s.xHat, nn.RMSNormEpsilonF32)
	kernels.MatVecInt8(s.qkv, ba.QKV.Data, D, ba.QKV.Scale, s.norm, nil)

	keys, values, ring := st.attnK[b], st.attnV[b], st.attnCap
	slot := pos % ring
	copy(keys[slot*D:(slot+1)*D], s.qkv[D:2*D])
	copy(values[slot*D:(slot+1)*D], s.qkv[2*D:])

	lo := max(nn.WindowStart(pos, q.attnWindow), pos-ring+1)
	scores := st.attnScores[:pos-lo+1]
	scale := float32(1 / math.Sqrt(float64(dh)))
	clear(s.att)
	for h := 0; h < heads; h++ {
		query := s.qkv[h*dh : (h+1)*dh]
		for j := lo; j <= pos; j++ {
			k := (j % ring) * D
			scores[j-lo] = kernels.DotF32(query, keys[k+h*dh:k+(h+1)*dh]) * scale
		}
		nn.SoftmaxInPlaceF32(scores, st.attnScratch[:len(scores)])
		out := s.att[h*dh : (h+1)*dh]
		for j := lo; j <= pos; j++ {
			v := (j % ring) * D
			kernels.AddScaledF32(out, values[v+h*dh:v+(h+1)*dh], scores[j-lo])
		}
	}
	kernels.MatVecInt8(s.proj, ba.Out.Data, D, ba.Out.Scale, s.att, nil)
	for d := 0; d < D; d++ {
		st.cur[d] += s.proj[d]
	}
}
