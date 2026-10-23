package ca

import (
	"github.com/mignon-ai/mignon-ai/internal/kernels"
	"github.com/mignon-ai/mignon-ai/internal/nn"
	"github.com/mignon-ai/mignon-ai/internal/optim"
	"github.com/mignon-ai/mignon-ai/internal/parallel"
)

// blockGemmPacks holds the weight matrices one block reads as gemm B operands, packed once per
// pass since every step of the block shares them: transposed for the forward products (X·Wᵀ)
// and as stored for the backward input gradients (dZ·W). Packing grows the buffers on first use.
type blockGemmPacks struct {
	l1Value, l1Gate, l2 nn.MatrixPack // the gated rule network
	selector            nn.MatrixPack // the latch gate projection
	attn                nn.AttentionPacks
}

// maxBlockMatrices bounds the matrices of one block: the GLU value, gate and layer-2 matrices,
// the latch selector and the attention QKV and output projections.
const maxBlockMatrices = 6

// packedMatrix pairs a weight matrix with the pack that holds it.
type packedMatrix struct {
	pack   *nn.MatrixPack
	matrix optim.Matrix
}

// packSpec packs one weight matrix (k×n, element (p, j) at weights[offset + p*rs + j*cs]) into dst.
type packSpec struct {
	dst          *kernels.GemmPackedB
	k, n, offset int
	rs, cs       int
}

func (s packSpec) pack(weights []float32) {
	s.dst.Pack(s.k, s.n, weights[s.offset:], s.rs, s.cs)
}

// forwardSpec packs Wᵀ (strides swapped) and backwardSpec packs W as stored; m.Rows is the
// matrix's output count.
func (pm packedMatrix) forwardSpec() packSpec {
	m := pm.matrix
	return packSpec{&pm.pack.Fwd, m.Cols, m.Rows, m.Offset, 1, m.Stride}
}

func (pm packedMatrix) backwardSpec() packSpec {
	m := pm.matrix
	return packSpec{&pm.pack.Bwd, m.Rows, m.Cols, m.Offset, m.Stride, 1}
}

// appendMatrices appends the matrices of block bl with the packs that hold them.
func (pk *blockGemmPacks) appendMatrices(ms []packedMatrix, L Layout, bl BlockLayout) []packedMatrix {
	H, in, out, D := L.HiddenSize, L.InDim, L.OutDim, L.StateSize
	ms = append(ms,
		packedMatrix{&pk.l1Value, optim.Matrix{Offset: bl.Layer1Offset, Rows: H, Cols: in, Stride: in + 1}},
		packedMatrix{&pk.l1Gate, optim.Matrix{Offset: l1GateOffset(L, bl), Rows: H, Cols: in, Stride: in + 1}},
		packedMatrix{&pk.l2, optim.Matrix{Offset: bl.Layer2Offset, Rows: out, Cols: H, Stride: H + 1}},
		packedMatrix{&pk.selector, optim.Matrix{Offset: bl.SelectiveOffset, Rows: D, Cols: D, Stride: D}},
	)
	if bl.AttnOffset >= 0 {
		proj := L.Attention(bl).Matrices() // the QKV projection, then the output projection
		ms = append(ms, packedMatrix{&pk.attn.QKV, proj[0]}, packedMatrix{&pk.attn.Out, proj[1]})
	}
	return ms
}

// headMatrix is the output projection (V×D) with the packs that hold it.
func (L Layout) headMatrix(pack *nn.MatrixPack) packedMatrix {
	return packedMatrix{pack, optim.Matrix{Offset: L.head().Off, Rows: L.VocabSize, Cols: L.StateSize, Stride: L.head().Stride}}
}

// l1GateOffset is the start of the gate half of the block's layer-1 weights.
func l1GateOffset(L Layout, bl BlockLayout) int {
	return bl.Layer1Offset + L.HiddenSize*(L.InDim+1)
}

// packAll packs every spec in order.
func packAll(specs []packSpec, weights []float32) {
	for _, s := range specs {
		s.pack(weights)
	}
}

// WeightPacks holds every weight matrix the dense passes multiply by, packed for gemm in both the
// forward and backward layouts. Packing depends only on the weights, so a training loop can pack
// once per optimizer step and share the read-only result across workers (see
// Workspace.UseSharedPacks) instead of every worker repacking for every sequence.
type WeightPacks struct {
	layout Layout
	blocks []blockGemmPacks
	head   nn.MatrixPack
}

// NewWeightPacks allocates empty packs for layout; call Refresh before using them.
func NewWeightPacks(layout Layout) *WeightPacks {
	return &WeightPacks{layout: layout, blocks: make([]blockGemmPacks, len(layout.Blocks))}
}

// Refresh repacks every matrix from weights using up to workers goroutines. It must be called
// after every change to the weights while workspaces share these packs.
func (p *WeightPacks) Refresh(weights []float32, workers int) {
	var specs []packSpec
	add := func(ms []packedMatrix) {
		for _, pm := range ms {
			specs = append(specs, pm.forwardSpec(), pm.backwardSpec())
		}
	}
	for b := range p.blocks {
		var buf [maxBlockMatrices]packedMatrix
		add(p.blocks[b].appendMatrices(buf[:0], p.layout, p.layout.Blocks[b]))
	}
	add([]packedMatrix{p.layout.headMatrix(&p.head)})
	parallel.For(len(specs), workers, func(lo, hi int) {
		packAll(specs[lo:hi], weights)
	})
}

// UseSharedPacks makes the workspace read its packs from p instead of packing the weights itself
// on every pass. The caller keeps p refreshed for the weights it passes; nil restores per-pass
// packing.
func (ws *Workspace) UseSharedPacks(p *WeightPacks) {
	ws.sharedPacks = p
}

// blockPacksFor returns block b's packs, packing the direction selected by spec first unless
// they are shared. spec picks forwardSpec or backwardSpec of each matrix.
func (ws *Workspace) blockPacksFor(b int, weights []float32, spec func(packedMatrix) packSpec) *blockGemmPacks {
	if ws.sharedPacks != nil {
		return &ws.sharedPacks.blocks[b]
	}
	pk := &ws.blockPacks[b]
	var buf [maxBlockMatrices]packedMatrix
	for _, pm := range pk.appendMatrices(buf[:0], ws.layout, ws.layout.Blocks[b]) {
		spec(pm).pack(weights)
	}
	return pk
}

// forwardPacks returns block b's packs for a forward pass over weights.
func (ws *Workspace) forwardPacks(b int, weights []float32) *blockGemmPacks {
	return ws.blockPacksFor(b, weights, packedMatrix.forwardSpec)
}

// backwardPacks returns block b's packs for a backward pass over weights.
func (ws *Workspace) backwardPacks(b int, weights []float32) *blockGemmPacks {
	return ws.blockPacksFor(b, weights, packedMatrix.backwardSpec)
}

// headPackFor returns the head's packs, packing the direction selected by spec first unless they
// are shared.
func (ws *Workspace) headPackFor(weights []float32, spec func(packedMatrix) packSpec) *nn.MatrixPack {
	if ws.sharedPacks != nil {
		return &ws.sharedPacks.head
	}
	spec(ws.layout.headMatrix(&ws.headPack)).pack(weights)
	return &ws.headPack
}
