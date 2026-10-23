package quant

import (
	"github.com/mignon-ai/mignon-ai/internal/ca"
)

// Quantize converts a ca.Model into Model
func Quantize(model *ca.Model, maxContext int) *Model {
	vocabSize := model.VocabSize()
	stateSize := model.StateSize()
	hiddenSize := model.HiddenSize()
	numBlocks := len(model.BlockConfigs())

	q := NewModel(vocabSize, stateSize, hiddenSize, maxContext, numBlocks)
	q.tiedEmbed = model.TiedEmbed()

	layout := model.Layout()
	q.attnHeads, q.attnWindow = layout.AttnHeads, layout.AttnWindow
	weights := model.Weights()

	// 1. Quantize Embeddings
	embedWeights := weights[layout.EmbedOffset : layout.EmbedOffset+vocabSize*stateSize]
	qtEmbed := NewTensorF32(embedWeights)
	q.embedWeights = qtEmbed.Data
	q.embedScale = qtEmbed.Scale

	// 2. Quantize Blocks
	inDim := layout.InDim
	outDim := layout.OutDim
	for bIdx, bl := range layout.Blocks {
		flatW1 := make([]float32, 2*hiddenSize*inDim)
		b1 := make([]float32, 2*hiddenSize)
		for h := 0; h < 2*hiddenSize; h++ {
			nOff := bl.Layer1Offset + h*(inDim+1)
			copy(flatW1[h*inDim:(h+1)*inDim], weights[nOff:nOff+inDim])
			b1[h] = weights[nOff+inDim]
		}

		flatW2 := make([]float32, outDim*hiddenSize)
		b2 := make([]float32, outDim)
		for s := 0; s < outDim; s++ {
			nOff := bl.Layer2Offset + s*(hiddenSize+1)
			copy(flatW2[s*hiddenSize:(s+1)*hiddenSize], weights[nOff:nOff+hiddenSize])
			b2[s] = weights[nOff+hiddenSize]
		}

		gamma := make([]float32, stateSize)
		for d := 0; d < stateSize; d++ {
			gamma[d] = weights[bl.GammaOffset+d]
		}

		ruleWSel := NewTensorF32(weights[bl.SelectiveOffset : bl.SelectiveOffset+stateSize*stateSize])

		q.blocks[bIdx] = Block{
			Steps:           bl.Steps,
			DilationBase:    bl.DilationBase,
			InitialDilation: bl.InitDilation,
			RuleW1:          NewTensorF32(flatW1),
			RuleB1:          b1,
			RuleW2:          NewTensorF32(flatW2),
			RuleB2:          b2,
			Gamma:           gamma,
			RuleWSelect:     ruleWSel,
		}
		if bl.AttnOffset >= 0 {
			q.blocks[bIdx].Attn = quantizeAttention(weights, layout.Attention(bl))
		}
	}

	// 3. Final RMSNorm
	q.finalNorm = make([]float32, stateSize)
	for d := 0; d < stateSize; d++ {
		q.finalNorm[d] = weights[layout.FinalNormOffset+d]
	}

	// 4. Quantize Head
	if model.TiedEmbed() {
		q.headB = make([]float32, vocabSize)
		for v := 0; v < vocabSize; v++ {
			q.headB[v] = weights[layout.HeadOffset+v]
		}
		q.headW = q.embedWeights
		q.headWScale = q.embedScale
	} else {
		flatHead := make([]float32, vocabSize*stateSize)
		q.headB = make([]float32, vocabSize)
		for v := 0; v < vocabSize; v++ {
			nOff := layout.HeadOffset + v*(stateSize+1)
			copy(flatHead[v*stateSize:(v+1)*stateSize], weights[nOff:nOff+stateSize])
			q.headB[v] = weights[nOff+stateSize]
		}
		qtHead := NewTensorF32(flatHead)
		q.headW = qtHead.Data
		q.headWScale = qtHead.Scale
	}

	return q
}

// Dequantize converts a Model back into a full-precision float32 ca.Model for fine-tuning.
func Dequantize(q *Model) *ca.Model {
	configs := make([]ca.BlockConfig, len(q.blocks))
	for i, b := range q.blocks {
		configs[i] = ca.BlockConfig{
			Steps:           b.Steps,
			DilationBase:    b.DilationBase,
			InitialDilation: b.InitialDilation,
		}
	}

	m := ca.New(ca.Config{
		VocabSize:  q.vocabSize,
		StateSize:  q.stateSize,
		HiddenSize: q.hiddenSize,
		Blocks:     configs,
		TiedEmbed:  q.tiedEmbed,
		AttnHeads:  q.attnHeads,
		AttnWindow: q.attnWindow,
	})
	layout := m.Layout()

	// 1. Dequantize Embedding weights
	copy(m.Weights()[layout.EmbedOffset:], Tensor{Data: q.embedWeights, Scale: q.embedScale}.DequantizeF32())

	// 2. Dequantize Blocks
	inDim := layout.InDim
	outDim := layout.OutDim
	for bIdx, qb := range q.blocks {
		bl := layout.Blocks[bIdx]
		w1 := qb.RuleW1.DequantizeF32()
		for h := 0; h < 2*q.hiddenSize; h++ {
			nOff := bl.Layer1Offset + h*(inDim+1)
			copy(m.Weights()[nOff:nOff+inDim], w1[h*inDim:(h+1)*inDim])
			m.Weights()[nOff+inDim] = float32(qb.RuleB1[h])
		}

		w2 := qb.RuleW2.DequantizeF32()
		for s := 0; s < outDim; s++ {
			nOff := bl.Layer2Offset + s*(q.hiddenSize+1)
			copy(m.Weights()[nOff:nOff+q.hiddenSize], w2[s*q.hiddenSize:(s+1)*q.hiddenSize])
			m.Weights()[nOff+q.hiddenSize] = float32(qb.RuleB2[s])
		}

		if len(qb.Gamma) == m.StateSize() {
			for d := 0; d < m.StateSize(); d++ {
				m.Weights()[bl.GammaOffset+d] = float32(qb.Gamma[d])
			}
		}

		copy(m.Weights()[bl.SelectiveOffset:bl.SelectiveOffset+m.StateSize()*m.StateSize()], qb.RuleWSelect.DequantizeF32())

		if qb.Attn != nil && bl.AttnOffset >= 0 {
			qb.Attn.dequantize(m.Weights(), layout.Attention(bl))
		}
	}

	// 3. Final RMSNorm
	if len(q.finalNorm) == m.StateSize() {
		for d := 0; d < m.StateSize(); d++ {
			m.Weights()[layout.FinalNormOffset+d] = float32(q.finalNorm[d])
		}
	}

	// 4. Output Head
	if q.tiedEmbed {
		for v := 0; v < q.vocabSize; v++ {
			m.Weights()[layout.HeadOffset+v] = float32(q.headB[v])
		}
	} else {
		headW := Tensor{Data: q.headW, Scale: q.headWScale}.DequantizeF32()
		for v := 0; v < q.vocabSize; v++ {
			nOff := layout.HeadOffset + v*(q.stateSize+1)
			copy(m.Weights()[nOff:nOff+q.stateSize], headW[v*q.stateSize:(v+1)*q.stateSize])
			m.Weights()[nOff+q.stateSize] = float32(q.headB[v])
		}
	}

	return m
}
