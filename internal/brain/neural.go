package brain

import (
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"math/rand"
	"sort"
	"sync"
)

const (
	padToken       = "<pad>"
	unkToken       = "<unk>"
	bosToken       = "<bos>"
	eosToken       = "<eos>"
	userToken      = "<user>"
	assistantToken = "<assistant>"
	selfToken      = "<self>"

	padID       = 0
	unkID       = 1
	bosID       = 2
	eosID       = 3
	userID      = 4
	assistantID = 5
	selfID      = 6

	reservedTokensCount = 7
	maxSeqLen           = 64
	rmsEps              = float32(1e-5)
	synapseVarFloor     = float32(1e-8)
	synapseWeightFloor  = float32(0.025)
)

// TrainingSample represents one token sequence with exact cross-entropy loss from LossStart onward.
type TrainingSample struct {
	IDs       []int
	LossStart int // index of first target token ( >= 1 )
}

// TransformerLayer holds weights & AdamW moments for one Pre-RMSNorm Decoder-Only GPT Block:
// x = x + MHA(RMSNorm1(x));  x = x + FFN_SiLU(RMSNorm2(x))
type TransformerLayer struct {
	Norm1, Norm1M, Norm1V []float32 // D
	WQ, WQM, WQV          []float32 // D * D
	WK, WKM, WKV          []float32 // D * D
	WV, WVM, WVV          []float32 // D * D
	WO, WOM, WOV          []float32 // D * D

	Norm2, Norm2M, Norm2V []float32 // D
	W1, W1M, W1V          []float32 // D * F
	B1, B1M, B1V          []float32 // F
	W2, W2M, W2V          []float32 // F * D
	B2, B2M, B2V          []float32 // D
}

// NeuralNet implements a Closed-Book Multi-Layer, Multi-Head Decoder-Only GPT Transformer
// with Pre-RMSNorm, SiLU Feed-Forward Key-Value Memory, Weight-Tied Full-Vocabulary Softmax,
// Multi-Goroutine AdamW Optimization, and Compact IEEE-754 Binary Serialization.
type NeuralNet struct {
	NumLayers     int
	NumHeads      int
	EmbedDim      int
	HiddenNeurons int // FFN intermediate dimension F
	MaxVocab      int
	AdamStep      int64

	TokenToID   map[string]int
	IDToToken   []string
	TokenCounts []int

	// Vocabulary-indexed parameters (weight-tied input & output embeddings)
	Embeddings []float32 // V * D
	EmbM       []float32
	EmbV       []float32
	BOut       []float32 // V
	BOutM      []float32
	BOutV      []float32

	// Learned positional embeddings
	PosEmb []float32 // maxSeqLen * D
	PosM   []float32
	PosV   []float32

	// Stacked GPT Transformer Decoder Blocks
	Layers []TransformerLayer

	// Final Pre-Unembedding RMSNorm
	NormF, NormFM, NormFV []float32 // D
}

// NewNeuralNet constructs an untrained Multi-Layer, Multi-Head Decoder-Only GPT Transformer.
// ActiveSynapses() starts at 0 until real backpropagation updates occur.
func NewNeuralNet(numLayers, numHeads, embedDim, ffnDim, maxVocab int, seed int64) *NeuralNet {
	if numLayers < 2 {
		numLayers = 2
	}
	if numHeads < 2 {
		numHeads = 4
	}
	if embedDim < 32 {
		embedDim = 48
	}
	if embedDim%numHeads != 0 {
		embedDim += numHeads - (embedDim % numHeads)
	}
	if ffnDim < embedDim*2 {
		ffnDim = embedDim * 4
	}
	if maxVocab < 512 {
		maxVocab = 4096
	}

	rng := rand.New(rand.NewSource(seed))
	L := numLayers
	D := embedDim
	F := ffnDim

	initGaussian := func(n int, scale float64) []float32 {
		s := make([]float32, n)
		for i := range s {
			s[i] = float32(rng.NormFloat64() * scale)
		}
		return s
	}
	initOnes := func(n int) []float32 {
		s := make([]float32, n)
		for i := range s {
			s[i] = 1.0
		}
		return s
	}

	// GPT-2 style residual-stream depth scaling: 1 / sqrt(2 * L) on residual projections WO and W2
	dScale := 0.5 / math.Sqrt(float64(D))
	resOScale := 0.5 / math.Sqrt(float64(2*L*D))
	res2Scale := 0.5 / math.Sqrt(float64(2*L*F))

	layers := make([]TransformerLayer, L)
	for l := 0; l < L; l++ {
		layers[l] = TransformerLayer{
			Norm1:  initOnes(D),
			Norm1M: make([]float32, D),
			Norm1V: make([]float32, D),
			WQ:     initGaussian(D*D, dScale),
			WQM:    make([]float32, D*D),
			WQV:    make([]float32, D*D),
			WK:     initGaussian(D*D, dScale),
			WKM:    make([]float32, D*D),
			WKV:    make([]float32, D*D),
			WV:     initGaussian(D*D, dScale),
			WVM:    make([]float32, D*D),
			WVV:    make([]float32, D*D),
			WO:     initGaussian(D*D, resOScale),
			WOM:    make([]float32, D*D),
			WOV:    make([]float32, D*D),

			Norm2:  initOnes(D),
			Norm2M: make([]float32, D),
			Norm2V: make([]float32, D),
			W1:     initGaussian(D*F, dScale),
			W1M:    make([]float32, D*F),
			W1V:    make([]float32, D*F),
			B1:     make([]float32, F),
			B1M:    make([]float32, F),
			B1V:    make([]float32, F),
			W2:     initGaussian(F*D, res2Scale),
			W2M:    make([]float32, F*D),
			W2V:    make([]float32, F*D),
			B2:     make([]float32, D),
			B2M:    make([]float32, D),
			B2V:    make([]float32, D),
		}
	}

	nn := &NeuralNet{
		NumLayers:     L,
		NumHeads:      numHeads,
		EmbedDim:      D,
		HiddenNeurons: F,
		MaxVocab:      maxVocab,
		TokenToID:     make(map[string]int, 512),
		IDToToken:     make([]string, 0, 512),
		TokenCounts:   make([]int, 0, 512),
		PosEmb:        initGaussian(maxSeqLen*D, 0.02),
		PosM:          make([]float32, maxSeqLen*D),
		PosV:          make([]float32, maxSeqLen*D),
		Layers:        layers,
		NormF:         initOnes(D),
		NormFM:        make([]float32, D),
		NormFV:        make([]float32, D),
	}

	for _, sp := range []string{padToken, unkToken, bosToken, eosToken, userToken, assistantToken, selfToken} {
		nn.ensureToken(sp, rng)
		nn.TokenCounts[len(nn.TokenCounts)-1] = 0
	}
	return nn
}

func (nn *NeuralNet) ensureToken(tok string, rng *rand.Rand) int {
	if id, ok := nn.TokenToID[tok]; ok {
		nn.TokenCounts[id]++
		return id
	}
	if len(nn.IDToToken) >= nn.MaxVocab {
		nn.TokenCounts[unkID]++
		return unkID
	}

	id := len(nn.IDToToken)
	nn.TokenToID[tok] = id
	nn.IDToToken = append(nn.IDToToken, tok)
	nn.TokenCounts = append(nn.TokenCounts, 1)

	scale := float32(0.45 / math.Sqrt(float64(nn.EmbedDim)))
	for d := 0; d < nn.EmbedDim; d++ {
		nn.Embeddings = append(nn.Embeddings, float32(rng.NormFloat64())*scale)
		nn.EmbM = append(nn.EmbM, 0)
		nn.EmbV = append(nn.EmbV, 0)
	}
	nn.BOut = append(nn.BOut, 0)
	nn.BOutM = append(nn.BOutM, 0)
	nn.BOutV = append(nn.BOutV, 0)
	return id
}

// RegisterTokens registers tokens in vocabulary up to MaxVocab and returns their IDs.
func (nn *NeuralNet) RegisterTokens(tokens []string, rng *rand.Rand) []int {
	ids := make([]int, len(tokens))
	for i, tok := range tokens {
		ids[i] = nn.ensureToken(tok, rng)
	}
	return ids
}

// LookupTokens maps tokens to existing IDs without mutating vocabulary.
func (nn *NeuralNet) LookupTokens(tokens []string) []int {
	ids := make([]int, len(tokens))
	for i, tok := range tokens {
		if id, ok := nn.TokenToID[tok]; ok {
			ids[i] = id
		} else {
			ids[i] = unkID
		}
	}
	return ids
}

// VocabSize returns total vocabulary slots including reserved structural tokens.
func (nn *NeuralNet) VocabSize() int { return len(nn.IDToToken) }

// LearnedWordCount returns the number of learned subwords/words beyond reserved tokens.
func (nn *NeuralNet) LearnedWordCount() int {
	if len(nn.IDToToken) <= reservedTokensCount {
		return 0
	}
	return len(nn.IDToToken) - reservedTokensCount
}

// KnowsToken reports whether tok has been trained at least once.
func (nn *NeuralNet) KnowsToken(tok string) bool {
	id, ok := nn.TokenToID[tok]
	if !ok || id < reservedTokensCount {
		return false
	}
	return nn.TokenCounts[id] > 0
}

// ActiveSynapses counts genuinely trained Transformer weights (AdamW v > floor and |w| > floor).
// Starts at 0 on an untrained brain!
func (nn *NeuralNet) ActiveSynapses() int {
	if nn.AdamStep == 0 {
		return 0
	}
	countTrained := func(w, v []float32) int {
		c := 0
		n := len(w)
		if len(v) < n {
			n = len(v)
		}
		for i := 0; i < n; i++ {
			if v[i] > synapseVarFloor && (w[i] > synapseWeightFloor || w[i] < -synapseWeightFloor) {
				c++
			}
		}
		return c
	}
	total := countTrained(nn.Embeddings, nn.EmbV)
	for l := range nn.Layers {
		ly := &nn.Layers[l]
		total += countTrained(ly.WQ, ly.WQV)
		total += countTrained(ly.WK, ly.WKV)
		total += countTrained(ly.WV, ly.WVV)
		total += countTrained(ly.WO, ly.WOV)
		total += countTrained(ly.W1, ly.W1V)
		total += countTrained(ly.W2, ly.W2V)
	}
	return total
}

// PruneToMaxVocab downsizes vocabulary tensors safely when importing onto a lower-RAM machine.
func (nn *NeuralNet) PruneToMaxVocab(newMaxVocab int) {
	if newMaxVocab < 512 {
		newMaxVocab = 512
	}
	nn.MaxVocab = newMaxVocab
	if len(nn.IDToToken) <= newMaxVocab {
		return
	}

	type tokFreq struct {
		oldID int
		count int
	}
	learned := make([]tokFreq, 0, len(nn.IDToToken)-reservedTokensCount)
	for id := reservedTokensCount; id < len(nn.IDToToken); id++ {
		learned = append(learned, tokFreq{oldID: id, count: nn.TokenCounts[id]})
	}
	sort.Slice(learned, func(i, j int) bool {
		if learned[i].count == learned[j].count {
			return learned[i].oldID < learned[j].oldID
		}
		return learned[i].count > learned[j].count
	})
	keepLearned := newMaxVocab - reservedTokensCount
	if keepLearned < len(learned) {
		learned = learned[:keepLearned]
	}

	keepIDs := make([]int, 0, newMaxVocab)
	for id := 0; id < reservedTokensCount; id++ {
		keepIDs = append(keepIDs, id)
	}
	for _, lf := range learned {
		keepIDs = append(keepIDs, lf.oldID)
	}

	D := nn.EmbedDim
	newTokToID := make(map[string]int, len(keepIDs))
	newIDToTok := make([]string, len(keepIDs))
	newCounts := make([]int, len(keepIDs))
	newEmb := make([]float32, len(keepIDs)*D)
	newEmbM := make([]float32, len(keepIDs)*D)
	newEmbV := make([]float32, len(keepIDs)*D)
	newBOut := make([]float32, len(keepIDs))
	newBOutM := make([]float32, len(keepIDs))
	newBOutV := make([]float32, len(keepIDs))

	for newID, oldID := range keepIDs {
		tok := nn.IDToToken[oldID]
		newTokToID[tok] = newID
		newIDToTok[newID] = tok
		newCounts[newID] = nn.TokenCounts[oldID]
		copy(newEmb[newID*D:(newID+1)*D], nn.Embeddings[oldID*D:(oldID+1)*D])
		copy(newEmbM[newID*D:(newID+1)*D], nn.EmbM[oldID*D:(oldID+1)*D])
		copy(newEmbV[newID*D:(newID+1)*D], nn.EmbV[oldID*D:(oldID+1)*D])
		newBOut[newID] = nn.BOut[oldID]
		newBOutM[newID] = nn.BOutM[oldID]
		newBOutV[newID] = nn.BOutV[oldID]
	}

	nn.TokenToID = newTokToID
	nn.IDToToken = newIDToTok
	nn.TokenCounts = newCounts
	nn.Embeddings = newEmb
	nn.EmbM = newEmbM
	nn.EmbV = newEmbV
	nn.BOut = newBOut
	nn.BOutM = newBOutM
	nn.BOutV = newBOutV
}

type layerGrad struct {
	dNorm1             []float32
	dWQ, dWK, dWV, dWO []float32
	dNorm2             []float32
	dW1, dB1, dW2, dB2 []float32
}

type gptGradAccum struct {
	dEmb        []float32
	touchedEmbs map[int]bool
	dBOut       []float32
	dPos        []float32
	layers      []layerGrad
	dNormF      []float32
	lossSum     float64
	predSteps   int
}

func newGPTGradAccum(V, L, D, F int) *gptGradAccum {
	lg := make([]layerGrad, L)
	for l := 0; l < L; l++ {
		lg[l] = layerGrad{
			dNorm1: make([]float32, D),
			dWQ:    make([]float32, D*D),
			dWK:    make([]float32, D*D),
			dWV:    make([]float32, D*D),
			dWO:    make([]float32, D*D),
			dNorm2: make([]float32, D),
			dW1:    make([]float32, D*F),
			dB1:    make([]float32, F),
			dW2:    make([]float32, F*D),
			dB2:    make([]float32, D),
		}
	}
	return &gptGradAccum{
		dEmb:        make([]float32, V*D),
		touchedEmbs: make(map[int]bool, 256),
		dBOut:       make([]float32, V),
		dPos:        make([]float32, maxSeqLen*D),
		layers:      lg,
		dNormF:      make([]float32, D),
	}
}

// TrainMiniBatch executes parallel multi-goroutine GPT Transformer backpropagation across
// samples in batches of batchSize using workerThreads CPU workers and AdamW optimization.
func (nn *NeuralNet) TrainMiniBatch(samples []TrainingSample, batchSize, workerThreads int, lr float32, seed int64) (float64, int) {
	if len(samples) == 0 || len(nn.IDToToken) <= reservedTokensCount {
		return 0, 0
	}
	if batchSize < 1 {
		batchSize = 16
	}
	if workerThreads < 1 {
		workerThreads = 1
	}

	var totalLoss float64
	totalSteps := 0

	for start := 0; start < len(samples); start += batchSize {
		end := start + batchSize
		if end > len(samples) {
			end = len(samples)
		}
		chunk := samples[start:end]

		workers := workerThreads
		if workers > len(chunk) {
			workers = len(chunk)
		}

		V := len(nn.IDToToken)
		L := nn.NumLayers
		D := nn.EmbedDim
		F := nn.HiddenNeurons

		accums := make([]*gptGradAccum, workers)
		var wg sync.WaitGroup
		wg.Add(workers)

		for wIdx := 0; wIdx < workers; wIdx++ {
			accums[wIdx] = newGPTGradAccum(V, L, D, F)
			wStart := (len(chunk) * wIdx) / workers
			wEnd := (len(chunk) * (wIdx + 1)) / workers

			go func(ga *gptGradAccum, sub []TrainingSample) {
				defer wg.Done()
				for _, smp := range sub {
					nn.backwardGPT(smp, ga)
				}
			}(accums[wIdx], chunk[wStart:wEnd])
		}
		wg.Wait()

		mainAcc := accums[0]
		for wIdx := 1; wIdx < workers; wIdx++ {
			other := accums[wIdx]
			mainAcc.lossSum += other.lossSum
			mainAcc.predSteps += other.predSteps
			for tid := range other.touchedEmbs {
				mainAcc.touchedEmbs[tid] = true
				off := tid * D
				for d := 0; d < D; d++ {
					mainAcc.dEmb[off+d] += other.dEmb[off+d]
				}
				mainAcc.dBOut[tid] += other.dBOut[tid]
			}
			addSlice(mainAcc.dPos, other.dPos)
			for l := 0; l < L; l++ {
				addSlice(mainAcc.layers[l].dNorm1, other.layers[l].dNorm1)
				addSlice(mainAcc.layers[l].dWQ, other.layers[l].dWQ)
				addSlice(mainAcc.layers[l].dWK, other.layers[l].dWK)
				addSlice(mainAcc.layers[l].dWV, other.layers[l].dWV)
				addSlice(mainAcc.layers[l].dWO, other.layers[l].dWO)
				addSlice(mainAcc.layers[l].dNorm2, other.layers[l].dNorm2)
				addSlice(mainAcc.layers[l].dW1, other.layers[l].dW1)
				addSlice(mainAcc.layers[l].dB1, other.layers[l].dB1)
				addSlice(mainAcc.layers[l].dW2, other.layers[l].dW2)
				addSlice(mainAcc.layers[l].dB2, other.layers[l].dB2)
			}
			addSlice(mainAcc.dNormF, other.dNormF)
		}

		if mainAcc.predSteps > 0 {
			invSteps := float32(1.0 / float64(mainAcc.predSteps))
			nn.applyAdamW(mainAcc, invSteps, lr)
			totalLoss += mainAcc.lossSum
			totalSteps += mainAcc.predSteps
		}
	}

	if totalSteps == 0 {
		return 0, 0
	}
	return totalLoss / float64(totalSteps), totalSteps
}

type layerCache struct {
	xIn     []float32 // T * D
	rms1    []float32 // T
	xNorm1  []float32 // T * D
	q, k, v []float32 // T * D
	attn    []float32 // Nh * T * T
	headOut []float32 // T * D
	mMid    []float32 // T * D
	rms2    []float32 // T
	mNorm2  []float32 // T * D
	z1      []float32 // T * F
	a1      []float32 // T * F
}

func (nn *NeuralNet) backwardGPT(smp TrainingSample, ga *gptGradAccum) {
	ids := smp.IDs
	if len(ids) > maxSeqLen {
		ids = ids[len(ids)-maxSeqLen:]
	}
	T := len(ids)
	lossStart := smp.LossStart
	if lossStart < 1 {
		lossStart = 1
	}
	if T < 2 || lossStart >= T {
		return
	}

	V := len(nn.IDToToken)
	L := nn.NumLayers
	Nh := nn.NumHeads
	D := nn.EmbedDim
	F := nn.HiddenNeurons
	headDim := D / Nh
	invSqrtHead := float32(1.0 / math.Sqrt(float64(headDim)))

	caches := make([]layerCache, L)
	curX := make([]float32, T*D)

	// 1. Input Token + Positional Embeddings
	for t := 0; t < T; t++ {
		tid := ids[t]
		if tid < 0 || tid >= V {
			tid = unkID
		}
		eOff := tid * D
		pOff := t * D
		for d := 0; d < D; d++ {
			curX[t*D+d] = nn.Embeddings[eOff+d] + nn.PosEmb[pOff+d]
		}
	}

	// 2. Forward through L GPT Transformer Blocks
	for l := 0; l < L; l++ {
		ly := &nn.Layers[l]
		c := layerCache{
			xIn:     make([]float32, T*D),
			rms1:    make([]float32, T),
			xNorm1:  make([]float32, T*D),
			q:       make([]float32, T*D),
			k:       make([]float32, T*D),
			v:       make([]float32, T*D),
			attn:    make([]float32, Nh*T*T),
			headOut: make([]float32, T*D),
			mMid:    make([]float32, T*D),
			rms2:    make([]float32, T),
			mNorm2:  make([]float32, T*D),
			z1:      make([]float32, T*F),
			a1:      make([]float32, T*F),
		}
		copy(c.xIn, curX)

		// Pre-RMSNorm 1 + Q, K, V
		for t := 0; t < T; t++ {
			c.rms1[t] = rmsNormForward(c.xIn[t*D:(t+1)*D], ly.Norm1, c.xNorm1[t*D:(t+1)*D], D)
			matVecMul(c.xNorm1[t*D:(t+1)*D], ly.WQ, c.q[t*D:(t+1)*D], D, D)
			matVecMul(c.xNorm1[t*D:(t+1)*D], ly.WK, c.k[t*D:(t+1)*D], D, D)
			matVecMul(c.xNorm1[t*D:(t+1)*D], ly.WV, c.v[t*D:(t+1)*D], D, D)
		}

		// Multi-Head Causal Self-Attention
		for h := 0; h < Nh; h++ {
			hOff := h * headDim
			aBase := h * T * T
			for t := 0; t < T; t++ {
				maxSc := float32(-1e30)
				qSlice := c.q[t*D+hOff : t*D+hOff+headDim]
				for j := 0; j <= t; j++ {
					kSlice := c.k[j*D+hOff : j*D+hOff+headDim]
					var dot float32
					for col := 0; col < headDim; col++ {
						dot += qSlice[col] * kSlice[col]
					}
					sc := dot * invSqrtHead
					c.attn[aBase+t*T+j] = sc
					if sc > maxSc {
						maxSc = sc
					}
				}
				var sumE float64
				for j := 0; j <= t; j++ {
					e := math.Exp(float64(c.attn[aBase+t*T+j] - maxSc))
					c.attn[aBase+t*T+j] = float32(e)
					sumE += e
				}
				invS := float32(1.0 / sumE)
				outSlice := c.headOut[t*D+hOff : t*D+hOff+headDim]
				for j := 0; j <= t; j++ {
					w := c.attn[aBase+t*T+j] * invS
					c.attn[aBase+t*T+j] = w
					vSlice := c.v[j*D+hOff : j*D+hOff+headDim]
					for col := 0; col < headDim; col++ {
						outSlice[col] += w * vSlice[col]
					}
				}
			}
		}

		// Output projection WO + Residual -> mMid, then Pre-RMSNorm 2 + SiLU FFN + Residual -> curX
		projBuf := make([]float32, D)
		ffnOut := make([]float32, D)
		for t := 0; t < T; t++ {
			matVecMul(c.headOut[t*D:(t+1)*D], ly.WO, projBuf, D, D)
			for d := 0; d < D; d++ {
				c.mMid[t*D+d] = c.xIn[t*D+d] + projBuf[d]
			}

			c.rms2[t] = rmsNormForward(c.mMid[t*D:(t+1)*D], ly.Norm2, c.mNorm2[t*D:(t+1)*D], D)
			mSlice := c.mNorm2[t*D : (t+1)*D]
			zSlice := c.z1[t*F : (t+1)*F]
			aSlice := c.a1[t*F : (t+1)*F]

			copy(zSlice, ly.B1)
			for d := 0; d < D; d++ {
				md := mSlice[d]
				wRow := ly.W1[d*F : (d+1)*F]
				for f := 0; f < F; f++ {
					zSlice[f] += md * wRow[f]
				}
			}
			for f := 0; f < F; f++ {
				s := zSlice[f]
				aSlice[f] = s * sigmoid32(s)
			}

			copy(ffnOut, ly.B2)
			for f := 0; f < F; f++ {
				af := aSlice[f]
				wRow := ly.W2[f*D : (f+1)*D]
				for d := 0; d < D; d++ {
					ffnOut[d] += af * wRow[d]
				}
			}
			for d := 0; d < D; d++ {
				curX[t*D+d] = c.mMid[t*D+d] + ffnOut[d]
			}
		}
		caches[l] = c
	}

	// 3. Final Pre-Unembedding RMSNorm
	rmsF := make([]float32, T)
	yFinal := make([]float32, T*D)
	for t := 0; t < T; t++ {
		rmsF[t] = rmsNormForward(curX[t*D:(t+1)*D], nn.NormF, yFinal[t*D:(t+1)*D], D)
	}

	// 4. Exact Full-Vocabulary Softmax Cross-Entropy Loss & Unembedding Gradients
	dyFinal := make([]float32, T*D)
	logits := make([]float32, V)
	probs := make([]float32, V)

	for t := lossStart - 1; t < T-1; t++ {
		target := ids[t+1]
		if target == padID {
			continue
		}

		ySlice := yFinal[t*D : (t+1)*D]
		maxL := float32(-1e30)
		for vid := 0; vid < V; vid++ {
			if vid == padID || vid == bosID || vid == userID || vid == assistantID {
				logits[vid] = -1e9
				continue
			}
			var dot float32
			eOff := vid * D
			for d := 0; d < D; d++ {
				dot += ySlice[d] * nn.Embeddings[eOff+d]
			}
			l := dot + nn.BOut[vid]
			logits[vid] = l
			if l > maxL {
				maxL = l
			}
		}

		var sumE float64
		for vid := 0; vid < V; vid++ {
			if logits[vid] <= -1e8 {
				probs[vid] = 0
				continue
			}
			e := math.Exp(float64(logits[vid] - maxL))
			probs[vid] = float32(e)
			sumE += e
		}
		invS := float32(1.0 / sumE)
		for vid := 0; vid < V; vid++ {
			probs[vid] *= invS
		}

		tp := float64(probs[target])
		if tp < 1e-9 {
			tp = 1e-9
		}
		ga.lossSum += -math.Log(tp)
		ga.predSteps++

		dySlice := dyFinal[t*D : (t+1)*D]
		for vid := 0; vid < V; vid++ {
			p := probs[vid]
			if vid != target && p < 1e-6 {
				continue
			}
			grad := p
			if vid == target {
				grad -= 1.0
			}
			ga.touchedEmbs[vid] = true
			ga.dBOut[vid] += grad
			eOff := vid * D
			for d := 0; d < D; d++ {
				dySlice[d] += grad * nn.Embeddings[eOff+d]
				ga.dEmb[eOff+d] += grad * ySlice[d]
			}
		}
	}

	// 5. Backward through Final RMSNorm -> dCurX
	dCurX := make([]float32, T*D)
	for t := 0; t < T; t++ {
		rmsNormBackward(
			curX[t*D:(t+1)*D],
			nn.NormF,
			rmsF[t],
			dyFinal[t*D:(t+1)*D],
			dCurX[t*D:(t+1)*D],
			ga.dNormF,
			D,
		)
	}

	// 6. Backward through L GPT Transformer Blocks (l = L-1 down to 0)
	dmMid := make([]float32, T*D)
	dmNorm2 := make([]float32, D)
	da1 := make([]float32, F)
	dz1 := make([]float32, F)
	dHeadOut := make([]float32, T*D)
	dq := make([]float32, T*D)
	dk := make([]float32, T*D)
	dv := make([]float32, T*D)
	dxNorm1 := make([]float32, D)
	dxIn := make([]float32, T*D)
	dAlpha := make([]float32, T)

	for l := L - 1; l >= 0; l-- {
		ly := &nn.Layers[l]
		lg := &ga.layers[l]
		c := &caches[l]

		copy(dmMid, dCurX)
		clearSlice(dHeadOut)
		clearSlice(dq)
		clearSlice(dk)
		clearSlice(dv)
		clearSlice(dxIn)

		// Sub-block B Backward: SiLU FFN + Pre-RMSNorm 2 + Residual
		for t := 0; t < T; t++ {
			dOut := dCurX[t*D : (t+1)*D]
			aSlice := c.a1[t*F : (t+1)*F]
			zSlice := c.z1[t*F : (t+1)*F]
			mNormSlice := c.mNorm2[t*D : (t+1)*D]

			for d := 0; d < D; d++ {
				lg.dB2[d] += dOut[d]
			}
			for f := 0; f < F; f++ {
				af := aSlice[f]
				row := f * D
				var sumDa float32
				for d := 0; d < D; d++ {
					g := dOut[d]
					lg.dW2[row+d] += af * g
					sumDa += g * ly.W2[row+d]
				}
				sig := sigmoid32(zSlice[f])
				siluGrad := sig * (1.0 + zSlice[f]*(1.0-sig))
				dz := clipGrad(sumDa*siluGrad, 4.0)
				dz1[f] = dz
				lg.dB1[f] += dz
				da1[f] = sumDa
			}
			_ = da1

			for d := 0; d < D; d++ {
				md := mNormSlice[d]
				row := d * F
				var sumDm float32
				for f := 0; f < F; f++ {
					g := dz1[f]
					lg.dW1[row+f] += md * g
					sumDm += g * ly.W1[row+f]
				}
				dmNorm2[d] = sumDm
			}

			var dx2 [256]float32
			dx2Slice := dx2[:D]
			clearSlice(dx2Slice)
			rmsNormBackward(c.mMid[t*D:(t+1)*D], ly.Norm2, c.rms2[t], dmNorm2, dx2Slice, lg.dNorm2, D)
			for d := 0; d < D; d++ {
				dmMid[t*D+d] = clipGrad(dmMid[t*D+d]+dx2Slice[d], 4.0)
			}

			// WO Backward from dmMid
			dmSlice := dmMid[t*D : (t+1)*D]
			hSlice := c.headOut[t*D : (t+1)*D]
			dhSlice := dHeadOut[t*D : (t+1)*D]
			for i := 0; i < D; i++ {
				hi := hSlice[i]
				row := i * D
				var sumDh float32
				for j := 0; j < D; j++ {
					g := dmSlice[j]
					lg.dWO[row+j] += hi * g
					sumDh += g * ly.WO[row+j]
				}
				dhSlice[i] = sumDh
			}
		}

		// Sub-block A Backward: Multi-Head Causal Self-Attention
		for h := 0; h < Nh; h++ {
			hOff := h * headDim
			aBase := h * T * T
			for t := 0; t < T; t++ {
				dhSlice := dHeadOut[t*D+hOff : t*D+hOff+headDim]
				var dotSum float32
				for j := 0; j <= t; j++ {
					w := c.attn[aBase+t*T+j]
					vSlice := c.v[j*D+hOff : j*D+hOff+headDim]
					dvSlice := dv[j*D+hOff : j*D+hOff+headDim]
					var daDotV float32
					for col := 0; col < headDim; col++ {
						daDotV += dhSlice[col] * vSlice[col]
						dvSlice[col] += w * dhSlice[col]
					}
					dAlpha[j] = daDotV
					dotSum += w * daDotV
				}
				qSlice := c.q[t*D+hOff : t*D+hOff+headDim]
				dqSlice := dq[t*D+hOff : t*D+hOff+headDim]
				for j := 0; j <= t; j++ {
					w := c.attn[aBase+t*T+j]
					dSc := w * (dAlpha[j] - dotSum) * invSqrtHead
					kSlice := c.k[j*D+hOff : j*D+hOff+headDim]
					dkSlice := dk[j*D+hOff : j*D+hOff+headDim]
					for col := 0; col < headDim; col++ {
						dqSlice[col] += dSc * kSlice[col]
						dkSlice[col] += dSc * qSlice[col]
					}
				}
			}
		}

		// WQ, WK, WV + Pre-RMSNorm 1 + Residual Backward
		for t := 0; t < T; t++ {
			xnSlice := c.xNorm1[t*D : (t+1)*D]
			dqt := dq[t*D : (t+1)*D]
			dkt := dk[t*D : (t+1)*D]
			dvt := dv[t*D : (t+1)*D]

			for i := 0; i < D; i++ {
				xi := xnSlice[i]
				row := i * D
				var sumDxn float32
				for j := 0; j < D; j++ {
					gq, gk, gv := dqt[j], dkt[j], dvt[j]
					lg.dWQ[row+j] += xi * gq
					lg.dWK[row+j] += xi * gk
					lg.dWV[row+j] += xi * gv
					sumDxn += gq*ly.WQ[row+j] + gk*ly.WK[row+j] + gv*ly.WV[row+j]
				}
				dxNorm1[i] = sumDxn
			}

			var dx1 [256]float32
			dx1Slice := dx1[:D]
			clearSlice(dx1Slice)
			rmsNormBackward(c.xIn[t*D:(t+1)*D], ly.Norm1, c.rms1[t], dxNorm1, dx1Slice, lg.dNorm1, D)
			for d := 0; d < D; d++ {
				dxIn[t*D+d] = clipGrad(dmMid[t*D+d]+dx1Slice[d], 4.0)
			}
		}

		copy(dCurX, dxIn)
	}

	// 7. Token + Positional Embeddings Backward
	for t := 0; t < T; t++ {
		tid := ids[t]
		if tid > 0 && tid < V {
			ga.touchedEmbs[tid] = true
			eOff := tid * D
			for d := 0; d < D; d++ {
				g := clipGrad(dCurX[t*D+d], 4.0)
				ga.dEmb[eOff+d] += g
				ga.dPos[t*D+d] += g
			}
		}
	}
}

func (nn *NeuralNet) applyAdamW(ga *gptGradAccum, invSteps, lr float32) {
	nn.AdamStep++
	const (
		beta1 = float32(0.9)
		beta2 = float32(0.98)
		eps   = float32(1e-8)
		wd    = float32(1e-4)
	)
	bc1 := float32(1.0 - math.Pow(float64(beta1), float64(nn.AdamStep)))
	bc2 := float32(1.0 - math.Pow(float64(beta2), float64(nn.AdamStep)))
	stepLR := lr * float32(math.Sqrt(float64(bc2))) / bc1

	updateSlice := func(w, m, v, g []float32, decay float32) {
		for i := range w {
			gi := clipGrad(g[i]*invSteps, 1.5)
			mi := beta1*m[i] + (1.0-beta1)*gi
			vi := beta2*v[i] + (1.0-beta2)*gi*gi
			m[i] = mi
			v[i] = vi
			w[i] -= stepLR*(mi/(float32(math.Sqrt(float64(vi)))+eps)) + lr*decay*w[i]
		}
	}

	D := nn.EmbedDim
	for tid := range ga.touchedEmbs {
		off := tid * D
		updateSlice(nn.Embeddings[off:off+D], nn.EmbM[off:off+D], nn.EmbV[off:off+D], ga.dEmb[off:off+D], wd)
		gi := clipGrad(ga.dBOut[tid]*invSteps, 1.5)
		mi := beta1*nn.BOutM[tid] + (1.0-beta1)*gi
		vi := beta2*nn.BOutV[tid] + (1.0-beta2)*gi*gi
		nn.BOutM[tid] = mi
		nn.BOutV[tid] = vi
		nn.BOut[tid] -= stepLR * (mi / (float32(math.Sqrt(float64(vi))) + eps))
	}

	updateSlice(nn.PosEmb, nn.PosM, nn.PosV, ga.dPos, wd)
	for l := range nn.Layers {
		ly := &nn.Layers[l]
		lg := &ga.layers[l]
		updateSlice(ly.Norm1, ly.Norm1M, ly.Norm1V, lg.dNorm1, 0)
		updateSlice(ly.WQ, ly.WQM, ly.WQV, lg.dWQ, wd)
		updateSlice(ly.WK, ly.WKM, ly.WKV, lg.dWK, wd)
		updateSlice(ly.WV, ly.WVM, ly.WVV, lg.dWV, wd)
		updateSlice(ly.WO, ly.WOM, ly.WOV, lg.dWO, wd)
		updateSlice(ly.Norm2, ly.Norm2M, ly.Norm2V, lg.dNorm2, 0)
		updateSlice(ly.W1, ly.W1M, ly.W1V, lg.dW1, wd)
		updateSlice(ly.B1, ly.B1M, ly.B1V, lg.dB1, 0)
		updateSlice(ly.W2, ly.W2M, ly.W2V, lg.dW2, wd)
		updateSlice(ly.B2, ly.B2M, ly.B2V, lg.dB2, 0)
	}
	updateSlice(nn.NormF, nn.NormFM, nn.NormFV, ga.dNormF, 0)
}

// ForwardFullVocab runs a pure Closed-Book Decoder-Only GPT Transformer pass across seqIDs
// and computes exact logits & softmax probabilities across EVERY token in vocabulary.
// Zero external databases, zero RAG lookup, zero copy heads!
func (nn *NeuralNet) ForwardFullVocab(seqIDs []int) ([]float32, []float32, float64, float64) {
	V := len(nn.IDToToken)
	if V <= reservedTokensCount || len(seqIDs) == 0 {
		return nil, nil, 99.0, 0
	}
	if len(seqIDs) > maxSeqLen {
		seqIDs = seqIDs[len(seqIDs)-maxSeqLen:]
	}
	T := len(seqIDs)
	L := nn.NumLayers
	Nh := nn.NumHeads
	D := nn.EmbedDim
	F := nn.HiddenNeurons
	headDim := D / Nh
	invSqrtHead := float32(1.0 / math.Sqrt(float64(headDim)))

	curX := make([]float32, T*D)
	for t := 0; t < T; t++ {
		tid := seqIDs[t]
		if tid < 0 || tid >= V {
			tid = unkID
		}
		eOff := tid * D
		pOff := t * D
		for d := 0; d < D; d++ {
			curX[t*D+d] = nn.Embeddings[eOff+d] + nn.PosEmb[pOff+d]
		}
	}

	xNorm := make([]float32, T*D)
	q := make([]float32, T*D)
	k := make([]float32, T*D)
	v := make([]float32, T*D)
	headOut := make([]float32, T*D)
	scBuf := make([]float32, T)
	projBuf := make([]float32, D)
	mNorm := make([]float32, D)
	aBuf := make([]float32, F)

	for l := 0; l < L; l++ {
		ly := &nn.Layers[l]
		clearSlice(headOut)

		for t := 0; t < T; t++ {
			_ = rmsNormForward(curX[t*D:(t+1)*D], ly.Norm1, xNorm[t*D:(t+1)*D], D)
			matVecMul(xNorm[t*D:(t+1)*D], ly.WQ, q[t*D:(t+1)*D], D, D)
			matVecMul(xNorm[t*D:(t+1)*D], ly.WK, k[t*D:(t+1)*D], D, D)
			matVecMul(xNorm[t*D:(t+1)*D], ly.WV, v[t*D:(t+1)*D], D, D)
		}

		for h := 0; h < Nh; h++ {
			hOff := h * headDim
			for t := 0; t < T; t++ {
				maxSc := float32(-1e30)
				qSlice := q[t*D+hOff : t*D+hOff+headDim]
				for j := 0; j <= t; j++ {
					kSlice := k[j*D+hOff : j*D+hOff+headDim]
					var dot float32
					for col := 0; col < headDim; col++ {
						dot += qSlice[col] * kSlice[col]
					}
					sc := dot * invSqrtHead
					scBuf[j] = sc
					if sc > maxSc {
						maxSc = sc
					}
				}
				var sumE float64
				for j := 0; j <= t; j++ {
					e := math.Exp(float64(scBuf[j] - maxSc))
					scBuf[j] = float32(e)
					sumE += e
				}
				invS := float32(1.0 / sumE)
				outSlice := headOut[t*D+hOff : t*D+hOff+headDim]
				for j := 0; j <= t; j++ {
					w := scBuf[j] * invS
					vSlice := v[j*D+hOff : j*D+hOff+headDim]
					for col := 0; col < headDim; col++ {
						outSlice[col] += w * vSlice[col]
					}
				}
			}
		}

		for t := 0; t < T; t++ {
			matVecMul(headOut[t*D:(t+1)*D], ly.WO, projBuf, D, D)
			for d := 0; d < D; d++ {
				curX[t*D+d] += projBuf[d]
			}
			_ = rmsNormForward(curX[t*D:(t+1)*D], ly.Norm2, mNorm, D)
			copy(aBuf, ly.B1)
			for d := 0; d < D; d++ {
				md := mNorm[d]
				wRow := ly.W1[d*F : (d+1)*F]
				for f := 0; f < F; f++ {
					aBuf[f] += md * wRow[f]
				}
			}
			for f := 0; f < F; f++ {
				s := aBuf[f]
				aBuf[f] = s * sigmoid32(s)
			}
			for d := 0; d < D; d++ {
				curX[t*D+d] += ly.B2[d]
			}
			tSlice := curX[t*D : (t+1)*D]
			for f := 0; f < F; f++ {
				af := aBuf[f]
				wRow := ly.W2[f*D : (f+1)*D]
				for d := 0; d < D; d++ {
					tSlice[d] += af * wRow[d]
				}
			}
		}
	}

	yLast := make([]float32, D)
	_ = rmsNormForward(curX[(T-1)*D:T*D], nn.NormF, yLast, D)

	logits := make([]float32, V)
	probs := make([]float32, V)
	maxL := float32(-1e30)

	for vid := 0; vid < V; vid++ {
		if vid == padID || vid == unkID || vid == bosID || vid == userID || vid == assistantID {
			logits[vid] = -1e9
			continue
		}
		var dot float32
		eOff := vid * D
		for d := 0; d < D; d++ {
			dot += yLast[d] * nn.Embeddings[eOff+d]
		}
		l := dot + nn.BOut[vid]
		logits[vid] = l
		if l > maxL {
			maxL = l
		}
	}

	var sumE float64
	for vid := 0; vid < V; vid++ {
		if logits[vid] <= -1e8 {
			probs[vid] = 0
			continue
		}
		e := math.Exp(float64(logits[vid] - maxL))
		probs[vid] = float32(e)
		sumE += e
	}
	invS := float32(1.0 / sumE)
	var entropy float64
	var maxProb float64
	for vid := 0; vid < V; vid++ {
		p := float64(probs[vid] * invS)
		probs[vid] = float32(p)
		if p > 1e-12 {
			entropy -= p * math.Log(p)
			if p > maxProb {
				maxProb = p
			}
		}
	}

	return probs, logits, entropy, maxProb
}

func rmsNormForward(x, gamma, out []float32, D int) float32 {
	var sumSq float32
	for d := 0; d < D; d++ {
		sumSq += x[d] * x[d]
	}
	invRMS := float32(1.0 / math.Sqrt(float64(sumSq/float32(D)+rmsEps)))
	for d := 0; d < D; d++ {
		out[d] = gamma[d] * x[d] * invRMS
	}
	return invRMS
}

func rmsNormBackward(x, gamma []float32, invRMS float32, dy, dx, dGamma []float32, D int) {
	var c float32
	for d := 0; d < D; d++ {
		xHat := x[d] * invRMS
		dGamma[d] += dy[d] * xHat
		c += (dy[d] * gamma[d]) * xHat
	}
	c /= float32(D)
	for d := 0; d < D; d++ {
		xHat := x[d] * invRMS
		u := dy[d] * gamma[d]
		dx[d] += invRMS * (u - xHat*c)
	}
}

// WriteBinary writes all GPT Transformer float32 weight & AdamW moment tensors in IEEE-754 binary.
func (nn *NeuralNet) WriteBinary(w io.Writer) error {
	header := []int32{
		int32(nn.NumLayers),
		int32(nn.NumHeads),
		int32(nn.EmbedDim),
		int32(nn.HiddenNeurons),
		int32(nn.MaxVocab),
	}
	for _, v := range header {
		if err := binary.Write(w, binary.LittleEndian, v); err != nil {
			return err
		}
	}
	if err := binary.Write(w, binary.LittleEndian, nn.AdamStep); err != nil {
		return err
	}
	if err := binary.Write(w, binary.LittleEndian, int32(len(nn.IDToToken))); err != nil {
		return err
	}
	for i, tok := range nn.IDToToken {
		tb := []byte(tok)
		if err := binary.Write(w, binary.LittleEndian, uint16(len(tb))); err != nil {
			return err
		}
		if _, err := w.Write(tb); err != nil {
			return err
		}
		if err := binary.Write(w, binary.LittleEndian, int32(nn.TokenCounts[i])); err != nil {
			return err
		}
	}

	topSlices := [][]float32{
		nn.Embeddings, nn.EmbM, nn.EmbV,
		nn.BOut, nn.BOutM, nn.BOutV,
		nn.PosEmb, nn.PosM, nn.PosV,
		nn.NormF, nn.NormFM, nn.NormFV,
	}
	for _, sl := range topSlices {
		if err := writeFloat32Slice(w, sl); err != nil {
			return err
		}
	}

	for l := range nn.Layers {
		ly := &nn.Layers[l]
		lSlices := [][]float32{
			ly.Norm1, ly.Norm1M, ly.Norm1V,
			ly.WQ, ly.WQM, ly.WQV,
			ly.WK, ly.WKM, ly.WKV,
			ly.WV, ly.WVM, ly.WVV,
			ly.WO, ly.WOM, ly.WOV,
			ly.Norm2, ly.Norm2M, ly.Norm2V,
			ly.W1, ly.W1M, ly.W1V,
			ly.B1, ly.B1M, ly.B1V,
			ly.W2, ly.W2M, ly.W2V,
			ly.B2, ly.B2M, ly.B2V,
		}
		for _, sl := range lSlices {
			if err := writeFloat32Slice(w, sl); err != nil {
				return err
			}
		}
	}
	return nil
}

// ReadNeuralNetBinary reads a GPT Transformer NeuralNet from compact IEEE-754 little-endian binary.
func ReadNeuralNetBinary(r io.Reader) (*NeuralNet, error) {
	var nl, nh, d32, f32, mv32, v32 int32
	var step int64
	for _, ptr := range []*int32{&nl, &nh, &d32, &f32, &mv32} {
		if err := binary.Read(r, binary.LittleEndian, ptr); err != nil {
			return nil, err
		}
	}
	if err := binary.Read(r, binary.LittleEndian, &step); err != nil {
		return nil, err
	}
	if err := binary.Read(r, binary.LittleEndian, &v32); err != nil {
		return nil, err
	}
	if nl <= 0 || nh <= 0 || d32 <= 0 || f32 <= 0 || v32 < reservedTokensCount || v32 > 200000 {
		return nil, fmt.Errorf("invalid GPT binary header L=%d Nh=%d D=%d F=%d V=%d", nl, nh, d32, f32, v32)
	}

	V := int(v32)
	nn := &NeuralNet{
		NumLayers:     int(nl),
		NumHeads:      int(nh),
		EmbedDim:      int(d32),
		HiddenNeurons: int(f32),
		MaxVocab:      int(mv32),
		AdamStep:      step,
		TokenToID:     make(map[string]int, V),
		IDToToken:     make([]string, V),
		TokenCounts:   make([]int, V),
		Layers:        make([]TransformerLayer, int(nl)),
	}

	for i := 0; i < V; i++ {
		var slen uint16
		if err := binary.Read(r, binary.LittleEndian, &slen); err != nil {
			return nil, err
		}
		buf := make([]byte, int(slen))
		if _, err := io.ReadFull(r, buf); err != nil {
			return nil, err
		}
		var cnt int32
		if err := binary.Read(r, binary.LittleEndian, &cnt); err != nil {
			return nil, err
		}
		tok := string(buf)
		nn.IDToToken[i] = tok
		nn.TokenToID[tok] = i
		nn.TokenCounts[i] = int(cnt)
	}

	topTargets := []*[]float32{
		&nn.Embeddings, &nn.EmbM, &nn.EmbV,
		&nn.BOut, &nn.BOutM, &nn.BOutV,
		&nn.PosEmb, &nn.PosM, &nn.PosV,
		&nn.NormF, &nn.NormFM, &nn.NormFV,
	}
	for _, dst := range topTargets {
		sl, err := readFloat32Slice(r)
		if err != nil {
			return nil, err
		}
		*dst = sl
	}

	for l := range nn.Layers {
		ly := &nn.Layers[l]
		lTargets := []*[]float32{
			&ly.Norm1, &ly.Norm1M, &ly.Norm1V,
			&ly.WQ, &ly.WQM, &ly.WQV,
			&ly.WK, &ly.WKM, &ly.WKV,
			&ly.WV, &ly.WVM, &ly.WVV,
			&ly.WO, &ly.WOM, &ly.WOV,
			&ly.Norm2, &ly.Norm2M, &ly.Norm2V,
			&ly.W1, &ly.W1M, &ly.W1V,
			&ly.B1, &ly.B1M, &ly.B1V,
			&ly.W2, &ly.W2M, &ly.W2V,
			&ly.B2, &ly.B2M, &ly.B2V,
		}
		for _, dst := range lTargets {
			sl, err := readFloat32Slice(r)
			if err != nil {
				return nil, err
			}
			*dst = sl
		}
	}
	return nn, nil
}

func writeFloat32Slice(w io.Writer, s []float32) error {
	if err := binary.Write(w, binary.LittleEndian, uint32(len(s))); err != nil {
		return err
	}
	buf := make([]byte, 16*1024)
	for start := 0; start < len(s); {
		n := len(s) - start
		if n > len(buf)/4 {
			n = len(buf) / 4
		}
		for i := 0; i < n; i++ {
			binary.LittleEndian.PutUint32(buf[i*4:], math.Float32bits(s[start+i]))
		}
		written, err := w.Write(buf[:n*4])
		if err != nil {
			return err
		}
		if written != n*4 {
			return io.ErrShortWrite
		}
		start += n
	}
	return nil
}

func readFloat32Slice(r io.Reader) ([]float32, error) {
	var n uint32
	if err := binary.Read(r, binary.LittleEndian, &n); err != nil {
		return nil, err
	}
	if n > 50_000_000 {
		return nil, fmt.Errorf("tensor slice too large: %d", n)
	}
	out := make([]float32, int(n))
	buf := make([]byte, 16*1024)
	for start := 0; start < len(out); {
		count := len(out) - start
		if count > len(buf)/4 {
			count = len(buf) / 4
		}
		if _, err := io.ReadFull(r, buf[:count*4]); err != nil {
			return nil, err
		}
		for i := 0; i < count; i++ {
			out[start+i] = math.Float32frombits(binary.LittleEndian.Uint32(buf[i*4:]))
		}
		start += count
	}
	return out, nil
}

func matVecMul(vec, mat, out []float32, rows, cols int) {
	clearSlice(out[:cols])
	for i := 0; i < rows; i++ {
		vi := vec[i]
		if vi == 0 {
			continue
		}
		row := mat[i*cols : (i+1)*cols]
		for j := 0; j < cols; j++ {
			out[j] += vi * row[j]
		}
	}
}

func addSlice(dst, src []float32) {
	for i := range dst {
		dst[i] += src[i]
	}
}

func clearSlice(s []float32) {
	for i := range s {
		s[i] = 0
	}
}

func sigmoid32(x float32) float32 {
	if x > 15 {
		return 1.0
	}
	if x < -15 {
		return 0.0
	}
	return float32(1.0 / (1.0 + math.Exp(-float64(x))))
}

func clipGrad(g, lim float32) float32 {
	if g > lim {
		return lim
	}
	if g < -lim {
		return -lim
	}
	return g
}
