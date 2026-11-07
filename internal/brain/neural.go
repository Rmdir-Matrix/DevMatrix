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
	padToken  = "<pad>"
	unkToken  = "<unk>"
	bosToken  = "<bos>"
	eosToken  = "<eos>"
	ctxToken  = "<ctx>"
	askToken  = "<ask>"
	ansToken  = "<ans>"
	selfToken = "<self>"

	padID  = 0
	unkID  = 1
	bosID  = 2
	eosID  = 3
	ctxID  = 4
	askID  = 5
	ansID  = 6
	selfID = 7

	reservedTokensCount = 8
	maxSeqLen           = 64
	negativeSampleCount = 48
	synapseVarFloor     = float32(1e-8)
	synapseWeightFloor  = float32(0.03)
)

// TrainingSample represents one token sequence with loss computed from LossStart onward.
type TrainingSample struct {
	IDs       []int
	LossStart int // index of first target position ( >= 1 )
}

// NeuralNet implements a Causal Self-Attention + GRU + Pointer-Generator Copy-Attention
// Neural Language Model with Weight-Tied Full-Vocabulary Output, Multi-Core AdamW, and Binary I/O.
type NeuralNet struct {
	EmbedDim      int
	HiddenNeurons int
	MaxVocab      int
	AdamStep      int64

	TokenToID   map[string]int
	IDToToken   []string
	TokenCounts []int

	// Vocabulary-indexed parameters (grow dynamically up to MaxVocab)
	Embeddings []float32 // V * D
	EmbM       []float32
	EmbV       []float32
	BOut       []float32 // V
	BOutM      []float32
	BOutV      []float32

	// Positional + Causal Self-Attention parameters
	PosEmb []float32 // maxSeqLen * D
	PosM   []float32
	PosV   []float32
	WQ     []float32 // D * D
	WQM    []float32
	WQV    []float32
	WK     []float32 // D * D
	WKM    []float32
	WKV    []float32
	WV     []float32 // D * D
	WVM    []float32
	WVV    []float32

	// GRU Recurrent Layer parameters (input dim = D + H, out dim = H)
	Wz  []float32 // (D+H) * H
	WzM []float32
	WzV []float32
	Bz  []float32 // H
	BzM []float32
	BzV []float32

	Wr  []float32 // (D+H) * H
	WrM []float32
	WrV []float32
	Br  []float32 // H
	BrM []float32
	BrV []float32

	Wh  []float32 // (D+H) * H
	WhM []float32
	WhV []float32
	Bh  []float32 // H
	BhM []float32
	BhV []float32

	// Linear Unembedding Projection H -> D (weight-tied against Embeddings)
	WO  []float32 // H * D
	WOM []float32
	WOV []float32
	BO  []float32 // D
	BOM []float32
	BOV []float32

	// Differentiable Pointer-Generator Copy-Attention Head over <ctx> span
	WCopy  []float32 // H * D
	WCopyM []float32
	WCopyV []float32
	WGate  []float32 // H
	WGateM []float32
	WGateV []float32
	BGate  []float32 // 1
	BGateM []float32
	BGateV []float32
}

// NewNeuralNet constructs an untrained Causal Self-Attention + GRU + Pointer-Generator model.
// ActiveSynapses() starts at 0 until real backpropagation updates occur.
func NewNeuralNet(embedDim, hiddenNeurons, maxVocab int, seed int64) *NeuralNet {
	if embedDim < 24 {
		embedDim = 32
	}
	if hiddenNeurons < 48 {
		hiddenNeurons = 96
	}
	if maxVocab < 512 {
		maxVocab = 4096
	}

	rng := rand.New(rand.NewSource(seed))
	D := embedDim
	H := hiddenNeurons
	DH := D + H

	initGaussian := func(n int, scale float64) []float32 {
		s := make([]float32, n)
		for i := range s {
			s[i] = float32(rng.NormFloat64() * scale)
		}
		return s
	}

	dScale := 0.30 / math.Sqrt(float64(D))
	dhScale := 0.30 / math.Sqrt(float64(DH))
	hScale := 0.30 / math.Sqrt(float64(H))

	nn := &NeuralNet{
		EmbedDim:      D,
		HiddenNeurons: H,
		MaxVocab:      maxVocab,
		TokenToID:     make(map[string]int, 512),
		IDToToken:     make([]string, 0, 512),
		TokenCounts:   make([]int, 0, 512),

		PosEmb: initGaussian(maxSeqLen*D, dScale*0.5),
		PosM:   make([]float32, maxSeqLen*D),
		PosV:   make([]float32, maxSeqLen*D),

		WQ:  initGaussian(D*D, dScale),
		WQM: make([]float32, D*D),
		WQV: make([]float32, D*D),
		WK:  initGaussian(D*D, dScale),
		WKM: make([]float32, D*D),
		WKV: make([]float32, D*D),
		WV:  initGaussian(D*D, dScale),
		WVM: make([]float32, D*D),
		WVV: make([]float32, D*D),

		Wz:  initGaussian(DH*H, dhScale),
		WzM: make([]float32, DH*H),
		WzV: make([]float32, DH*H),
		Bz:  make([]float32, H),
		BzM: make([]float32, H),
		BzV: make([]float32, H),

		Wr:  initGaussian(DH*H, dhScale),
		WrM: make([]float32, DH*H),
		WrV: make([]float32, DH*H),
		Br:  make([]float32, H),
		BrM: make([]float32, H),
		BrV: make([]float32, H),

		Wh:  initGaussian(DH*H, dhScale),
		WhM: make([]float32, DH*H),
		WhV: make([]float32, DH*H),
		Bh:  make([]float32, H),
		BhM: make([]float32, H),
		BhV: make([]float32, H),

		WO:  initGaussian(H*D, hScale),
		WOM: make([]float32, H*D),
		WOV: make([]float32, H*D),
		BO:  make([]float32, D),
		BOM: make([]float32, D),
		BOV: make([]float32, D),

		WCopy:  initGaussian(H*D, hScale),
		WCopyM: make([]float32, H*D),
		WCopyV: make([]float32, H*D),
		WGate:  initGaussian(H, hScale),
		WGateM: make([]float32, H),
		WGateV: make([]float32, H),
		BGate:  []float32{0.0},
		BGateM: []float32{0.0},
		BGateV: []float32{0.0},
	}

	for _, sp := range []string{padToken, unkToken, bosToken, eosToken, ctxToken, askToken, ansToken, selfToken} {
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

	scale := float32(0.35 / math.Sqrt(float64(nn.EmbedDim)))
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

// ActiveSynapses counts genuinely trained synaptic connections: weights that have received
// real gradient updates (AdamW second moment v > floor) and exceed magnitude threshold.
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
	total := 0
	total += countTrained(nn.Embeddings, nn.EmbV)
	total += countTrained(nn.WQ, nn.WQV)
	total += countTrained(nn.WK, nn.WKV)
	total += countTrained(nn.WV, nn.WVV)
	total += countTrained(nn.Wz, nn.WzV)
	total += countTrained(nn.Wr, nn.WrV)
	total += countTrained(nn.Wh, nn.WhV)
	total += countTrained(nn.WO, nn.WOV)
	total += countTrained(nn.WCopy, nn.WCopyV)
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

type gradAccum struct {
	dEmb          []float32
	touchedEmbs   map[int]bool
	dBOut         []float32
	dPos          []float32
	dWQ, dWK, dWV []float32
	dWz, dBz      []float32
	dWr, dBr      []float32
	dWh, dBh      []float32
	dWO, dBO      []float32
	dWCopy        []float32
	dWGate        []float32
	dBGate        []float32
	lossSum       float64
	predSteps     int
}

func newGradAccum(V, D, H int) *gradAccum {
	DH := D + H
	return &gradAccum{
		dEmb:        make([]float32, V*D),
		touchedEmbs: make(map[int]bool, 128),
		dBOut:       make([]float32, V),
		dPos:        make([]float32, maxSeqLen*D),
		dWQ:         make([]float32, D*D),
		dWK:         make([]float32, D*D),
		dWV:         make([]float32, D*D),
		dWz:         make([]float32, DH*H),
		dBz:         make([]float32, H),
		dWr:         make([]float32, DH*H),
		dBr:         make([]float32, H),
		dWh:         make([]float32, DH*H),
		dBh:         make([]float32, H),
		dWO:         make([]float32, H*D),
		dBO:         make([]float32, D),
		dWCopy:      make([]float32, H*D),
		dWGate:      make([]float32, H),
		dBGate:      make([]float32, 1),
	}
}

// TrainMiniBatch executes parallel multi-goroutine Causal Self-Attention + GRU + Pointer-Generator
// BPTT across samples in batches of batchSize using workerThreads CPU workers and AdamW optimization.
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
		D := nn.EmbedDim
		H := nn.HiddenNeurons

		accums := make([]*gradAccum, workers)
		var wg sync.WaitGroup
		wg.Add(workers)

		for wIdx := 0; wIdx < workers; wIdx++ {
			accums[wIdx] = newGradAccum(V, D, H)
			wStart := (len(chunk) * wIdx) / workers
			wEnd := (len(chunk) * (wIdx + 1)) / workers
			wSeed := seed + int64(start*31+wIdx*997)

			go func(ga *gradAccum, sub []TrainingSample, s int64) {
				defer wg.Done()
				rng := rand.New(rand.NewSource(s))
				for _, smp := range sub {
					nn.backwardSample(smp, ga, rng)
				}
			}(accums[wIdx], chunk[wStart:wEnd], wSeed)
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
			addSlice(mainAcc.dWQ, other.dWQ)
			addSlice(mainAcc.dWK, other.dWK)
			addSlice(mainAcc.dWV, other.dWV)
			addSlice(mainAcc.dWz, other.dWz)
			addSlice(mainAcc.dBz, other.dBz)
			addSlice(mainAcc.dWr, other.dWr)
			addSlice(mainAcc.dBr, other.dBr)
			addSlice(mainAcc.dWh, other.dWh)
			addSlice(mainAcc.dBh, other.dBh)
			addSlice(mainAcc.dWO, other.dWO)
			addSlice(mainAcc.dBO, other.dBO)
			addSlice(mainAcc.dWCopy, other.dWCopy)
			addSlice(mainAcc.dWGate, other.dWGate)
			addSlice(mainAcc.dBGate, other.dBGate)
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

// findCtxSpan returns end index of <ctx> tokens if seqIDs starts with <ctx> ... <ask>
func findCtxSpan(ids []int) int {
	if len(ids) < 3 || ids[0] != ctxID {
		return 0
	}
	for i := 1; i < len(ids); i++ {
		if ids[i] == askID {
			if i > 1 {
				return i
			}
			return 0
		}
	}
	return 0
}

func (nn *NeuralNet) backwardSample(smp TrainingSample, ga *gradAccum, rng *rand.Rand) {
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
	D := nn.EmbedDim
	H := nn.HiddenNeurons
	DH := D + H
	invSqrtD := float32(1.0 / math.Sqrt(float64(D)))
	ctxEnd := findCtxSpan(ids)

	x := make([]float32, T*D)
	q := make([]float32, T*D)
	k := make([]float32, T*D)
	v := make([]float32, T*D)
	attn := make([]float32, T*T)
	u := make([]float32, T*D)
	du := make([]float32, T*D)

	zGate := make([]float32, T*H)
	rGate := make([]float32, T*H)
	hCand := make([]float32, T*H)
	hState := make([]float32, T*H)
	dhOut := make([]float32, T*H)

	// 1. Embeddings + Positional + Q, K, V
	for t := 0; t < T; t++ {
		tid := ids[t]
		eOff := tid * D
		pOff := t * D
		for d := 0; d < D; d++ {
			x[t*D+d] = nn.Embeddings[eOff+d] + nn.PosEmb[pOff+d]
		}
		matVecMul(x[t*D:(t+1)*D], nn.WQ, q[t*D:(t+1)*D], D, D)
		matVecMul(x[t*D:(t+1)*D], nn.WK, k[t*D:(t+1)*D], D, D)
		matVecMul(x[t*D:(t+1)*D], nn.WV, v[t*D:(t+1)*D], D, D)
	}

	// 2. Causal Scaled Dot-Product Self-Attention + Residual u_t = x_t + a_t
	for t := 0; t < T; t++ {
		maxSc := float32(-1e30)
		for j := 0; j <= t; j++ {
			var dot float32
			for d := 0; d < D; d++ {
				dot += q[t*D+d] * k[j*D+d]
			}
			sc := dot * invSqrtD
			attn[t*T+j] = sc
			if sc > maxSc {
				maxSc = sc
			}
		}
		var sumExp float64
		for j := 0; j <= t; j++ {
			e := math.Exp(float64(attn[t*T+j] - maxSc))
			attn[t*T+j] = float32(e)
			sumExp += e
		}
		invS := float32(1.0 / sumExp)
		copy(u[t*D:(t+1)*D], x[t*D:(t+1)*D])
		for j := 0; j <= t; j++ {
			w := attn[t*T+j] * invS
			attn[t*T+j] = w
			for d := 0; d < D; d++ {
				u[t*D+d] += w * v[j*D+d]
			}
		}
	}

	// 3. GRU Forward Pass across t = 0..T-1
	cVec := make([]float32, DH)
	gVec := make([]float32, DH)
	for t := 0; t < T; t++ {
		copy(cVec[:D], u[t*D:(t+1)*D])
		if t > 0 {
			copy(cVec[D:], hState[(t-1)*H:t*H])
		} else {
			clearSlice(cVec[D:])
		}

		for j := 0; j < H; j++ {
			sz := nn.Bz[j]
			sr := nn.Br[j]
			for i := 0; i < DH; i++ {
				ci := cVec[i]
				idx := i*H + j
				sz += ci * nn.Wz[idx]
				sr += ci * nn.Wr[idx]
			}
			zGate[t*H+j] = sigmoid32(sz)
			rGate[t*H+j] = sigmoid32(sr)
		}

		copy(gVec[:D], u[t*D:(t+1)*D])
		if t > 0 {
			for j := 0; j < H; j++ {
				gVec[D+j] = rGate[t*H+j] * hState[(t-1)*H+j]
			}
		} else {
			clearSlice(gVec[D:])
		}

		for j := 0; j < H; j++ {
			sh := nn.Bh[j]
			for i := 0; i < DH; i++ {
				sh += gVec[i] * nn.Wh[i*H+j]
			}
			cand := float32(math.Tanh(float64(sh)))
			hCand[t*H+j] = cand
			var prevH float32
			if t > 0 {
				prevH = hState[(t-1)*H+j]
			}
			zt := zGate[t*H+j]
			hState[t*H+j] = (1.0-zt)*prevH + zt*cand
		}
	}

	// 4. Output Unembedding + Differentiable Pointer-Generator Copy-Attention over <ctx>
	oVec := make([]float32, D)
	doVec := make([]float32, D)
	qCopy := make([]float32, D)
	dqCopy := make([]float32, D)
	var copyAlpha []float32
	if ctxEnd > 1 {
		copyAlpha = make([]float32, ctxEnd)
	}

	cands := make([]int, 0, negativeSampleCount+16)
	logits := make([]float32, 0, negativeSampleCount+16)
	probs := make([]float32, 0, negativeSampleCount+16)

	negCap := negativeSampleCount
	if V-4 < negCap {
		negCap = V - 4
	}
	if negCap < 1 {
		negCap = 1
	}

	for t := lossStart - 1; t < T-1; t++ {
		target := ids[t+1]
		if target == padID {
			continue
		}

		hSlice := hState[t*H : (t+1)*H]
		for d := 0; d < D; d++ {
			s := nn.BO[d]
			for j := 0; j < H; j++ {
				s += hSlice[j] * nn.WO[j*D+d]
			}
			oVec[d] = s
		}

		cands = cands[:0]
		cands = append(cands, target)
		addUniqueCand := func(cid int) {
			if cid <= bosID && cid != unkID {
				return
			}
			if cid == ctxID || cid == askID || cid == ansID {
				return
			}
			for _, ex := range cands {
				if ex == cid {
					return
				}
			}
			cands = append(cands, cid)
		}
		addUniqueCand(eosID)
		for sIdx := 0; sIdx < T && len(cands) < 14; sIdx++ {
			addUniqueCand(ids[sIdx])
		}
		for tries := 0; len(cands) < negCap+4 && tries < (negCap+4)*3; tries++ {
			addUniqueCand(rng.Intn(V))
		}

		logits = logits[:len(cands)]
		probs = probs[:len(cands)]
		maxL := float32(-1e30)
		var negLogOffset float32
		if len(cands) > 1 && V > len(cands) {
			negLogOffset = float32(math.Log(float64(V-1) / float64(len(cands)-1)))
		}
		for kIdx, cid := range cands {
			var dot float32
			eOff := cid * D
			for d := 0; d < D; d++ {
				dot += oVec[d] * nn.Embeddings[eOff+d]
			}
			l := dot + nn.BOut[cid]
			if kIdx > 0 {
				l += negLogOffset
			}
			logits[kIdx] = l
			if l > maxL {
				maxL = l
			}
		}

		var sumE float64
		for kIdx, l := range logits {
			e := math.Exp(float64(l - maxL))
			probs[kIdx] = float32(e)
			sumE += e
		}
		invE := float32(1.0 / sumE)
		for kIdx := range probs {
			probs[kIdx] *= invE
		}

		pVocabTarget := float64(probs[0])
		wVocab := float32(1.0)

		// Pointer-Generator Copy-Attention when <ctx> span is present
		if ctxEnd > 1 {
			matVecMul(hSlice, nn.WCopy, qCopy, H, D)
			maxC := float32(-1e30)
			for j := 1; j < ctxEnd; j++ {
				var dot float32
				for d := 0; d < D; d++ {
					dot += qCopy[d] * u[j*D+d]
				}
				sc := dot * invSqrtD
				copyAlpha[j] = sc
				if sc > maxC {
					maxC = sc
				}
			}
			var sumC float64
			for j := 1; j < ctxEnd; j++ {
				e := math.Exp(float64(copyAlpha[j] - maxC))
				copyAlpha[j] = float32(e)
				sumC += e
			}
			invC := float32(1.0 / sumC)
			var pCopyTarget float64
			for j := 1; j < ctxEnd; j++ {
				copyAlpha[j] *= invC
				if ids[j] == target {
					pCopyTarget += float64(copyAlpha[j])
				}
			}

			gateLogit := nn.BGate[0]
			for j := 0; j < H; j++ {
				gateLogit += hSlice[j] * nn.WGate[j]
			}
			gCopy := float64(sigmoid32(gateLogit))

			pMix := (1.0-gCopy)*pVocabTarget + gCopy*pCopyTarget
			if pMix < 1e-9 {
				pMix = 1e-9
			}
			ga.lossSum += -math.Log(pMix)
			ga.predSteps++

			wCopy := float64(0)
			if pCopyTarget > 0 {
				wCopy = (gCopy * pCopyTarget) / pMix
			}
			wVocab = float32(1.0 - wCopy)

			// Gate gradients: dL/daGate = gCopy - wCopy
			daGate := float32(gCopy - wCopy)
			ga.dBGate[0] += daGate
			for j := 0; j < H; j++ {
				ga.dWGate[j] += hSlice[j] * daGate
				dhOut[t*H+j] += daGate * nn.WGate[j]
			}

			// Copy-attention gradients when target appears in <ctx>
			if wCopy > 1e-6 && pCopyTarget > 1e-9 {
				clearSlice(dqCopy)
				wCopy32 := float32(wCopy)
				invPC := float32(1.0 / pCopyTarget)
				for j := 1; j < ctxEnd; j++ {
					var matchFrac float32
					if ids[j] == target {
						matchFrac = copyAlpha[j] * invPC
					}
					dSc := wCopy32 * (copyAlpha[j] - matchFrac) * invSqrtD
					for d := 0; d < D; d++ {
						dqCopy[d] += dSc * u[j*D+d]
						du[j*D+d] += dSc * qCopy[d]
					}
				}
				for j := 0; j < H; j++ {
					hj := hSlice[j]
					row := j * D
					var sumDh float32
					for d := 0; d < D; d++ {
						g := dqCopy[d]
						ga.dWCopy[row+d] += hj * g
						sumDh += g * nn.WCopy[row+d]
					}
					dhOut[t*H+j] += sumDh
				}
			}
		} else {
			tp := pVocabTarget
			if tp < 1e-9 {
				tp = 1e-9
			}
			ga.lossSum += -math.Log(tp)
			ga.predSteps++
		}

		clearSlice(doVec)
		for kIdx, cid := range cands {
			grad := probs[kIdx]
			if kIdx == 0 {
				grad -= 1.0
			}
			grad *= wVocab
			ga.touchedEmbs[cid] = true
			ga.dBOut[cid] += grad
			eOff := cid * D
			for d := 0; d < D; d++ {
				doVec[d] += grad * nn.Embeddings[eOff+d]
				ga.dEmb[eOff+d] += grad * oVec[d]
			}
		}

		for d := 0; d < D; d++ {
			doVec[d] = clipGrad(doVec[d], 3.0)
			ga.dBO[d] += doVec[d]
		}
		for j := 0; j < H; j++ {
			hj := hSlice[j]
			var sumDh float32
			row := j * D
			for d := 0; d < D; d++ {
				g := doVec[d]
				ga.dWO[row+d] += hj * g
				sumDh += g * nn.WO[row+d]
			}
			dhOut[t*H+j] += sumDh
		}
	}

	// 5. GRU Backpropagation Through Time (t = T-1 down to 0)
	dhNext := make([]float32, H)
	dhCur := make([]float32, H)
	daz := make([]float32, H)
	dar := make([]float32, H)
	dah := make([]float32, H)
	dg := make([]float32, DH)
	dc := make([]float32, DH)

	for t := T - 1; t >= 0; t-- {
		for j := 0; j < H; j++ {
			dhCur[j] = clipGrad(dhOut[t*H+j]+dhNext[j], 4.0)
		}

		copy(cVec[:D], u[t*D:(t+1)*D])
		if t > 0 {
			copy(cVec[D:], hState[(t-1)*H:t*H])
		} else {
			clearSlice(cVec[D:])
		}
		copy(gVec[:D], u[t*D:(t+1)*D])
		if t > 0 {
			for j := 0; j < H; j++ {
				gVec[D+j] = rGate[t*H+j] * hState[(t-1)*H+j]
			}
		} else {
			clearSlice(gVec[D:])
		}

		for j := 0; j < H; j++ {
			var prevH float32
			if t > 0 {
				prevH = hState[(t-1)*H+j]
			}
			zt := zGate[t*H+j]
			cand := hCand[t*H+j]

			dCand := dhCur[j] * zt
			dah[j] = dCand * (1.0 - cand*cand)
			ga.dBh[j] += dah[j]

			dz := dhCur[j] * (cand - prevH)
			daz[j] = dz * zt * (1.0 - zt)
			ga.dBz[j] += daz[j]
		}

		for i := 0; i < DH; i++ {
			var sumDg float32
			gi := gVec[i]
			row := i * H
			for j := 0; j < H; j++ {
				g := dah[j]
				sumDg += g * nn.Wh[row+j]
				ga.dWh[row+j] += gi * g
			}
			dg[i] = sumDg
		}

		for j := 0; j < H; j++ {
			var prevH float32
			if t > 0 {
				prevH = hState[(t-1)*H+j]
			}
			rt := rGate[t*H+j]
			dr := dg[D+j] * prevH
			dar[j] = dr * rt * (1.0 - rt)
			ga.dBr[j] += dar[j]
		}

		for i := 0; i < DH; i++ {
			var sumDc float32
			ci := cVec[i]
			row := i * H
			for j := 0; j < H; j++ {
				gz := daz[j]
				gr := dar[j]
				sumDc += gz*nn.Wz[row+j] + gr*nn.Wr[row+j]
				ga.dWz[row+j] += ci * gz
				ga.dWr[row+j] += ci * gr
			}
			dc[i] = sumDc
		}

		for d := 0; d < D; d++ {
			du[t*D+d] = clipGrad(du[t*D+d]+dg[d]+dc[d], 4.0)
		}
		if t > 0 {
			for j := 0; j < H; j++ {
				zt := zGate[t*H+j]
				rt := rGate[t*H+j]
				dhNext[j] = dhCur[j]*(1.0-zt) + dg[D+j]*rt + dc[D+j]
			}
		}
	}

	// 6. Causal Self-Attention Backward (t = 0..T-1)
	dq := make([]float32, T*D)
	dk := make([]float32, T*D)
	dv := make([]float32, T*D)
	dAlpha := make([]float32, T)

	for t := 0; t < T; t++ {
		daSlice := du[t*D : (t+1)*D]
		var dotSum float32
		for j := 0; j <= t; j++ {
			var daDotV float32
			for d := 0; d < D; d++ {
				daDotV += daSlice[d] * v[j*D+d]
				dv[j*D+d] += attn[t*T+j] * daSlice[d]
			}
			dAlpha[j] = daDotV
			dotSum += attn[t*T+j] * daDotV
		}
		for j := 0; j <= t; j++ {
			dSc := attn[t*T+j] * (dAlpha[j] - dotSum) * invSqrtD
			for d := 0; d < D; d++ {
				dq[t*D+d] += dSc * k[j*D+d]
				dk[j*D+d] += dSc * q[t*D+d]
			}
		}
	}

	dx := make([]float32, D)
	for t := 0; t < T; t++ {
		copy(dx, du[t*D:(t+1)*D])
		xt := x[t*D : (t+1)*D]
		dqt := dq[t*D : (t+1)*D]
		dkt := dk[t*D : (t+1)*D]
		dvt := dv[t*D : (t+1)*D]

		for i := 0; i < D; i++ {
			var sumDx float32
			xi := xt[i]
			row := i * D
			for j := 0; j < D; j++ {
				gq, gk, gv := dqt[j], dkt[j], dvt[j]
				ga.dWQ[row+j] += xi * gq
				ga.dWK[row+j] += xi * gk
				ga.dWV[row+j] += xi * gv
				sumDx += gq*nn.WQ[row+j] + gk*nn.WK[row+j] + gv*nn.WV[row+j]
			}
			dx[i] += sumDx
		}

		tid := ids[t]
		if tid != padID {
			ga.touchedEmbs[tid] = true
			eOff := tid * D
			for d := 0; d < D; d++ {
				g := clipGrad(dx[d], 4.0)
				ga.dEmb[eOff+d] += g
				ga.dPos[t*D+d] += g
			}
		}
	}
}

func (nn *NeuralNet) applyAdamW(ga *gradAccum, invSteps, lr float32) {
	nn.AdamStep++
	const (
		beta1 = float32(0.9)
		beta2 = float32(0.995)
		eps   = float32(1e-8)
		wd    = float32(2e-4)
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
	updateSlice(nn.WQ, nn.WQM, nn.WQV, ga.dWQ, wd)
	updateSlice(nn.WK, nn.WKM, nn.WKV, ga.dWK, wd)
	updateSlice(nn.WV, nn.WVM, nn.WVV, ga.dWV, wd)
	updateSlice(nn.Wz, nn.WzM, nn.WzV, ga.dWz, wd)
	updateSlice(nn.Bz, nn.BzM, nn.BzV, ga.dBz, 0)
	updateSlice(nn.Wr, nn.WrM, nn.WrV, ga.dWr, wd)
	updateSlice(nn.Br, nn.BrM, nn.BrV, ga.dBr, 0)
	updateSlice(nn.Wh, nn.WhM, nn.WhV, ga.dWh, wd)
	updateSlice(nn.Bh, nn.BhM, nn.BhV, ga.dBh, 0)
	updateSlice(nn.WO, nn.WOM, nn.WOV, ga.dWO, wd)
	updateSlice(nn.BO, nn.BOM, nn.BOV, ga.dBO, 0)
	updateSlice(nn.WCopy, nn.WCopyM, nn.WCopyV, ga.dWCopy, wd)
	updateSlice(nn.WGate, nn.WGateM, nn.WGateV, ga.dWGate, wd)
	updateSlice(nn.BGate, nn.BGateM, nn.BGateV, ga.dBGate, 0)
}

// ForwardFullVocab runs Causal Self-Attention + GRU + Pointer-Generator Copy-Attention across
// seqIDs and computes probabilities + effective log-probabilities across EVERY token in vocabulary.
// Returns (probs[V], effectiveLogProbs[V], entropy, maxProb).
func (nn *NeuralNet) ForwardFullVocab(seqIDs []int) ([]float32, []float32, float64, float64) {
	V := len(nn.IDToToken)
	if V <= reservedTokensCount || len(seqIDs) == 0 {
		return nil, nil, 99.0, 0
	}
	if len(seqIDs) > maxSeqLen {
		seqIDs = seqIDs[len(seqIDs)-maxSeqLen:]
	}
	T := len(seqIDs)
	D := nn.EmbedDim
	H := nn.HiddenNeurons
	DH := D + H
	invSqrtD := float32(1.0 / math.Sqrt(float64(D)))
	ctxEnd := findCtxSpan(seqIDs)

	x := make([]float32, T*D)
	q := make([]float32, T*D)
	k := make([]float32, T*D)
	v := make([]float32, T*D)
	u := make([]float32, T*D)

	for t := 0; t < T; t++ {
		tid := seqIDs[t]
		if tid < 0 || tid >= V {
			tid = unkID
		}
		eOff := tid * D
		pOff := t * D
		for d := 0; d < D; d++ {
			x[t*D+d] = nn.Embeddings[eOff+d] + nn.PosEmb[pOff+d]
		}
		matVecMul(x[t*D:(t+1)*D], nn.WQ, q[t*D:(t+1)*D], D, D)
		matVecMul(x[t*D:(t+1)*D], nn.WK, k[t*D:(t+1)*D], D, D)
		matVecMul(x[t*D:(t+1)*D], nn.WV, v[t*D:(t+1)*D], D, D)
	}

	scBuf := make([]float32, T)
	for t := 0; t < T; t++ {
		maxSc := float32(-1e30)
		for j := 0; j <= t; j++ {
			var dot float32
			for d := 0; d < D; d++ {
				dot += q[t*D+d] * k[j*D+d]
			}
			sc := dot * invSqrtD
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
		copy(u[t*D:(t+1)*D], x[t*D:(t+1)*D])
		for j := 0; j <= t; j++ {
			w := scBuf[j] * invS
			for d := 0; d < D; d++ {
				u[t*D+d] += w * v[j*D+d]
			}
		}
	}

	hCur := make([]float32, H)
	hPrev := make([]float32, H)
	cVec := make([]float32, DH)
	gVec := make([]float32, DH)
	zG := make([]float32, H)
	rG := make([]float32, H)

	for t := 0; t < T; t++ {
		copy(hPrev, hCur)
		copy(cVec[:D], u[t*D:(t+1)*D])
		copy(cVec[D:], hPrev)

		for j := 0; j < H; j++ {
			sz := nn.Bz[j]
			sr := nn.Br[j]
			for i := 0; i < DH; i++ {
				ci := cVec[i]
				idx := i*H + j
				sz += ci * nn.Wz[idx]
				sr += ci * nn.Wr[idx]
			}
			zG[j] = sigmoid32(sz)
			rG[j] = sigmoid32(sr)
		}

		copy(gVec[:D], u[t*D:(t+1)*D])
		for j := 0; j < H; j++ {
			gVec[D+j] = rG[j] * hPrev[j]
		}
		for j := 0; j < H; j++ {
			sh := nn.Bh[j]
			for i := 0; i < DH; i++ {
				sh += gVec[i] * nn.Wh[i*H+j]
			}
			cand := float32(math.Tanh(float64(sh)))
			hCur[j] = (1.0-zG[j])*hPrev[j] + zG[j]*cand
		}
	}

	oVec := make([]float32, D)
	for d := 0; d < D; d++ {
		s := nn.BO[d]
		for j := 0; j < H; j++ {
			s += hCur[j] * nn.WO[j*D+d]
		}
		oVec[d] = s
	}

	logits := make([]float32, V)
	probs := make([]float32, V)
	maxL := float32(-1e30)

	for vid := 0; vid < V; vid++ {
		if vid == padID || vid == unkID || vid == bosID || vid == ctxID || vid == askID || vid == ansID {
			logits[vid] = -1e9
			continue
		}
		var dot float32
		eOff := vid * D
		for d := 0; d < D; d++ {
			dot += oVec[d] * nn.Embeddings[eOff+d]
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

	// Blend Pointer-Generator Copy-Attention distribution over <ctx> span with coverage tracking
	if ctxEnd > 1 {
		ansIdx := -1
		for idx := ctxEnd; idx < T; idx++ {
			if seqIDs[idx] == ansID {
				ansIdx = idx
				break
			}
		}
		emittedCount := make(map[int]int, 16)
		lastCtxPos := 0
		if ansIdx >= 0 {
			for idx := ansIdx + 1; idx < T; idx++ {
				tokID := seqIDs[idx]
				emittedCount[tokID]++
				for j := lastCtxPos + 1; j < ctxEnd; j++ {
					if seqIDs[j] == tokID {
						lastCtxPos = j
						break
					}
				}
			}
		}

		qCopy := make([]float32, D)
		matVecMul(hCur, nn.WCopy, qCopy, H, D)
		copySc := make([]float32, ctxEnd)
		maxC := float32(-1e30)
		for j := 1; j < ctxEnd; j++ {
			var dot float32
			for d := 0; d < D; d++ {
				dot += qCopy[d] * u[j*D+d]
			}
			sc := dot * invSqrtD
			// Coverage mechanism (See et al., 2017): discourage re-copying covered positions & favor sequential flow
			if used := emittedCount[seqIDs[j]]; used > 0 && j <= lastCtxPos {
				sc -= float32(used) * 2.5
			}
			if j == lastCtxPos+1 {
				sc += 1.35
			} else if j == lastCtxPos+2 {
				sc += 0.55
			}
			copySc[j] = sc
			if sc > maxC {
				maxC = sc
			}
		}
		var sumC float64
		for j := 1; j < ctxEnd; j++ {
			e := math.Exp(float64(copySc[j] - maxC))
			copySc[j] = float32(e)
			sumC += e
		}
		invC := float32(1.0 / sumC)

		gateLogit := nn.BGate[0]
		for j := 0; j < H; j++ {
			gateLogit += hCur[j] * nn.WGate[j]
		}
		gCopy := sigmoid32(gateLogit)

		for vid := 0; vid < V; vid++ {
			probs[vid] *= (1.0 - gCopy)
		}
		for j := 1; j < ctxEnd; j++ {
			cid := seqIDs[j]
			if cid >= 0 && cid < V && logits[cid] > -1e8 {
				probs[cid] += gCopy * copySc[j] * invC
			}
		}
	}

	var entropy float64
	var maxProb float64
	for vid := 0; vid < V; vid++ {
		if logits[vid] <= -1e8 {
			continue
		}
		p := float64(probs[vid])
		if p > 1e-12 {
			logits[vid] = float32(math.Log(p))
			entropy -= p * math.Log(p)
			if p > maxProb {
				maxProb = p
			}
		} else {
			logits[vid] = -28.0
		}
	}

	return probs, logits, entropy, maxProb
}

// EmbedText computes an L2-normalized neural representation over subwords for RAG vector search.
func (nn *NeuralNet) EmbedText(text string) []float32 {
	D := nn.EmbedDim
	vec := make([]float32, D)
	subwords := SubwordTokenize(text)
	matched := 0
	for _, sw := range subwords {
		if stopWords[sw] {
			continue
		}
		id, ok := nn.TokenToID[sw]
		if !ok || id < reservedTokensCount || nn.TokenCounts[id] == 0 {
			continue
		}
		off := id * D
		for d := 0; d < D; d++ {
			vec[d] += nn.Embeddings[off+d]
		}
		matched++
	}
	if matched == 0 {
		return vec
	}
	var normSq float64
	for d := 0; d < D; d++ {
		vec[d] /= float32(matched)
		normSq += float64(vec[d] * vec[d])
	}
	if normSq > 1e-12 {
		inv := float32(1.0 / math.Sqrt(normSq))
		for d := 0; d < D; d++ {
			vec[d] *= inv
		}
	}
	return vec
}

// CosineSimilarity computes cosine similarity between two L2-normalized float32 vectors.
func CosineSimilarity(a, b []float32) float32 {
	if len(a) == 0 || len(a) != len(b) {
		return 0
	}
	var dot float32
	for i := range a {
		dot += a[i] * b[i]
	}
	return dot
}

// WriteBinary writes all float32 weight & AdamW moment tensors in compact IEEE-754 little-endian binary.
func (nn *NeuralNet) WriteBinary(w io.Writer) error {
	if err := binary.Write(w, binary.LittleEndian, int32(nn.EmbedDim)); err != nil {
		return err
	}
	if err := binary.Write(w, binary.LittleEndian, int32(nn.HiddenNeurons)); err != nil {
		return err
	}
	if err := binary.Write(w, binary.LittleEndian, int32(nn.MaxVocab)); err != nil {
		return err
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

	slices := [][]float32{
		nn.Embeddings, nn.EmbM, nn.EmbV,
		nn.BOut, nn.BOutM, nn.BOutV,
		nn.PosEmb, nn.PosM, nn.PosV,
		nn.WQ, nn.WQM, nn.WQV,
		nn.WK, nn.WKM, nn.WKV,
		nn.WV, nn.WVM, nn.WVV,
		nn.Wz, nn.WzM, nn.WzV, nn.Bz, nn.BzM, nn.BzV,
		nn.Wr, nn.WrM, nn.WrV, nn.Br, nn.BrM, nn.BrV,
		nn.Wh, nn.WhM, nn.WhV, nn.Bh, nn.BhM, nn.BhV,
		nn.WO, nn.WOM, nn.WOV, nn.BO, nn.BOM, nn.BOV,
		nn.WCopy, nn.WCopyM, nn.WCopyV,
		nn.WGate, nn.WGateM, nn.WGateV,
		nn.BGate, nn.BGateM, nn.BGateV,
	}
	for _, sl := range slices {
		if err := writeFloat32Slice(w, sl); err != nil {
			return err
		}
	}
	return nil
}

// ReadNeuralNetBinary reads a NeuralNet from compact IEEE-754 little-endian binary stream.
func ReadNeuralNetBinary(r io.Reader) (*NeuralNet, error) {
	var d32, h32, mv32, v32 int32
	var step int64
	if err := binary.Read(r, binary.LittleEndian, &d32); err != nil {
		return nil, err
	}
	if err := binary.Read(r, binary.LittleEndian, &h32); err != nil {
		return nil, err
	}
	if err := binary.Read(r, binary.LittleEndian, &mv32); err != nil {
		return nil, err
	}
	if err := binary.Read(r, binary.LittleEndian, &step); err != nil {
		return nil, err
	}
	if err := binary.Read(r, binary.LittleEndian, &v32); err != nil {
		return nil, err
	}
	if d32 <= 0 || h32 <= 0 || v32 < reservedTokensCount || v32 > 200000 {
		return nil, fmt.Errorf("invalid neural header dims D=%d H=%d V=%d", d32, h32, v32)
	}

	V := int(v32)
	nn := &NeuralNet{
		EmbedDim:      int(d32),
		HiddenNeurons: int(h32),
		MaxVocab:      int(mv32),
		AdamStep:      step,
		TokenToID:     make(map[string]int, V),
		IDToToken:     make([]string, V),
		TokenCounts:   make([]int, V),
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

	targets := []*[]float32{
		&nn.Embeddings, &nn.EmbM, &nn.EmbV,
		&nn.BOut, &nn.BOutM, &nn.BOutV,
		&nn.PosEmb, &nn.PosM, &nn.PosV,
		&nn.WQ, &nn.WQM, &nn.WQV,
		&nn.WK, &nn.WKM, &nn.WKV,
		&nn.WV, &nn.WVM, &nn.WVV,
		&nn.Wz, &nn.WzM, &nn.WzV, &nn.Bz, &nn.BzM, &nn.BzV,
		&nn.Wr, &nn.WrM, &nn.WrV, &nn.Br, &nn.BrM, &nn.BrV,
		&nn.Wh, &nn.WhM, &nn.WhV, &nn.Bh, &nn.BhM, &nn.BhV,
		&nn.WO, &nn.WOM, &nn.WOV, &nn.BO, &nn.BOM, &nn.BOV,
		&nn.WCopy, &nn.WCopyM, &nn.WCopyV,
		&nn.WGate, &nn.WGateM, &nn.WGateV,
		&nn.BGate, &nn.BGateM, &nn.BGateV,
	}
	for _, dst := range targets {
		sl, err := readFloat32Slice(r)
		if err != nil {
			return nil, err
		}
		*dst = sl
	}
	return nn, nil
}

func writeFloat32Slice(w io.Writer, s []float32) error {
	if err := binary.Write(w, binary.LittleEndian, uint32(len(s))); err != nil {
		return err
	}
	buf := make([]byte, len(s)*4)
	for i, v := range s {
		binary.LittleEndian.PutUint32(buf[i*4:(i+1)*4], math.Float32bits(v))
	}
	_, err := w.Write(buf)
	return err
}

func readFloat32Slice(r io.Reader) ([]float32, error) {
	var n uint32
	if err := binary.Read(r, binary.LittleEndian, &n); err != nil {
		return nil, err
	}
	if n > 50_000_000 {
		return nil, fmt.Errorf("tensor slice too large: %d", n)
	}
	buf := make([]byte, int(n)*4)
	if _, err := io.ReadFull(r, buf); err != nil {
		return nil, err
	}
	out := make([]float32, int(n))
	for i := range out {
		out[i] = math.Float32frombits(binary.LittleEndian.Uint32(buf[i*4 : (i+1)*4]))
	}
	return out, nil
}

func matVecMul(vec, mat, out []float32, rows, cols int) {
	for j := 0; j < cols; j++ {
		var sum float32
		for i := 0; i < rows; i++ {
			sum += vec[i] * mat[i*cols+j]
		}
		out[j] = sum
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
