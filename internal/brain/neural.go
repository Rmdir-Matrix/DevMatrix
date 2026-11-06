package brain

import (
	"math"
	"math/rand"
)

const (
	padToken     = "<pad>"
	unkToken     = "<unk>"
	bosToken     = "<bos>"
	eosToken     = "<eos>"
	sepToken     = "<sep>"
	thoughtToken = "<thought>"
	selfToken    = "<self>"

	padID     = 0
	unkID     = 1
	bosID     = 2
	eosID     = 3
	sepID     = 4
	thoughtID = 5
	selfID    = 6

	reservedTokensCount = 7
	negativeSampleCount = 28
	synapseThreshold    = float32(0.045)
)

// NeuralNet is a conditioned positional feed-forward neural language & embedding model.
// Input layout (4 * EmbedDim): [PromptConditionMean, Pos(t-3), Pos(t-2), Pos(t-1)].
type NeuralNet struct {
	EmbedDim      int            `json:"embed_dim"`
	HiddenNeurons int            `json:"hidden_neurons"`
	MaxVocab      int            `json:"max_vocab"`
	TokenToID     map[string]int `json:"token_to_id"`
	IDToToken     []string       `json:"id_to_token"`
	TokenCounts   []int          `json:"token_counts"`

	Embeddings []float32 `json:"embeddings"`
	W1         []float32 `json:"w1"`
	B1         []float32 `json:"b1"`
	W2         []float32 `json:"w2"`
	B2         []float32 `json:"b2"`
}

// NewNeuralNet initializes an untrained neural network sized to the hardware profile.
// Vocabulary starts with ONLY structural markers (0 words learned yet).
func NewNeuralNet(embedDim, hiddenNeurons, maxVocab int, seed int64) *NeuralNet {
	if embedDim < 16 {
		embedDim = 32
	}
	if hiddenNeurons < 32 {
		hiddenNeurons = 64
	}
	if maxVocab < 512 {
		maxVocab = 4096
	}

	rng := rand.New(rand.NewSource(seed))
	inDim := 4 * embedDim
	w1Scale := float32(1.0 / math.Sqrt(float64(inDim)))

	w1 := make([]float32, inDim*hiddenNeurons)
	for i := range w1 {
		w1[i] = float32(rng.NormFloat64()) * w1Scale * 0.30
	}

	nn := &NeuralNet{
		EmbedDim:      embedDim,
		HiddenNeurons: hiddenNeurons,
		MaxVocab:      maxVocab,
		TokenToID:     make(map[string]int, 256),
		IDToToken:     make([]string, 0, 256),
		TokenCounts:   make([]int, 0, 256),
		W1:            w1,
		B1:            make([]float32, hiddenNeurons),
	}

	for _, sp := range []string{padToken, unkToken, bosToken, eosToken, sepToken, thoughtToken, selfToken} {
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

	embScale := float32(1.0 / math.Sqrt(float64(nn.EmbedDim)))
	w2Scale := float32(1.0 / math.Sqrt(float64(nn.HiddenNeurons)))

	for d := 0; d < nn.EmbedDim; d++ {
		nn.Embeddings = append(nn.Embeddings, float32(rng.NormFloat64())*embScale*0.25)
	}
	for h := 0; h < nn.HiddenNeurons; h++ {
		nn.W2 = append(nn.W2, float32(rng.NormFloat64())*w2Scale*0.20)
	}
	nn.B2 = append(nn.B2, 0)
	return id
}

// RegisterTokens adds any unseen tokens from the slice into the vocabulary up to MaxVocab.
func (nn *NeuralNet) RegisterTokens(tokens []string, rng *rand.Rand) {
	for _, tok := range tokens {
		nn.ensureToken(tok, rng)
	}
}

// LearnedWordCount returns the number of real words/punctuation learned beyond structural markers.
func (nn *NeuralNet) LearnedWordCount() int {
	if len(nn.IDToToken) <= reservedTokensCount {
		return 0
	}
	return len(nn.IDToToken) - reservedTokensCount
}

// VocabSize returns total vocabulary slots including structural tokens.
func (nn *NeuralNet) VocabSize() int {
	return len(nn.IDToToken)
}

// KnowsToken reports whether a token has been trained into the neural vocabulary at least once.
func (nn *NeuralNet) KnowsToken(tok string) bool {
	id, ok := nn.TokenToID[tok]
	if !ok || id < reservedTokensCount {
		return false
	}
	return nn.TokenCounts[id] > 0
}

// ActiveSynapses counts strong learned synaptic weights (|weight| > threshold) across all layers.
func (nn *NeuralNet) ActiveSynapses() int {
	count := 0
	for _, w := range nn.W1 {
		if w > synapseThreshold || w < -synapseThreshold {
			count++
		}
	}
	for _, w := range nn.W2 {
		if w > synapseThreshold || w < -synapseThreshold {
			count++
		}
	}
	for _, w := range nn.Embeddings {
		if w > synapseThreshold || w < -synapseThreshold {
			count++
		}
	}
	return count
}

// TrainSentence trains an unconditioned prose sentence: <bos> tokens... <eos>.
func (nn *NeuralNet) TrainSentence(tokens []string, lr float32, rng *rand.Rand) (float64, int) {
	return nn.TrainConditioned(nil, tokens, lr, rng)
}

// TrainConditioned runs forward + sampled-softmax cross-entropy backpropagation on
// [condTokens -> <sep> -> targetTokens -> <eos>] (or [<bos> -> targetTokens -> <eos>] if condTokens is empty).
func (nn *NeuralNet) TrainConditioned(condTokens []string, targetTokens []string, lr float32, rng *rand.Rand) (float64, int) {
	if len(targetTokens) == 0 {
		return 0, 0
	}

	condIDs := make([]int, 0, len(condTokens)+1)
	for _, tok := range condTokens {
		if id, ok := nn.TokenToID[tok]; ok {
			condIDs = append(condIDs, id)
		} else {
			condIDs = append(condIDs, nn.ensureToken(tok, rng))
		}
	}

	seq := make([]int, 0, len(targetTokens)+2)
	if len(condIDs) > 0 {
		seq = append(seq, sepID)
	} else {
		seq = append(seq, bosID)
		condIDs = append(condIDs, bosID)
	}
	for _, tok := range targetTokens {
		if id, ok := nn.TokenToID[tok]; ok {
			seq = append(seq, id)
		} else {
			seq = append(seq, nn.ensureToken(tok, rng))
		}
	}
	seq = append(seq, eosID)

	vocabLen := len(nn.IDToToken)
	if vocabLen <= reservedTokensCount+1 {
		return 0, 0
	}

	D := nn.EmbedDim
	H := nn.HiddenNeurons
	inDim := 4 * D
	if len(nn.W1) != inDim*H {
		// Upgrade legacy W1 tensor cleanly if needed
		nn.resizeInputDim(inDim, H, rng)
	}

	x := make([]float32, inDim)
	h := make([]float32, H)
	dh := make([]float32, H)
	dz := make([]float32, H)
	dx := make([]float32, inDim)

	availNeg := vocabLen - 3
	negCap := negativeSampleCount
	if availNeg < negCap {
		negCap = availNeg
	}
	if negCap < 1 {
		negCap = 1
	}

	cands := make([]int, 0, negCap+2)
	logits := make([]float32, 0, negCap+2)
	probs := make([]float32, 0, negCap+2)

	var totalLoss float64
	steps := 0
	weightDecay := float32(1e-5)

	for pos := 1; pos < len(seq); pos++ {
		target := seq[pos]
		prev1 := seq[pos-1]
		prev2 := padID
		if pos >= 2 {
			prev2 = seq[pos-2]
		}
		prev3 := padID
		if pos >= 3 {
			prev3 = seq[pos-3]
		}

		// Slot 0: Prompt/Conditioning Mean Vector
		for d := 0; d < D; d++ {
			x[d] = 0
		}
		for _, cid := range condIDs {
			off := cid * D
			for d := 0; d < D; d++ {
				x[d] += nn.Embeddings[off+d]
			}
		}
		invCond := 1.0 / float32(len(condIDs))
		for d := 0; d < D; d++ {
			x[d] *= invCond
		}

		// Slots 1, 2, 3: Positional embeddings [pos-3, pos-2, pos-1]
		copy(x[1*D:2*D], nn.Embeddings[prev3*D:(prev3+1)*D])
		copy(x[2*D:3*D], nn.Embeddings[prev2*D:(prev2+1)*D])
		copy(x[3*D:4*D], nn.Embeddings[prev1*D:(prev1+1)*D])

		// Hidden forward: h = tanh(x * W1 + B1)
		for j := 0; j < H; j++ {
			sum := nn.B1[j]
			for i := 0; i < inDim; i++ {
				sum += x[i] * nn.W1[i*H+j]
			}
			h[j] = float32(math.Tanh(float64(sum)))
		}

		// Candidate set: [0]=target, always include eosID if target!=eosID, plus sampled negatives
		cands = cands[:0]
		cands = append(cands, target)
		if target != eosID {
			cands = append(cands, eosID)
		}
		maxTries := (negCap + 2) * 4
		for tries := 0; len(cands) < negCap+1 && tries < maxTries; tries++ {
			neg := rng.Intn(vocabLen)
			if neg == target || neg == padID || neg == bosID || neg == sepID {
				continue
			}
			dup := false
			for _, existing := range cands {
				if existing == neg {
					dup = true
					break
				}
			}
			if !dup {
				cands = append(cands, neg)
			}
		}

		logits = logits[:len(cands)]
		probs = probs[:len(cands)]
		maxLogit := float32(-1e30)
		for k, cid := range cands {
			w2Off := cid * H
			s := nn.B2[cid]
			for j := 0; j < H; j++ {
				s += h[j] * nn.W2[w2Off+j]
			}
			logits[k] = s
			if s > maxLogit {
				maxLogit = s
			}
		}

		var sumExp float64
		for k, s := range logits {
			e := math.Exp(float64(s - maxLogit))
			probs[k] = float32(e)
			sumExp += e
		}
		invSum := float32(1.0 / sumExp)
		for k := range probs {
			probs[k] *= invSum
		}

		targetProb := float64(probs[0])
		if targetProb < 1e-9 {
			targetProb = 1e-9
		}
		totalLoss += -math.Log(targetProb)
		steps++

		// Backpropagation through output layer
		for j := 0; j < H; j++ {
			dh[j] = 0
		}
		for k, cid := range cands {
			grad := probs[k]
			if k == 0 {
				grad -= 1.0
			}
			w2Off := cid * H
			for j := 0; j < H; j++ {
				oldW := nn.W2[w2Off+j]
				dh[j] += grad * oldW
				nn.W2[w2Off+j] = oldW - lr*(grad*h[j]+weightDecay*oldW)
			}
			nn.B2[cid] -= lr * grad
		}

		// Hidden gradient dz = dh * (1 - h^2)
		for j := 0; j < H; j++ {
			g := dh[j] * (1.0 - h[j]*h[j])
			if g > 3.0 {
				g = 3.0
			} else if g < -3.0 {
				g = -3.0
			}
			dz[j] = g
			nn.B1[j] -= lr * g
		}

		// Backpropagate into W1 and input vector dx
		for i := 0; i < inDim; i++ {
			var sumDx float32
			rowOff := i * H
			xi := x[i]
			for j := 0; j < H; j++ {
				oldW1 := nn.W1[rowOff+j]
				sumDx += dz[j] * oldW1
				nn.W1[rowOff+j] = oldW1 - lr*(xi*dz[j]+weightDecay*oldW1)
			}
			dx[i] = sumDx
		}

		// Update positional embeddings (skip padID=0)
		if prev3 != padID {
			off := prev3 * D
			for d := 0; d < D; d++ {
				nn.Embeddings[off+d] -= lr * dx[1*D+d]
			}
		}
		if prev2 != padID {
			off := prev2 * D
			for d := 0; d < D; d++ {
				nn.Embeddings[off+d] -= lr * dx[2*D+d]
			}
		}
		off1 := prev1 * D
		for d := 0; d < D; d++ {
			nn.Embeddings[off1+d] -= lr * dx[3*D+d]
		}

		// Update prompt conditioning embeddings
		condScale := lr * invCond
		for _, cid := range condIDs {
			if cid == padID {
				continue
			}
			off := cid * D
			for d := 0; d < D; d++ {
				nn.Embeddings[off+d] -= condScale * dx[d]
			}
		}
	}

	return totalLoss, steps
}

// ForwardLogits computes neural output logits for candidate token IDs given conditioning IDs and [prev3, prev2, prev1].
func (nn *NeuralNet) ForwardLogits(condIDs []int, prev3, prev2, prev1 int, candidateIDs []int) []float32 {
	out := make([]float32, len(candidateIDs))
	if len(candidateIDs) == 0 || len(nn.IDToToken) == 0 {
		return out
	}

	D := nn.EmbedDim
	H := nn.HiddenNeurons
	inDim := 4 * D
	if len(nn.W1) != inDim*H {
		return out
	}

	x := make([]float32, inDim)
	if len(condIDs) > 0 {
		valid := 0
		for _, cid := range condIDs {
			if cid >= 0 && cid < len(nn.IDToToken) {
				off := cid * D
				for d := 0; d < D; d++ {
					x[d] += nn.Embeddings[off+d]
				}
				valid++
			}
		}
		if valid > 0 {
			inv := 1.0 / float32(valid)
			for d := 0; d < D; d++ {
				x[d] *= inv
			}
		}
	} else {
		copy(x[:D], nn.Embeddings[bosID*D:(bosID+1)*D])
	}

	copySafe := func(slot int, tid int) {
		if tid >= 0 && tid < len(nn.IDToToken) {
			copy(x[slot*D:(slot+1)*D], nn.Embeddings[tid*D:(tid+1)*D])
		}
	}
	copySafe(1, prev3)
	copySafe(2, prev2)
	copySafe(3, prev1)

	h := make([]float32, H)
	for j := 0; j < H; j++ {
		sum := nn.B1[j]
		for i := 0; i < inDim; i++ {
			sum += x[i] * nn.W1[i*H+j]
		}
		h[j] = float32(math.Tanh(float64(sum)))
	}

	for k, cid := range candidateIDs {
		if cid < 0 || cid >= len(nn.IDToToken) {
			out[k] = -20.0
			continue
		}
		w2Off := cid * H
		s := nn.B2[cid]
		for j := 0; j < H; j++ {
			s += h[j] * nn.W2[w2Off+j]
		}
		out[k] = s
	}
	return out
}

func (nn *NeuralNet) resizeInputDim(inDim, H int, rng *rand.Rand) {
	scale := float32(1.0 / math.Sqrt(float64(inDim)))
	w1 := make([]float32, inDim*H)
	for i := range w1 {
		w1[i] = float32(rng.NormFloat64()) * scale * 0.30
	}
	nn.W1 = w1
}

// EmbedText computes an L2-normalized mean-pooled neural vector representation for semantic retrieval.
func (nn *NeuralNet) EmbedText(text string) []float32 {
	D := nn.EmbedDim
	vec := make([]float32, D)
	toks := ExtractKeywords(text, 24)
	if len(toks) == 0 {
		toks = Tokenize(text)
	}
	matched := 0
	for _, tok := range toks {
		id, ok := nn.TokenToID[tok]
		if !ok || id < reservedTokensCount {
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

// CosineSimilarity computes cosine similarity between two L2-normalized embedding vectors.
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
