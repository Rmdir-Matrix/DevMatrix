package brain

import (
	"math"
	"math/rand"
)

const (
	padToken = "<pad>"
	unkToken = "<unk>"
	bosToken = "<bos>"
	eosToken = "<eos>"

	negativeSampleCount = 24
	synapseThreshold    = float32(0.045)
)

// NeuralNet is a genuine trainable feed-forward context neural language and embedding model.
// Weights grow dynamically up to MaxVocab so memory remains strictly within the user's RAM profile.
type NeuralNet struct {
	EmbedDim      int            `json:"embed_dim"`
	HiddenNeurons int            `json:"hidden_neurons"`
	MaxVocab      int            `json:"max_vocab"`
	TokenToID     map[string]int `json:"token_to_id"`
	IDToToken     []string       `json:"id_to_token"`
	TokenCounts   []int          `json:"token_counts"`

	// Flattened float32 tensors for low memory overhead and fast CPU cache locality:
	// Embeddings: len(IDToToken) * EmbedDim
	// W1:         (2 * EmbedDim) * HiddenNeurons
	// B1:         HiddenNeurons
	// W2:         len(IDToToken) * HiddenNeurons
	// B2:         len(IDToToken)
	Embeddings []float32 `json:"embeddings"`
	W1         []float32 `json:"w1"`
	B1         []float32 `json:"b1"`
	W2         []float32 `json:"w2"`
	B2         []float32 `json:"b2"`
}

// NewNeuralNet initializes a new neural network sized to the hardware profile.
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
	inDim := 2 * embedDim
	w1Scale := float32(1.0 / math.Sqrt(float64(inDim)))

	w1 := make([]float32, inDim*hiddenNeurons)
	for i := range w1 {
		w1[i] = float32(rng.NormFloat64()) * w1Scale * 0.35
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

	for _, sp := range []string{padToken, unkToken, bosToken, eosToken} {
		nn.ensureToken(sp, rng)
	}
	return nn
}

func (nn *NeuralNet) ensureToken(tok string, rng *rand.Rand) int {
	if id, ok := nn.TokenToID[tok]; ok {
		nn.TokenCounts[id]++
		return id
	}
	if len(nn.IDToToken) >= nn.MaxVocab {
		nn.TokenCounts[1]++ // <unk>
		return 1
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

// VocabSize returns the number of learned tokens in the neural vocabulary.
func (nn *NeuralNet) VocabSize() int {
	return len(nn.IDToToken)
}

// ActiveSynapses counts strong learned synaptic connections (|weight| > threshold) across layers.
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

// TrainSentence runs forward + sampled-softmax cross-entropy backpropagation across a token slice.
// Returns (sumLoss, tokenPredictions).
func (nn *NeuralNet) TrainSentence(tokens []string, lr float32, rng *rand.Rand) (float64, int) {
	if len(tokens) == 0 {
		return 0, 0
	}

	ids := make([]int, 0, len(tokens)+2)
	ids = append(ids, 2) // <bos>
	for _, tok := range tokens {
		if id, ok := nn.TokenToID[tok]; ok {
			ids = append(ids, id)
		} else {
			ids = append(ids, nn.ensureToken(tok, rng))
		}
	}
	ids = append(ids, 3) // <eos>

	vocabLen := len(nn.IDToToken)
	if vocabLen < 5 {
		return 0, 0
	}

	D := nn.EmbedDim
	H := nn.HiddenNeurons
	inDim := 2 * D

	x := make([]float32, inDim)
	h := make([]float32, H)
	dh := make([]float32, H)
	dz := make([]float32, H)
	dx := make([]float32, inDim)

	negCap := negativeSampleCount
	if vocabLen-2 < negCap {
		negCap = vocabLen - 2
	}
	if negCap < 1 {
		return 0, 0
	}
	cands := make([]int, 0, negCap+1)
	logits := make([]float32, 0, negCap+1)
	probs := make([]float32, 0, negCap+1)

	var totalLoss float64
	steps := 0
	weightDecay := float32(1e-5)

	for pos := 1; pos < len(ids); pos++ {
		target := ids[pos]
		prev := ids[pos-1]

		// Build input vector x = [context_mean(pos-5..pos-2), embed(prev)]
		for i := range x {
			x[i] = 0
		}
		ctxStart := pos - 5
		if ctxStart < 0 {
			ctxStart = 0
		}
		ctxCount := 0
		for cPos := ctxStart; cPos < pos-1; cPos++ {
			cid := ids[cPos]
			off := cid * D
			for d := 0; d < D; d++ {
				x[d] += nn.Embeddings[off+d]
			}
			ctxCount++
		}
		if ctxCount > 0 {
			inv := 1.0 / float32(ctxCount)
			for d := 0; d < D; d++ {
				x[d] *= inv
			}
		} else {
			// Use <bos> embedding if at first token
			bosOff := 2 * D
			copy(x[:D], nn.Embeddings[bosOff:bosOff+D])
		}
		prevOff := prev * D
		copy(x[D:2*D], nn.Embeddings[prevOff:prevOff+D])

		// Forward hidden layer: h = tanh(x * W1 + B1)
		for j := 0; j < H; j++ {
			sum := nn.B1[j]
			for i := 0; i < inDim; i++ {
				sum += x[i] * nn.W1[i*H+j]
			}
			h[j] = float32(math.Tanh(float64(sum)))
		}

		// Candidate set: index 0 is true target, followed by sampled negative tokens
		cands = cands[:0]
		cands = append(cands, target)
		maxTries := (negCap + 1) * 4
		for tries := 0; len(cands) < negCap+1 && tries < maxTries; tries++ {
			neg := rng.Intn(vocabLen)
			if neg == target || neg == 0 {
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

		// Compute candidate logits
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

		// Backpropagation through sampled output layer
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

		// Hidden activation derivative dz = dh * (1 - h^2) with gradient clipping
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

		// Backprop into W1 and input vector dx
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

		// Update immediate previous token embedding
		for d := 0; d < D; d++ {
			nn.Embeddings[prevOff+d] -= lr * dx[D+d]
		}
		// Update context window embeddings
		if ctxCount > 0 {
			ctxScale := lr / float32(ctxCount)
			for cPos := ctxStart; cPos < pos-1; cPos++ {
				cid := ids[cPos]
				off := cid * D
				for d := 0; d < D; d++ {
					nn.Embeddings[off+d] -= ctxScale * dx[d]
				}
			}
		}
	}

	return totalLoss, steps
}

// ScoreCandidates scores candidate next tokens given recent token history using forward inference.
func (nn *NeuralNet) ScoreCandidates(history []string, candidates []string) []float32 {
	out := make([]float32, len(candidates))
	if len(candidates) == 0 || len(nn.IDToToken) == 0 {
		return out
	}

	D := nn.EmbedDim
	H := nn.HiddenNeurons
	x := make([]float32, 2*D)

	histIDs := make([]int, 0, len(history)+1)
	histIDs = append(histIDs, 2) // <bos>
	for _, tok := range history {
		if id, ok := nn.TokenToID[tok]; ok {
			histIDs = append(histIDs, id)
		} else {
			histIDs = append(histIDs, 1)
		}
	}

	prev := histIDs[len(histIDs)-1]
	ctxStart := len(histIDs) - 5
	if ctxStart < 0 {
		ctxStart = 0
	}
	ctxCount := 0
	for i := ctxStart; i < len(histIDs)-1; i++ {
		off := histIDs[i] * D
		for d := 0; d < D; d++ {
			x[d] += nn.Embeddings[off+d]
		}
		ctxCount++
	}
	if ctxCount > 0 {
		inv := 1.0 / float32(ctxCount)
		for d := 0; d < D; d++ {
			x[d] *= inv
		}
	} else {
		copy(x[:D], nn.Embeddings[2*D:3*D])
	}
	copy(x[D:2*D], nn.Embeddings[prev*D:(prev+1)*D])

	h := make([]float32, H)
	for j := 0; j < H; j++ {
		sum := nn.B1[j]
		for i := 0; i < 2*D; i++ {
			sum += x[i] * nn.W1[i*H+j]
		}
		h[j] = float32(math.Tanh(float64(sum)))
	}

	for k, cand := range candidates {
		cid, ok := nn.TokenToID[cand]
		if !ok {
			out[k] = -10.0
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
		if !ok || id <= 3 {
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

// CosineSimilarity computes the cosine similarity between two normalized embedding vectors.
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
