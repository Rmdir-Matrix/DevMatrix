package brain

import (
	"fmt"
	"math"
	"math/rand"
)

// ParameterCount counts model weights and biases, not optimizer state.
func (nn *NeuralNet) ParameterCount() int64 {
	n := len(nn.Embeddings) + len(nn.BOut) + len(nn.PosEmb) + len(nn.NormF)
	for i := range nn.Layers {
		l := &nn.Layers[i]
		n += len(l.Norm1) + len(l.WQ) + len(l.WK) + len(l.WV) + len(l.WO) + len(l.Norm2) + len(l.W1) + len(l.B1) + len(l.W2) + len(l.B2)
	}
	return int64(n)
}

// widenFFN preserves the old function exactly: new inputs are initialized but their
// outgoing weights are zero. Adam moments for existing neurons remain unchanged.
// Call only between batches, never while training or inference workers are running.
func (nn *NeuralNet) widenFFN(newF int, seed int64) {
	oldF, d := nn.HiddenNeurons, nn.EmbedDim
	if newF <= oldF {
		return
	}
	rng := rand.New(rand.NewSource(seed))
	scale := 0.5 / math.Sqrt(float64(d))
	for i := range nn.Layers {
		l := &nn.Layers[i]
		expandRows := func(old []float32, random bool) []float32 {
			dst := make([]float32, d*newF)
			for j := 0; j < d; j++ {
				copy(dst[j*newF:j*newF+oldF], old[j*oldF:(j+1)*oldF])
				if random {
					for k := oldF; k < newF; k++ {
						dst[j*newF+k] = float32(rng.NormFloat64() * scale)
					}
				}
			}
			return dst
		}
		extend := func(old []float32, n int) []float32 { dst := make([]float32, n); copy(dst, old); return dst }
		l.W1 = expandRows(l.W1, true)
		l.W1M = expandRows(l.W1M, false)
		l.W1V = expandRows(l.W1V, false)
		l.B1 = extend(l.B1, newF)
		l.B1M = extend(l.B1M, newF)
		l.B1V = extend(l.B1V, newF)
		l.W2 = extend(l.W2, newF*d)
		l.W2M = extend(l.W2M, newF*d)
		l.W2V = extend(l.W2V, newF*d)
	}
	nn.HiddenNeurons = newF
}

// TryGrow spends one earned training milestone. Estimates include weights, Adam
// state and per-worker gradients, but are not a hard process memory guarantee.
func (b *Brain) TryGrow() (string, bool) {
	if b.Net == nil {
		return "No model loaded.", false
	}
	if b.GrowthEvents >= b.TrainingItems/8 {
		return "Train on more material to earn a growth milestone (one per 8 trained items).", false
	}
	nn := b.Net
	maxF := b.GrowthMaxWidth
	if maxF <= 0 {
		return "Growth limits are not configured; reload the model with a hardware profile.", false
	}
	if nn.HiddenNeurons >= maxF {
		return "The current hardware profile's growth limit has been reached.", false
	}
	oldF := nn.HiddenNeurons
	step := maxInt(16, nn.EmbedDim/2)
	newF := minInt(oldF+step, maxF)
	added := int64(len(nn.Layers)) * int64(newF-oldF) * int64(2*nn.EmbedDim+1)
	workers := maxInt(b.WorkerThreads, 1)
	// Reserve for the full configured vocabulary, not only the current one.
	futureVocab := int64(maxInt(nn.MaxVocab-len(nn.IDToToken), 0)) * int64(nn.EmbedDim+1)
	estimate := (nn.ParameterCount()+added+futureVocab)*int64(12+4*(workers+1)) + 32*1024*1024
	if b.GrowthRAMMB <= 0 || estimate > int64(b.GrowthRAMMB)*1024*1024*3/5 {
		return "Growth deferred: estimated training memory needs more headroom under the configured RAM budget.", false
	}
	nn.widenFFN(newF, int64(b.TrainingItems)*7919+nn.AdamStep)
	b.GrowthEvents++
	return fmt.Sprintf("FFN width %d → %d; parameters %d → %d", oldF, newF, nn.ParameterCount()-added, nn.ParameterCount()), true
}
