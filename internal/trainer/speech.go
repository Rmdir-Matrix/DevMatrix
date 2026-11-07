package trainer

import (
	"github.com/Rmdir-Matrix/DevMatrix/internal/brain"
)

// IntelligenceStage returns a short UI badge describing Matrix's current training phase.
func IntelligenceStage(level int, dictsDone, dictsTotal, grammarDone, grammarTotal int) string {
	if dictsTotal > 0 && dictsDone < dictsTotal {
		return "Studying Dictionaries"
	}
	if grammarTotal > 0 && grammarDone < grammarTotal {
		return "Learning Grammar"
	}
	switch {
	case level < 25:
		return "Curious Learner"
	case level < 50:
		return "Focused Student"
	case level < 100:
		return "Analytical Scholar"
	default:
		return "Neural Polymath"
	}
}

// GenerateTrainingThought decodes a live status reflection 100% from Matrix's
// Causal Self-Attention + GRU neural weights (zero hardcoded thought strings or templates).
func GenerateTrainingThought(b *brain.Brain, rep brain.TrainReport) string {
	return b.GenerateThoughtFromWeights(rep.ArticleTitle, rep.ArticleExcerpt)
}
