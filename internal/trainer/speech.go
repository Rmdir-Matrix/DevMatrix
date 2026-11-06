package trainer

import (
	"github.com/Rmdir-Matrix/DevMatrix/internal/brain"
)

// IntelligenceStage returns a short UI badge describing Matrix's training phase.
func IntelligenceStage(level int, grammarDone int, grammarTotal int) string {
	if grammarTotal > 0 && grammarDone < grammarTotal {
		return "Learning Words & Grammar"
	}
	switch {
	case level < 10:
		return "Early Reader"
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

// GenerateTrainingThought decodes a live status reflection directly from Matrix's trained
// neural language model (zero hardcoded thought strings).
func GenerateTrainingThought(b *brain.Brain, rep brain.TrainStepReport) string {
	return b.GenerateModelThought(rep.SampleKeywords)
}
