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

// GenerateTrainingThought decodes a live status reflection directly from Matrix's trained
// neural language model (zero hardcoded thought strings).
func GenerateTrainingThought(b *brain.Brain, rep brain.TrainStepReport) string {
	return b.GenerateModelThought(rep.SampleKeywords)
}
