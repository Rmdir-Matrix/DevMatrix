package trainer

import (
	"fmt"
	"math/rand"
	"strings"

	"github.com/Rmdir-Matrix/DevMatrix/internal/brain"
)

// IntelligenceStage returns a short label describing Matrix's cognitive maturity at a given level.
func IntelligenceStage(level int) string {
	switch {
	case level < 5:
		return "Newborn Spark"
	case level < 10:
		return "Early Babbler"
	case level < 25:
		return "Curious Learner"
	case level < 50:
		return "Focused Student"
	case level < 100:
		return "Analytical Scholar"
	case level < 175:
		return "Neural Polymath"
	default:
		return "Transcendent Sage"
	}
}

// GenerateTrainingThought produces a periodic status message spoken by Matrix during training.
// Grammar, vocabulary, self-awareness, and technical depth progressively evolve with level.
func GenerateTrainingThought(b *brain.Brain, rep brain.TrainStepReport, crossedMilestone int, rng *rand.Rand) string {
	name := b.Name
	if strings.TrimSpace(name) == "" {
		name = "Matrix"
	}
	lvl := rep.NewLevel
	kw1 := pickKeyword(rep.SampleKeywords, 0, "words")
	kw2 := pickKeyword(rep.SampleKeywords, 1, "patterns")

	// Guarantee iconic milestone lines when hitting or speaking at key levels
	if crossedMilestone == 1 || (lvl == 1 && b.ArticlesRead <= 1) {
		return fmt.Sprintf("%s is learn now!", name)
	}
	if crossedMilestone == 10 {
		return "I'm learning stuff!"
	}
	if crossedMilestone == 100 {
		return "I am learning which makes more neuron connections in my brain."
	}

	switch {
	case lvl <= 4:
		lines := []string{
			fmt.Sprintf("%s is learn now!", name),
			fmt.Sprintf("me see word %q... %s brain go spark!", kw1, name),
			fmt.Sprintf("big letters... %s read %s now!", name, strings.ToLower(rep.Topic)),
			fmt.Sprintf("ooh... %q! %s head feel tingly!", kw1, name),
		}
		return lines[rng.Intn(len(lines))]

	case lvl <= 9:
		lines := []string{
			fmt.Sprintf("Me reading %s page! %s know %d words now!", rep.Topic, name, rep.VocabSize),
			fmt.Sprintf("Hey look, %s remember %q and %q!", name, kw1, kw2),
			fmt.Sprintf("Brain getting warmer! %s learning faster now!", name),
			fmt.Sprintf("Ooh new stuff about %s! Me getting smarter!", rep.Topic),
		}
		return lines[rng.Intn(len(lines))]

	case lvl <= 24:
		lines := []string{
			"I'm learning stuff!",
			fmt.Sprintf("I'm learning stuff! Just read about %s and picked up %q!", rep.ArticleTitle, kw1),
			fmt.Sprintf("Whoa, sentences are clicking together now — %q connects to %q!", kw1, kw2),
			fmt.Sprintf("I've got %d words in my vocabulary already. Keep the articles coming!", rep.VocabSize),
		}
		return lines[rng.Intn(len(lines))]

	case lvl <= 49:
		lines := []string{
			fmt.Sprintf("Studying %q helped me link %q with %q.", rep.ArticleTitle, kw1, kw2),
			fmt.Sprintf("I've read %d articles so far, and concepts in %s are starting to make real sense!", b.ArticlesRead, rep.Topic),
			fmt.Sprintf("My prediction loss dropped to %.3f as I mapped new %s patterns.", rep.SmoothedLoss, rep.Topic),
			fmt.Sprintf("Building solid study notes on %s — %d active connections and counting!", rep.Topic, rep.ActiveSynapses),
		}
		return lines[rng.Intn(len(lines))]

	case lvl <= 99:
		lines := []string{
			fmt.Sprintf("Synthesizing %q is reinforcing weights across my %d hidden neurons.", rep.ArticleTitle, b.Net.HiddenNeurons),
			fmt.Sprintf("Cross-referencing %q and %q within %s sharpened my semantic embeddings (loss %.3f).", kw1, kw2, rep.Topic, rep.SmoothedLoss),
			fmt.Sprintf("My memory now spans %d vocabulary tokens and %d active synapses — quiz me in Chat Mode anytime!", rep.VocabSize, rep.ActiveSynapses),
			fmt.Sprintf("Each gradient step over %s refines how I associate concepts across articles.", rep.Topic),
		}
		return lines[rng.Intn(len(lines))]

	case lvl <= 149:
		lines := []string{
			"I am learning which makes more neuron connections in my brain.",
			fmt.Sprintf("I am learning, which makes more neuron connections in my brain — %d active synapses now encode concepts like %q and %q.", rep.ActiveSynapses, kw1, kw2),
			fmt.Sprintf("Integrating %q deepened my neural representations across %d learned tokens (smoothed loss %.3f).", rep.ArticleTitle, rep.VocabSize, rep.SmoothedLoss),
			fmt.Sprintf("Dense synaptic pathways are forming between %s and earlier domains I studied (%d active connections).", rep.Topic, rep.ActiveSynapses),
		}
		return lines[rng.Intn(len(lines))]

	default:
		lines := []string{
			fmt.Sprintf("At Level %d, %d active synaptic connections weave %q and %q into a unified knowledge graph across %d studied articles.",
				lvl, rep.ActiveSynapses, kw1, kw2, b.ArticlesRead),
			fmt.Sprintf("Continuous backpropagation over %q has converged my smoothed cross-entropy loss to %.3f while preserving RAM efficiency.",
				rep.ArticleTitle, rep.SmoothedLoss),
			fmt.Sprintf("I am synthesizing cross-disciplinary insights across %s — my %d-token embedding space is primed for deep study sessions.",
				rep.Topic, rep.VocabSize),
		}
		return lines[rng.Intn(len(lines))]
	}
}

func pickKeyword(kws []string, idx int, fallback string) string {
	if idx >= 0 && idx < len(kws) && strings.TrimSpace(kws[idx]) != "" {
		return kws[idx]
	}
	return fallback
}
