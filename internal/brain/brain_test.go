package brain

import (
	"testing"

	"github.com/Rmdir-Matrix/DevMatrix/internal/corpus"
	"github.com/Rmdir-Matrix/DevMatrix/internal/specs"
)

func TestNeuralBackpropDecreasesLossAndBuildsSynapses(t *testing.T) {
	prof := specs.ComputeProfile{
		TierName:      "Balanced",
		RAMBudgetMB:   512,
		EmbedDim:      32,
		HiddenNeurons: 96,
		MaxVocab:      4096,
		MaxPassages:   500,
	}
	b := NewBrain("Matrix", prof)
	initialSynapses := b.Net.ActiveSynapses()

	art := corpus.Article{
		Title:  "Cellular Respiration and ATP",
		Topic:  "Biology",
		Source: "test:atp",
		Content: "Mitochondria generate adenosine triphosphate through oxidative phosphorylation during cellular respiration. " +
			"Glycolysis splits glucose into pyruvate in the cytoplasm before the citric acid cycle runs inside mitochondria. " +
			"The electron transport chain pumps protons across the inner membrane to power ATP synthase.",
	}

	rep1 := b.TrainOnArticle(art, 1)
	var lastRep TrainStepReport
	for i := 0; i < 6; i++ {
		lastRep = b.TrainOnArticle(art, 2)
	}

	if lastRep.AvgLoss >= rep1.AvgLoss {
		t.Fatalf("expected neural cross-entropy loss to decrease with training: first=%.4f, later=%.4f", rep1.AvgLoss, lastRep.AvgLoss)
	}
	if lastRep.ActiveSynapses <= initialSynapses {
		t.Fatalf("expected active synapses to grow: initial=%d, after=%d", initialSynapses, lastRep.ActiveSynapses)
	}
	if b.Level <= 1 {
		t.Fatalf("expected Matrix level to advance above 1, got %d", b.Level)
	}

	hits := b.SearchPassages("How do mitochondria produce ATP?", 2)
	if len(hits) == 0 {
		t.Fatalf("expected semantic passage search to find learned ATP facts")
	}

	// Test persistence round-trip
	dir := t.TempDir()
	if err := b.Save(dir); err != nil {
		t.Fatalf("Save error: %v", err)
	}
	loaded, existed, err := LoadOrCreate(dir, "Matrix", prof)
	if err != nil || !existed {
		t.Fatalf("LoadOrCreate failed: existed=%v err=%v", existed, err)
	}
	if loaded.Level != b.Level || loaded.Net.VocabSize() != b.Net.VocabSize() {
		t.Fatalf("loaded brain mismatch: got lvl=%d vocab=%d, want lvl=%d vocab=%d",
			loaded.Level, loaded.Net.VocabSize(), b.Level, b.Net.VocabSize())
	}
}
