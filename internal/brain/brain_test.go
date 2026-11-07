package brain

import (
	"strings"
	"testing"

	"github.com/Rmdir-Matrix/DevMatrix/internal/corpus"
	"github.com/Rmdir-Matrix/DevMatrix/internal/specs"
)

func TestCausalAttentionGRUBackpropZeroInitSynapsesSubwordsAndBinaryIO(t *testing.T) {
	prof := specs.ComputeProfile{
		TierName:      "Balanced",
		RAMBudgetMB:   512,
		EmbedDim:      32,
		HiddenNeurons: 96,
		MaxVocab:      4096,
		MaxPassages:   500,
		BatchSize:     16,
		WorkerThreads: 2,
	}
	b := NewBrain("Matrix", prof)

	// Fix 5: Brand-new brain MUST start at 0 active synapses (not Gaussian noise!)
	if got := b.Net.ActiveSynapses(); got != 0 {
		t.Fatalf("expected brand-new untrained brain to have 0 active synapses, got %d", got)
	}

	// Fix 4: Verify subword morpheme decomposition & clean detokenization
	sw := SubwordTokenize("Scientists are learning connections carefully.")
	joined := DetokenizeSubwords(sw, "Matrix")
	if !strings.Contains(joined, "learning") || !strings.Contains(joined, "carefully") {
		t.Fatalf("expected subwords %v to detokenize cleanly, got %q", sw, joined)
	}

	art := corpus.Article{
		Title:  "Cellular Respiration and ATP",
		Topic:  "Biology",
		Source: "test:atp",
		Content: "Mitochondria generate adenosine triphosphate through oxidative phosphorylation during cellular respiration. " +
			"Glycolysis splits glucose into pyruvate in the cytoplasm before the citric acid cycle runs inside mitochondria. " +
			"The electron transport chain pumps protons across the inner membrane to power ATP synthase.",
	}

	rep1 := b.TrainOnArticle(art, 1)
	var lastRep TrainReport
	for i := 0; i < 6; i++ {
		lastRep = b.TrainOnArticle(art, 2)
	}

	if lastRep.AvgLoss >= rep1.AvgLoss {
		t.Fatalf("expected Causal Attention + GRU cross-entropy loss to decrease: first=%.4f, later=%.4f", rep1.AvgLoss, lastRep.AvgLoss)
	}
	if lastRep.ActiveSynapses <= 0 {
		t.Fatalf("expected active synapses to grow above 0 after AdamW updates, got %d", lastRep.ActiveSynapses)
	}
	if b.Level <= 1 {
		t.Fatalf("expected Matrix level to advance above 1, got %d", b.Level)
	}

	// Fix 4: Subword inflection generalization ("pumping" -> "pump" + "##ing")
	// Even if only "pumps" ("pump" + "##s") and "...ing" ("during" or another "-ing" word) were seen
	b.TrainOnArticle(corpus.Article{
		Title:   "Learning Morphology",
		Topic:   "Language",
		Source:  "test:morph",
		Content: "Students are studying and learning how enzymes pump ions.",
	}, 2)
	_, unknown := EncodeWithVocab("pumping", b.Name, b.Net.KnowsToken)
	if len(unknown) != 0 {
		t.Fatalf("expected subword tokenizer to decompose inflected 'pumping' into known root+suffix, got unknown=%v", unknown)
	}

	hits := b.SearchPassages("How do mitochondria produce ATP?", 2)
	if len(hits) == 0 {
		t.Fatalf("expected semantic passage search to find learned ATP facts")
	}

	// Fix 2: Emergent training thought comes purely from neural weights
	thought := b.GenerateThoughtFromWeights(art.Title, rep1.ArticleExcerpt)
	if strings.TrimSpace(thought) == "" {
		t.Fatalf("expected non-empty neural thought decoded from trained weights")
	}

	// Fix 6: Compact binary persistence round-trip & low-RAM vocabulary pruning
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

	// Verify PruneToMaxVocab shrinks vocabulary tensors safely when capped
	loaded.Net.PruneToMaxVocab(512)
	if loaded.Net.MaxVocab != 512 || loaded.Net.VocabSize() > 512 {
		t.Fatalf("expected PruneToMaxVocab(512) to cap vocab, got MaxVocab=%d VocabSize=%d",
			loaded.Net.MaxVocab, loaded.Net.VocabSize())
	}
}
