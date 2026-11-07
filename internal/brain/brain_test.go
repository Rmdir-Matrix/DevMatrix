package brain

import (
	"strings"
	"testing"

	"github.com/Rmdir-Matrix/DevMatrix/internal/corpus"
	"github.com/Rmdir-Matrix/DevMatrix/internal/specs"
)

func TestClosedBookGPTTransformerZeroLookupAndBinaryIO(t *testing.T) {
	prof := specs.ComputeProfile{
		TierName:      "Balanced",
		RAMBudgetMB:   512,
		NumLayers:     3,
		NumHeads:      4,
		EmbedDim:      64,
		HiddenNeurons: 256,
		MaxVocab:      4096,
		MaxPassages:   384,
		BatchSize:     12,
		WorkerThreads: 2,
	}
	b := NewBrain("Matrix", prof)

	if got := b.Net.ActiveSynapses(); got != 0 {
		t.Fatalf("expected brand-new untrained GPT brain to have 0 active synapses, got %d", got)
	}

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
			"Glycolysis splits glucose into pyruvate in the cytoplasm before the citric acid cycle runs inside mitochondria.",
	}

	rep1 := b.TrainOnArticle(art, 1)
	var lastRep TrainReport
	for i := 0; i < 4; i++ {
		lastRep = b.TrainOnArticle(art, 2)
	}

	if lastRep.AvgLoss >= rep1.AvgLoss {
		t.Fatalf("expected GPT Transformer cross-entropy loss to decrease: first=%.4f, later=%.4f", rep1.AvgLoss, lastRep.AvgLoss)
	}
	if lastRep.ActiveSynapses <= 0 {
		t.Fatalf("expected active synapses > 0 after AdamW updates, got %d", lastRep.ActiveSynapses)
	}

	// Prove 100% closed-book weight storage: wipe training replay buffers completely,
	// then ask a question and verify the answer STILL comes out of the GPT weights alone!
	b.CoreReplay = nil
	b.RecentReplay = nil
	ans, known, reason := b.GenerateModelReply("explain mitochondria")
	if !known || !strings.Contains(strings.ToLower(ans), "mitochondria") {
		t.Fatalf("expected pure closed-book GPT weights to answer 'explain mitochondria' with zero replay/lookup, got known=%v reason=%q ans=%q",
			known, reason, ans)
	}

	thought := b.GenerateThoughtFromWeights(art.Title, rep1.ArticleExcerpt)
	if strings.TrimSpace(thought) == "" {
		t.Fatalf("expected non-empty neural thought decoded from trained GPT weights")
	}

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

	loaded.Net.PruneToMaxVocab(512)
	if loaded.Net.MaxVocab != 512 || loaded.Net.VocabSize() > 512 {
		t.Fatalf("expected PruneToMaxVocab(512) to cap vocab, got MaxVocab=%d VocabSize=%d",
			loaded.Net.MaxVocab, loaded.Net.VocabSize())
	}
}
