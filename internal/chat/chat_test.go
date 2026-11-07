package chat

import (
	"context"
	"strings"
	"testing"

	"github.com/Rmdir-Matrix/DevMatrix/internal/brain"
	"github.com/Rmdir-Matrix/DevMatrix/internal/corpus"
	"github.com/Rmdir-Matrix/DevMatrix/internal/specs"
)

func TestDictionariesFirstThenGrammarAndFullVocabNeuralChat(t *testing.T) {
	prof, _ := specs.BuildProfile(specs.HardwareInfo{TotalRAMMB: 4096, AvailableRAMMB: 2048, CPUCores: 4}, 512)
	b := brain.NewBrain("Matrix", prof)

	// 1. Untrained Matrix has 0 synapses and ZERO fixed answers
	if _, known, _ := Respond(b, "hello"); known {
		t.Fatalf("expected untrained Matrix NOT to know 'hello'")
	}
	if _, known, _ := Respond(b, "define hypothesis"); known {
		t.Fatalf("expected untrained Matrix NOT to know 'define hypothesis'")
	}

	cr, err := corpus.NewReader(t.TempDir())
	if err != nil {
		t.Fatalf("NewReader: %v", err)
	}
	cr.DisableLiveFetch = true

	// 2. Train on all 8 Core Dictionary Chapters FIRST + Grammar Lessons SECOND
	for i := 0; i < cr.DictionaryCount(); i++ {
		dictVol, _ := cr.NextArticle(context.Background(), "", b.SeenSources)
		if !strings.HasPrefix(dictVol.Source, "dictionary:") {
			t.Fatalf("expected dictionary chapter at step %d, got %q", i, dictVol.Source)
		}
		b.TrainOnArticle(dictVol, 2)
	}
	for i := 0; i < cr.GrammarCount(); i++ {
		lesson, _ := cr.NextArticle(context.Background(), "", b.SeenSources)
		b.TrainOnArticle(lesson, 4)
	}

	// Verify conversational reply emerges from neural weights without hardcoded maps
	hayReply, knownHAY, hayReason := Respond(b, "how are you")
	if !knownHAY || strings.TrimSpace(hayReply) == "" {
		t.Fatalf("expected Matrix to decode 'how are you' after grammar curriculum, got known=%v reason=%q reply=%q",
			knownHAY, hayReason, hayReply)
	}

	// Teach/verify specific lexicon word via /define flow
	hypoArt, err := cr.FetchDictionaryEntry(context.Background(), "hypothesis")
	if err != nil {
		t.Fatalf("FetchDictionaryEntry(hypothesis): %v", err)
	}
	b.TrainOnArticle(hypoArt, 4)

	defAns, knownDef, defReason := Respond(b, "define hypothesis")
	if !knownDef || strings.TrimSpace(defAns) == "" {
		t.Fatalf("expected Matrix to know 'hypothesis' after dictionary training, reason=%q", defReason)
	}

	// Specialized unseen word ("chloroplasts") is unknown before reading Biology
	if _, knownChloroBefore, _ := Respond(b, "what are chloroplasts"); knownChloroBefore {
		t.Fatalf("expected 'chloroplasts' to be unknown before studying Biology article")
	}

	// 4. Train on Photosynthesis article -> now "chloroplasts" is decoded from full-vocab neural weights!
	art := corpus.Article{
		Title:   "Photosynthesis Overview",
		Topic:   "Biology",
		Source:  "test:photo",
		Content: "Chloroplasts contain chlorophyll pigments that absorb sunlight during photosynthesis to produce sugars.",
	}
	b.TrainOnArticle(art, 5)

	chloroAns, knownChloroAfter, reason := Respond(b, "what are chloroplasts")
	if !knownChloroAfter {
		t.Fatalf("expected chloroplasts to be known after training, got reason=%q", reason)
	}
	if strings.TrimSpace(chloroAns) == "" {
		t.Fatalf("expected non-empty model-decoded chloroplasts explanation")
	}

	// Unseen topic ("sushi") must STILL be honestly reported as unknown!
	if _, knownSushi, _ := Respond(b, "tell me about sushi"); knownSushi {
		t.Fatalf("expected unseen topic 'sushi' to be unknown")
	}
}
