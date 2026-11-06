package chat

import (
	"context"
	"strings"
	"testing"

	"github.com/Rmdir-Matrix/DevMatrix/internal/brain"
	"github.com/Rmdir-Matrix/DevMatrix/internal/corpus"
	"github.com/Rmdir-Matrix/DevMatrix/internal/specs"
)

func TestDictionariesFirstThenGrammarAndPureModelChat(t *testing.T) {
	prof, _ := specs.BuildProfile(specs.HardwareInfo{TotalRAMMB: 4096, AvailableRAMMB: 2048, CPUCores: 2}, 512)
	b := brain.NewBrain("Matrix", prof)

	// 1. Untrained Matrix has ZERO fixed answers
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

	// 2. Train on all 12 Dictionary Volumes FIRST
	for i := 0; i < cr.DictionaryCount(); i++ {
		dictVol, _ := cr.NextArticle(context.Background(), "", b.SeenSources)
		if !strings.HasPrefix(dictVol.Source, "dictionary:") {
			t.Fatalf("expected dictionary volume at step %d, got %q", i, dictVol.Source)
		}
		b.TrainOnArticle(dictVol, 3)
	}

	defAns, knownDef, defReason := Respond(b, "define hypothesis")
	if !knownDef {
		t.Fatalf("expected Matrix to know 'hypothesis' after dictionary training, reason=%q", defReason)
	}
	if !strings.Contains(strings.ToLower(defAns), "hypothesis") || !strings.Contains(strings.ToLower(defAns), "noun") {
		t.Fatalf("expected dictionary definition decoded from model, got: %s", defAns)
	}

	// 3. Complete Grammar Lessons (1..10) SECOND
	for i := 0; i < cr.GrammarCount(); i++ {
		lesson, _ := cr.NextArticle(context.Background(), "", b.SeenSources)
		b.TrainOnArticle(lesson, 3)
	}

	hayReply, knownHAY, _ := Respond(b, "how are you")
	if !knownHAY || strings.TrimSpace(hayReply) == "" {
		t.Fatalf("expected Matrix to decode 'how are you' after grammar curriculum, got known=%v reply=%q", knownHAY, hayReply)
	}

	// Specialized article word ("chloroplasts") is still unknown before reading Biology
	if _, knownChloroBefore, _ := Respond(b, "what are chloroplasts"); knownChloroBefore {
		t.Fatalf("expected 'chloroplasts' to be unknown before studying Biology article")
	}

	// 4. Train on Photosynthesis article -> now "chloroplasts" is decoded from the model!
	art := corpus.Article{
		Title:   "Photosynthesis Overview",
		Topic:   "Biology",
		Source:  "test:photo",
		Content: "Chloroplasts contain chlorophyll pigments that absorb sunlight during photosynthesis to produce sugars.",
	}
	b.TrainOnArticle(art, 3)

	chloroAns, knownChloroAfter, reason := Respond(b, "what are chloroplasts")
	if !knownChloroAfter {
		t.Fatalf("expected chloroplasts to be known after training, got reason=%q", reason)
	}
	if !strings.Contains(strings.ToLower(chloroAns), "chloroplast") && !strings.Contains(strings.ToLower(chloroAns), "chlorophyll") {
		t.Fatalf("expected model-decoded chloroplasts explanation, got: %s", chloroAns)
	}

	// Unseen topic ("sushi") must STILL be honestly reported as unknown!
	if _, knownSushi, _ := Respond(b, "tell me about sushi"); knownSushi {
		t.Fatalf("expected unseen topic 'sushi' to be unknown")
	}
}
