package chat

import (
	"context"
	"strings"
	"testing"

	"github.com/Rmdir-Matrix/DevMatrix/internal/brain"
	"github.com/Rmdir-Matrix/DevMatrix/internal/corpus"
	"github.com/Rmdir-Matrix/DevMatrix/internal/specs"
)

func TestPureModelOutputAndUnknownDetection(t *testing.T) {
	prof, _ := specs.BuildProfile(specs.HardwareInfo{TotalRAMMB: 4096, AvailableRAMMB: 2048, CPUCores: 2}, 512)
	b := brain.NewBrain("Matrix", prof)

	// 1. Untrained Matrix has ZERO fixed answers — even "hello" and "how are you" are unknown!
	if _, known, _ := Respond(b, "hello"); known {
		t.Fatalf("expected untrained Matrix NOT to know 'hello'")
	}
	if _, known, _ := Respond(b, "how are you"); known {
		t.Fatalf("expected untrained Matrix NOT to know 'how are you'")
	}

	cr, err := corpus.NewReader(t.TempDir())
	if err != nil {
		t.Fatalf("NewReader: %v", err)
	}
	cr.DisableLiveFetch = true

	// 2. Train on Grammar Lesson 1 ONLY: knows "hello", still does NOT know "how are you"!
	lesson1, _ := cr.NextArticle(context.Background(), "", b.SeenSources)
	b.TrainOnArticle(lesson1, 3)

	hello1, knownHello1, _ := Respond(b, "hello")
	if !knownHello1 || strings.TrimSpace(hello1) == "" {
		t.Fatalf("expected Matrix to decode 'hello' after Lesson 1, got known=%v reply=%q", knownHello1, hello1)
	}
	if _, knownHAY1, _ := Respond(b, "how are you"); knownHAY1 {
		t.Fatalf("expected Matrix after Lesson 1 ONLY not to know 'how are you' yet")
	}

	// 3. Complete remaining Grammar Lessons (2..10)
	for i := 1; i < cr.GrammarCount(); i++ {
		lesson, _ := cr.NextArticle(context.Background(), "", b.SeenSources)
		b.TrainOnArticle(lesson, 3)
	}

	hayReply, knownHAY, _ := Respond(b, "how are you")
	if !knownHAY || strings.TrimSpace(hayReply) == "" {
		t.Fatalf("expected Matrix to decode 'how are you' after Grammar Lesson 3+, got known=%v reply=%q", knownHAY, hayReply)
	}

	// Before reading Biology, "photosynthesis" must be reported as unknown!
	if _, knownPhotoBefore, _ := Respond(b, "what is photosynthesis"); knownPhotoBefore {
		t.Fatalf("expected 'photosynthesis' to be unknown before studying Biology article")
	}

	// 4. Train on Photosynthesis article -> now "what is photosynthesis" is decoded from the model!
	art := corpus.Article{
		Title:   "Photosynthesis Overview",
		Topic:   "Biology",
		Source:  "test:photo",
		Content: "Photosynthesis is the biochemical process inside chloroplasts where chlorophyll absorbs sunlight to produce sugars.",
	}
	b.TrainOnArticle(art, 3)

	photoAns, knownPhotoAfter, reason := Respond(b, "what is photosynthesis")
	if !knownPhotoAfter {
		t.Fatalf("expected photosynthesis to be known after training, got reason=%q", reason)
	}
	lowerAns := strings.ToLower(photoAns)
	if !strings.Contains(lowerAns, "photosynthesis") && !strings.Contains(lowerAns, "chloroplast") && !strings.Contains(lowerAns, "sunlight") {
		t.Fatalf("expected model-decoded photosynthesis explanation, got: %s", photoAns)
	}

	// Unseen topic ("sushi") must STILL be honestly reported as unknown!
	if _, knownSushi, _ := Respond(b, "tell me about sushi"); knownSushi {
		t.Fatalf("expected unseen topic 'sushi' to be unknown")
	}
}
