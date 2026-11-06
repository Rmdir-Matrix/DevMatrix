package corpus

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestReaderGrammarFirstThenLocalAndEncyclopedia(t *testing.T) {
	dir := t.TempDir()
	r, err := NewReader(dir)
	if err != nil {
		t.Fatalf("NewReader failed: %v", err)
	}
	r.DisableLiveFetch = true

	if r.GrammarCount() != 10 {
		t.Fatalf("expected 10 foundational grammar lessons, got %d", r.GrammarCount())
	}
	if r.SeedCount() < 20 {
		t.Fatalf("expected >= 20 embedded seed articles, got %d", r.SeedCount())
	}

	seen := make(map[string]bool)

	// 1. Fresh Matrix must learn Grammar Lessons 1..10 in strict order FIRST
	for i := 0; i < r.GrammarCount(); i++ {
		lesson, err := r.NextArticle(context.Background(), "", seen)
		if err != nil {
			t.Fatalf("NextArticle grammar %d failed: %v", i, err)
		}
		if !strings.HasPrefix(lesson.Source, "curriculum:") {
			t.Fatalf("step %d expected curriculum:* lesson first, got %q", i, lesson.Source)
		}
		seen[lesson.Source] = true
	}

	// 2. Once grammar foundation is complete, local study notes get priority
	notePath := filepath.Join(r.NotesDir(), "mitochondria_review.md")
	noteBody := "Mitochondria generate ATP via oxidative phosphorylation across the inner mitochondrial membrane during aerobic cellular respiration."
	if err := os.WriteFile(notePath, []byte(noteBody), 0o644); err != nil {
		t.Fatalf("write note: %v", err)
	}

	localArt, err := r.NextArticle(context.Background(), "", seen)
	if err != nil {
		t.Fatalf("NextArticle local failed: %v", err)
	}
	if localArt.Source != "local:mitochondria_review.md" {
		t.Fatalf("expected local note after grammar curriculum, got source %q", localArt.Source)
	}

	// 3. Topic-targeted study works directly as well
	physArt, err := r.NextArticle(context.Background(), "Physics", seen)
	if err != nil || len(strings.Fields(physArt.Content)) < 40 {
		t.Fatalf("NextArticle Physics failed: %+v err=%v", physArt, err)
	}
}
