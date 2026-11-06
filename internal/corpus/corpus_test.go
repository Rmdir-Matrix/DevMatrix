package corpus

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestReaderEmbeddedAndLocalNotes(t *testing.T) {
	dir := t.TempDir()
	r, err := NewReader(dir)
	if err != nil {
		t.Fatalf("NewReader failed: %v", err)
	}
	r.DisableLiveFetch = true

	if r.SeedCount() < 20 {
		t.Fatalf("expected >= 20 embedded seed articles, got %d", r.SeedCount())
	}

	seen := make(map[string]bool)
	art, err := r.NextArticle(context.Background(), "Physics", seen)
	if err != nil {
		t.Fatalf("NextArticle failed: %v", err)
	}
	if art.Title == "" || len(strings.Fields(art.Content)) < 40 {
		t.Fatalf("unexpected short article: %+v", art)
	}

	// Drop a custom local study note and verify it gets prioritized first
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
		t.Fatalf("expected local note priority, got source %q", localArt.Source)
	}
}
