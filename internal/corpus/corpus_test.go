package corpus

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestReaderDictionariesFirstThenGrammarThenArticles(t *testing.T) {
	dir := t.TempDir()
	r, err := NewReader(dir)
	if err != nil {
		t.Fatalf("NewReader failed: %v", err)
	}
	r.DisableLiveFetch = true

	if r.DictionaryCount() != 12 {
		t.Fatalf("expected 12 dictionary volumes, got %d", r.DictionaryCount())
	}
	if r.GrammarCount() != 10 {
		t.Fatalf("expected 10 foundational grammar lessons, got %d", r.GrammarCount())
	}
	if r.SeedCount() < 20 {
		t.Fatalf("expected >= 20 embedded seed articles, got %d", r.SeedCount())
	}

	seen := make(map[string]bool)

	// 1. Fresh Matrix MUST train on all 12 Dictionary Volumes FIRST
	for i := 0; i < r.DictionaryCount(); i++ {
		dictVol, err := r.NextArticle(context.Background(), "", seen)
		if err != nil {
			t.Fatalf("NextArticle dictionary %d failed: %v", i, err)
		}
		if !strings.HasPrefix(dictVol.Source, "dictionary:") {
			t.Fatalf("step %d expected dictionary:* volume first, got %q", i, dictVol.Source)
		}
		seen[dictVol.Source] = true
	}

	// 2. Next, Matrix MUST train on all 10 Grammar Lessons SECOND
	for i := 0; i < r.GrammarCount(); i++ {
		lesson, err := r.NextArticle(context.Background(), "", seen)
		if err != nil {
			t.Fatalf("NextArticle grammar %d failed: %v", i, err)
		}
		if !strings.HasPrefix(lesson.Source, "curriculum:") {
			t.Fatalf("step %d expected curriculum:* lesson second, got %q", i, lesson.Source)
		}
		seen[lesson.Source] = true
	}

	// 3. Once dictionaries + grammar are complete, local study notes & real articles follow
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
		t.Fatalf("expected local note after dictionaries + grammar, got source %q", localArt.Source)
	}

	// 4. Verify FetchDictionaryEntry finds embedded dictionary entries offline
	entry, err := r.FetchDictionaryEntry(context.Background(), "hypothesis")
	if err != nil || !strings.Contains(entry.Content, "hypothesis") {
		t.Fatalf("FetchDictionaryEntry(hypothesis) failed: %+v err=%v", entry, err)
	}
}
