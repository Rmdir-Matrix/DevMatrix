package corpus

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestReader4800WordLexiconFirstThenGrammarThenArticles(t *testing.T) {
	dir := t.TempDir()
	r, err := NewReader(dir)
	if err != nil {
		t.Fatalf("NewReader failed: %v", err)
	}
	r.DisableLiveFetch = true

	if r.DictionaryWordCount() < 4000 {
		t.Fatalf("expected >= 4000 real English dictionary words, got %d", r.DictionaryWordCount())
	}
	if r.DictionaryCount() != 6 {
		t.Fatalf("expected 6 Phase-1 core dictionary chapters (240 foundational words), got %d", r.DictionaryCount())
	}
	if r.ExtendedDictionaryCount() < 50 {
		t.Fatalf("expected >= 50 extended dictionary chapters, got %d", r.ExtendedDictionaryCount())
	}
	if r.GrammarCount() != 8 {
		t.Fatalf("expected 8 foundational grammar lessons, got %d", r.GrammarCount())
	}
	if r.SeedCount() < 20 {
		t.Fatalf("expected >= 20 embedded seed articles, got %d", r.SeedCount())
	}

	seen := make(map[string]bool)

	// 1. Fresh Matrix MUST train on all 8 Core Dictionary Chapters FIRST
	for i := 0; i < r.DictionaryCount(); i++ {
		dictVol, err := r.NextArticle(context.Background(), "", seen)
		if err != nil {
			t.Fatalf("NextArticle dictionary %d failed: %v", i, err)
		}
		if !strings.HasPrefix(dictVol.Source, "dictionary:core") {
			t.Fatalf("step %d expected dictionary:core* chapter first, got %q", i, dictVol.Source)
		}
		seen[dictVol.Source] = true
	}

	// 2. Next, Matrix MUST train on all 8 Grammar Lessons SECOND (and zero planted THOUGHT: lines)
	for i := 0; i < r.GrammarCount(); i++ {
		lesson, err := r.NextArticle(context.Background(), "", seen)
		if err != nil {
			t.Fatalf("NextArticle grammar %d failed: %v", i, err)
		}
		if !strings.HasPrefix(lesson.Source, "curriculum:") && !strings.HasPrefix(lesson.Source, "grammar:") {
			t.Fatalf("step %d expected grammar/curriculum lesson second, got %q", i, lesson.Source)
		}
		if strings.Contains(lesson.Content, "THOUGHT:") {
			t.Fatalf("grammar lesson %q must not contain planted THOUGHT: strings", lesson.Title)
		}
		seen[lesson.Source] = true
	}

	// 3. Once core dictionaries + grammar are complete, local study notes & real articles follow
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

	// 4. Verify FetchDictionaryEntry finds embedded lexicon entries offline across the 4,800-word index
	for _, word := range []string{"hypothesis", "theory", "analyze", "logic"} {
		entry, err := r.FetchDictionaryEntry(context.Background(), word)
		if err != nil || !strings.Contains(strings.ToLower(entry.Content), word) {
			t.Fatalf("FetchDictionaryEntry(%q) failed: %+v err=%v", word, entry, err)
		}
	}
}
