package main

import (
	"bytes"
	"path/filepath"
	"strings"
	"testing"

	"github.com/Rmdir-Matrix/DevMatrix/internal/brain"
	"github.com/Rmdir-Matrix/DevMatrix/internal/chat"
	"github.com/Rmdir-Matrix/DevMatrix/internal/corpus"
	"github.com/Rmdir-Matrix/DevMatrix/internal/share"
	"github.com/Rmdir-Matrix/DevMatrix/internal/specs"
	"github.com/Rmdir-Matrix/DevMatrix/internal/trainer"
	"github.com/Rmdir-Matrix/DevMatrix/internal/ui"
)

func TestFullEndToEndDictionariesGrammarNeuralChatAndShare(t *testing.T) {
	dir := t.TempDir()
	hw := specs.DetectHardware()
	prof, _ := specs.BuildProfile(hw, 512)
	specs.ApplyRuntimeLimits(prof)

	cr, err := corpus.NewReader(dir)
	if err != nil {
		t.Fatalf("NewReader: %v", err)
	}
	cr.DisableLiveFetch = true

	b := brain.NewBrain("Matrix", prof)

	// Verify brand-new brain starts at 0 synapses and knows nothing yet
	if b.Net.ActiveSynapses() != 0 {
		t.Fatalf("expected 0 active synapses at birth, got %d", b.Net.ActiveSynapses())
	}
	if _, known, _ := chat.Respond(b, "hello"); known {
		t.Fatalf("brand-new brain should not know 'hello' before training")
	}

	var trainLog bytes.Buffer
	// Train across all 8 Core Dictionaries + all 8 Grammar Lessons + 12 Phase-3 items
	err = trainer.RunLoop(&trainLog, ui.NewTheme(false), b, cr, trainer.Options{
		MaxArticles: 28,
		StepDelay:   0,
		DataDir:     dir,
	}, make(chan struct{}))
	if err != nil {
		t.Fatalf("RunLoop error: %v", err)
	}

	if cr.DictionaryCompleted(b.SeenSources) != cr.DictionaryCount() {
		t.Fatalf("expected all %d core dictionary chapters completed first, got %d",
			cr.DictionaryCount(), cr.DictionaryCompleted(b.SeenSources))
	}
	if cr.GrammarCompleted(b.SeenSources) != cr.GrammarCount() {
		t.Fatalf("expected all %d grammar lessons completed second, got %d",
			cr.GrammarCount(), cr.GrammarCompleted(b.SeenSources))
	}
	if b.Net.ActiveSynapses() <= 0 {
		t.Fatalf("expected active synapses > 0 after multi-core AdamW training")
	}

	// Teach neural networks article and verify pure full-vocab neural reply
	b.TrainOnArticle(corpus.Article{
		Title:   "Backpropagation and Neural Networks",
		Topic:   "Computer Science",
		Source:  "test:nn",
		Content: "Backpropagation computes gradients across neural network layers so gradient descent reduces prediction loss.",
	}, 4)

	reply, known, reason := chat.Respond(b, "Explain backpropagation and neural networks")
	if !known || strings.TrimSpace(reply) == "" {
		t.Fatalf("expected trained model reply on neural networks, got known=%v reason=%q", known, reason)
	}
	if _, knownUnseen, _ := chat.Respond(b, "How do you cook ramen noodles?"); knownUnseen {
		t.Fatalf("expected unseen topic 'ramen' to return known=false")
	}

	// Verify exporting compact binary .matrix with custom name and importing on another PC
	sharePath := filepath.Join(dir, "legend.matrix")
	savedFile, sz, err := share.ExportMatrix(b, "StudyLegendMatrix", sharePath)
	if err != nil || sz == 0 {
		t.Fatalf("ExportMatrix failed: sz=%d err=%v", sz, err)
	}

	friendDir := filepath.Join(dir, "friend_data")
	imported, summary, err := share.ImportMatrix(savedFile, friendDir, prof)
	if err != nil {
		t.Fatalf("ImportMatrix failed: %v", err)
	}
	if imported.Name != "StudyLegendMatrix" || summary.ExportedName != "StudyLegendMatrix" {
		t.Fatalf("expected StudyLegendMatrix, got %q", imported.Name)
	}
}
