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

func TestFullEndToEndGrammarFirstModelChatAndShare(t *testing.T) {
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

	// Verify brand-new brain knows nothing yet
	if _, known, _ := chat.Respond(b, "hello"); known {
		t.Fatalf("brand-new brain should not know 'hello' before training")
	}

	var trainLog bytes.Buffer
	err = trainer.RunLoop(&trainLog, ui.NewTheme(false), b, cr, trainer.Options{
		MaxArticles: 135,
		StepDelay:   0,
		DataDir:     dir,
	}, make(chan struct{}))
	if err != nil {
		t.Fatalf("RunLoop error: %v", err)
	}

	if b.DictionariesDone != cr.DictionaryCount() {
		t.Fatalf("expected all %d dictionary volumes completed first, got %d", cr.DictionaryCount(), b.DictionariesDone)
	}
	if b.GrammarDone != cr.GrammarCount() {
		t.Fatalf("expected all %d grammar lessons completed second, got %d", cr.GrammarCount(), b.GrammarDone)
	}
	if b.Level < 100 {
		t.Fatalf("expected Matrix to reach Level >= 100, got Level %d", b.Level)
	}

	// Verify pure model replies on trained topics and honest unknown on unseen topics
	reply, known, reason := chat.Respond(b, "Explain backpropagation and neural networks")
	if !known || strings.TrimSpace(reply) == "" {
		t.Fatalf("expected trained model reply on neural networks, got known=%v reason=%q", known, reason)
	}
	if _, knownUnseen, _ := chat.Respond(b, "How do you cook ramen noodles?"); knownUnseen {
		t.Fatalf("expected unseen topic 'ramen' to return known=false")
	}

	// Verify exporting with custom name and importing on another PC
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
