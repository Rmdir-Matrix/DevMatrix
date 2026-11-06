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

func TestFullEndToEndLifecycleLevel1To100ChatAndShare(t *testing.T) {
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
	var trainLog bytes.Buffer

	// Train enough articles to progress from Level 1 -> Level 10 -> Level 100+
	err = trainer.RunLoop(&trainLog, ui.NewTheme(false), b, cr, trainer.Options{
		MaxArticles: 135,
		StepDelay:   0,
		DataDir:     dir,
	}, make(chan struct{}))
	if err != nil {
		t.Fatalf("RunLoop error: %v", err)
	}

	outStr := trainLog.String()
	for _, want := range []string{
		"Matrix is learn now!",
		"I'm learning stuff!",
		"I am learning which makes more neuron connections in my brain.",
	} {
		if !strings.Contains(outStr, want) {
			t.Fatalf("expected training log to contain milestone %q", want)
		}
	}
	if b.Level < 100 {
		t.Fatalf("expected Matrix to reach Level >= 100, got Level %d", b.Level)
	}

	// Verify Chat Mode answers study questions intelligently at Level 100+
	reply := chat.Respond(b, "Explain backpropagation and neural networks")
	if !strings.Contains(strings.ToLower(reply), "neural") || !strings.Contains(strings.ToLower(reply), "synapses") {
		t.Fatalf("expected rich Level 100+ chat synthesis, got:\n%s", reply)
	}

	// Verify exporting with custom name and importing on another user's PC
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
	if imported.Level != b.Level {
		t.Fatalf("expected imported level %d, got %d", b.Level, imported.Level)
	}
}
