package trainer

import (
	"bytes"
	"strings"
	"testing"

	"github.com/Rmdir-Matrix/DevMatrix/internal/brain"
	"github.com/Rmdir-Matrix/DevMatrix/internal/corpus"
	"github.com/Rmdir-Matrix/DevMatrix/internal/specs"
	"github.com/Rmdir-Matrix/DevMatrix/internal/ui"
)

func TestModelGeneratedThoughtsEvolveFromGrammarToArticles(t *testing.T) {
	dir := t.TempDir()
	cr, err := corpus.NewReader(dir)
	if err != nil {
		t.Fatalf("NewReader: %v", err)
	}
	cr.DisableLiveFetch = true

	prof, _ := specs.BuildProfile(specs.HardwareInfo{TotalRAMMB: 4096, AvailableRAMMB: 2048, CPUCores: 2}, 512)
	b := brain.NewBrain("Matrix", prof)

	// Untrained model must produce empty thought (no hardcoded strings!)
	if got := GenerateTrainingThought(b, brain.TrainStepReport{}); got != "" {
		t.Fatalf("expected empty thought on untrained brain, got %q", got)
	}

	var out bytes.Buffer
	err = RunLoop(&out, ui.NewTheme(false), b, cr, Options{
		MaxArticles: 15,
		StepDelay:   0,
		DataDir:     dir,
	}, make(chan struct{}))
	if err != nil {
		t.Fatalf("RunLoop error: %v", err)
	}

	got := out.String()
	if !strings.Contains(got, "[GRAMMAR]") {
		t.Fatalf("expected [GRAMMAR] curriculum stage first, got:\n%s", got)
	}
	if !strings.Contains(got, "[ARTICLE]") {
		t.Fatalf("expected transition to [ARTICLE] after 10 grammar lessons, got:\n%s", got)
	}
	if !strings.Contains(got, "💬 [Matrix • Level 1]:") {
		t.Fatalf("expected model-decoded Level 1 thought in output, got:\n%s", got)
	}
}
