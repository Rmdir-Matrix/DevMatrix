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

func TestDictionariesFirstThenGrammarThenArticles(t *testing.T) {
	dir := t.TempDir()
	cr, err := corpus.NewReader(dir)
	if err != nil {
		t.Fatalf("NewReader: %v", err)
	}
	cr.DisableLiveFetch = true

	prof, _ := specs.BuildProfile(specs.HardwareInfo{TotalRAMMB: 4096, AvailableRAMMB: 2048, CPUCores: 2}, 512)
	b := brain.NewBrain("Matrix", prof)

	if got := GenerateTrainingThought(b, brain.TrainReport{}); got != "" {
		t.Fatalf("expected empty thought on untrained brain, got %q", got)
	}

	var out bytes.Buffer
	err = RunLoop(&out, ui.NewTheme(false), b, cr, Options{
		MaxArticles: 20, // 8 core dictionaries + 8 grammar lessons + 4 Phase-3 items
		StepDelay:   0,
		DataDir:     dir,
	}, make(chan struct{}))
	if err != nil {
		t.Fatalf("RunLoop error: %v", err)
	}

	got := out.String()
	dictIdx := strings.Index(got, "[DICT]")
	gramIdx := strings.Index(got, "[GRAMMAR]")

	if dictIdx < 0 || gramIdx < 0 {
		t.Fatalf("expected [DICT] and [GRAMMAR] stages in output:\n%s", got)
	}
	if !(dictIdx < gramIdx) {
		t.Fatalf("expected strict order [DICT] (%d) -> [GRAMMAR] (%d)", dictIdx, gramIdx)
	}
}
