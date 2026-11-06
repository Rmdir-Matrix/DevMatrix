package trainer

import (
	"bytes"
	"math/rand"
	"strings"
	"testing"

	"github.com/Rmdir-Matrix/DevMatrix/internal/brain"
	"github.com/Rmdir-Matrix/DevMatrix/internal/corpus"
	"github.com/Rmdir-Matrix/DevMatrix/internal/specs"
	"github.com/Rmdir-Matrix/DevMatrix/internal/ui"
)

func TestMilestoneMessages(t *testing.T) {
	prof, _ := specs.BuildProfile(specs.HardwareInfo{TotalRAMMB: 4096, AvailableRAMMB: 2048, CPUCores: 2}, 512)
	b := brain.NewBrain("Matrix", prof)
	rng := rand.New(rand.NewSource(42))

	rep := brain.TrainStepReport{
		ArticleTitle:   "Neural Networks",
		Topic:          "AI",
		NewLevel:       1,
		VocabSize:      120,
		ActiveSynapses: 4200,
		SampleKeywords: []string{"neuron", "synapse"},
	}

	m1 := GenerateTrainingThought(b, rep, 1, rng)
	if m1 != "Matrix is learn now!" {
		t.Fatalf("Level 1 milestone = %q, want %q", m1, "Matrix is learn now!")
	}

	rep.NewLevel = 10
	m10 := GenerateTrainingThought(b, rep, 10, rng)
	if m10 != "I'm learning stuff!" {
		t.Fatalf("Level 10 milestone = %q, want %q", m10, "I'm learning stuff!")
	}

	rep.NewLevel = 100
	m100 := GenerateTrainingThought(b, rep, 100, rng)
	if m100 != "I am learning which makes more neuron connections in my brain." {
		t.Fatalf("Level 100 milestone = %q, want %q", m100, "I am learning which makes more neuron connections in my brain.")
	}
}

func TestRunLoopProgressesAndEmitsThoughts(t *testing.T) {
	dir := t.TempDir()
	cr, err := corpus.NewReader(dir)
	if err != nil {
		t.Fatalf("NewReader: %v", err)
	}
	cr.DisableLiveFetch = true

	prof, _ := specs.BuildProfile(specs.HardwareInfo{TotalRAMMB: 4096, AvailableRAMMB: 2048, CPUCores: 2}, 512)
	b := brain.NewBrain("Matrix", prof)

	var out bytes.Buffer
	err = RunLoop(&out, ui.NewTheme(false), b, cr, Options{
		MaxArticles: 8,
		StepDelay:   0,
		DataDir:     dir,
	}, make(chan struct{}))
	if err != nil {
		t.Fatalf("RunLoop error: %v", err)
	}

	got := out.String()
	if !strings.Contains(got, "Matrix is learn now!") {
		t.Fatalf("expected Level 1 message in training output, got:\n%s", got)
	}
	if !strings.Contains(got, "I'm learning stuff!") {
		t.Fatalf("expected Level 10 milestone message after 8 articles, got:\n%s", got)
	}
	if b.Level < 10 {
		t.Fatalf("expected level >= 10 after 8 articles, got %d", b.Level)
	}
}
