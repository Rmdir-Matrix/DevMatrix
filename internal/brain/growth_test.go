package brain

import (
	"math"
	"math/rand"
	"testing"

	"github.com/Rmdir-Matrix/DevMatrix/internal/corpus"
	"github.com/Rmdir-Matrix/DevMatrix/internal/specs"
)

func growthProfile() specs.ComputeProfile {
	return specs.ComputeProfile{RAMBudgetMB: 384, NumLayers: 2, NumHeads: 4, EmbedDim: 32, HiddenNeurons: 64, MaxVocab: 512, WorkerThreads: 2, BatchSize: 4}
}

func TestGrowthPreservesPredictionsAndTrainsNewWeights(t *testing.T) {
	b := NewBrain("Matrix", growthProfile())
	ids := b.Net.RegisterTokens([]string{"one", "two", "three"}, rand.New(rand.NewSource(3)))
	seq := append([]int{bosID}, ids...)
	_, before, _, _ := b.Net.ForwardFullVocab(seq)
	oldF := b.Net.HiddenNeurons
	oldParams := b.Net.ParameterCount()
	b.TrainingItems = 8
	if msg, ok := b.TryGrow(); !ok {
		t.Fatal(msg)
	}
	if b.Net.HiddenNeurons <= oldF || b.Net.ParameterCount() <= oldParams {
		t.Fatal("growth did not add parameters")
	}
	_, after, _, _ := b.Net.ForwardFullVocab(seq)
	for i := range before {
		if math.Abs(float64(before[i]-after[i])) > 1e-6 {
			t.Fatalf("growth changed prediction %d: %g vs %g", i, before[i], after[i])
		}
	}
	if _, ok := b.TryGrow(); ok {
		t.Fatal("spent the same milestone twice")
	}
	b.Net.TrainMiniBatch([]TrainingSample{{IDs: seq, LossStart: 1}}, 1, 2, 0.003, 4)
	l := b.Net.Layers[0]
	learned := false
	for _, w := range l.W2[oldF*b.Net.EmbedDim:] {
		if w != 0 {
			learned = true
			break
		}
	}
	if !learned {
		t.Fatal("new neurons did not learn")
	}
	raw, err := b.MarshalCompactBinary()
	if err != nil {
		t.Fatal(err)
	}
	restored, err := UnmarshalCompactBinary(raw)
	if err != nil {
		t.Fatal(err)
	}
	restored.ApplyProfileCaps(growthProfile())
	if restored.Net.HiddenNeurons != b.Net.HiddenNeurons || restored.GrowthEvents != 1 || restored.TrainingItems != 8 {
		t.Fatal("growth state did not round trip")
	}
	_, v, _, _ := b.Net.ForwardFullVocab(seq)
	_, v2, _, _ := restored.Net.ForwardFullVocab(seq)
	for i := range v {
		if v[i] != v2[i] {
			t.Fatalf("checkpoint changed weights at %d", i)
		}
	}
}

func TestGrowthRespectsBudgetAndWidthLimit(t *testing.T) {
	p := growthProfile()
	b := NewBrain("Matrix", p)
	b.TrainingItems = 8
	b.GrowthRAMMB = 1
	if _, ok := b.TryGrow(); ok {
		t.Fatal("grew beyond estimated memory budget")
	}
	b.GrowthRAMMB = 384
	b.GrowthMaxWidth = b.Net.HiddenNeurons
	if _, ok := b.TryGrow(); ok {
		t.Fatal("grew beyond width limit")
	}
	if b.GrowthEvents != 0 {
		t.Fatal("denied growth spent milestone")
	}
}

func TestGrowthTriggeredByTrainingAndPersistsInSave(t *testing.T) {
	p := growthProfile()
	p.WorkerThreads = 1
	b := NewBrain("Matrix", p)
	// Empty material cannot create a training milestone.
	b.TrainOnArticle(corpus.Article{}, 1)
	if b.TrainingItems != 0 {
		t.Fatalf("empty article earned growth: %d", b.TrainingItems)
	}
	art := corpus.Article{Title: "Testing", Content: "Stars produce energy inside their cores.", Source: "test:training"}
	initial := b.Net.ParameterCount()
	var report TrainReport
	for i := 0; i < 8; i++ {
		report = b.TrainOnArticle(art, 1)
	}
	if report.GrowthMessage == "" || b.Net.ParameterCount() <= initial {
		t.Fatalf("no growth reported: %+v", report)
	}
	dir := t.TempDir()
	if err := b.Save(dir); err != nil {
		t.Fatal(err)
	}
	loaded, err := LoadFile(BrainFilePath(dir, b.Name), p)
	if err != nil {
		t.Fatal(err)
	}
	if loaded.TrainingItems != 8 || loaded.GrowthEvents != 1 || loaded.Net.ParameterCount() != b.Net.ParameterCount() {
		t.Fatal("save lost growth")
	}
}
