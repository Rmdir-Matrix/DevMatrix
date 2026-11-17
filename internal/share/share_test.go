package share

import (
	"path/filepath"
	"testing"

	"github.com/Rmdir-Matrix/DevMatrix/internal/brain"
	"github.com/Rmdir-Matrix/DevMatrix/internal/corpus"
	"github.com/Rmdir-Matrix/DevMatrix/internal/specs"
)

func TestExportAndImportMatrixWithCustomNameAndLowRAMPruning(t *testing.T) {
	highProf, _ := specs.BuildProfile(specs.HardwareInfo{TotalRAMMB: 16384, AvailableRAMMB: 8192, CPUCores: 4}, 1024)
	srcBrain := brain.NewBrain("Matrix", highProf)

	srcBrain.TrainOnArticle(corpus.Article{
		Title:   "Quantum Superposition",
		Topic:   "Physics",
		Source:  "test:quantum",
		Content: "Quantum particles exhibit wave-particle duality and exist in linear superposition until measured.",
	}, 2)

	tmpDir := t.TempDir()
	outFile := filepath.Join(tmpDir, "shared_buddy.matrix")

	savedPath, sz, err := ExportMatrix(srcBrain, "QuantumBuddy", outFile)
	if err != nil {
		t.Fatalf("ExportMatrix failed: %v", err)
	}
	if sz <= 0 {
		t.Fatalf("expected positive exported file size, got %d", sz)
	}

	// Import onto a Featherweight 96 MB RAM profile and verify caps/pruning apply cleanly
	featherProf, _ := specs.BuildProfile(specs.HardwareInfo{TotalRAMMB: 1024, AvailableRAMMB: 512, CPUCores: 1}, 96)
	destDataDir := filepath.Join(tmpDir, "friend_pc_data")
	imported, sum, err := ImportMatrix(savedPath, destDataDir, featherProf)
	if err != nil {
		t.Fatalf("ImportMatrix failed: %v", err)
	}
	if imported.Name != "QuantumBuddy" || sum.ExportedName != "QuantumBuddy" {
		t.Fatalf("expected exported name QuantumBuddy, got brain=%q summary=%q", imported.Name, sum.ExportedName)
	}
	if imported.Net.MaxVocab != featherProf.MaxVocab || imported.MaxReplay != featherProf.MaxPassages {
		t.Fatalf("expected Featherweight caps (%d vocab, %d replay), got (%d vocab, %d replay)",
			featherProf.MaxVocab, featherProf.MaxPassages, imported.Net.MaxVocab, imported.MaxReplay)
	}
	if imported.Level != srcBrain.Level || imported.ArticlesRead != 1 {
		t.Fatalf("imported stats mismatch: lvl=%d articles=%d", imported.Level, imported.ArticlesRead)
	}

	savedList, err := ListSavedBrains(destDataDir, featherProf)
	if err != nil || len(savedList) != 1 {
		t.Fatalf("ListSavedBrains expected 1 brain, got %d (err=%v)", len(savedList), err)
	}
}

func TestExportContainsLicenseAndSourceCard(t *testing.T) {
	prof, _ := specs.BuildProfile(specs.HardwareInfo{TotalRAMMB: 4096, AvailableRAMMB: 2048, CPUCores: 2}, 256)
	b := brain.NewBrain("Study", prof)
	b.RecordSourceCredits(corpus.CreditsForArticle(corpus.Article{Source: "https://en.wikipedia.org/wiki/Computer"}))
	b.RecordSourceCredits(corpus.CreditsForArticle(corpus.Article{Source: "http://arxiv.org/abs/1234.5678"}))
	out := filepath.Join(t.TempDir(), "card.matrix")
	path, _, err := ExportMatrix(b, "Study", out)
	if err != nil {
		t.Fatal(err)
	}
	imported, summary, err := ImportMatrix(path, t.TempDir(), prof)
	if err != nil {
		t.Fatal(err)
	}
	if summary.License != "CC BY-SA 4.0" || !summary.SourcesComplete || len(summary.Sources) != 2 || len(imported.SourceCredits) != 2 {
		t.Fatalf("bad card: %+v", summary)
	}
	if summary.LicenseURL != "https://creativecommons.org/licenses/by-sa/4.0/" {
		t.Fatal(summary.LicenseURL)
	}
	b.SourceHistoryComplete = false
	path, _, err = ExportMatrix(b, "Old", filepath.Join(t.TempDir(), "old.matrix"))
	if err != nil {
		t.Fatal(err)
	}
	_, summary, err = ImportMatrix(path, t.TempDir(), prof)
	if err != nil || summary.SourcesComplete {
		t.Fatalf("legacy history incorrectly complete: %+v %v", summary, err)
	}
}
