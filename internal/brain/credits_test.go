package brain

import (
	"github.com/Rmdir-Matrix/DevMatrix/internal/corpus"
	"github.com/Rmdir-Matrix/DevMatrix/internal/specs"
	"testing"
)

func TestSourceCreditsPersistAndLegacyFlag(t *testing.T) {
	prof, _ := specs.BuildProfile(specs.HardwareInfo{TotalRAMMB: 4096, AvailableRAMMB: 2000, CPUCores: 2}, 256)
	b := NewBrain("Matrix", prof)
	if !b.SourceHistoryComplete {
		t.Fatal("new brain should track provenance")
	}
	credits := corpus.CreditsForArticle(corpus.Article{Source: "https://en.wikipedia.org/wiki/Computer"})
	b.RecordSourceCredits(credits)
	b.RecordSourceCredits(credits)
	if len(b.SourceCredits) != 1 {
		t.Fatal("credits not deduplicated")
	}
	raw, err := b.MarshalCompactBinary()
	if err != nil {
		t.Fatal(err)
	}
	loaded, err := UnmarshalCompactBinary(raw)
	if err != nil {
		t.Fatal(err)
	}
	if !loaded.SourceHistoryComplete || len(loaded.SourceCredits) != 1 || loaded.SourceCredits[0].URL != credits[0].URL {
		t.Fatalf("lost credits: %+v", loaded.SourceCredits)
	}
	legacy := Brain{SeenSources: map[string]bool{"dictionary:live-batch-1": true, "https://en.wikibooks.org/wiki/Physics": true}}
	c := legacy.SharingCredits()
	if legacy.SourceHistoryComplete || len(c) != 2 {
		t.Fatalf("legacy claims full provenance: %+v", c)
	}
}
func TestSourceCreditsAreBounded(t *testing.T) {
	b := Brain{SourceHistoryComplete: true, SourceCredits: make([]corpus.SourceCredit, maxSourceCredits)}
	b.RecordSourceCredits([]corpus.SourceCredit{{Name: "extra"}})
	if b.SourceHistoryComplete || len(b.SourceCredits) != maxSourceCredits {
		t.Fatal("credit cap or incomplete flag failed")
	}
}
