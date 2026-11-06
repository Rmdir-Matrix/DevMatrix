package chat

import (
	"strings"
	"testing"

	"github.com/Rmdir-Matrix/DevMatrix/internal/brain"
	"github.com/Rmdir-Matrix/DevMatrix/internal/corpus"
	"github.com/Rmdir-Matrix/DevMatrix/internal/specs"
)

func TestChatLevelAwareResponsesAndDirectTeaching(t *testing.T) {
	prof, _ := specs.BuildProfile(specs.HardwareInfo{TotalRAMMB: 4096, AvailableRAMMB: 2048, CPUCores: 2}, 512)
	b := brain.NewBrain("Matrix", prof)

	// Untrained Level 1 greeting should reflect baby-AI level
	greet1 := OpeningGreeting(b)
	if !strings.Contains(strings.ToLower(greet1), "level 1") {
		t.Fatalf("expected Level 1 greeting, got: %s", greet1)
	}

	// Teach Matrix a fact directly in chat
	ack := Respond(b, "remember: Photosynthesis occurs inside chloroplasts using chlorophyll pigments to capture sunlight.")
	if !strings.Contains(ack, "XP") && !strings.Contains(ack, "synapses") && !strings.Contains(ack, "connections") {
		t.Fatalf("unexpected teaching ack: %s", ack)
	}

	// Train further to reach higher level and ask about photosynthesis
	art := corpus.Article{
		Title:   "Photosynthesis Overview",
		Topic:   "Biology",
		Source:  "test:photo",
		Content: "Photosynthesis occurs inside chloroplasts where chlorophyll absorbs solar energy to fix carbon dioxide during the Calvin cycle.",
	}
	for i := 0; i < 6; i++ {
		b.TrainOnArticle(art, 2)
	}

	ans := Respond(b, "Where does photosynthesis occur?")
	if !strings.Contains(strings.ToLower(ans), "chloroplast") {
		t.Fatalf("expected answer to mention chloroplasts, got: %s", ans)
	}
}
