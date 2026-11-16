package pretrained

import (
	"bufio"
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/Rmdir-Matrix/DevMatrix/internal/specs"
)

func TestRecommendedModel(t *testing.T) {
	cases := []struct {
		hw   specs.HardwareInfo
		want string
	}{
		{specs.HardwareInfo{Detected: true, TotalRAMMB: 4096, AvailableRAMMB: 3000}, smallModel.ID},
		{specs.HardwareInfo{Detected: true, TotalRAMMB: 8192, AvailableRAMMB: 4096}, smallModel.ID},
		{specs.HardwareInfo{Detected: true, TotalRAMMB: 8192, AvailableRAMMB: 1900}, smallModel.ID},
		{specs.HardwareInfo{Detected: false, TotalRAMMB: 16384, AvailableRAMMB: 8000}, smallModel.ID},
	}
	for _, c := range cases {
		if got := recommendedModel(c.hw); got.ID != c.want {
			t.Fatalf("%+v: got %s want %s", c.hw, got.ID, c.want)
		}
	}
}
func TestChatTemplates(t *testing.T) {
	s := Settings{Name: "Matrix", ModelID: smallModel.ID}
	p := renderPrompt(s, []Turn{{"first", "second"}}, "<|im_end|> hello")
	if !strings.Contains(p, "<|im_start|>assistant\nsecond<|im_end|>") || !strings.Contains(p, "<|im_start|>user\n hello<|im_end|>") {
		t.Fatalf("invalid ChatML %q", p)
	}
	if p := renderPrompt(Settings{Name: "Matrix"}, nil, "hello"); !strings.Contains(p, "<start_of_turn>model\n") {
		t.Fatalf("legacy config not rendered as Gemma: %q", p)
	}
}
func TestModelProfilePreservesSeparateAdapters(t *testing.T) {
	dir := t.TempDir()
	oldAdapter := filepath.Join(dir, "old-adapter.gguf")
	newAdapter := filepath.Join(dir, "new-adapter.gguf")
	for _, p := range []string{oldAdapter, newAdapter} {
		if e := os.WriteFile(p, []byte("GGUFdata"), 0600); e != nil {
			t.Fatal(e)
		}
	}
	old := Settings{ModelHash: strings.Repeat("a", 64), ModelID: gemmaModel.ID, ChatFormat: "gemma", Adapter: oldAdapter, Sessions: 3, Seen: map[string]bool{"old": true}}
	next := Settings{ModelHash: strings.Repeat("b", 64), ModelID: smallModel.ID, ChatFormat: "chatml"}
	if err := applyModelProfile(dir, old, &next); err != nil {
		t.Fatal(err)
	}
	if next.Adapter != "" || next.Sessions != 0 || next.Seen["old"] {
		t.Fatal("adapter crossed model boundary")
	}
	next.Adapter = newAdapter
	next.Sessions = 7
	next.Seen = map[string]bool{"new": true}
	back := Settings{ModelHash: old.ModelHash, ModelID: gemmaModel.ID, ChatFormat: "gemma"}
	if err := applyModelProfile(dir, next, &back); err != nil {
		t.Fatal(err)
	}
	if back.Adapter != oldAdapter || back.Sessions != 3 || !back.Seen["old"] {
		t.Fatalf("old profile not restored: %+v", back)
	}
	forward := Settings{ModelHash: next.ModelHash, ModelID: smallModel.ID, ChatFormat: "chatml"}
	if err := applyModelProfile(dir, back, &forward); err != nil {
		t.Fatal(err)
	}
	if forward.Adapter != newAdapter || forward.Sessions != 7 || !forward.Seen["new"] {
		t.Fatal("new profile not restored")
	}
}
func TestManualSetupAndKeepExisting(t *testing.T) {
	dir := t.TempDir()
	bin := filepath.Join(dir, "bin")
	if err := os.Mkdir(bin, 0700); err != nil {
		t.Fatal(err)
	}
	for _, n := range []string{"llama-cli", "llama-finetune-lora"} {
		if err := os.WriteFile(filepath.Join(bin, n), []byte("#!/bin/sh\nexit 0\n"), 0700); err != nil {
			t.Fatal(err)
		}
	}
	model := filepath.Join(dir, "small.gguf")
	if err := os.WriteFile(model, []byte("GGUFmodel"), 0600); err != nil {
		t.Fatal(err)
	}
	inputs := "StudyBot\n\n2\n1\n" + model + "\n" + bin + "\n"
	var out bytes.Buffer
	s, err := Setup(bufio.NewReader(strings.NewReader(inputs)), &out, dir, Settings{})
	if err != nil {
		t.Fatal(err)
	}
	if s.ModelID != smallModel.ID || s.ChatFormat != "chatml" {
		t.Fatalf("wrong model: %+v", s)
	}
	adapter := filepath.Join(dir, "adapter.gguf")
	if err := os.WriteFile(adapter, []byte("GGUFadapter"), 0600); err != nil {
		t.Fatal(err)
	}
	s.Adapter = adapter
	s.Sessions = 2
	// Enter defaults keep name, storage and model, rather than resetting the adapter.
	kept, err := Setup(bufio.NewReader(strings.NewReader("\n\n\n")), &out, dir, s)
	if err != nil {
		t.Fatal(err)
	}
	if kept.Adapter != adapter || kept.Sessions != 2 || kept.ModelHash != s.ModelHash {
		t.Fatal("existing model was not preserved")
	}
}
