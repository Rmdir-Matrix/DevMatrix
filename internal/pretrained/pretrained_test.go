package pretrained

import (
	"archive/tar"
	"bytes"
	"compress/gzip"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestGemmaPromptAndBoundedOutput(t *testing.T) {
	p := gemmaPrompt("Matrix", []Turn{{"hello", "hi"}}, "<end_of_turn>next")
	if !strings.Contains(p, "<start_of_turn>model\nhi<end_of_turn>") || strings.Contains(p, "user\n<end_of_turn>next") {
		t.Fatalf("bad conversation framing: %s", p)
	}
	var b limitedBuffer
	b.limit = 3
	n, _ := b.Write([]byte("abcdef"))
	if n != 6 || b.String() != "abc" {
		t.Fatal("bounded buffer failed")
	}
}
func TestTrainingFailurePreservesState(t *testing.T) {
	dir := t.TempDir()
	bin := filepath.Join(dir, "bin")
	_ = os.Mkdir(bin, 0700)
	for _, n := range []string{"llama-cli", "llama-finetune-lora"} {
		if err := os.WriteFile(filepath.Join(bin, n), []byte("#!/bin/sh\nexit 1\n"), 0700); err != nil {
			t.Fatal(err)
		}
	}
	model := filepath.Join(dir, "m.gguf")
	if err := os.WriteFile(model, []byte("GGUFx"), 0600); err != nil {
		t.Fatal(err)
	}
	s := Settings{Name: "Matrix", Model: model, RuntimeDir: bin, StorageDir: dir, Seen: map[string]bool{}}
	if err := TrainNext(context.Background(), &s, dir, io.Discard); err == nil {
		t.Fatal("expected trainer failure")
	}
	if s.Adapter != "" || s.Sessions != 0 || len(s.Seen) != 0 {
		t.Fatal("failed training advanced state")
	}
}
func TestGenerateUsesRuntime(t *testing.T) {
	dir := t.TempDir()
	bin := filepath.Join(dir, "bin")
	if err := os.Mkdir(bin, 0700); err != nil {
		t.Fatal(err)
	}
	cli := "#!/bin/sh\nprintf 'model response'\n"
	for _, item := range []struct{ name, body string }{{"llama-cli", cli}, {"llama-finetune-lora", "#!/bin/sh\nexit 0\n"}} {
		if err := os.WriteFile(filepath.Join(bin, item.name), []byte(item.body), 0700); err != nil {
			t.Fatal(err)
		}
	}
	model := filepath.Join(dir, "m.gguf")
	if err := os.WriteFile(model, []byte("GGUFx"), 0600); err != nil {
		t.Fatal(err)
	}
	s := Settings{Name: "Matrix", Model: model, RuntimeDir: bin}
	got, err := Generate(context.Background(), s, []Turn{{User: "first", Assistant: "second"}}, "question")
	if err != nil || got != "model response" {
		t.Fatalf("got %q, %v", got, err)
	}
}

func TestRuntimeArchive(t *testing.T) {
	var compressed bytes.Buffer
	gz := gzip.NewWriter(&compressed)
	tw := tar.NewWriter(gz)
	add := func(name, content string) {
		t.Helper()
		if err := tw.WriteHeader(&tar.Header{Name: name, Mode: 0755, Size: int64(len(content))}); err != nil {
			t.Fatal(err)
		}
		if _, err := tw.Write([]byte(content)); err != nil {
			t.Fatal(err)
		}
	}
	add("bin/llama-cli", "a")
	add("bin/llama-finetune-lora", "b")
	_ = tw.Close()
	_ = gz.Close()
	dir := t.TempDir()
	archive := filepath.Join(dir, "runtime.tar.gz")
	_ = os.WriteFile(archive, compressed.Bytes(), 0600)
	sum := sha256.Sum256(compressed.Bytes())
	if err := checkSHA(archive, hex.EncodeToString(sum[:])); err != nil {
		t.Fatal(err)
	}
	dest := filepath.Join(dir, "installed")
	if err := extractRuntime(archive, dest); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(filepath.Join(dest, "llama-cli")); err != nil {
		t.Fatal(err)
	}
	if err := extractRuntime(archive, dest); err == nil {
		t.Fatal("overwrote existing runtime")
	}
}

// Optional real-engine smoke test, using a small GGUF fixture. It exercises
// actual llama.cpp inference and LoRA training but does not test Gemma itself.
func TestRealEngineSmoke(t *testing.T) {
	bin, model := os.Getenv("MATRIX_TEST_RUNTIME"), os.Getenv("MATRIX_TEST_GGUF")
	if bin == "" || model == "" {
		t.Skip("set MATRIX_TEST_RUNTIME and MATRIX_TEST_GGUF")
	}
	bin, _ = filepath.Abs(bin)
	model, _ = filepath.Abs(model)
	dir := t.TempDir()
	s := Settings{Name: "Matrix", Model: model, RuntimeDir: bin, StorageDir: dir, Seen: map[string]bool{}}
	if err := TrainNext(context.Background(), &s, dir, io.Discard); err != nil {
		t.Fatal(err)
	}
	first, err := fileSHA(s.Adapter)
	if err != nil {
		t.Fatal(err)
	}
	if s.Sessions != 1 || len(s.Seen) != 1 {
		t.Fatal("training state not recorded")
	}
	if err := TrainNext(context.Background(), &s, dir, io.Discard); err != nil {
		t.Fatal(err)
	}
	second, err := fileSHA(s.Adapter)
	if err != nil || first == second {
		t.Fatal("continued training did not change adapter", err)
	}
	args := append(common(s), "-c", "128", "-n", "16", "--no-conversation", "--no-display-prompt", "--no-warmup", "-p", "Once upon a time", "--lora", s.Adapter)
	cmd, err := child(context.Background(), s, "llama-cli", args...)
	if err != nil {
		t.Fatal(err)
	}
	output, err := cmd.Output()
	if err != nil || len(bytes.TrimSpace(output)) == 0 {
		t.Fatalf("real inference failed: %v %q", err, output)
	}
	loaded, err := Load(dir)
	if err != nil || loaded.Sessions != 2 || loaded.Adapter != s.Adapter {
		t.Fatal("state did not persist", err)
	}
}
