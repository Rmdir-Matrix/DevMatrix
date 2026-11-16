// Package pretrained runs Gemma locally through the CPU-only QVAC llama.cpp
// tools. The base weights remain frozen; training writes a separate LoRA.
package pretrained

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"time"

	"github.com/Rmdir-Matrix/DevMatrix/internal/corpus"
	"github.com/Rmdir-Matrix/DevMatrix/internal/specs"
)

const modelFile = "gemma-3-270m-it-Q4_K_M.gguf"
const defaultModelURL = "https://huggingface.co/unsloth/gemma-3-270m-it-GGUF/resolve/37cbb6e/gemma-3-270m-it-Q4_K_M.gguf"
const modelSHA = "b1baabd6b729e4041822220d3e648e00d99cac5df86b10dffb77bcccf0688e39"
const runtimeURL = "https://github.com/tetherto/qvac-fabric-llm.cpp/releases/download/b7349/llama-b7349-bin-ubuntu-x64.tar.gz"
const runtimeSHA = "f9490b999f5f49535dea67d7fe5c839b76b011a93703c53226f8330f4b53eb23"

type Settings struct {
	Name       string          `json:"name"`
	Model      string          `json:"model"`
	RuntimeDir string          `json:"runtime_dir"`
	StorageDir string          `json:"storage_dir"`
	ModelHash  string          `json:"model_hash"`
	Adapter    string          `json:"adapter"`
	Seen       map[string]bool `json:"seen"`
	Sessions   int             `json:"sessions"`
}

func settingsPath(dir string) string { return filepath.Join(dir, "pretrained.json") }
func Load(dir string) (Settings, error) {
	var s Settings
	b, e := os.ReadFile(settingsPath(dir))
	if e != nil {
		return s, e
	}
	e = json.Unmarshal(b, &s)
	if s.Seen == nil {
		s.Seen = map[string]bool{}
	}
	return s, e
}
func Save(dir string, s Settings) error {
	b, e := json.MarshalIndent(s, "", "  ")
	if e != nil {
		return e
	}
	p := settingsPath(dir)
	if e = os.WriteFile(p+".tmp", b, 0600); e != nil {
		return e
	}
	return os.Rename(p+".tmp", p)
}
func prompt(r *bufio.Reader, w io.Writer, label string) (string, error) {
	fmt.Fprint(w, label)
	s, e := r.ReadString('\n')
	if e != nil && len(s) == 0 {
		return "", e
	}
	return strings.TrimSpace(s), nil
}
func ggufFile(p string) error {
	f, e := os.Open(p)
	if e != nil {
		return e
	}
	defer f.Close()
	var b [4]byte
	if _, e = io.ReadFull(f, b[:]); e != nil {
		return e
	}
	if string(b[:]) != "GGUF" {
		return fmt.Errorf("%s is not a GGUF file", p)
	}
	return nil
}
func Validate(s Settings) error {
	if strings.TrimSpace(s.Name) == "" {
		return fmt.Errorf("missing Matrix name")
	}
	if err := ggufFile(s.Model); err != nil {
		return fmt.Errorf("model: %w", err)
	}
	for _, file := range []string{"llama-cli", "llama-finetune-lora"} {
		info, err := os.Stat(filepath.Join(s.RuntimeDir, file))
		if err != nil {
			return fmt.Errorf("missing runtime %s: %w", file, err)
		}
		if !info.Mode().IsRegular() || info.Mode()&0111 == 0 {
			return fmt.Errorf("runtime %s is not executable", file)
		}
	}
	if s.Adapter != "" {
		if err := ggufFile(s.Adapter); err != nil {
			return fmt.Errorf("adapter: %w", err)
		}
	}
	return nil
}

// Setup lets users opt in to pinned downloads, or provide already downloaded files.
func Setup(r *bufio.Reader, w io.Writer, dir string, old Settings) (Settings, error) {
	if runtime.GOOS != "linux" || runtime.GOARCH != "amd64" {
		return old, fmt.Errorf("pretrained setup currently supports Linux x86-64 only")
	}
	name, e := prompt(r, w, "  Study buddy name [Matrix]: ")
	if e != nil {
		return old, e
	}
	if name == "" {
		name = "Matrix"
	}
	if len(name) > 64 {
		name = name[:64]
	}
	storage, e := prompt(r, w, "  SSD storage folder ["+dir+"]: ")
	if e != nil {
		return old, e
	}
	if storage == "" {
		storage = dir
	}
	storage, e = filepath.Abs(storage)
	if e != nil {
		return old, e
	}
	if e = os.MkdirAll(storage, 0700); e != nil {
		return old, e
	}
	fmt.Fprintln(w, "\n  Setup stores the model, runtime and trained adapters under:", storage)
	fmt.Fprintln(w, "  Gemma 3 270M IT Q4 (~253 MB) is subject to Google's Gemma terms:")
	fmt.Fprintln(w, "  https://ai.google.dev/gemma/terms")
	fmt.Fprintln(w, "  QVAC llama.cpp CPU tools (~36 MB) are downloaded separately. Internet required.")
	fmt.Fprintln(w, "  [1] Download model and runtime  [2] Use existing files  [Q] Cancel")
	choice, e := prompt(r, w, "  Choice [1]: ")
	if e != nil {
		return old, e
	}
	s := Settings{Name: name, StorageDir: storage, Seen: map[string]bool{}}
	switch strings.ToLower(choice) {
	case "", "1":
		agree, e := prompt(r, w, "  Accept the Gemma terms and download? [y/N]: ")
		if e != nil {
			return old, e
		}
		if !strings.EqualFold(agree, "y") && !strings.EqualFold(agree, "yes") {
			return old, fmt.Errorf("setup cancelled; no model downloaded")
		}
		s.Model = filepath.Join(storage, "models", modelFile)
		s.RuntimeDir = filepath.Join(storage, "runtime-b7349")
		if e = os.MkdirAll(filepath.Dir(s.Model), 0700); e != nil {
			return old, e
		}
		fmt.Fprintln(w, "  Downloading and verifying the model (this may take a while)...")
		if e = downloadVerified(context.Background(), defaultModelURL, s.Model, modelSHA, 400<<20); e != nil {
			return old, fmt.Errorf("model download: %w", e)
		}
		if _, statErr := os.Stat(s.RuntimeDir); os.IsNotExist(statErr) {
			fmt.Fprintln(w, "  Downloading and verifying the CPU runtime...")
			archive := filepath.Join(storage, "runtime-b7349.tar.gz")
			if e = downloadVerified(context.Background(), runtimeURL, archive, runtimeSHA, 80<<20); e != nil {
				return old, fmt.Errorf("runtime download: %w", e)
			}
			defer os.Remove(archive)
			if e = extractRuntime(archive, s.RuntimeDir); e != nil {
				return old, fmt.Errorf("runtime install: %w", e)
			}
		} else if statErr != nil {
			return old, statErr
		} else {
			fmt.Fprintln(w, "  Reusing existing CPU runtime.")
		}
	case "2":
		s.Model, e = prompt(r, w, "  Path to Gemma 3 270M IT GGUF: ")
		if e != nil {
			return old, e
		}
		s.Model, e = filepath.Abs(s.Model)
		if e != nil {
			return old, e
		}
		s.RuntimeDir, e = prompt(r, w, "  Directory containing llama-cli and llama-finetune-lora: ")
		if e != nil {
			return old, e
		}
		s.RuntimeDir, e = filepath.Abs(s.RuntimeDir)
		if e != nil {
			return old, e
		}
	default:
		return old, fmt.Errorf("setup cancelled")
	}
	if err := ggufFile(s.Model); err != nil {
		return old, err
	}
	s.ModelHash, e = fileSHA(s.Model)
	if e != nil {
		return old, e
	}
	if old.ModelHash != "" && old.ModelHash == s.ModelHash && old.Adapter != "" {
		s.Adapter = old.Adapter
		s.Seen = old.Seen
		s.Sessions = old.Sessions
	}
	if e = Validate(s); e != nil {
		return old, e
	}
	if e = Save(dir, s); e != nil {
		return old, e
	}
	return s, nil
}

// child uses a strict Linux address-space limit in addition to the free-RAM
// check. It is not a guarantee about total system RAM or other processes.
func child(ctx context.Context, s Settings, tool string, args ...string) (*exec.Cmd, error) {
	if runtime.GOOS != "linux" {
		return nil, fmt.Errorf("pretrained runtime requires Linux")
	}
	limit, e := exec.LookPath("prlimit")
	if e != nil {
		return nil, fmt.Errorf("install util-linux (prlimit) to enforce a runtime memory limit: %w", e)
	}
	hw := specs.DetectHardware()
	minAvail := 1024
	if tool == "llama-finetune-lora" {
		minAvail = 1900
	}
	if hw.Detected && hw.AvailableRAMMB < minAvail {
		return nil, fmt.Errorf("not enough available RAM (%d MB); close programs to free at least %d MB", hw.AvailableRAMMB, minAvail)
	}
	mb := 1536
	if tool == "llama-finetune-lora" {
		mb = 1792
	}
	// Restrict to no more than half total RAM (leaves room for OS and Matrix).
	if hw.Detected && hw.TotalRAMMB/2 < mb {
		mb = hw.TotalRAMMB / 2
	}
	if mb < 512 {
		return nil, fmt.Errorf("not enough total RAM for local inference")
	}
	binary := filepath.Join(s.RuntimeDir, tool)
	all := append([]string{fmt.Sprintf("--as=%d", int64(mb)*1024*1024), "--", binary}, args...)
	cmd := exec.CommandContext(ctx, limit, all...)
	cmd.Env = append(os.Environ(), "LD_LIBRARY_PATH="+s.RuntimeDir)
	return cmd, nil
}
func common(s Settings) []string {
	return []string{"-m", s.Model, "-ngl", "0", "-t", "2", "-tb", "2", "--no-warmup"}
}

type Turn struct{ User, Assistant string }

func gemmaPrompt(name string, history []Turn, input string) string {
	clean := func(s string) string {
		s = strings.ReplaceAll(s, "<start_of_turn>", "")
		s = strings.ReplaceAll(s, "<end_of_turn>", "")
		return s
	}
	var p strings.Builder
	p.WriteString("<start_of_turn>user\nYou are " + clean(name) + ", a helpful study buddy. Give clear answers and admit uncertainty.\n")
	for _, t := range history {
		p.WriteString(clean(t.User))
		p.WriteString("<end_of_turn>\n<start_of_turn>model\n")
		p.WriteString(clean(t.Assistant))
		p.WriteString("<end_of_turn>\n<start_of_turn>user\n")
	}
	p.WriteString(clean(input))
	p.WriteString("<end_of_turn>\n<start_of_turn>model\n")
	return p.String()
}
func Generate(ctx context.Context, s Settings, history []Turn, input string) (string, error) {
	if err := Validate(s); err != nil {
		return "", err
	}
	if len(input) > 2000 {
		return "", fmt.Errorf("message is too long (max 2000 bytes)")
	}
	if len(history) > 2 {
		history = history[len(history)-2:]
	}
	args := append(common(s), "-c", "1024", "-b", "64", "-ub", "32", "-n", "160", "--no-conversation", "--no-display-prompt", "--simple-io", "-p", gemmaPrompt(s.Name, history, input))
	if s.Adapter != "" {
		args = append(args, "--lora", s.Adapter)
	}
	cmd, err := child(ctx, s, "llama-cli", args...)
	if err != nil {
		return "", err
	}
	var stdout, stderr limitedBuffer
	stdout.limit = 64 << 10
	stderr.limit = 64 << 10
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	if err = cmd.Run(); err != nil {
		return "", fmt.Errorf("model failed: %w (%s)", err, strings.TrimSpace(stderr.String()))
	}
	output := strings.TrimSpace(stdout.String())
	if output == "" {
		return "", fmt.Errorf("model returned no text: %s", strings.TrimSpace(stderr.String()))
	}
	return output, nil
}

// TrainNext performs a real CPU LoRA update on one article. No retrieval is
// added to chat. Only publish the new adapter and advance curriculum on success.
func TrainNext(ctx context.Context, s *Settings, dir string, out io.Writer) error {
	if err := Validate(*s); err != nil {
		return err
	}
	cr, err := corpus.NewReader(dir)
	if err != nil {
		return err
	}
	readCtx, cancel := context.WithTimeout(ctx, 30*time.Second)
	defer cancel()
	art, err := cr.NextArticle(readCtx, "", s.Seen)
	if err != nil {
		return err
	}
	content := strings.TrimSpace(art.Content)
	if len(content) < 512 {
		return fmt.Errorf("training text is too short; no update made")
	}
	if len(content) > 24000 {
		content = content[:24000]
	}
	storage := s.StorageDir
	if storage == "" {
		storage = dir
	}
	if err := os.MkdirAll(storage, 0700); err != nil {
		return err
	}
	data, err := os.CreateTemp(storage, "training-*.txt")
	if err != nil {
		return err
	}
	defer os.Remove(data.Name())
	if _, err = data.WriteString(content + "\n"); err != nil {
		data.Close()
		return err
	}
	if err = data.Close(); err != nil {
		return err
	}
	nextFile, err := os.CreateTemp(storage, "adapter-next-*.gguf")
	if err != nil {
		return err
	}
	next := nextFile.Name()
	if err = nextFile.Close(); err != nil {
		return err
	}
	_ = os.Remove(next)
	defer os.Remove(next)
	args := append(common(*s), "-f", data.Name(), "-c", "128", "-b", "32", "-ub", "32", "-fa", "off", "--lora-rank", "4", "--lora-alpha", "8", "--lora-modules", "attn_q,attn_v", "--num-epochs", "1", "--learning-rate", "0.00001", "--checkpoint-save-steps", "0", "--output-adapter", next)
	if s.Adapter != "" {
		args = append(args, "--lora", s.Adapter)
	}
	cmd, err := child(ctx, *s, "llama-finetune-lora", args...)
	if err != nil {
		return err
	}
	fmt.Fprintf(out, "  Fine-tuning on %q (%s). CPU training can be slow.\n", art.Title, art.Source)
	cmd.Stdout = out
	cmd.Stderr = out
	if err = cmd.Run(); err != nil {
		return fmt.Errorf("training failed; previous adapter preserved: %w", err)
	}
	if err = ggufFile(next); err != nil {
		return fmt.Errorf("trainer did not produce a valid adapter: %w", err)
	}
	// Distinct immutable generations avoid corrupting the active adapter if a save fails.
	final := strings.TrimSuffix(next, ".gguf") + "-saved.gguf"
	if err = os.Rename(next, final); err != nil {
		return err
	}
	updated := *s
	updated.Adapter = final
	updated.Sessions++
	updated.Seen = make(map[string]bool, len(s.Seen)+1)
	for k, v := range s.Seen {
		updated.Seen[k] = v
	}
	updated.Seen[art.Source] = true
	if err = Save(dir, updated); err != nil {
		return fmt.Errorf("adapter written but settings could not be saved: %w", err)
	}
	*s = updated
	fmt.Fprintf(out, "  Saved LoRA adapter: %s\n", final)
	return nil
}

type limitedBuffer struct {
	b     strings.Builder
	limit int
}

func (b *limitedBuffer) Write(p []byte) (int, error) {
	n := len(p)
	remaining := b.limit - b.b.Len()
	if remaining > 0 {
		if len(p) > remaining {
			p = p[:remaining]
		}
		_, _ = b.b.Write(p)
	}
	return n, nil
}
func (b *limitedBuffer) String() string { return b.b.String() }

func Run(r *bufio.Reader, w io.Writer, dir string, forceSetup, autoChat, autoTrain bool) error {
	s, err := Load(dir)
	if err != nil && !os.IsNotExist(err) {
		return err
	}
	if forceSetup || os.IsNotExist(err) {
		s, err = Setup(r, w, dir, s)
		if err != nil {
			return err
		}
	}
	if err = Validate(s); err != nil {
		return fmt.Errorf("run --setup to configure the pretrained model: %w", err)
	}
	if s.ModelHash != "" {
		if err := checkSHA(s.Model, s.ModelHash); err != nil {
			return fmt.Errorf("model changed or is corrupted; run --setup again: %w", err)
		}
	}
	fmt.Fprintf(w, "\n  %s uses a local pretrained GGUF model. LoRA sessions: %d.\n", s.Name, s.Sessions)
	fmt.Fprintln(w, "  CPU only; the runtime has an address-space limit, not a whole-system RAM guarantee.")
	action := ""
	if autoChat {
		action = "2"
	} else if autoTrain {
		action = "1"
	}
	for {
		if action == "" {
			fmt.Fprintln(w, "\n  [1] Train one item  [2] Chat  [3] Setup  [Q] Quit")
			action, err = prompt(r, w, "  Select [2]: ")
			if err != nil {
				return nil
			}
			if action == "" {
				action = "2"
			}
		}
		switch strings.ToLower(action) {
		case "1", "train":
			if err = TrainNext(context.Background(), &s, dir, w); err != nil {
				fmt.Fprintln(w, "  Training error:", err)
			}
		case "2", "chat":
			fmt.Fprintln(w, "  Chat is closed-book. Type /clear, /train or /back.")
			var history []Turn
			for {
				text, e := prompt(r, w, "  You > ")
				if e != nil {
					return nil
				}
				switch strings.ToLower(text) {
				case "/back", "/quit":
					goto chatdone
				case "/clear":
					history = nil
					fmt.Fprintln(w, "  Context cleared.")
					continue
				case "/train":
					if e = TrainNext(context.Background(), &s, dir, w); e != nil {
						fmt.Fprintln(w, "  Training error:", e)
					}
					continue
				case "":
					continue
				}
				fmt.Fprintln(w, "  Thinking...")
				answer, e := Generate(context.Background(), s, history, text)
				if e != nil {
					fmt.Fprintln(w, "  Model error:", e)
					continue
				}
				fmt.Fprintf(w, "  %s > %s\n", s.Name, answer)
				history = append(history, Turn{text, answer})
				if len(history) > 2 {
					history = history[len(history)-2:]
				}
			}
		chatdone:
			;
		case "3", "setup":
			if updated, e := Setup(r, w, dir, s); e != nil {
				fmt.Fprintln(w, "  Setup:", e)
			} else {
				s = updated
			}
		case "q", "quit", "exit":
			return nil
		default:
			fmt.Fprintln(w, "  Choose 1, 2, 3 or Q.")
		}
		action = ""
	}
}
