// Package pretrained runs small local models through the CPU-only QVAC llama.cpp
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
	"strconv"
	"strings"
	"time"

	"github.com/Rmdir-Matrix/DevMatrix/internal/corpus"
	"github.com/Rmdir-Matrix/DevMatrix/internal/specs"
)

const runtimeURL = "https://github.com/tetherto/qvac-fabric-llm.cpp/releases/download/b7349/llama-b7349-bin-ubuntu-x64.tar.gz"
const runtimeSHA = "f9490b999f5f49535dea67d7fe5c839b76b011a93703c53226f8330f4b53eb23"

type Settings struct {
	Name       string          `json:"name"`
	Model      string          `json:"model"`
	RuntimeDir string          `json:"runtime_dir"`
	StorageDir string          `json:"storage_dir"`
	ModelHash  string          `json:"model_hash"`
	ModelID    string          `json:"model_id,omitempty"`
	ChatFormat string          `json:"chat_format,omitempty"`
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
	defaultName := old.Name
	if defaultName == "" {
		defaultName = "Matrix"
	}
	name, e := prompt(r, w, "  Study buddy name ["+defaultName+"]: ")
	if e != nil {
		return old, e
	}
	if name == "" {
		name = defaultName
	}
	if len(name) > 64 {
		name = name[:64]
	}
	defaultStorage := old.StorageDir
	if defaultStorage == "" {
		defaultStorage = dir
	}
	storage, e := prompt(r, w, "  SSD storage folder ["+defaultStorage+"]: ")
	if e != nil {
		return old, e
	}
	if storage == "" {
		storage = defaultStorage
	}
	storage, e = filepath.Abs(storage)
	if e != nil {
		return old, e
	}
	if e = os.MkdirAll(storage, 0700); e != nil {
		return old, e
	}
	fmt.Fprintln(w, "\n  Setup stores the model, runtime and trained adapters under:", storage)
	hw := specs.DetectHardware()
	selected := recommendedModel(hw)
	if hw.Detected {
		fmt.Fprintf(w, "  Detected RAM: %s total, %s available; CPU cores: %d.\n", specs.FormatRAM(hw.TotalRAMMB), specs.FormatRAM(hw.AvailableRAMMB), hw.CPUCores)
	} else {
		fmt.Fprintln(w, "  RAM detection failed; using the smallest model recommendation.")
	}
	fmt.Fprintf(w, "  Recommended: %s (~%d MB download).\n", selected.Name, selected.SizeMB)
	fmt.Fprintln(w, "  Smaller models save memory but can give weaker answers. RAM use depends on workload.")
	fmt.Fprintln(w, "  QVAC CPU tools (~36 MB) are downloaded separately. Internet required.")
	choiceDefault := "1"
	if old.Model != "" {
		choiceDefault = "3"
		fmt.Fprintf(w, "  Current model: %s\n", old.Model)
	}
	fmt.Fprintln(w, "  [1] Download recommended model  [2] Use existing files  [3] Keep current model  [Q] Cancel")
	choice, e := prompt(r, w, "  Choice ["+choiceDefault+"]: ")
	if e != nil {
		return old, e
	}
	if choice == "" {
		choice = choiceDefault
	}
	s := Settings{Name: name, StorageDir: storage, Seen: map[string]bool{}}
	switch strings.ToLower(choice) {
	case "1":
		fmt.Fprintln(w, "  Model license:", selected.License)
		agree, e := prompt(r, w, "  Accept the model license and download? [y/N]: ")
		if e != nil {
			return old, e
		}
		if !strings.EqualFold(agree, "y") && !strings.EqualFold(agree, "yes") {
			return old, fmt.Errorf("setup cancelled; no model downloaded")
		}
		s.ModelID, s.ChatFormat = selected.ID, selected.Format
		s.Model = filepath.Join(storage, "models", selected.Filename)
		s.RuntimeDir = filepath.Join(storage, "runtime-b7349")
		if e = os.MkdirAll(filepath.Dir(s.Model), 0700); e != nil {
			return old, e
		}
		fmt.Fprintln(w, "  Downloading and verifying the model (this may take a while)...")
		if e = downloadVerified(context.Background(), selected.URL, s.Model, selected.SHA, 400<<20); e != nil {
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
		fmt.Fprintln(w, "  Chat format: [1] SmolLM2 / ChatML  [2] Gemma 3")
		format, ferr := prompt(r, w, "  Chat format ["+map[bool]string{true: "1", false: "2"}[selected.Format == "chatml"]+"]: ")
		if ferr != nil {
			return old, ferr
		}
		if format == "" {
			if selected.Format == "chatml" {
				format = "1"
			} else {
				format = "2"
			}
		}
		switch format {
		case "1":
			s.ModelID, s.ChatFormat = smallModel.ID, smallModel.Format
		case "2":
			s.ModelID, s.ChatFormat = gemmaModel.ID, gemmaModel.Format
		default:
			return old, fmt.Errorf("invalid chat format")
		}
		s.Model, e = prompt(r, w, "  Path to the GGUF file: ")
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
	case "3":
		if old.Model == "" {
			return old, fmt.Errorf("no current model; run setup again")
		}
		s.Model, s.RuntimeDir, s.ModelID, s.ChatFormat = old.Model, old.RuntimeDir, old.ModelID, old.ChatFormat
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
	if e = applyModelProfile(dir, old, &s); e != nil {
		return old, e
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
func childWithLimit(ctx context.Context, s Settings, tool string, mb int, args ...string) (*exec.Cmd, error) {
	if runtime.GOOS != "linux" {
		return nil, fmt.Errorf("pretrained runtime requires Linux")
	}
	limit, e := exec.LookPath("prlimit")
	if e != nil {
		return nil, fmt.Errorf("install util-linux (prlimit) to enforce a runtime memory limit: %w", e)
	}
	binary := filepath.Join(s.RuntimeDir, tool)
	all := append([]string{fmt.Sprintf("--as=%d", int64(mb)*1024*1024), "--", binary}, args...)
	cmd := exec.CommandContext(ctx, limit, all...)
	cmd.Env = append(os.Environ(), "LD_LIBRARY_PATH="+s.RuntimeDir)
	return cmd, nil
}
func child(ctx context.Context, s Settings, tool string, args ...string) (*exec.Cmd, error) {
	hw := specs.DetectHardware()
	mb := 1536
	if hw.Detected && hw.AvailableRAMMB < 1024 {
		return nil, fmt.Errorf("not enough available RAM (%d MB); close programs to free at least 1024 MB", hw.AvailableRAMMB)
	}
	if hw.Detected && hw.TotalRAMMB/2 < mb {
		mb = hw.TotalRAMMB / 2
	}
	if mb < 512 {
		return nil, fmt.Errorf("not enough total RAM for local inference")
	}
	return childWithLimit(ctx, s, tool, mb, args...)
}

func common(s Settings) []string {
	threads := runtime.NumCPU()
	if threads > 2 {
		threads = 2
	}
	if threads < 1 {
		threads = 1
	}
	n := strconv.Itoa(threads)
	return []string{"-m", s.Model, "-ngl", "0", "-t", n, "-tb", n, "--no-warmup"}
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
func renderPrompt(s Settings, history []Turn, input string) string {
	if modelFormat(s) != "chatml" {
		return gemmaPrompt(s.Name, history, input)
	}
	clean := func(v string) string {
		v = strings.ReplaceAll(v, "<|im_start|>", "")
		v = strings.ReplaceAll(v, "<|im_end|>", "")
		return v
	}
	var p strings.Builder
	p.WriteString("<|im_start|>system\nYou are " + clean(s.Name) + ", a helpful study buddy. Give clear answers and admit uncertainty.<|im_end|>\n")
	for _, t := range history {
		p.WriteString("<|im_start|>user\n" + clean(t.User) + "<|im_end|>\n<|im_start|>assistant\n" + clean(t.Assistant) + "<|im_end|>\n")
	}
	p.WriteString("<|im_start|>user\n" + clean(input) + "<|im_end|>\n<|im_start|>assistant\n")
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
	args := append(common(s), "-c", "1024", "-b", "64", "-ub", "32", "-n", "160", "--no-conversation", "--no-display-prompt", "--simple-io", "-p", renderPrompt(s, history, input))
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
	args := append(common(*s), "-f", data.Name(), "-c", "64", "-b", "8", "-ub", "8", "-fa", "off", "--lora-rank", "4", "--lora-alpha", "8", "--lora-modules", "attn_q,attn_v", "--num-epochs", "1", "--learning-rate", "0.00001", "--checkpoint-save-steps", "0", "--output-adapter", next)
	if s.Adapter != "" {
		args = append(args, "--lora", s.Adapter)
	}
	budget, err := trainingBudgetFor(specs.DetectHardware(), *s)
	if err != nil {
		return err
	}
	cmd, err := childWithLimit(ctx, *s, "llama-finetune-lora", budget.addressMB, args...)
	if err != nil {
		return err
	}
	fmt.Fprintf(out, "  Low-memory training on %q (%s). This may be slow.\n", art.Title, art.Source)
	fmt.Fprintf(out, "  Low-memory settings: context 64, batch 8; trainer address-space limit %d MiB.\n", budget.addressMB)
	fmt.Fprintf(out, "  Training stops if available RAM falls below %d MiB or trainer RSS exceeds %d MiB.\n", trainingReserveMB, budget.rssMB)
	fmt.Fprintln(out, "  Sampling can miss spikes; previous weights remain safe if training stops.")
	cmd.Stdout = out
	cmd.Stderr = out
	if err = runTrainingGuarded(cmd, readTrainingMemory, 100*time.Millisecond, budget.rssMB); err != nil {
		return fmt.Errorf("training stopped; previous adapter preserved: %w", err)
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
	hw := specs.DetectHardware()
	recommend := recommendedModel(hw)
	if modelID(s) == gemmaModel.ID && recommend.ID == smallModel.ID && !forceSetup {
		fmt.Fprintf(w, "\n  Hardware check: %s is smaller and recommended for this PC. Your current model and adapter will be kept unless you choose to switch.\n", recommend.Name)
		answer, e := prompt(r, w, "  Open setup to switch models (then choose 1)? [y/N]: ")
		if e != nil && e != io.EOF {
			return e
		}
		if strings.EqualFold(answer, "y") || strings.EqualFold(answer, "yes") {
			if updated, e := Setup(r, w, dir, s); e != nil {
				fmt.Fprintln(w, "  Setup:", e)
			} else {
				s = updated
			}
		}
	}
	fmt.Fprintf(w, "\n  %s uses model %s. LoRA sessions: %d.\n", s.Name, filepath.Base(s.Model), s.Sessions)
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
