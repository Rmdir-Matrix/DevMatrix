package share

import (
	"bufio"
	"bytes"
	"compress/gzip"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/Rmdir-Matrix/DevMatrix/internal/brain"
	"github.com/Rmdir-Matrix/DevMatrix/internal/config"
	"github.com/Rmdir-Matrix/DevMatrix/internal/specs"
	"github.com/Rmdir-Matrix/DevMatrix/internal/ui"
)

const ShareFormatVersion = "matrix-share-v1"

// ShareSummary holds human-readable metadata inside a .matrix portable file.
type ShareSummary struct {
	ExportedName   string   `json:"exported_name"`
	Level          int      `json:"level"`
	XP             int64    `json:"xp"`
	ArticlesRead   int      `json:"articles_read"`
	ActiveSynapses int      `json:"active_synapses"`
	VocabSize      int      `json:"vocab_size"`
	TopTopics      []string `json:"top_topics"`
}

type shareEnvelope struct {
	Format         string       `json:"format"`
	ExportedAt     string       `json:"exported_at"`
	Summary        ShareSummary `json:"summary"`
	SHA256Checksum string       `json:"sha256_checksum"`
	BrainPayload   []byte       `json:"brain_payload"`
}

// ExportMatrix packages a Brain under exportName into a single portable .matrix file.
// Returns (resolvedFilePath, sizeBytes, error).
func ExportMatrix(b *brain.Brain, exportName string, outPath string) (string, int64, error) {
	chosenName := config.SanitizeMatrixName(exportName)
	if chosenName == "" {
		chosenName = b.Name
	}

	// Clone metadata name cleanly for the exported payload without mutating caller unexpectedly
	origName := b.Name
	b.Name = chosenName
	rawBrain, err := json.Marshal(b)
	b.Name = origName
	if err != nil {
		return "", 0, fmt.Errorf("serialize brain: %w", err)
	}

	sum := sha256.Sum256(rawBrain)
	env := shareEnvelope{
		Format:     ShareFormatVersion,
		ExportedAt: time.Now().UTC().Format(time.RFC3339),
		Summary: ShareSummary{
			ExportedName:   chosenName,
			Level:          b.Level,
			XP:             b.XP,
			ArticlesRead:   b.ArticlesRead,
			ActiveSynapses: b.Net.ActiveSynapses(),
			VocabSize:      b.Net.VocabSize(),
			TopTopics:      b.TopTopics(5),
		},
		SHA256Checksum: hex.EncodeToString(sum[:]),
		BrainPayload:   rawBrain,
	}

	if strings.TrimSpace(outPath) == "" {
		outPath = fileSlug(chosenName) + ".matrix"
	} else if !strings.HasSuffix(strings.ToLower(outPath), ".matrix") {
		outPath += ".matrix"
	}
	absOut, err := filepath.Abs(outPath)
	if err != nil {
		return "", 0, err
	}
	if err := os.MkdirAll(filepath.Dir(absOut), 0o755); err != nil {
		return "", 0, err
	}

	var buf bytes.Buffer
	gw, _ := gzip.NewWriterLevel(&buf, gzip.BestCompression)
	if err := json.NewEncoder(gw).Encode(env); err != nil {
		_ = gw.Close()
		return "", 0, err
	}
	if err := gw.Close(); err != nil {
		return "", 0, err
	}

	if err := os.WriteFile(absOut, buf.Bytes(), 0o644); err != nil {
		return "", 0, err
	}
	return absOut, int64(buf.Len()), nil
}

// ImportMatrix loads and verifies a shared .matrix archive, adapts its memory caps safely
// to the local machine's hardware profile, and saves it into dataDir/brains.
func ImportMatrix(filePath string, dataDir string, prof specs.ComputeProfile) (*brain.Brain, ShareSummary, error) {
	f, err := os.Open(strings.TrimSpace(filePath))
	if err != nil {
		return nil, ShareSummary{}, err
	}
	defer f.Close()

	gr, err := gzip.NewReader(f)
	if err != nil {
		return nil, ShareSummary{}, fmt.Errorf("invalid .matrix archive (gzip): %w", err)
	}
	defer gr.Close()

	var env shareEnvelope
	if err := json.NewDecoder(gr).Decode(&env); err != nil {
		return nil, ShareSummary{}, fmt.Errorf("invalid .matrix manifest: %w", err)
	}
	if env.Format != ShareFormatVersion {
		return nil, ShareSummary{}, fmt.Errorf("unsupported .matrix format %q", env.Format)
	}

	sum := sha256.Sum256(env.BrainPayload)
	if hex.EncodeToString(sum[:]) != env.SHA256Checksum {
		return nil, ShareSummary{}, fmt.Errorf("checksum mismatch: .matrix file appears corrupted")
	}

	var loaded brain.Brain
	if err := json.Unmarshal(env.BrainPayload, &loaded); err != nil {
		return nil, ShareSummary{}, fmt.Errorf("decode brain weights: %w", err)
	}
	if strings.TrimSpace(env.Summary.ExportedName) != "" {
		loaded.Name = config.SanitizeMatrixName(env.Summary.ExportedName)
	}
	loaded.ApplyProfileCaps(prof)

	if err := loaded.Save(dataDir); err != nil {
		return nil, ShareSummary{}, err
	}
	return &loaded, env.Summary, nil
}

// ListSavedBrains returns summaries of all saved Matrix brains inside dataDir/brains.
func ListSavedBrains(dataDir string, prof specs.ComputeProfile) ([]ShareSummary, error) {
	brainsDir := filepath.Join(dataDir, "brains")
	entries, err := os.ReadDir(brainsDir)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}
		return nil, err
	}

	var list []ShareSummary
	for _, e := range entries {
		if e.IsDir() || !strings.HasSuffix(e.Name(), ".brain.gz") {
			continue
		}
		full := filepath.Join(brainsDir, e.Name())
		f, err := os.Open(full)
		if err != nil {
			continue
		}
		gr, err := gzip.NewReader(f)
		if err != nil {
			_ = f.Close()
			continue
		}
		var b brain.Brain
		decErr := json.NewDecoder(gr).Decode(&b)
		_ = gr.Close()
		_ = f.Close()
		if decErr != nil || b.Net == nil {
			continue
		}
		b.ApplyProfileCaps(prof)
		list = append(list, ShareSummary{
			ExportedName:   b.Name,
			Level:          b.Level,
			XP:             b.XP,
			ArticlesRead:   b.ArticlesRead,
			ActiveSynapses: b.Net.ActiveSynapses(),
			VocabSize:      b.Net.VocabSize(),
			TopTopics:      b.TopTopics(3),
		})
	}
	return list, nil
}

// RunInteractiveMenu displays the Export / Import / Rename / Switch Matrixes hub.
func RunInteractiveMenu(
	reader *bufio.Reader,
	out io.Writer,
	theme ui.Theme,
	dataDir string,
	cfg *config.AppConfig,
	activeBrain **brain.Brain,
) error {
	for {
		b := *activeBrain
		fmt.Fprintln(out)
		fmt.Fprintln(out, theme.BoldCyan("  === SHARE, EXPORT & MANAGE MATRIXES ==="))
		fmt.Fprintf(out, "  Active Matrix: %s (Level %d • %d synapses • %d articles)\n\n",
			theme.BoldGreen(b.Name), b.Level, b.Net.ActiveSynapses(), b.ArticlesRead)
		fmt.Fprintf(out, "    %s Export %s (save a named .matrix file to share with friends)\n", theme.BoldGreen("[1]"), b.Name)
		fmt.Fprintf(out, "    %s Import a shared .matrix file\n", theme.BoldCyan("[2]"))
		fmt.Fprintf(out, "    %s Rename current Matrix\n", theme.Yellow("[3]"))
		fmt.Fprintf(out, "    %s Switch / Create another Matrix\n", theme.Magenta("[4]"))
		fmt.Fprintf(out, "    %s Back to Main Menu\n\n", theme.Dim("[B]"))

		choice, err := ui.Prompt(reader, out, "  Select option: ")
		if err != nil || strings.EqualFold(choice, "b") || choice == "" {
			return nil
		}

		switch strings.ToLower(choice) {
		case "1", "export":
			exportName, _ := ui.Prompt(reader, out, fmt.Sprintf("  Export name for this Matrix [%s]: ", b.Name))
			if strings.TrimSpace(exportName) == "" {
				exportName = b.Name
			}
			defaultFile := fileSlug(exportName) + ".matrix"
			destPath, _ := ui.Prompt(reader, out, fmt.Sprintf("  Output file path [%s]: ", defaultFile))
			if strings.TrimSpace(destPath) == "" {
				destPath = defaultFile
			}
			savedPath, sz, exErr := ExportMatrix(b, exportName, destPath)
			if exErr != nil {
				fmt.Fprintln(out, "  "+theme.Red("Export failed: "+exErr.Error()))
				continue
			}
			fmt.Fprintf(out, "  %s Exported %q (Lv.%d) → %s (%.1f KB)\n",
				theme.BoldGreen("✓"), config.SanitizeMatrixName(exportName), b.Level, savedPath, float64(sz)/1024.0)
			fmt.Fprintln(out, "  "+theme.Dim("Share this single .matrix file with anyone running Matrix so they can import it!"))

		case "2", "import":
			printLocalMatrixFilesHint(out, theme)
			inPath, _ := ui.Prompt(reader, out, "  Path to .matrix file to import: ")
			if strings.TrimSpace(inPath) == "" {
				continue
			}
			imported, sum, imErr := ImportMatrix(inPath, dataDir, cfg.Profile)
			if imErr != nil {
				fmt.Fprintln(out, "  "+theme.Red("Import failed: "+imErr.Error()))
				continue
			}
			*activeBrain = imported
			cfg.ActiveMatrixName = imported.Name
			_ = config.Save(dataDir, *cfg)
			fmt.Fprintf(out, "  %s Imported & activated %q! (Level %d • %d articles • %d active synapses)\n",
				theme.BoldGreen("✓"), sum.ExportedName, imported.Level, imported.ArticlesRead, imported.Net.ActiveSynapses())

		case "3", "rename":
			newName, _ := ui.Prompt(reader, out, fmt.Sprintf("  New name for %s: ", b.Name))
			newName = config.SanitizeMatrixName(newName)
			if newName == "" || newName == b.Name {
				continue
			}
			b.Name = newName
			_ = b.Save(dataDir)
			cfg.ActiveMatrixName = newName
			_ = config.Save(dataDir, *cfg)
			fmt.Fprintf(out, "  %s Renamed active Matrix to %q!\n", theme.BoldGreen("✓"), newName)

		case "4", "switch":
			saved, _ := ListSavedBrains(dataDir, cfg.Profile)
			if len(saved) > 0 {
				fmt.Fprintln(out, "\n  Saved Matrixes on this PC:")
				for i, s := range saved {
					fmt.Fprintf(out, "    [%d] %s — Level %d (%d articles, %d synapses)\n",
						i+1, theme.BoldGreen(s.ExportedName), s.Level, s.ArticlesRead, s.ActiveSynapses)
				}
			}
			targetName, _ := ui.Prompt(reader, out, "  Enter Matrix name to switch to (or type a brand-new name to create one): ")
			targetName = config.SanitizeMatrixName(targetName)
			if targetName == "" {
				continue
			}
			_ = b.Save(dataDir)
			nextBrain, existed, lErr := brain.LoadOrCreate(dataDir, targetName, cfg.Profile)
			if lErr != nil {
				fmt.Fprintln(out, "  "+theme.Red("Could not load/create Matrix: "+lErr.Error()))
				continue
			}
			_ = nextBrain.Save(dataDir)
			*activeBrain = nextBrain
			cfg.ActiveMatrixName = nextBrain.Name
			_ = config.Save(dataDir, *cfg)
			if existed {
				fmt.Fprintf(out, "  %s Switched to %q (Level %d)!\n", theme.BoldGreen("✓"), nextBrain.Name, nextBrain.Level)
			} else {
				fmt.Fprintf(out, "  %s Created brand-new Level 1 Matrix %q!\n", theme.BoldGreen("✓"), nextBrain.Name)
			}
		}
	}
}

func printLocalMatrixFilesHint(out io.Writer, theme ui.Theme) {
	matches, _ := filepath.Glob("*.matrix")
	if len(matches) > 0 {
		fmt.Fprintf(out, "  %s Found in current folder: %s\n", theme.Dim("Hint:"), strings.Join(matches, ", "))
	}
}

func fileSlug(name string) string {
	s := strings.ToLower(strings.TrimSpace(name))
	var b strings.Builder
	for _, r := range s {
		if (r >= 'a' && r <= 'z') || (r >= '0' && r <= '9') {
			b.WriteRune(r)
		} else if r == ' ' || r == '-' || r == '_' {
			b.WriteRune('-')
		}
	}
	res := strings.Trim(b.String(), "-")
	if res == "" {
		return "matrix"
	}
	return res
}
