package main

import (
	"bufio"
	"context"
	"flag"
	"fmt"
	"io"
	"os"
	"runtime"
	"strings"

	"github.com/Rmdir-Matrix/DevMatrix/internal/brain"
	"github.com/Rmdir-Matrix/DevMatrix/internal/chat"
	"github.com/Rmdir-Matrix/DevMatrix/internal/config"
	"github.com/Rmdir-Matrix/DevMatrix/internal/corpus"
	"github.com/Rmdir-Matrix/DevMatrix/internal/platform"
	"github.com/Rmdir-Matrix/DevMatrix/internal/share"
	"github.com/Rmdir-Matrix/DevMatrix/internal/specs"
	"github.com/Rmdir-Matrix/DevMatrix/internal/trainer"
	"github.com/Rmdir-Matrix/DevMatrix/internal/ui"
)

func main() {
	dataDirFlag := flag.String("data-dir", "", "Directory for Matrix saves and configuration (default: ./matrix_data)")
	forceSetup := flag.Bool("setup", false, "Run the interactive RAM & Matrix Name setup wizard on startup")
	autoTrain := flag.Bool("train", false, "Immediately start Mode 1 (AI Article Training) on startup")
	autoChat := flag.Bool("chat", false, "Immediately start Mode 2 (Study Buddy Chat) on startup")
	exportPath := flag.String("export", "", "Export active Matrix to a shareable .matrix file and exit")
	exportName := flag.String("export-name", "", "Custom name to embed when using --export")
	importPath := flag.String("import", "", "Import a shared .matrix file on startup")
	showSpecsOnly := flag.Bool("specs", false, "Print detected hardware specs and auto-optimized AI profile, then exit")
	noColor := flag.Bool("no-color", false, "Disable ANSI terminal colors")
	flag.Parse()

	colorOK := platform.InitConsole() && !*noColor
	theme := ui.NewTheme(colorOK)

	hw := specs.DetectHardware()
	if *showSpecsOnly {
		printSpecsDiagnostic(os.Stdout, theme, hw)
		return
	}

	dataDir, err := config.ResolveDataDir(*dataDirFlag)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error initializing data folder: %v\n", err)
		os.Exit(1)
	}

	cfg, setupDone, err := config.Load(dataDir, hw)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Warning: could not read existing config (%v); starting setup.\n", err)
	}

	// Non-interactive CLI --export shortcut
	if strings.TrimSpace(*exportPath) != "" {
		b, _, lErr := brain.LoadOrCreate(dataDir, cfg.ActiveMatrixName, cfg.Profile)
		if lErr != nil {
			fmt.Fprintf(os.Stderr, "Error loading Matrix: %v\n", lErr)
			os.Exit(1)
		}
		saved, sz, exErr := share.ExportMatrix(b, *exportName, *exportPath)
		if exErr != nil {
			fmt.Fprintf(os.Stderr, "Export error: %v\n", exErr)
			os.Exit(1)
		}
		fmt.Printf("Exported %s (Lv.%d) -> %s (%.1f KB)\n", b.Name, b.Level, saved, float64(sz)/1024.0)
		return
	}

	reader := bufio.NewReader(os.Stdin)
	fmt.Print(theme.Banner(cfg.ActiveMatrixName))

	if !setupDone || *forceSetup {
		cfg, err = config.RunSetupWizard(reader, os.Stdout, theme, hw, cfg)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Setup interrupted: %v\n", err)
			os.Exit(1)
		}
		if err := config.Save(dataDir, cfg); err != nil {
			fmt.Fprintf(os.Stderr, "Error saving configuration: %v\n", err)
			os.Exit(1)
		}
	}

	// Enforce strict Go runtime soft memory cap and CPU worker bounds
	specs.ApplyRuntimeLimits(cfg.Profile)

	articleReader, err := corpus.NewReader(dataDir)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error loading study corpus: %v\n", err)
		os.Exit(1)
	}

	var activeBrain *brain.Brain
	if strings.TrimSpace(*importPath) != "" {
		imp, sum, imErr := share.ImportMatrix(*importPath, dataDir, cfg.Profile)
		if imErr != nil {
			fmt.Fprintf(os.Stderr, "Import error: %v\n", imErr)
			os.Exit(1)
		}
		activeBrain = imp
		cfg.ActiveMatrixName = imp.Name
		_ = config.Save(dataDir, cfg)
		fmt.Printf("  %s Imported shared Matrix %q (Level %d • %d articles)!\n",
			theme.BoldGreen("✓"), sum.ExportedName, imp.Level, imp.ArticlesRead)
	} else {
		var existed bool
		activeBrain, existed, err = brain.LoadOrCreate(dataDir, cfg.ActiveMatrixName, cfg.Profile)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Warning: starting fresh brain (%v)\n", err)
			activeBrain = brain.NewBrain(cfg.ActiveMatrixName, cfg.Profile)
		} else if !existed {
			_ = activeBrain.Save(dataDir)
		}
	}

	if *autoTrain {
		_ = trainer.RunInteractive(reader, os.Stdout, theme, activeBrain, articleReader, dataDir, true)
	} else if *autoChat {
		_ = chat.RunInteractive(reader, os.Stdout, theme, activeBrain, articleReader, dataDir)
	}

	runMainMenu(reader, os.Stdout, theme, dataDir, hw, cfg, activeBrain, articleReader)
}

func printSpecsDiagnostic(out io.Writer, theme ui.Theme, hw specs.HardwareInfo) {
	prof, _ := specs.BuildProfile(hw, 0)
	fmt.Fprintln(out, theme.BoldCyan("Matrix Hardware & Auto-Optimizer Diagnostic"))
	fmt.Fprintf(out, "  OS / Arch       : %s / %s\n", hw.OS, hw.Arch)
	fmt.Fprintf(out, "  CPU Cores       : %d\n", hw.CPUCores)
	fmt.Fprintf(out, "  Total RAM       : %s (auto-detected: %v)\n", specs.FormatRAM(hw.TotalRAMMB), hw.Detected)
	fmt.Fprintf(out, "  Available RAM   : %s\n", specs.FormatRAM(hw.AvailableRAMMB))
	fmt.Fprintf(out, "  Recommended Cap : %d MB (%s Tier)\n", prof.RAMBudgetMB, prof.TierName)
	fmt.Fprintf(out, "  Model Capacity  : %d embed dim • %d hidden neurons • %d max vocab • %d threads\n",
		prof.EmbedDim, prof.HiddenNeurons, prof.MaxVocab, prof.WorkerThreads)
}

func printBrainDiagnostic(out io.Writer, theme ui.Theme, b *brain.Brain, cr *corpus.Reader) {
	fmt.Fprintln(out)
	fmt.Fprintln(out, theme.BoldGreen("Matrix Neural Brain & Corpus Status"))
	fmt.Fprintf(out, "  Buddy Name & ID : %s (#%s) — %s\n", theme.Bold(b.Name), b.ID, trainer.IntelligenceStage(b.Level))
	fmt.Fprintf(out, "  Level & XP      : Level %d (%d XP  •  next level at %d XP)\n",
		b.Level, b.XP, brain.XPForLevel(b.Level+1))
	fmt.Fprintf(out, "  Neural Architecture : %d embed dim → %d hidden neurons → %d/%d vocabulary tokens\n",
		b.Net.EmbedDim, b.Net.HiddenNeurons, b.Net.VocabSize(), b.Net.MaxVocab)
	fmt.Fprintf(out, "  Active Synapses : %d learned neuron connections\n", b.Net.ActiveSynapses())
	fmt.Fprintf(out, "  Training Stats  : %d articles read • %d tokens trained • smoothed loss: %.4f\n",
		b.ArticlesRead, b.TokensTrained, b.SmoothedLoss)
	fmt.Fprintf(out, "  Study Memory    : %d/%d indexed passages\n", len(b.Passages), b.MaxPassages)
	fmt.Fprintf(out, "  Article Sources : Live Wikipedia API + %d built-in study articles + local notes (%s)\n",
		cr.SeedCount(), cr.NotesDir())
	if topics := b.TopTopics(5); len(topics) > 0 {
		fmt.Fprintf(out, "  Top Topics      : %s\n", strings.Join(topics, ", "))
	}
}

func printStatusBar(out io.Writer, theme ui.Theme, cfg config.AppConfig, b *brain.Brain) {
	var mem runtime.MemStats
	runtime.ReadMemStats(&mem)
	usedMB := float64(mem.Alloc) / (1024.0 * 1024.0)

	fmt.Fprintln(out)
	fmt.Fprintf(out, "  %s %s (Lv.%d • %s)  |  %s %d synapses • %d articles  |  %s %.1f/%d MB\n",
		theme.Dim("Buddy:"), theme.BoldGreen(b.Name), b.Level, trainer.IntelligenceStage(b.Level),
		theme.Dim("Brain:"), b.Net.ActiveSynapses(), b.ArticlesRead,
		theme.Dim("RAM:"), usedMB, cfg.Profile.RAMBudgetMB,
	)
	fmt.Fprintln(out, theme.Dim("  ──────────────────────────────────────────────────────────────────────────────────"))
}

func runMainMenu(
	reader *bufio.Reader,
	out io.Writer,
	theme ui.Theme,
	dataDir string,
	hw specs.HardwareInfo,
	cfg config.AppConfig,
	activeBrain *brain.Brain,
	articleReader *corpus.Reader,
) {
	for {
		printStatusBar(out, theme, cfg, activeBrain)
		fmt.Fprintf(out, "    %s Training Mode %s\n", theme.BoldGreen("[1]"), theme.Dim("(Easy AI article training — press Enter to start!)"))
		fmt.Fprintf(out, "    %s Chat Mode     %s\n", theme.BoldCyan("[2]"), theme.Dim("(Talk, study & quiz with "+activeBrain.Name+")"))
		fmt.Fprintf(out, "    %s Share / Export%s\n", theme.Magenta("[3]"), theme.Dim("(Export name, share .matrix files, or switch Matrixes)"))
		fmt.Fprintf(out, "    %s Brain & Specs %s\n", theme.Yellow("[4]"), theme.Dim("(Inspect neural stats or change RAM / Name)"))
		fmt.Fprintf(out, "    %s Quit\n\n", theme.Dim("[Q]"))

		choice, err := ui.Prompt(reader, out, "  Select mode [1]: ")
		if err != nil {
			_ = activeBrain.Save(dataDir)
			fmt.Fprintln(out)
			return
		}

		switch strings.ToLower(choice) {
		case "", "1", "t", "train":
			if err := trainer.RunInteractive(reader, out, theme, activeBrain, articleReader, dataDir, false); err != nil {
				fmt.Fprintln(out, "  "+theme.Red("Training error: "+err.Error()))
			}
		case "2", "c", "chat":
			if err := chat.RunInteractive(reader, out, theme, activeBrain, articleReader, dataDir); err != nil {
				fmt.Fprintln(out, "  "+theme.Red("Chat error: "+err.Error()))
			}
		case "3", "e", "export", "share":
			if err := share.RunInteractiveMenu(reader, out, theme, dataDir, &cfg, &activeBrain); err != nil {
				fmt.Fprintln(out, "  "+theme.Red("Share menu error: "+err.Error()))
			}
		case "4", "s", "setup", "specs", "brain":
			printSpecsDiagnostic(out, theme, hw)
			printBrainDiagnostic(out, theme, activeBrain, articleReader)

			sub, sErr := ui.Prompt(reader, out, "\n  Options: [T] Train 1 sample article  |  [S] Re-run RAM/Name setup  |  [Enter] Back: ")
			if sErr != nil {
				continue
			}
			switch strings.ToLower(sub) {
			case "t":
				art, aErr := articleReader.NextArticle(context.Background(), "", activeBrain.SeenSources)
				if aErr == nil {
					rep := activeBrain.TrainOnArticle(art, 2)
					_ = activeBrain.Save(dataDir)
					fmt.Fprintf(out, "  %s Studied %q [%s] (%d tokens) → Loss: %.4f | Synapses: %d | Level: %d\n",
						theme.BoldGreen("✓"), rep.ArticleTitle, rep.Topic, rep.TokensRead,
						rep.SmoothedLoss, rep.ActiveSynapses, rep.NewLevel)
				}
			case "s":
				updated, wErr := config.RunSetupWizard(reader, out, theme, hw, cfg)
				if wErr == nil {
					cfg = updated
					_ = config.Save(dataDir, cfg)
					specs.ApplyRuntimeLimits(cfg.Profile)
					activeBrain, _, _ = brain.LoadOrCreate(dataDir, cfg.ActiveMatrixName, cfg.Profile)
					_ = activeBrain.Save(dataDir)
				}
			}
		case "q", "quit", "exit":
			_ = activeBrain.Save(dataDir)
			fmt.Fprintf(out, "\n  %s Saved & powered down %s. See you next study session!\n\n",
				theme.BoldGreen("⚡"), activeBrain.Name)
			return
		default:
			fmt.Fprintln(out, "  "+theme.Red("Unknown command. Choose 1, 2, 3, 4, or Q."))
		}
	}
}
