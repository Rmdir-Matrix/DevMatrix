package main

import (
	"bufio"
	"flag"
	"fmt"
	"io"
	"os"
	"runtime"
	"strings"

	"github.com/Rmdir-Matrix/DevMatrix/internal/config"
	"github.com/Rmdir-Matrix/DevMatrix/internal/platform"
	"github.com/Rmdir-Matrix/DevMatrix/internal/specs"
	"github.com/Rmdir-Matrix/DevMatrix/internal/ui"
)

func main() {
	dataDirFlag := flag.String("data-dir", "", "Directory for Matrix saves and configuration (default: ./matrix_data)")
	forceSetup := flag.Bool("setup", false, "Run the interactive RAM & Matrix Name setup wizard on startup")
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

	runMainMenu(reader, os.Stdout, theme, dataDir, hw, cfg)
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

func printStatusBar(out io.Writer, theme ui.Theme, hw specs.HardwareInfo, cfg config.AppConfig) {
	var mem runtime.MemStats
	runtime.ReadMemStats(&mem)
	usedMB := float64(mem.Alloc) / (1024.0 * 1024.0)

	fmt.Fprintln(out)
	fmt.Fprintf(out, "  %s %s  |  %s %s (%d MB cap, %.1f MB in use)  |  %s %d neurons, %d threads\n",
		theme.Dim("Buddy:"), theme.BoldGreen(cfg.ActiveMatrixName),
		theme.Dim("Spec Tier:"), theme.BoldCyan(cfg.Profile.TierName), cfg.Profile.RAMBudgetMB, usedMB,
		theme.Dim("Brain Spec:"), cfg.Profile.HiddenNeurons, cfg.Profile.WorkerThreads,
	)
	fmt.Fprintln(out, theme.Dim("  ─────────────────────────────────────────────────────────────────────────"))
}

func runMainMenu(reader *bufio.Reader, out io.Writer, theme ui.Theme, dataDir string, hw specs.HardwareInfo, cfg config.AppConfig) {
	for {
		printStatusBar(out, theme, hw, cfg)
		fmt.Fprintf(out, "    %s Training Mode %s\n", theme.BoldGreen("[1]"), theme.Dim("(Read real articles & train AI neural connections)"))
		fmt.Fprintf(out, "    %s Chat Mode     %s\n", theme.BoldCyan("[2]"), theme.Dim("(Talk & study with "+cfg.ActiveMatrixName+")"))
		fmt.Fprintf(out, "    %s Share / Export%s\n", theme.Magenta("[3]"), theme.Dim("(Export, import, or switch named Matrixes)"))
		fmt.Fprintf(out, "    %s Setup & Specs %s\n", theme.Yellow("[4]"), theme.Dim("(View hardware diagnostics or change RAM / Name)"))
		fmt.Fprintf(out, "    %s Quit\n\n", theme.Dim("[Q]"))

		choice, err := ui.Prompt(reader, out, "  Select mode: ")
		if err != nil {
			fmt.Fprintln(out)
			return
		}

		switch strings.ToLower(choice) {
		case "1", "t", "train":
			fmt.Fprintln(out, "  "+theme.Yellow("→ Training Mode & AI Brain Engine are queued next in Steps 2 & 3!"))
		case "2", "c", "chat":
			fmt.Fprintln(out, "  "+theme.Yellow("→ Study Buddy Chat Mode is queued in Step 4!"))
		case "3", "e", "export", "share":
			fmt.Fprintln(out, "  "+theme.Yellow("→ Matrix Export/Import Sharing is queued in Step 4!"))
		case "4", "s", "setup", "specs":
			printSpecsDiagnostic(out, theme, hw)
			reCfg, rErr := ui.Prompt(reader, out, "\n  Re-run setup wizard to change Name or RAM limit? [y/N]: ")
			if rErr == nil && (strings.EqualFold(reCfg, "y") || strings.EqualFold(reCfg, "yes")) {
				updated, wErr := config.RunSetupWizard(reader, out, theme, hw, cfg)
				if wErr == nil {
					cfg = updated
					_ = config.Save(dataDir, cfg)
					specs.ApplyRuntimeLimits(cfg.Profile)
				}
			}
		case "q", "quit", "exit":
			fmt.Fprintf(out, "\n  %s Powering down %s. See you next study session!\n\n",
				theme.BoldGreen("⚡"), cfg.ActiveMatrixName)
			return
		default:
			fmt.Fprintln(out, "  "+theme.Red("Unknown command. Choose 1, 2, 3, 4, or Q."))
		}
	}
}
