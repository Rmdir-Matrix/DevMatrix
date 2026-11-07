package config

import (
	"bufio"
	"fmt"
	"io"

	"github.com/Rmdir-Matrix/DevMatrix/internal/specs"
	"github.com/Rmdir-Matrix/DevMatrix/internal/ui"
)

// RunSetupWizard guides the user through naming their Matrix and configuring RAM safely.
func RunSetupWizard(reader *bufio.Reader, out io.Writer, theme ui.Theme, hw specs.HardwareInfo, current AppConfig) (AppConfig, error) {
	fmt.Fprintln(out)
	fmt.Fprintln(out, theme.BoldCyan("  === MATRIX INITIALIZATION & HARDWARE SETUP ==="))

	// 1. Show detected specs
	if hw.Detected {
		fmt.Fprintf(out, "  %s OS/Arch: %s/%s  |  CPU Cores: %d\n",
			theme.Green("✓"), hw.OS, hw.Arch, hw.CPUCores)
		fmt.Fprintf(out, "  %s System RAM Detected: %s total  (%s currently available)\n",
			theme.Green("✓"),
			theme.Bold(specs.FormatRAM(hw.TotalRAMMB)),
			specs.FormatRAM(hw.AvailableRAMMB),
		)
	} else {
		fmt.Fprintf(out, "  %s Hardware auto-probe limited (%s/%s, %d CPU cores). Safe defaults ready.\n",
			theme.Yellow("!"), hw.OS, hw.Arch, hw.CPUCores)
	}
	fmt.Fprintln(out)

	// 2. Ask for Matrix name
	defaultName := current.ActiveMatrixName
	if defaultName == "" {
		defaultName = DefaultMatrixName
	}
	nameAns, err := ui.Prompt(
		reader,
		out,
		fmt.Sprintf("  Name your study buddy [%s]: ", theme.BoldGreen(defaultName)),
	)
	if err != nil && err != io.EOF {
		return current, err
	}
	chosenName := defaultName
	if nameAns != "" {
		chosenName = SanitizeMatrixName(nameAns)
	}

	// 3. Ask for RAM optimization mode
	autoProfile, _ := specs.BuildProfile(hw, 0)
	fmt.Fprintln(out)
	fmt.Fprintln(out, theme.Bold("  How should "+chosenName+" manage RAM with a conservative target?"))
	fmt.Fprintf(out, "    %s Auto-Optimize from detected specs %s\n",
		theme.BoldGreen("[1]"),
		theme.Dim(fmt.Sprintf("(Recommended: %d MB target • %s Tier • %d neurons)",
			autoProfile.RAMBudgetMB, autoProfile.TierName, autoProfile.HiddenNeurons)),
	)
	fmt.Fprintf(out, "    %s Enter my PC's total RAM %s\n",
		theme.BoldCyan("[2]"),
		theme.Dim("(Matrix will calculate a safe ~15-20% budget automatically)"),
	)
	fmt.Fprintf(out, "    %s Set a RAM target for Matrix %s\n",
		theme.Yellow("[3]"),
		theme.Dim("(e.g. 256MB, 512MB, 1024MB, 2GB)"),
	)

	var ramMode string
	var userEnteredMB int
	var chosenProfile specs.ComputeProfile

	for {
		choice, err := ui.Prompt(reader, out, "  Select option [1]: ")
		if err != nil && err != io.EOF {
			return current, err
		}
		if choice == "" || choice == "1" {
			ramMode = "auto"
			userEnteredMB = 0
			chosenProfile = autoProfile
			break
		}
		if choice == "2" {
			mb, note, pErr := askRAMValue(reader, out, theme, "  Enter your PC's total RAM (e.g. 4GB, 8GB, 16GB): ")
			if pErr != nil {
				if pErr == io.EOF {
					ramMode = "auto"
					chosenProfile = autoProfile
					break
				}
				continue
			}
			if note != "" {
				fmt.Fprintln(out, "  "+theme.Dim(note))
			}
			safeBudget := specs.RecommendBudgetFromSystemRAM(mb, int(float64(mb)*0.65))
			prof, warn := specs.BuildProfile(hw, safeBudget)
			if warn != "" {
				fmt.Fprintln(out, "  "+theme.Yellow("Note: "+warn))
			}
			ramMode = "system_ram"
			userEnteredMB = mb
			chosenProfile = prof
			break
		}
		if choice == "3" {
			mb, note, pErr := askRAMValue(reader, out, theme, "  Enter RAM target for Matrix (e.g. 256MB, 512MB, 1024MB, 2GB): ")
			if pErr != nil {
				if pErr == io.EOF {
					ramMode = "auto"
					chosenProfile = autoProfile
					break
				}
				continue
			}
			if note != "" {
				fmt.Fprintln(out, "  "+theme.Dim(note))
			}
			prof, warn := specs.BuildProfile(hw, mb)
			if warn != "" {
				fmt.Fprintln(out, "  "+theme.Yellow("Note: "+warn))
			}
			ramMode = "custom_cap"
			userEnteredMB = mb
			chosenProfile = prof
			break
		}
		fmt.Fprintln(out, "  "+theme.Red("Please enter 1, 2, or 3."))
	}

	updated := current
	updated.ActiveMatrixName = chosenName
	updated.RAMMode = ramMode
	updated.UserEnteredRAMMB = userEnteredMB
	updated.Profile = chosenProfile
	updated.SetupCompleted = true

	fmt.Fprintln(out, "  "+theme.Dim("Memory target is soft; total process usage may exceed it."))
	fmt.Fprintln(out)
	fmt.Fprintf(out, "  %s %s configured! Tier: %s | RAM Target: %d MB | Hidden Neurons: %d | CPU Threads: %d\n",
		theme.BoldGreen("✓"),
		theme.Bold(chosenName),
		theme.BoldCyan(chosenProfile.TierName),
		chosenProfile.RAMBudgetMB,
		chosenProfile.HiddenNeurons,
		chosenProfile.WorkerThreads,
	)
	return updated, nil
}

func askRAMValue(reader *bufio.Reader, out io.Writer, theme ui.Theme, label string) (int, string, error) {
	for {
		ans, err := ui.Prompt(reader, out, label)
		if err != nil {
			return 0, "", err
		}
		mb, note, pErr := specs.ParseRAMInput(ans)
		if pErr != nil {
			fmt.Fprintln(out, "  "+theme.Red(pErr.Error()))
			continue
		}
		return mb, note, nil
	}
}
