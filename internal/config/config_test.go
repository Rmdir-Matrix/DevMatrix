package config

import (
	"bufio"
	"bytes"
	"strings"
	"testing"

	"github.com/Rmdir-Matrix/DevMatrix/internal/specs"
	"github.com/Rmdir-Matrix/DevMatrix/internal/ui"
)

func TestConfigSaveAndLoad(t *testing.T) {
	dir := t.TempDir()
	hw := specs.HardwareInfo{
		TotalRAMMB:     8192,
		AvailableRAMMB: 4096,
		CPUCores:       4,
		OS:             "linux",
		Arch:           "amd64",
		Detected:       true,
	}

	cfg, exists, err := Load(dir, hw)
	if err != nil {
		t.Fatalf("Load fresh error: %v", err)
	}
	if exists {
		t.Fatalf("expected fresh config SetupCompleted=false")
	}
	if cfg.ActiveMatrixName != DefaultMatrixName {
		t.Fatalf("expected default name %q, got %q", DefaultMatrixName, cfg.ActiveMatrixName)
	}

	prof, _ := specs.BuildProfile(hw, 512)
	cfg.ActiveMatrixName = "NeoBuddy"
	cfg.RAMMode = "custom_cap"
	cfg.UserEnteredRAMMB = 512
	cfg.Profile = prof
	cfg.SetupCompleted = true

	if err := Save(dir, cfg); err != nil {
		t.Fatalf("Save error: %v", err)
	}

	loaded, exists2, err := Load(dir, hw)
	if err != nil {
		t.Fatalf("Load saved error: %v", err)
	}
	if !exists2 {
		t.Fatalf("expected SetupCompleted=true")
	}
	if loaded.ActiveMatrixName != "NeoBuddy" {
		t.Fatalf("expected NeoBuddy, got %q", loaded.ActiveMatrixName)
	}
	if loaded.Profile.RAMBudgetMB != 512 {
		t.Fatalf("expected 512 MB, got %d", loaded.Profile.RAMBudgetMB)
	}
}

func TestRunSetupWizardAutoAndCustom(t *testing.T) {
	hw := specs.HardwareInfo{
		TotalRAMMB:     16384,
		AvailableRAMMB: 8192,
		CPUCores:       8,
		OS:             "windows",
		Arch:           "amd64",
		Detected:       true,
	}
	initial := AppConfig{ActiveMatrixName: DefaultMatrixName}

	// Simulate user pressing Enter for default name ("Matrix") and Enter for [1] Auto-Optimize
	inAuto := bufio.NewReader(strings.NewReader("\n\n"))
	var outAuto bytes.Buffer
	cfgAuto, err := RunSetupWizard(inAuto, &outAuto, ui.NewTheme(false), hw, initial)
	if err != nil {
		t.Fatalf("RunSetupWizard auto error: %v", err)
	}
	if !cfgAuto.SetupCompleted || cfgAuto.ActiveMatrixName != "Matrix" || cfgAuto.RAMMode != "auto" {
		t.Fatalf("unexpected auto wizard result: %+v", cfgAuto)
	}

	// Simulate user entering custom name "StudyBot", option 2 (PC RAM), and "8GB"
	inSys := bufio.NewReader(strings.NewReader("StudyBot\n2\n8GB\n"))
	var outSys bytes.Buffer
	cfgSys, err := RunSetupWizard(inSys, &outSys, ui.NewTheme(false), hw, initial)
	if err != nil {
		t.Fatalf("RunSetupWizard system_ram error: %v", err)
	}
	if cfgSys.ActiveMatrixName != "StudyBot" || cfgSys.RAMMode != "system_ram" || cfgSys.UserEnteredRAMMB != 8192 {
		t.Fatalf("unexpected system_ram wizard result: %+v", cfgSys)
	}
}

