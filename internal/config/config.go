package config

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/Rmdir-Matrix/DevMatrix/internal/specs"
)

const (
	DefaultMatrixName = "Matrix"
	DefaultDataDir    = "matrix_data"
	ConfigFileName    = "config.json"
)

// AppConfig stores the user's hardware optimization settings and active Matrix preferences.
type AppConfig struct {
	ActiveMatrixName string               `json:"active_matrix_name"`
	RAMMode          string               `json:"ram_mode"` // "auto", "system_ram", "custom_cap"
	UserEnteredRAMMB int                  `json:"user_entered_ram_mb,omitempty"`
	Profile          specs.ComputeProfile `json:"profile"`
	SetupCompleted   bool                 `json:"setup_completed"`
	CreatedAt        string               `json:"created_at"`
	UpdatedAt        string               `json:"updated_at"`
}

// ResolveDataDir returns the storage directory for configs and saved Matrix brains.
func ResolveDataDir(override string) (string, error) {
	dir := strings.TrimSpace(override)
	if dir == "" {
		dir = DefaultDataDir
	}
	abs, err := filepath.Abs(dir)
	if err != nil {
		return "", err
	}
	if err := os.MkdirAll(abs, 0o755); err != nil {
		return "", fmt.Errorf("create data dir %s: %w", abs, err)
	}
	return abs, nil
}

// ConfigPath returns the path to config.json inside dataDir.
func ConfigPath(dataDir string) string {
	return filepath.Join(dataDir, ConfigFileName)
}

// Load reads config.json if it exists, or returns a fresh uninitialized AppConfig.
func Load(dataDir string, hw specs.HardwareInfo) (AppConfig, bool, error) {
	path := ConfigPath(dataDir)
	data, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			prof, _ := specs.BuildProfile(hw, 0)
			now := time.Now().UTC().Format(time.RFC3339)
			return AppConfig{
				ActiveMatrixName: DefaultMatrixName,
				RAMMode:          "auto",
				Profile:          prof,
				SetupCompleted:   false,
				CreatedAt:        now,
				UpdatedAt:        now,
			}, false, nil
		}
		return AppConfig{}, false, err
	}

	var cfg AppConfig
	if err := json.Unmarshal(data, &cfg); err != nil {
		return AppConfig{}, false, fmt.Errorf("parse %s: %w", path, err)
	}
	if strings.TrimSpace(cfg.ActiveMatrixName) == "" {
		cfg.ActiveMatrixName = DefaultMatrixName
	}
	// Re-validate profile against current machine so moving folder between PCs stays safe
	budget := cfg.Profile.RAMBudgetMB
	if cfg.RAMMode == "auto" {
		budget = 0
	}
	cfg.Profile, _ = specs.BuildProfile(hw, budget)
	return cfg, cfg.SetupCompleted, nil
}

// Save writes config.json atomically.
func Save(dataDir string, cfg AppConfig) error {
	if err := os.MkdirAll(dataDir, 0o755); err != nil {
		return err
	}
	if strings.TrimSpace(cfg.ActiveMatrixName) == "" {
		cfg.ActiveMatrixName = DefaultMatrixName
	}
	if cfg.CreatedAt == "" {
		cfg.CreatedAt = time.Now().UTC().Format(time.RFC3339)
	}
	cfg.UpdatedAt = time.Now().UTC().Format(time.RFC3339)

	raw, err := json.MarshalIndent(cfg, "", "  ")
	if err != nil {
		return err
	}

	path := ConfigPath(dataDir)
	tmp := path + ".tmp"
	if err := os.WriteFile(tmp, raw, 0o644); err != nil {
		return err
	}
	return os.Rename(tmp, path)
}

// SanitizeMatrixName cleans user input into a friendly display name.
func SanitizeMatrixName(input string) string {
	cleaned := strings.TrimSpace(input)
	if cleaned == "" {
		return DefaultMatrixName
	}
	if len(cleaned) > 36 {
		cleaned = strings.TrimSpace(cleaned[:36])
	}
	return cleaned
}
