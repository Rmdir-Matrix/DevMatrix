package specs

import (
	"fmt"
	"math"
	"runtime"
	"runtime/debug"
	"strconv"
	"strings"
)

const (
	MinRAMBudgetMB     = 64
	DefaultAutoMaxMB   = 2048
	FallbackTotalRAMMB = 4096
	FallbackAvailRAMMB = 2048
)

// HardwareInfo holds detected host hardware specifications.
type HardwareInfo struct {
	TotalRAMMB     int    `json:"total_ram_mb"`
	AvailableRAMMB int    `json:"available_ram_mb"`
	CPUCores       int    `json:"cpu_cores"`
	OS             string `json:"os"`
	Arch           string `json:"arch"`
	Detected       bool   `json:"detected"`
}

// ComputeProfile defines the safe memory budget and neural model sizing derived from specs.
type ComputeProfile struct {
	TierName      string `json:"tier_name"`
	RAMBudgetMB   int    `json:"ram_budget_mb"`
	EmbedDim      int    `json:"embed_dim"`
	HiddenNeurons int    `json:"hidden_neurons"`
	MaxVocab      int    `json:"max_vocab"`
	MaxPassages   int    `json:"max_passages"`
	BatchSize     int    `json:"batch_size"`
	WorkerThreads int    `json:"worker_threads"`
}

// DetectHardware probes the current machine's RAM and CPU core count.
func DetectHardware() HardwareInfo {
	cores := runtime.NumCPU()
	if cores < 1 {
		cores = 1
	}

	totalMB, availMB, ok := detectOSMemoryMB()
	if !ok || totalMB <= 0 {
		totalMB = FallbackTotalRAMMB
		availMB = FallbackAvailRAMMB
		ok = false
	}
	if availMB <= 0 || availMB > totalMB {
		availMB = totalMB / 2
	}

	return HardwareInfo{
		TotalRAMMB:     totalMB,
		AvailableRAMMB: availMB,
		CPUCores:       cores,
		OS:             runtime.GOOS,
		Arch:           runtime.GOARCH,
		Detected:       ok,
	}
}

// RecommendBudgetFromSystemRAM calculates a safe memory ceiling (~20% of available RAM,
// or ~15% of total RAM) so Matrix never overwhelms the host system.
func RecommendBudgetFromSystemRAM(totalRAMMB, availRAMMB int) int {
	base := int(float64(totalRAMMB) * 0.15)
	if availRAMMB > 0 {
		fromAvail := int(float64(availRAMMB) * 0.22)
		if fromAvail < base || base == 0 {
			base = fromAvail
		}
	}

	// Round to clean 32 MB steps
	base = (base / 32) * 32
	if base < MinRAMBudgetMB {
		base = MinRAMBudgetMB
	}
	if base > DefaultAutoMaxMB {
		base = DefaultAutoMaxMB
	}
	return base
}

// BuildProfile maps a target RAM budget (clamped safely against host hardware) and CPU count
// into concrete AI training & memory capacities.
func BuildProfile(hw HardwareInfo, requestedBudgetMB int) (ComputeProfile, string) {
	warning := ""
	budget := requestedBudgetMB
	if budget <= 0 {
		budget = RecommendBudgetFromSystemRAM(hw.TotalRAMMB, hw.AvailableRAMMB)
	}

	if budget < MinRAMBudgetMB {
		warning = fmt.Sprintf("Requested %d MB is below minimum; raised to %d MB.", budget, MinRAMBudgetMB)
		budget = MinRAMBudgetMB
	}

	// Safety guard: never allow budget to exceed 75% of detected total RAM (or 80% of available if reasonable)
	if hw.Detected && hw.TotalRAMMB > 0 {
		safeCap := int(float64(hw.TotalRAMMB) * 0.75)
		if safeCap < MinRAMBudgetMB {
			safeCap = MinRAMBudgetMB
		}
		if budget > safeCap {
			warning = fmt.Sprintf(
				"Capped RAM budget from %d MB down to %d MB (75%% of detected %s RAM) to protect your system.",
				budget, safeCap, FormatRAM(hw.TotalRAMMB),
			)
			budget = safeCap
		}
	}

	cores := hw.CPUCores
	if cores < 1 {
		cores = 1
	}

	switch {
	case budget <= 128:
		return ComputeProfile{
			TierName:      "Featherweight",
			RAMBudgetMB:   budget,
			EmbedDim:      24,
			HiddenNeurons: 48,
			MaxVocab:      4096,
			MaxPassages:   400,
			BatchSize:     8,
			WorkerThreads: 1,
		}, warning

	case budget <= 384:
		return ComputeProfile{
			TierName:      "Lite",
			RAMBudgetMB:   budget,
			EmbedDim:      32,
			HiddenNeurons: 96,
			MaxVocab:      8192,
			MaxPassages:   1200,
			BatchSize:     16,
			WorkerThreads: minInt(cores, 2),
		}, warning

	case budget <= 1024:
		return ComputeProfile{
			TierName:      "Balanced",
			RAMBudgetMB:   budget,
			EmbedDim:      48,
			HiddenNeurons: 160,
			MaxVocab:      16384,
			MaxPassages:   3500,
			BatchSize:     24,
			WorkerThreads: minInt(cores, 4),
		}, warning

	case budget <= 2048:
		return ComputeProfile{
			TierName:      "Performance",
			RAMBudgetMB:   budget,
			EmbedDim:      64,
			HiddenNeurons: 256,
			MaxVocab:      30000,
			MaxPassages:   8000,
			BatchSize:     32,
			WorkerThreads: minInt(cores, 6),
		}, warning

	default:
		return ComputeProfile{
			TierName:      "Ultra",
			RAMBudgetMB:   budget,
			EmbedDim:      96,
			HiddenNeurons: 384,
			MaxVocab:      50000,
			MaxPassages:   15000,
			BatchSize:     48,
			WorkerThreads: minInt(cores, 8),
		}, warning
	}
}

// ApplyRuntimeLimits enforces the compute profile's RAM ceiling and thread count in the Go runtime.
func ApplyRuntimeLimits(profile ComputeProfile) {
	if profile.WorkerThreads > 0 {
		runtime.GOMAXPROCS(profile.WorkerThreads)
	}
	if profile.RAMBudgetMB >= MinRAMBudgetMB {
		// Leave a small 10% headroom inside the soft memory limit so Go GC sweeps before hitting the ceiling
		softLimitBytes := int64(float64(profile.RAMBudgetMB) * 0.90 * 1024 * 1024)
		debug.SetMemoryLimit(softLimitBytes)
	}
}

// ParseRAMInput parses human-friendly memory strings like "512", "512MB", "8GB", "1.5 gb", or "8" (bare <=32 treated as GB).
// Returns (megabytes, interpretedNote, error).
func ParseRAMInput(raw string) (int, string, error) {
	s := strings.TrimSpace(strings.ToLower(raw))
	if s == "" {
		return 0, "", fmt.Errorf("empty RAM input")
	}

	multiplier := 1.0
	hasUnit := false

	switch {
	case strings.HasSuffix(s, "gib"):
		s = strings.TrimSpace(strings.TrimSuffix(s, "gib"))
		multiplier = 1024.0
		hasUnit = true
	case strings.HasSuffix(s, "gb"):
		s = strings.TrimSpace(strings.TrimSuffix(s, "gb"))
		multiplier = 1024.0
		hasUnit = true
	case strings.HasSuffix(s, "g"):
		s = strings.TrimSpace(strings.TrimSuffix(s, "g"))
		multiplier = 1024.0
		hasUnit = true
	case strings.HasSuffix(s, "mib"):
		s = strings.TrimSpace(strings.TrimSuffix(s, "mib"))
		multiplier = 1.0
		hasUnit = true
	case strings.HasSuffix(s, "mb"):
		s = strings.TrimSpace(strings.TrimSuffix(s, "mb"))
		multiplier = 1.0
		hasUnit = true
	case strings.HasSuffix(s, "m"):
		s = strings.TrimSpace(strings.TrimSuffix(s, "m"))
		multiplier = 1.0
		hasUnit = true
	}

	val, err := strconv.ParseFloat(s, 64)
	if err != nil || math.IsNaN(val) || math.IsInf(val, 0) || val <= 0 {
		return 0, "", fmt.Errorf("invalid RAM value %q (try e.g. 512MB, 1024MB, or 8GB)", raw)
	}

	note := ""
	// If user entered a bare small number like "4", "8", "16", or "32", interpret as GB
	if !hasUnit && val <= 32 {
		mb := int(math.Round(val * 1024.0))
		note = fmt.Sprintf("Interpreted %q as %.0f GB (%d MB).", strings.TrimSpace(raw), val, mb)
		return mb, note, nil
	}

	mb := int(math.Round(val * multiplier))
	if mb <= 0 {
		return 0, "", fmt.Errorf("RAM value must be greater than 0 MB")
	}
	return mb, note, nil
}

// FormatRAM formats megabytes nicely (e.g. "512 MB" or "15.8 GB (16192 MB)").
func FormatRAM(mb int) string {
	if mb >= 1024 {
		return fmt.Sprintf("%.1f GB (%d MB)", float64(mb)/1024.0, mb)
	}
	return fmt.Sprintf("%d MB", mb)
}

func minInt(a, b int) int {
	if a < b {
		return a
	}
	return b
}
