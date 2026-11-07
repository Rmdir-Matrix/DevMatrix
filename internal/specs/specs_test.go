package specs

import "testing"

func TestDetectHardware(t *testing.T) {
	hw := DetectHardware()
	if hw.TotalRAMMB <= 0 {
		t.Fatalf("expected positive TotalRAMMB, got %d", hw.TotalRAMMB)
	}
	if hw.AvailableRAMMB <= 0 || hw.AvailableRAMMB > hw.TotalRAMMB {
		t.Fatalf("invalid AvailableRAMMB %d (total=%d)", hw.AvailableRAMMB, hw.TotalRAMMB)
	}
	if hw.CPUCores < 1 {
		t.Fatalf("expected CPUCores >= 1, got %d", hw.CPUCores)
	}
}

func TestParseRAMInput(t *testing.T) {
	cases := []struct {
		in     string
		wantMB int
		wantOK bool
	}{
		{"512", 512, true},
		{"512MB", 512, true},
		{"1024 mb", 1024, true},
		{"2GB", 2048, true},
		{"1.5gb", 1536, true},
		{"8", 8192, true}, // bare small number <= 32 interpreted as GB
		{"16", 16384, true},
		{"256m", 256, true},
		{"", 0, false},
		{"-100", 0, false},
		{"abc", 0, false},
	}

	for _, tc := range cases {
		got, _, err := ParseRAMInput(tc.in)
		if tc.wantOK && err != nil {
			t.Errorf("ParseRAMInput(%q) unexpected error: %v", tc.in, err)
			continue
		}
		if !tc.wantOK && err == nil {
			t.Errorf("ParseRAMInput(%q) expected error, got %d", tc.in, got)
			continue
		}
		if tc.wantOK && got != tc.wantMB {
			t.Errorf("ParseRAMInput(%q) = %d, want %d", tc.in, got, tc.wantMB)
		}
	}
}

func TestBuildProfileSafetyClamp(t *testing.T) {
	hw := HardwareInfo{
		TotalRAMMB:     4096,
		AvailableRAMMB: 2048,
		CPUCores:       4,
		Detected:       true,
	}

	// Requesting 16 GB on a 4 GB system should clamp down safely to 75% (3072 MB)
	prof, warn := BuildProfile(hw, 16384)
	if prof.RAMBudgetMB != 3072 {
		t.Fatalf("expected safety clamp to 3072 MB, got %d MB", prof.RAMBudgetMB)
	}
	if warn == "" {
		t.Fatalf("expected safety warning when clamping RAM budget")
	}

	// Auto budget (0) should pick a non-overwhelming slice
	autoProf, _ := BuildProfile(hw, 0)
	if autoProf.RAMBudgetMB < MinRAMBudgetMB || autoProf.RAMBudgetMB > 1024 {
		t.Fatalf("unexpected auto budget %d MB for 4 GB system", autoProf.RAMBudgetMB)
	}
}

func TestAutomaticBudgetForFourGB(t *testing.T) {
	for _, available := range []int{4096, 2000, 1000} {
		got := RecommendBudgetFromSystemRAM(4096, available)
		if got > 512 || got < MinRAMBudgetMB {
			t.Fatalf("unsafe automatic budget: %d", got)
		}
	}
}
