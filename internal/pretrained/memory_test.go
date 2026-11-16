package pretrained

import (
	"errors"
	"os/exec"
	"strings"
	"testing"
	"time"

	"github.com/Rmdir-Matrix/DevMatrix/internal/specs"
)

func TestTrainingAddressLimit(t *testing.T) {
	cases := []struct {
		hw   specs.HardwareInfo
		want int
		fail bool
	}{
		{specs.HardwareInfo{TotalRAMMB: 4096, AvailableRAMMB: 3000, Detected: true}, 1024, false},
		{specs.HardwareInfo{TotalRAMMB: 4096, AvailableRAMMB: 1500, Detected: true}, 732, false},
		{specs.HardwareInfo{TotalRAMMB: 4096, AvailableRAMMB: 1400, Detected: true}, 0, true},
		{specs.HardwareInfo{TotalRAMMB: 1024, AvailableRAMMB: 1800, Detected: true}, 0, true},
		{specs.HardwareInfo{TotalRAMMB: 4096, AvailableRAMMB: 3000, Detected: false}, 0, true},
	}
	for _, c := range cases {
		got, err := trainingAddressLimit(c.hw)
		if (err != nil) != c.fail || got != c.want {
			t.Fatalf("%+v: got %d, %v", c.hw, got, err)
		}
	}
}
func TestMemoryGuardStopsTraining(t *testing.T) {
	tests := []struct {
		name   string
		sample memorySampler
		want   string
	}{
		{"low available", func(int) (int, int, error) { return 500, 30, nil }, "available RAM"},
		{"high RSS", func(int) (int, int, error) { return 2000, 900, nil }, "trainer RAM"},
		{"read failure", func(int) (int, int, error) { return 0, 0, errors.New("failed") }, "memory monitor failed"},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			cmd := exec.Command("sleep", "10")
			start := time.Now()
			err := runTrainingGuarded(cmd, tc.sample, 5*time.Millisecond, trainingRSSLimitMB)
			if err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("got %v", err)
			}
			if time.Since(start) > time.Second {
				t.Fatal("guard did not stop process promptly")
			}
		})
	}
}
func TestMemoryGuardAllowsCompletedProcess(t *testing.T) {
	cmd := exec.Command("true")
	if err := runTrainingGuarded(cmd, func(int) (int, int, error) { return 2000, 50, nil }, time.Millisecond, trainingRSSLimitMB); err != nil {
		t.Fatal(err)
	}
}

func TestSmolTrainingBudget(t *testing.T) {
	cases := []struct {
		avail, address, rss int
		fail                bool
	}{
		{3000, 1024, 768, false}, {1400, 1024, 632, false},
		{1152, 1024, 384, false}, {1151, 0, 0, true},
	}
	for _, c := range cases {
		hw := specs.HardwareInfo{Detected: true, TotalRAMMB: 4096, AvailableRAMMB: c.avail}
		got, err := trainingBudgetFor(hw, Settings{ModelHash: smallModel.SHA})
		if (err != nil) != c.fail || got.addressMB != c.address || got.rssMB != c.rss {
			t.Fatalf("avail %d: %+v %v", c.avail, got, err)
		}
	}
	hw := specs.HardwareInfo{Detected: true, TotalRAMMB: 4096, AvailableRAMMB: 1200}
	if _, err := trainingBudgetFor(hw, Settings{ModelID: smallModel.ID}); err == nil {
		t.Fatal("unverified model received smaller budget")
	}
	if _, err := trainingBudgetFor(specs.HardwareInfo{}, Settings{ModelHash: smallModel.SHA}); err == nil {
		t.Fatal("unknown RAM accepted")
	}
}
func TestDynamicRSSLimit(t *testing.T) {
	cmd := exec.Command("sleep", "10")
	err := runTrainingGuarded(cmd, func(int) (int, int, error) { return 2000, 400, nil }, 5*time.Millisecond, 384)
	if err == nil || !strings.Contains(err.Error(), "limit 384") {
		t.Fatalf("guard: %v", err)
	}
}
