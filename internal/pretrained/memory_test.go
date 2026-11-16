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
			err := runTrainingGuarded(cmd, tc.sample, 5*time.Millisecond)
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
	if err := runTrainingGuarded(cmd, func(int) (int, int, error) { return 2000, 50, nil }, time.Millisecond); err != nil {
		t.Fatal(err)
	}
}
