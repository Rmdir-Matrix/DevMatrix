package pretrained

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/Rmdir-Matrix/DevMatrix/internal/specs"
)

// These are conservative guardrails, not guarantees of total system memory.
// RSS includes mapped model pages but excludes other processes; address-space
// limits include mappings and can prevent a run even with enough physical RAM.
const trainingReserveMB = 768
const trainingRSSLimitMB = 768

// Bound virtual memory and refuse to start when there is insufficient headroom.
func trainingAddressLimit(hw specs.HardwareInfo) (int, error) {
	if !hw.Detected {
		return 0, fmt.Errorf("cannot check available system memory; training is disabled for safety")
	}
	if hw.AvailableRAMMB < trainingReserveMB+640 {
		return 0, fmt.Errorf("not enough available RAM (%d MB); close programs to free at least %d MB", hw.AvailableRAMMB, trainingReserveMB+640)
	}
	limit := 1024
	if hw.AvailableRAMMB-trainingReserveMB < limit {
		limit = hw.AvailableRAMMB - trainingReserveMB
	}
	if hw.TotalRAMMB/2 < limit {
		limit = hw.TotalRAMMB / 2
	}
	if limit < 640 {
		return 0, fmt.Errorf("not enough total RAM for low-memory training")
	}
	return limit, nil
}

// The catalog's exact SmolLM2 weights get a separate policy; manually chosen
// files cannot opt in by name alone. Virtual address space is not resident RAM.
type trainingBudget struct {
	addressMB int
	rssMB     int
}

func trainingBudgetFor(hw specs.HardwareInfo, s Settings) (trainingBudget, error) {
	if s.ModelHash != smallModel.SHA {
		limit, err := trainingAddressLimit(hw)
		return trainingBudget{addressMB: limit, rssMB: trainingRSSLimitMB}, err
	}
	if !hw.Detected {
		return trainingBudget{}, fmt.Errorf("cannot check available system memory; training is disabled for safety")
	}
	const minTrainerRSS = 384
	if hw.AvailableRAMMB < trainingReserveMB+minTrainerRSS {
		return trainingBudget{}, fmt.Errorf("not enough available RAM (%d MiB); SmolLM2 needs at least %d MiB available (including %d MiB system reserve)", hw.AvailableRAMMB, trainingReserveMB+minTrainerRSS, trainingReserveMB)
	}
	addressMB := 1024
	if hw.TotalRAMMB/2 < addressMB {
		addressMB = hw.TotalRAMMB / 2
	}
	if addressMB < 640 {
		return trainingBudget{}, fmt.Errorf("not enough total RAM for low-memory training")
	}
	rssMB := hw.AvailableRAMMB - trainingReserveMB
	if rssMB > trainingRSSLimitMB {
		rssMB = trainingRSSLimitMB
	}
	return trainingBudget{addressMB: addressMB, rssMB: rssMB}, nil
}

type memorySampler func(pid int) (availableMB, rssMB int, err error)

func readTrainingMemory(pid int) (int, int, error) {
	hw := specs.DetectHardware()
	if !hw.Detected {
		return 0, 0, fmt.Errorf("cannot check available RAM")
	}
	contents, err := os.ReadFile(filepath.Join("/proc", strconv.Itoa(pid), "status"))
	if err != nil {
		return 0, 0, err
	}
	for _, line := range strings.Split(string(contents), "\n") {
		fields := strings.Fields(line)
		if len(fields) >= 2 && fields[0] == "VmRSS:" {
			kb, e := strconv.ParseUint(fields[1], 10, 64)
			if e != nil {
				return 0, 0, e
			}
			return hw.AvailableRAMMB, int((kb + 1023) / 1024), nil
		}
	}
	return 0, 0, fmt.Errorf("cannot read trainer memory usage")
}

func runTrainingGuarded(cmd *exec.Cmd, sample memorySampler, interval time.Duration, rssLimitMB int) error {
	if err := cmd.Start(); err != nil {
		return err
	}
	done, finished := make(chan struct{}), make(chan struct{})
	reason := make(chan error, 1)
	go func() {
		defer close(finished)
		ticker := time.NewTicker(interval)
		defer ticker.Stop()
		for {
			select {
			case <-done:
				return
			case <-ticker.C:
				available, rss, err := sample(cmd.Process.Pid)
				if os.IsNotExist(err) {
					return
				} // process has just exited
				var stop error
				if err != nil {
					stop = fmt.Errorf("memory monitor failed: %w", err)
				} else if available < trainingReserveMB {
					stop = fmt.Errorf("available RAM dropped to %d MB (reserve %d MB)", available, trainingReserveMB)
				} else if rss > rssLimitMB {
					stop = fmt.Errorf("trainer RAM reached %d MB (limit %d MB)", rss, rssLimitMB)
				}
				if stop != nil {
					if cmd.Process.Kill() == nil {
						reason <- stop
					}
					return
				}
			}
		}
	}()
	err := cmd.Wait()
	close(done)
	<-finished
	select {
	case why := <-reason:
		return why
	default:
		return err
	}
}
