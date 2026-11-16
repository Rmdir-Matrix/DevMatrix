//go:build linux

package specs

import (
	"bufio"
	"os"
	"strconv"
	"strings"
	"syscall"
)

func detectOSMemoryMB() (totalMB int, availMB int, ok bool) {
	if f, err := os.Open("/proc/meminfo"); err == nil {
		defer f.Close()

		var memTotalKB, memAvailKB, memFreeKB, buffersKB, cachedKB int64
		scanner := bufio.NewScanner(f)
		for scanner.Scan() {
			line := scanner.Text()
			fields := strings.Fields(line)
			if len(fields) < 2 {
				continue
			}
			val, err := strconv.ParseInt(fields[1], 10, 64)
			if err != nil {
				continue
			}
			switch fields[0] {
			case "MemTotal:":
				memTotalKB = val
			case "MemAvailable:":
				memAvailKB = val
			case "MemFree:":
				memFreeKB = val
			case "Buffers:":
				buffersKB = val
			case "Cached:":
				cachedKB = val
			}
		}

		if memTotalKB > 0 {
			if memAvailKB <= 0 {
				memAvailKB = memFreeKB + buffersKB + cachedKB
			}
			totalMB = int(memTotalKB / 1024)
			availMB = int(memAvailKB / 1024)

			// A container may have much less memory available than the host reports.
			// Use limit minus current usage (conservatively ignoring reclaimable cache).
			cgMaxMB := readCgroupMemoryMaxMB()
			cgCurrentMB := readCgroupMemoryCurrentMB()
			totalMB, availMB = capCgroupMemory(totalMB, availMB, cgMaxMB, cgCurrentMB)
			return totalMB, availMB, true
		}
	}

	var info syscall.Sysinfo_t
	if err := syscall.Sysinfo(&info); err == nil && info.Totalram > 0 {
		unit := uint64(info.Unit)
		if unit == 0 {
			unit = 1
		}
		totalMB = int((uint64(info.Totalram) * unit) / (1024 * 1024))
		availMB = int((uint64(info.Freeram+info.Bufferram) * unit) / (1024 * 1024))
		return totalMB, availMB, totalMB > 0
	}

	return 0, 0, false
}

func readCgroupMemoryMaxMB() int {
	data, err := os.ReadFile("/sys/fs/cgroup/memory.max")
	if err != nil {
		return 0
	}
	s := strings.TrimSpace(string(data))
	if s == "" || s == "max" {
		return 0
	}
	bytesVal, err := strconv.ParseUint(s, 10, 64)
	if err != nil || bytesVal == 0 || bytesVal > (1<<50) {
		return 0
	}
	return int(bytesVal / (1024 * 1024))
}

// capCgroupMemory applies a cgroup v2 limit and current usage to host values.
func capCgroupMemory(total, available, limit, used int) (int, int) {
	if limit <= 0 {
		return total, available
	}
	if total > limit {
		total = limit
	}
	remaining := limit / 2 // conservative fallback if current usage is unavailable
	if used >= 0 {
		remaining = limit - used
	}
	if remaining < 1 {
		remaining = 1
	} // don't trigger DetectHardware's invalid-value fallback
	if available > remaining {
		available = remaining
	}
	if available > total {
		available = total
	}
	return total, available
}
func readCgroupMemoryCurrentMB() int {
	data, err := os.ReadFile("/sys/fs/cgroup/memory.current")
	if err != nil {
		return -1
	}
	val, err := strconv.ParseUint(strings.TrimSpace(string(data)), 10, 64)
	if err != nil || val > 1<<50 {
		return -1
	}
	return int((val + 1024*1024 - 1) / (1024 * 1024))
}
