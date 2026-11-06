//go:build !linux && !windows

package specs

func detectOSMemoryMB() (totalMB int, availMB int, ok bool) {
	return 0, 0, false
}
