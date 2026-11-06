//go:build windows

package specs

import (
	"syscall"
	"unsafe"
)

type memoryStatusEx struct {
	dwLength                uint32
	dwMemoryLoad            uint32
	ullTotalPhys            uint64
	ullAvailPhys            uint64
	ullTotalPageFile        uint64
	ullAvailPageFile        uint64
	ullTotalVirtual         uint64
	ullAvailVirtual         uint64
	ullAvailExtendedVirtual uint64
}

func detectOSMemoryMB() (totalMB int, availMB int, ok bool) {
	k32 := syscall.NewLazyDLL("kernel32.dll")
	proc := k32.NewProc("GlobalMemoryStatusEx")

	var mem memoryStatusEx
	mem.dwLength = uint32(unsafe.Sizeof(mem))

	r1, _, _ := proc.Call(uintptr(unsafe.Pointer(&mem)))
	if r1 == 0 || mem.ullTotalPhys == 0 {
		return 0, 0, false
	}

	totalMB = int(mem.ullTotalPhys / (1024 * 1024))
	availMB = int(mem.ullAvailPhys / (1024 * 1024))
	if totalMB <= 0 {
		return 0, 0, false
	}
	return totalMB, availMB, true
}
