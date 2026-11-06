//go:build windows

package platform

import (
	"os"
	"syscall"
	"unsafe"
)

const enableVirtualTerminalProcessing uint32 = 0x0004

var (
	kernel32       = syscall.NewLazyDLL("kernel32.dll")
	procGetMode    = kernel32.NewProc("GetConsoleMode")
	procSetMode    = kernel32.NewProc("SetConsoleMode")
	procSetOutCP   = kernel32.NewProc("SetConsoleOutputCP")
	procSetInCP    = kernel32.NewProc("SetConsoleCP")
)

// InitConsole enables UTF-8 codepage and ANSI Virtual Terminal colors on Windows consoles.
// Returns true if ANSI color sequences are supported.
func InitConsole() bool {
	if os.Getenv("NO_COLOR") != "" {
		return false
	}

	// Enable UTF-8 output/input code page (65001) for clean box-drawing and symbols
	_, _, _ = procSetOutCP.Call(65001)
	_, _, _ = procSetInCP.Call(65001)

	stdoutSupported := enableVT(os.Stdout.Fd())
	_ = enableVT(os.Stderr.Fd())

	if stdoutSupported {
		return true
	}
	// Windows Terminal / ConEmu / VSCode terminal even when stdout handle mode check fails
	if os.Getenv("WT_SESSION") != "" || os.Getenv("TERM_PROGRAM") != "" || os.Getenv("TERM") != "" {
		return true
	}
	return false
}

func enableVT(fd uintptr) bool {
	var mode uint32
	r1, _, _ := procGetMode.Call(fd, uintptr(unsafe.Pointer(&mode)))
	if r1 == 0 {
		return false
	}
	if mode&enableVirtualTerminalProcessing != 0 {
		return true
	}
	r2, _, _ := procSetMode.Call(fd, uintptr(mode|enableVirtualTerminalProcessing))
	return r2 != 0
}
