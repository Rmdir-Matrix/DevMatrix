//go:build !windows

package platform

import "os"

// InitConsole checks whether ANSI colors should be enabled on Unix-like terminals.
func InitConsole() bool {
	if os.Getenv("NO_COLOR") != "" {
		return false
	}
	if os.Getenv("TERM") == "dumb" {
		return false
	}
	return true
}
