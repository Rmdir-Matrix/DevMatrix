package ui

import (
	"bufio"
	"fmt"
	"io"
	"strings"
)

// Theme controls terminal color formatting with graceful plain-text fallback.
type Theme struct {
	ColorEnabled bool
}

func NewTheme(colorEnabled bool) Theme {
	return Theme{ColorEnabled: colorEnabled}
}

func (t Theme) wrap(code, text string) string {
	if !t.ColorEnabled || text == "" {
		return text
	}
	return "\x1b[" + code + "m" + text + "\x1b[0m"
}

func (t Theme) Green(s string) string     { return t.wrap("38;5;46", s) }
func (t Theme) BoldGreen(s string) string { return t.wrap("1;38;5;46", s) }
func (t Theme) Cyan(s string) string      { return t.wrap("38;5;51", s) }
func (t Theme) BoldCyan(s string) string  { return t.wrap("1;38;5;51", s) }
func (t Theme) Yellow(s string) string    { return t.wrap("38;5;220", s) }
func (t Theme) Magenta(s string) string   { return t.wrap("38;5;213", s) }
func (t Theme) Red(s string) string       { return t.wrap("1;31", s) }
func (t Theme) Bold(s string) string      { return t.wrap("1", s) }
func (t Theme) Dim(s string) string       { return t.wrap("2", s) }

// Banner renders the Matrix Study Buddy CLI header.
func (t Theme) Banner(matrixName string) string {
	art := []string{
		"  ███╗   ███╗ █████╗ ████████╗██████╗ ██╗██╗  ██╗",
		"  ████╗ ████║██╔══██╗╚══██╔══╝██╔══██╗██║╚██╗██╔╝",
		"  ██╔████╔██║███████║   ██║   ██████╔╝██║ ╚███╔╝ ",
		"  ██║╚██╔╝██║██╔══██║   ██║   ██╔══██╗██║ ██╔██╗ ",
		"  ██║ ╚═╝ ██║██║  ██║   ██║   ██║  ██║██║██╔╝ ██╗",
		"  ╚═╝     ╚═╝╚═╝  ╚═╝   ╚═╝   ╚═╝  ╚═╝╚═╝╚═╝  ╚═╝",
	}
	var b strings.Builder
	b.WriteString("\n")
	for _, line := range art {
		b.WriteString(t.BoldGreen(line) + "\n")
	}
	subtitle := fmt.Sprintf("  [ AI Study Buddy CLI  •  Active Buddy: %s ]", matrixName)
	b.WriteString(t.Cyan(subtitle) + "\n")
	b.WriteString(t.Dim("  ─────────────────────────────────────────────────────") + "\n")
	return b.String()
}

// Prompt prints a prompt string and reads one trimmed line from reader.
func Prompt(reader *bufio.Reader, out io.Writer, label string) (string, error) {
	fmt.Fprint(out, label)
	line, err := reader.ReadString('\n')
	if err != nil && err != io.EOF {
		return "", err
	}
	trimmed := strings.TrimRight(line, "\r\n")
	if err == io.EOF && trimmed == "" {
		return "", io.EOF
	}
	return strings.TrimSpace(trimmed), nil
}
