package trainer

import (
	"bufio"
	"context"
	"fmt"
	"io"
	"os"
	"os/signal"
	"strings"
	"time"

	"github.com/Rmdir-Matrix/DevMatrix/internal/brain"
	"github.com/Rmdir-Matrix/DevMatrix/internal/corpus"
	"github.com/Rmdir-Matrix/DevMatrix/internal/ui"
)

// Options configures a Training Mode session.
type Options struct {
	TopicFilter string
	MaxArticles int           // 0 = run continuously until user presses Enter or Ctrl+C
	StepDelay   time.Duration // visual pacing between articles in interactive mode
	DataDir     string
}

var majorMilestones = []int{1, 5, 10, 25, 50, 75, 100, 150, 200, 300, 500}

// RunInteractive launches Easy Training Mode from the CLI menu.
// Returns switchToChat=true if the user typed "chat" or "c" while pausing training.
func RunInteractive(
	reader *bufio.Reader,
	out io.Writer,
	theme ui.Theme,
	b *brain.Brain,
	cr *corpus.Reader,
	dataDir string,
	autoStart bool,
) (bool, error) {
	topic := ""
	grammarDone := cr.GrammarCompleted(b.SeenSources)
	grammarTotal := cr.GrammarCount()

	if !autoStart {
		fmt.Fprintln(out)
		fmt.Fprintln(out, theme.BoldGreen("  === MODE 1: EASY AI TRAINING MODE ==="))
		fmt.Fprintf(out, "  %s Stage: %s (Level %d • Grammar %d/%d • %d learned words • %d synapses)\n",
			theme.Green("•"),
			theme.BoldCyan(IntelligenceStage(b.Level, grammarDone, grammarTotal)),
			b.Level, grammarDone, grammarTotal, b.Net.LearnedWordCount(), b.Net.ActiveSynapses())
		if grammarDone < grammarTotal {
			fmt.Fprintf(out, "  %s Curriculum: %s learns core words & grammar FIRST (%d/%d), then real articles!\n",
				theme.Green("•"), b.Name, grammarDone, grammarTotal)
		} else {
			fmt.Fprintf(out, "  %s Sources: Live Wikipedia articles + Local notes (%s) + Built-in Encyclopedia\n",
				theme.Green("•"), theme.Dim(cr.NotesDir()))
		}

		ans, err := ui.Prompt(reader, out, "  Focus topic (press Enter for Words/Grammar → Real Articles Auto-Study): ")
		if err != nil && err != io.EOF {
			return false, err
		}
		topic = strings.TrimSpace(ans)
	}

	fmt.Fprintln(out)
	if topic == "" {
		fmt.Fprintf(out, "  %s %s is powered ON and training! %s\n\n",
			theme.BoldGreen("▶"), theme.Bold(b.Name),
			theme.Yellow("(Press [Enter] to pause, or type 'chat' + [Enter] to talk to "+b.Name+")"))
	} else {
		fmt.Fprintf(out, "  %s %s is studying %q! %s\n\n",
			theme.BoldGreen("▶"), theme.Bold(b.Name), topic,
			theme.Yellow("(Press [Enter] to pause, or type 'chat' + [Enter] to talk to "+b.Name+")"))
	}

	stopCh := make(chan struct{})
	cmdCh := make(chan string, 1)
	go func() {
		line, _ := reader.ReadString('\n')
		cmdCh <- strings.ToLower(strings.TrimSpace(line))
		close(stopCh)
	}()

	err := RunLoop(out, theme, b, cr, Options{
		TopicFilter: topic,
		MaxArticles: 0,
		StepDelay:   380 * time.Millisecond,
		DataDir:     dataDir,
	}, stopCh)

	select {
	case cmd := <-cmdCh:
		if cmd == "chat" || cmd == "c" || cmd == "2" {
			return true, err
		}
	default:
	}
	return false, err
}

// RunLoop executes the curriculum + real article-reading and neural training loop.
func RunLoop(
	out io.Writer,
	theme ui.Theme,
	b *brain.Brain,
	cr *corpus.Reader,
	opts Options,
	stopCh <-chan struct{},
) error {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, os.Interrupt)
	defer signal.Stop(sigCh)

	sessionArticles := 0
	sessionStartLevel := b.Level
	spokenFirst := false

	for {
		if opts.MaxArticles > 0 && sessionArticles >= opts.MaxArticles {
			break
		}

		select {
		case <-stopCh:
			goto finish
		case <-sigCh:
			fmt.Fprintln(out, "\n  "+theme.Yellow("Interrupt received — saving neural weights safely..."))
			goto finish
		default:
		}

		art, err := cr.NextArticle(ctx, opts.TopicFilter, b.SeenSources)
		if err != nil {
			fmt.Fprintln(out, "  "+theme.Red("Article reader error: "+err.Error()))
			break
		}

		wasFreshStart := b.ArticlesRead == 0 && !spokenFirst
		epochs := 2
		if strings.HasPrefix(art.Source, "curriculum:") {
			epochs = 3 // Extra pass on core grammar & vocabulary lessons
		}
		rep := b.TrainOnArticle(art, epochs)
		sessionArticles++

		badge := sourceBadge(theme, rep.Source)
		fmt.Fprintf(out, "  %s %-42s  %s\n",
			badge,
			truncateTitle(rep.ArticleTitle, 42),
			theme.Dim(fmt.Sprintf("(%d words vocab • loss %.3f • %d synapses • Lv.%d)",
				rep.VocabSize, rep.SmoothedLoss, rep.ActiveSynapses, rep.NewLevel)),
		)

		crossed := detectCrossedMilestone(rep.OldLevel, rep.NewLevel)
		if wasFreshStart {
			spokenFirst = true
			if msg := GenerateTrainingThought(b, rep); msg != "" {
				printMatrixThought(out, theme, b.Name, 1, msg)
			}
		} else if crossed > 1 {
			if msg := GenerateTrainingThought(b, rep); msg != "" {
				printMatrixThought(out, theme, b.Name, crossed, msg)
			}
		} else if sessionArticles%3 == 0 {
			if msg := GenerateTrainingThought(b, rep); msg != "" {
				printMatrixThought(out, theme, b.Name, rep.NewLevel, msg)
			}
		}

		if opts.DataDir != "" && sessionArticles%5 == 0 {
			_ = b.Save(opts.DataDir)
		}

		if opts.StepDelay > 0 {
			select {
			case <-stopCh:
				goto finish
			case <-sigCh:
				fmt.Fprintln(out, "\n  "+theme.Yellow("Interrupt received — saving neural weights safely..."))
				goto finish
			case <-time.After(opts.StepDelay):
			}
		}
	}

finish:
	if opts.DataDir != "" {
		if err := b.Save(opts.DataDir); err != nil {
			return err
		}
	}
	fmt.Fprintln(out)
	fmt.Fprintf(out, "  %s Training paused & saved! Completed %d items (Lv.%d → Lv.%d • %d words learned • %d active neuron connections).\n",
		theme.BoldGreen("✓"), sessionArticles, sessionStartLevel, b.Level, b.Net.LearnedWordCount(), b.Net.ActiveSynapses())
	return nil
}

func detectCrossedMilestone(oldLvl, newLvl int) int {
	if newLvl <= oldLvl {
		return 0
	}
	highest := 0
	for _, m := range majorMilestones {
		if oldLvl < m && newLvl >= m {
			highest = m
		}
	}
	return highest
}

func printMatrixThought(out io.Writer, theme ui.Theme, name string, level int, message string) {
	prefix := fmt.Sprintf("💬 [%s • Level %d]:", name, level)
	fmt.Fprintf(out, "     %s %s\n", theme.BoldGreen(prefix), theme.BoldCyan(fmt.Sprintf("%q", message)))
}

func sourceBadge(theme ui.Theme, src string) string {
	switch {
	case strings.HasPrefix(src, "curriculum:"):
		return theme.Yellow("[GRAMMAR]")
	case strings.HasPrefix(src, "http://"), strings.HasPrefix(src, "https://"), strings.HasPrefix(src, "wikipedia:"):
		return theme.BoldCyan("[WEB]    ")
	case strings.HasPrefix(src, "local:"):
		return theme.Magenta("[NOTES]  ")
	default:
		return theme.Green("[ARTICLE]")
	}
}

func truncateTitle(s string, maxLen int) string {
	runes := []rune(strings.TrimSpace(s))
	if len(runes) <= maxLen {
		return fmt.Sprintf("%q", string(runes))
	}
	return fmt.Sprintf("%q", string(runes[:maxLen-1])+"…")
}
