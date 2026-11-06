package trainer

import (
	"bufio"
	"context"
	"fmt"
	"io"
	"math/rand"
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
func RunInteractive(
	reader *bufio.Reader,
	out io.Writer,
	theme ui.Theme,
	b *brain.Brain,
	cr *corpus.Reader,
	dataDir string,
	autoStart bool,
) error {
	topic := ""
	if !autoStart {
		fmt.Fprintln(out)
		fmt.Fprintln(out, theme.BoldGreen("  === MODE 1: EASY AI TRAINING MODE ==="))
		fmt.Fprintf(out, "  %s Stage: %s (Level %d • %d active neuron connections)\n",
			theme.Green("•"), theme.BoldCyan(IntelligenceStage(b.Level)), b.Level, b.Net.ActiveSynapses())
		fmt.Fprintf(out, "  %s Sources: Live Wikipedia articles + Local notes (%s) + Built-in Encyclopedia\n",
			theme.Green("•"), theme.Dim(cr.NotesDir()))

		ans, err := ui.Prompt(reader, out, "  Focus topic (press Enter for Everything / Easy Auto-Study): ")
		if err != nil && err != io.EOF {
			return err
		}
		topic = strings.TrimSpace(ans)
	}

	fmt.Fprintln(out)
	if topic == "" {
		fmt.Fprintf(out, "  %s %s is powered ON and reading real articles! %s\n\n",
			theme.BoldGreen("▶"), theme.Bold(b.Name), theme.Yellow("(Press [Enter] anytime to pause & return)"))
	} else {
		fmt.Fprintf(out, "  %s %s is studying %q! %s\n\n",
			theme.BoldGreen("▶"), theme.Bold(b.Name), topic, theme.Yellow("(Press [Enter] anytime to pause & return)"))
	}

	stopCh := make(chan struct{})
	go func() {
		_, _ = reader.ReadString('\n')
		close(stopCh)
	}()

	return RunLoop(out, theme, b, cr, Options{
		TopicFilter: topic,
		MaxArticles: 0,
		StepDelay:   380 * time.Millisecond,
		DataDir:     dataDir,
	}, stopCh)
}

// RunLoop executes the real article-reading and neural training loop until stopped or MaxArticles reached.
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

	rng := rand.New(rand.NewSource(time.Now().UnixNano()))
	sessionArticles := 0
	sessionStartLevel := b.Level
	spokenFirstLevel1 := false

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

		wasFreshLevel1 := b.Level == 1 && b.ArticlesRead == 0 && !spokenFirstLevel1
		rep := b.TrainOnArticle(art, 2)
		sessionArticles++

		badge := sourceBadge(theme, rep.Source)
		fmt.Fprintf(out, "  %s Read %-42s  %s\n",
			badge,
			truncateTitle(rep.ArticleTitle, 42),
			theme.Dim(fmt.Sprintf("(%d tokens • loss %.3f • %d synapses • Lv.%d)",
				rep.TokensRead, rep.SmoothedLoss, rep.ActiveSynapses, rep.NewLevel)),
		)

		// Determine if Matrix should speak a level-appropriate thought right now
		crossed := detectCrossedMilestone(rep.OldLevel, rep.NewLevel)
		if wasFreshLevel1 {
			spokenFirstLevel1 = true
			// Emit the signature Level 1 line first!
			lvl1Rep := rep
			lvl1Rep.NewLevel = 1
			msg := GenerateTrainingThought(b, lvl1Rep, 1, rng)
			printMatrixThought(out, theme, b.Name, 1, msg)
		}

		if crossed > 1 {
			mRep := rep
			mRep.NewLevel = crossed
			msg := GenerateTrainingThought(b, mRep, crossed, rng)
			printMatrixThought(out, theme, b.Name, crossed, msg)
		} else if !wasFreshLevel1 && (sessionArticles%3 == 0 || rep.NewLevel > rep.OldLevel && sessionArticles%2 == 0) {
			msg := GenerateTrainingThought(b, rep, 0, rng)
			printMatrixThought(out, theme, b.Name, rep.NewLevel, msg)
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
	fmt.Fprintf(out, "  %s Training paused & saved! Studied %d articles this session (Lv.%d → Lv.%d • %d active neuron connections).\n",
		theme.BoldGreen("✓"), sessionArticles, sessionStartLevel, b.Level, b.Net.ActiveSynapses())
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
	case strings.HasPrefix(src, "http://"), strings.HasPrefix(src, "https://"), strings.HasPrefix(src, "wikipedia:"):
		return theme.BoldCyan("[WEB]   ")
	case strings.HasPrefix(src, "local:"):
		return theme.Magenta("[NOTES] ")
	default:
		return theme.Green("[STUDY] ")
	}
}

func truncateTitle(s string, maxLen int) string {
	runes := []rune(strings.TrimSpace(s))
	if len(runes) <= maxLen {
		return fmt.Sprintf("%q", string(runes))
	}
	return fmt.Sprintf("%q", string(runes[:maxLen-1])+"…")
}
