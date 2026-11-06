package chat

import (
	"bufio"
	"context"
	"fmt"
	"io"
	"strings"

	"github.com/Rmdir-Matrix/DevMatrix/internal/brain"
	"github.com/Rmdir-Matrix/DevMatrix/internal/corpus"
	"github.com/Rmdir-Matrix/DevMatrix/internal/trainer"
	"github.com/Rmdir-Matrix/DevMatrix/internal/ui"
)

// RunInteractive starts Mode 2: interactive Study Buddy Chat Mode.
// All conversational and study answers are decoded directly from the trained model,
// except when the model does not know the words/concepts yet.
func RunInteractive(
	reader *bufio.Reader,
	out io.Writer,
	theme ui.Theme,
	b *brain.Brain,
	cr *corpus.Reader,
	dataDir string,
) (bool, error) {
	dictsDone := cr.DictionaryCompleted(b.SeenSources)
	dictsTotal := cr.DictionaryCount()
	grammarDone := cr.GrammarCompleted(b.SeenSources)
	grammarTotal := cr.GrammarCount()

	fmt.Fprintln(out)
	fmt.Fprintln(out, theme.BoldCyan("  === MODE 2: STUDY BUDDY CHAT MODE (PURE MODEL OUTPUT) ==="))
	fmt.Fprintf(out, "  %s Connected to %s (Level %d • %s • %d words learned • %d synapses)\n",
		theme.Cyan("•"), theme.BoldGreen(b.Name), b.Level,
		trainer.IntelligenceStage(b.Level, dictsDone, dictsTotal, grammarDone, grammarTotal),
		b.Net.LearnedWordCount(), b.Net.ActiveSynapses())
	fmt.Fprintln(out, "  "+theme.Dim("Tools: /define <word> • /study <topic> • /quiz [topic] • remember: <fact> • /summary • /brain • /train • /back"))
	fmt.Fprintln(out)

	// Initial prompt uses the model itself (or reports honestly if untrained)
	if initReply, known, unknownReason := b.GenerateModelReply("hello"); known {
		printBuddyReply(out, theme, b.Name, b.Level, initReply)
	} else {
		printUnknownReply(out, theme, b.Name, b.Level, unknownReason)
	}

	for {
		fmt.Fprintln(out)
		userMsg, err := ui.Prompt(reader, out, theme.Bold("  You > "))
		if err != nil {
			_ = b.Save(dataDir)
			return false, nil
		}
		if userMsg == "" {
			continue
		}

		lower := strings.ToLower(strings.TrimSpace(userMsg))
		if lower == "/back" || lower == "/exit" || lower == "/quit" || lower == "exit" || lower == "quit" {
			_ = b.Save(dataDir)
			fmt.Fprintf(out, "  %s Returning to Main Menu (progress saved).\n", theme.Green("✓"))
			return false, nil
		}
		if lower == "/train" {
			_ = b.Save(dataDir)
			return true, nil
		}
		if strings.HasPrefix(lower, "/help") {
			printChatHelp(out, theme, b.Name)
			continue
		}
		if strings.HasPrefix(lower, "/brain") {
			fmt.Fprintf(out, "  %s Level %d | Words Learned: %d/%d | Hidden Neurons: %d | Active Synapses: %d | Items Read: %d | Loss: %.4f\n",
				theme.Cyan("[Brain Stats]"), b.Level, b.Net.LearnedWordCount(), b.Net.MaxVocab,
				b.Net.HiddenNeurons, b.Net.ActiveSynapses(), b.ArticlesRead, b.SmoothedLoss)
			continue
		}
		if strings.HasPrefix(lower, "/summary") || strings.HasPrefix(lower, "/topics") {
			fmt.Fprintf(out, "  %s %s\n", theme.Cyan("[Study Summary]"), BuildSummaryReport(b))
			continue
		}
		if strings.HasPrefix(lower, "/define") {
			word := strings.TrimSpace(userMsg[len("/define"):])
			runQuickDictionaryStudy(out, theme, b, cr, dataDir, word)
			continue
		}
		if strings.HasPrefix(lower, "/study") {
			topic := strings.TrimSpace(userMsg[len("/study"):])
			runQuickChatStudy(out, theme, b, cr, dataDir, topic)
			continue
		}
		if strings.HasPrefix(lower, "/quiz") {
			topic := strings.TrimSpace(userMsg[len("/quiz"):])
			runInteractiveQuiz(reader, out, theme, b, dataDir, topic)
			continue
		}

		reply, known, unknownReason := Respond(b, userMsg)
		if known {
			printBuddyReply(out, theme, b.Name, b.Level, reply)
		} else {
			printUnknownReply(out, theme, b.Name, b.Level, unknownReason)
		}
		_ = b.Save(dataDir)
	}
}

// Respond either trains on an explicit "remember:/teach:/note:" fact, or decodes an answer
// strictly from the trained neural language model (returning known=false if unknown).
func Respond(b *brain.Brain, userInput string) (reply string, known bool, unknownReason string) {
	trimmed := strings.TrimSpace(userInput)
	lower := strings.ToLower(trimmed)

	// Allow user to explicitly teach a new sentence into the neural network
	for _, prefix := range []string{"remember:", "teach:", "note:", "learn:"} {
		if strings.HasPrefix(lower, prefix) {
			fact := strings.TrimSpace(trimmed[len(prefix):])
			if len(strings.Fields(fact)) < 3 {
				return "", false, "Provide a full sentence after '" + prefix + "' so I can train my weights on it."
			}
			b.TrainOnArticle(corpus.Article{
				Title:   "User Taught Fact",
				Topic:   "Personal Study Notes",
				Source:  "chat:user-taught",
				Content: fact,
			}, 3)
			// Decode confirmation or echo from the newly trained fact itself!
			kws := brain.ExtractKeywords(fact, 3)
			probe := fact
			if len(kws) > 0 {
				probe = strings.Join(kws, " ")
			}
			gen, ok, msg := b.GenerateModelReply(probe)
			if ok {
				return gen, true, ""
			}
			return "", false, msg
		}
	}

	reply, known, unknownReason = b.GenerateModelReply(trimmed)
	if known {
		b.AwardStudyXP(6)
	}
	return reply, known, unknownReason
}

// BuildSummaryReport returns telemetry of studied domains and recent titles.
func BuildSummaryReport(b *brain.Brain) string {
	if b.ArticlesRead == 0 {
		return "0 lessons or articles studied yet."
	}
	topics := b.TopTopics(5)
	recent := b.RecentTitles
	if len(recent) > 4 {
		recent = recent[len(recent)-4:]
	}
	return fmt.Sprintf("Studied %d items (%d words in vocabulary, %d active synapses). Domains: %s. Recent: %s.",
		b.ArticlesRead, b.Net.LearnedWordCount(), b.Net.ActiveSynapses(),
		strings.Join(topics, ", "), strings.Join(recent, " • "))
}

func runQuickDictionaryStudy(out io.Writer, theme ui.Theme, b *brain.Brain, cr *corpus.Reader, dataDir, word string) {
	if word == "" {
		fmt.Fprintln(out, "  "+theme.Yellow("Usage: /define <word>  (for example: /define hypothesis)"))
		return
	}
	art, err := cr.FetchDictionaryEntry(context.Background(), word)
	if err != nil {
		printUnknownReply(out, theme, b.Name, b.Level,
			fmt.Sprintf("Could not find a dictionary entry for %q (%v).", word, err))
		return
	}
	rep := b.TrainOnArticle(art, 3)
	_ = b.Save(dataDir)
	fmt.Fprintf(out, "     %s Trained on %q (%d words vocab • loss %.3f • Lv.%d)\n",
		theme.Green("✓"), rep.ArticleTitle, rep.VocabSize, rep.SmoothedLoss, rep.NewLevel)

	if gen, ok, reason := b.GenerateModelReply("define " + word); ok {
		printBuddyReply(out, theme, b.Name, b.Level, gen)
	} else {
		printUnknownReply(out, theme, b.Name, b.Level, reason)
	}
}

func runQuickChatStudy(out io.Writer, theme ui.Theme, b *brain.Brain, cr *corpus.Reader, dataDir, topic string) {
	if topic == "" {
		fmt.Fprintln(out, "  "+theme.Yellow("Usage: /study <topic>  (for example: /study Thermodynamics)"))
		return
	}
	fmt.Fprintf(out, "  %s %s is reading and training on %q...\n", theme.Cyan("📖"), b.Name, topic)
	ctx := context.Background()
	for i := 0; i < 2; i++ {
		art, err := cr.NextArticle(ctx, topic, b.SeenSources)
		if err != nil {
			break
		}
		rep := b.TrainOnArticle(art, 2)
		fmt.Fprintf(out, "     %s Trained on %q (%d words vocab • loss %.3f • Lv.%d)\n",
			theme.Green("✓"), rep.ArticleTitle, rep.VocabSize, rep.SmoothedLoss, rep.NewLevel)
	}
	_ = b.Save(dataDir)

	if gen, ok, reason := b.GenerateModelReply(topic); ok {
		printBuddyReply(out, theme, b.Name, b.Level, gen)
	} else {
		printUnknownReply(out, theme, b.Name, b.Level, reason)
	}
}

func runInteractiveQuiz(reader *bufio.Reader, out io.Writer, theme ui.Theme, b *brain.Brain, dataDir, topicHint string) {
	p, answer, masked, ok := b.PickQuizPassage(topicHint)
	if !ok {
		printUnknownReply(out, theme, b.Name, b.Level,
			"I haven't studied enough domain articles yet to build a quiz. Complete the grammar lessons and read some articles first!")
		return
	}

	fmt.Fprintln(out)
	fmt.Fprintf(out, "  %s %s [%s — %s]\n", theme.BoldGreen("🧠 ACTIVE RECALL QUIZ"), theme.Dim("from"), p.Topic, p.Title)
	fmt.Fprintf(out, "     Fill in the missing concept:\n     %s\n", theme.BoldCyan(masked))
	guess, err := ui.Prompt(reader, out, "  Your Answer > ")
	if err != nil {
		return
	}

	if strings.EqualFold(strings.TrimSpace(guess), answer) {
		b.AwardStudyXP(45)
		_ = b.Save(dataDir)
		fmt.Fprintf(out, "  %s Correct: %q (+45 XP)\n", theme.BoldGreen("✓"), answer)
	} else {
		b.AwardStudyXP(15)
		_ = b.Save(dataDir)
		fmt.Fprintf(out, "  %s Expected: %q  |  Source sentence: %s\n", theme.Yellow("•"), answer, p.Text)
	}
}

func printChatHelp(out io.Writer, theme ui.Theme, name string) {
	fmt.Fprintln(out)
	fmt.Fprintln(out, theme.BoldCyan("  Study Buddy Chat Guide (100% Model-Generated Replies):"))
	fmt.Fprintln(out, "    • Talk naturally     — Every answer is decoded word-by-word from "+name+"'s trained model")
	fmt.Fprintln(out, "                           (unless "+name+" hasn't learned those words/concepts yet!)")
	fmt.Fprintln(out, "    • remember: <fact>   — Train "+name+"'s weights directly on a new sentence")
	fmt.Fprintln(out, "    • /study <topic>     — Fetch & train on real articles about <topic>")
	fmt.Fprintln(out, "    • /quiz [topic]      — Active-recall fill-in-the-blank quiz from learned passages")
	fmt.Fprintln(out, "    • /summary • /brain  — View studied topics or neural network telemetry")
	fmt.Fprintln(out, "    • /train • /back     — Switch to Training Mode or return to Main Menu")
}

func printBuddyReply(out io.Writer, theme ui.Theme, name string, level int, text string) {
	tag := fmt.Sprintf("  %s [Lv.%d] > ", name, level)
	fmt.Fprintf(out, "%s%s\n", theme.BoldGreen(tag), text)
}

func printUnknownReply(out io.Writer, theme ui.Theme, name string, level int, reason string) {
	tag := fmt.Sprintf("  %s [Lv.%d • Unknown] > ", name, level)
	fmt.Fprintf(out, "%s%s\n", theme.Yellow(tag), reason)
}
