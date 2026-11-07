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

// RunInteractive starts Mode 2: interactive Closed-Book GPT Study Buddy Chat Mode.
// 100% of responses are generated directly from the Multi-Layer GPT Transformer weights — zero lookup tables!
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
	fmt.Fprintln(out, theme.BoldCyan("  === MODE 2: CLOSED-BOOK GPT CHAT MODE (100% WEIGHTS • ZERO LOOKUP) ==="))
	fmt.Fprintf(out, "  %s Connected to %s (Level %d • %s • %dL/%dH GPT • %d words • %d synapses)\n",
		theme.Cyan("•"), theme.BoldGreen(b.Name), b.Level,
		trainer.IntelligenceStage(b.Level, dictsDone, dictsTotal, grammarDone, grammarTotal),
		b.Net.NumLayers, b.Net.NumHeads, b.Net.LearnedWordCount(), b.Net.ActiveSynapses())
	fmt.Fprintln(out, "  "+theme.Dim("Tools: /define <word> • /study <topic> • /quiz [topic] • remember: <fact> • /clear • /brain • /train • /back"))
	fmt.Fprintln(out)

	var history []brain.ChatTurn

	if initReply, known, unknownReason := b.GenerateModelReplyWithHistory(history, "hello"); known {
		printBuddyReply(out, theme, b.Name, b.Level, initReply)
		history = append(history, brain.ChatTurn{User: "hello", Assistant: initReply})
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
			fmt.Fprintf(out, "  %s Returning to Main Menu (GPT weights saved).\n", theme.Green("✓"))
			return false, nil
		}
		if lower == "/train" {
			_ = b.Save(dataDir)
			return true, nil
		}
		if lower == "/clear" {
			history = nil
			fmt.Fprintf(out, "  %s Cleared conversation context window.\n", theme.Green("✓"))
			continue
		}
		if strings.HasPrefix(lower, "/help") {
			printChatHelp(out, theme, b.Name)
			continue
		}
		if strings.HasPrefix(lower, "/brain") {
			fmt.Fprintf(out, "  %s Level %d | GPT: %d Layers × %d Heads (%dD → %d FFN) | Vocab: %d/%d | Synapses: %d | Loss: %.4f\n",
				theme.Cyan("[GPT Stats]"), b.Level, b.Net.NumLayers, b.Net.NumHeads, b.Net.EmbedDim,
				b.Net.HiddenNeurons, b.Net.LearnedWordCount(), b.Net.MaxVocab, b.Net.ActiveSynapses(), b.SmoothedLoss)
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

		reply, known, unknownReason := RespondWithHistory(b, &history, userMsg)
		if known {
			printBuddyReply(out, theme, b.Name, b.Level, reply)
		} else {
			printUnknownReply(out, theme, b.Name, b.Level, unknownReason)
		}
		_ = b.Save(dataDir)
	}
}

// Respond runs a stateless closed-book GPT response (or trains weights on explicit remember:/teach: input).
func Respond(b *brain.Brain, userInput string) (reply string, known bool, unknownReason string) {
	var h []brain.ChatTurn
	return RespondWithHistory(b, &h, userInput)
}

// RespondWithHistory maintains a sliding ChatGPT-style conversation window and generates
// 100% of answers directly from the GPT Transformer weights (zero lookup databases!).
func RespondWithHistory(b *brain.Brain, history *[]brain.ChatTurn, userInput string) (reply string, known bool, unknownReason string) {
	trimmed := strings.TrimSpace(userInput)
	lower := strings.ToLower(trimmed)

	for _, prefix := range []string{"remember:", "teach:", "note:", "learn:"} {
		if strings.HasPrefix(lower, prefix) {
			fact := strings.TrimSpace(trimmed[len(prefix):])
			if len(strings.Fields(fact)) < 3 {
				return "", false, "Provide a full sentence after '" + prefix + "' so I can train my GPT weights on it."
			}
			b.TrainOnArticle(corpus.Article{
				Title:   "User Taught Fact",
				Topic:   "Personal Study Notes",
				Source:  "chat:user-taught",
				Content: fact,
			}, 4)
			kws := brain.ExtractKeywords(fact, 2)
			probe := fact
			if len(kws) > 0 {
				probe = "explain " + kws[0]
			}
			gen, ok, msg := b.GenerateModelReply(probe)
			if ok {
				return gen, true, ""
			}
			return "", false, msg
		}
	}

	var histSlice []brain.ChatTurn
	if history != nil {
		histSlice = *history
	}
	reply, known, unknownReason = b.GenerateModelReplyWithHistory(histSlice, trimmed)
	if known {
		b.AwardStudyXP(6)
		if history != nil {
			*history = append(*history, brain.ChatTurn{User: trimmed, Assistant: reply})
			if len(*history) > 4 {
				*history = (*history)[len(*history)-4:]
			}
		}
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
	return fmt.Sprintf("Trained GPT weights on %d items (%d words in vocabulary, %d active synapses). Domains: %s. Recent: %s.",
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
	rep := b.TrainOnArticle(art, 4)
	_ = b.Save(dataDir)
	fmt.Fprintf(out, "     %s Trained GPT weights on %q (%d words vocab • loss %.3f • Lv.%d)\n",
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
	fmt.Fprintf(out, "  %s %s is reading and training GPT layers on %q...\n", theme.Cyan("📖"), b.Name, topic)
	ctx := context.Background()
	for i := 0; i < 2; i++ {
		art, err := cr.NextArticle(ctx, topic, b.SeenSources)
		if err != nil {
			break
		}
		rep := b.TrainOnArticle(art, 3)
		fmt.Fprintf(out, "     %s Trained GPT weights on %q (%d words vocab • loss %.3f • Lv.%d)\n",
			theme.Green("✓"), rep.ArticleTitle, rep.VocabSize, rep.SmoothedLoss, rep.NewLevel)
	}
	_ = b.Save(dataDir)

	if gen, ok, reason := b.GenerateModelReply("explain " + topic); ok {
		printBuddyReply(out, theme, b.Name, b.Level, gen)
	} else {
		printUnknownReply(out, theme, b.Name, b.Level, reason)
	}
}

func runInteractiveQuiz(reader *bufio.Reader, out io.Writer, theme ui.Theme, b *brain.Brain, dataDir, topicHint string) {
	concept, ok := b.PickQuizConcept(topicHint)
	if !ok {
		printUnknownReply(out, theme, b.Name, b.Level,
			"My GPT vocabulary is still small. Train me on dictionaries, grammar, and articles first!")
		return
	}

	fmt.Fprintln(out)
	fmt.Fprintf(out, "  %s Explain or define %q in your own words:\n",
		theme.BoldGreen("🧠 CLOSED-BOOK RECALL QUIZ"), concept)
	guess, err := ui.Prompt(reader, out, "  Your Explanation > ")
	if err != nil || strings.TrimSpace(guess) == "" {
		return
	}

	modelAns, mOK, _ := b.GenerateModelReply("explain " + concept)
	if !mOK {
		modelAns, _, _ = b.GenerateModelReply("define " + concept)
	}
	b.AwardStudyXP(30)
	_ = b.Save(dataDir)
	fmt.Fprintf(out, "  %s +30 Study XP! %s's GPT recall on %q: %s\n",
		theme.BoldGreen("✓"), b.Name, concept, modelAns)
}

func printChatHelp(out io.Writer, theme ui.Theme, name string) {
	fmt.Fprintln(out)
	fmt.Fprintln(out, theme.BoldCyan("  Closed-Book GPT Chat Guide (100% Transformer Weights • Zero Lookup):"))
	fmt.Fprintln(out, "    • Talk naturally     — Every word is generated strictly from "+name+"'s Multi-Layer GPT weights")
	fmt.Fprintln(out, "    • remember: <fact>   — Train "+name+"'s GPT layers directly on a new sentence")
	fmt.Fprintln(out, "    • /define <word>     — Fetch & train "+name+"'s weights on a dictionary definition")
	fmt.Fprintln(out, "    • /study <topic>     — Fetch & train "+name+"'s weights on articles about <topic>")
	fmt.Fprintln(out, "    • /quiz [topic]      — Active-recall concept check against "+name+"'s GPT weights")
	fmt.Fprintln(out, "    • /clear             — Clear the current chat context window")
	fmt.Fprintln(out, "    • /summary • /brain  — View studied topics or GPT Transformer telemetry")
	fmt.Fprintln(out, "    • /train • /back     — Switch to Training Mode or return to Main Menu")
}

func printBuddyReply(out io.Writer, theme ui.Theme, name string, level int, text string) {
	tag := fmt.Sprintf("  %s [Lv.%d] > ", name, level)
	fmt.Fprintf(out, "%s%s\n", theme.BoldGreen(tag), text)
}

func printUnknownReply(out io.Writer, theme ui.Theme, name string, level int, reason string) {
	tag := fmt.Sprintf("  %s [Lv.%d • Uncertain] > ", name, level)
	fmt.Fprintf(out, "%s%s\n", theme.Yellow(tag), reason)
}
