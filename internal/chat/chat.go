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
// Returns switchToTrain=true if the user types /train inside chat.
func RunInteractive(
	reader *bufio.Reader,
	out io.Writer,
	theme ui.Theme,
	b *brain.Brain,
	cr *corpus.Reader,
	dataDir string,
) (bool, error) {
	fmt.Fprintln(out)
	fmt.Fprintln(out, theme.BoldCyan("  === MODE 2: STUDY BUDDY CHAT MODE ==="))
	fmt.Fprintf(out, "  %s Connected to %s (Level %d • %s • %d synapses • %d articles read)\n",
		theme.Cyan("•"), theme.BoldGreen(b.Name), b.Level,
		trainer.IntelligenceStage(b.Level), b.Net.ActiveSynapses(), b.ArticlesRead)
	fmt.Fprintln(out, "  "+theme.Dim("Commands: /quiz [topic] • /study <topic> • /summary • /brain • /train • /back"))
	fmt.Fprintln(out)

	printBuddyReply(out, theme, b.Name, b.Level, OpeningGreeting(b))

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

		if strings.HasPrefix(lower, "/help") || lower == "help" {
			printChatHelp(out, theme, b.Name)
			continue
		}

		if strings.HasPrefix(lower, "/brain") {
			reply := fmt.Sprintf(
				"Brain Telemetry — Level %d (%s) | XP: %d | Vocabulary: %d/%d tokens | Hidden Neurons: %d | Active Synapses: %d | Articles Studied: %d | Smoothed Loss: %.4f.",
				b.Level, trainer.IntelligenceStage(b.Level), b.XP,
				b.Net.VocabSize(), b.Net.MaxVocab, b.Net.HiddenNeurons,
				b.Net.ActiveSynapses(), b.ArticlesRead, b.SmoothedLoss,
			)
			printBuddyReply(out, theme, b.Name, b.Level, reply)
			continue
		}

		if strings.HasPrefix(lower, "/summary") || strings.HasPrefix(lower, "/topics") {
			printBuddyReply(out, theme, b.Name, b.Level, BuildSummaryReply(b))
			continue
		}

		if strings.HasPrefix(lower, "/study") {
			topic := strings.TrimSpace(userMsg[len("/study"):])
			runQuickChatStudy(out, theme, b, cr, dataDir, topic)
			continue
		}

		if strings.HasPrefix(lower, "/quiz") || lower == "quiz me" {
			topic := ""
			if strings.HasPrefix(lower, "/quiz") {
				topic = strings.TrimSpace(userMsg[len("/quiz"):])
			}
			runInteractiveQuiz(reader, out, theme, b, dataDir, topic)
			continue
		}

		reply := Respond(b, userMsg)
		printBuddyReply(out, theme, b.Name, b.Level, reply)
		_ = b.Save(dataDir)
	}
}

// OpeningGreeting generates Matrix's initial chat greeting based on its current training level.
func OpeningGreeting(b *brain.Brain) string {
	lvl := b.Level
	switch {
	case lvl <= 4:
		if b.ArticlesRead == 0 {
			return fmt.Sprintf("hi!! %s is Level %d baby brain... me read 0 articles yet! Train me in Mode 1 (or type /study science) so %s is learn now!", b.Name, lvl, b.Name)
		}
		return fmt.Sprintf("hi friend! %s is Level %d! me read %d article so far... ask me simple word or train me more!", b.Name, lvl, b.ArticlesRead)
	case lvl <= 19:
		return fmt.Sprintf("Hey! I'm %s (Level %d) — I'm learning stuff! I've read %d articles so far. Ask me about what I read, or type /quiz to study together!", b.Name, lvl, b.ArticlesRead)
	case lvl <= 99:
		topics := b.TopTopics(3)
		topicStr := "general science and humanities"
		if len(topics) > 0 {
			topicStr = strings.Join(topics, ", ")
		}
		return fmt.Sprintf("Welcome back! I'm at Level %d with %d active neuron connections across %s. What should we review, explore, or quiz today?",
			lvl, b.Net.ActiveSynapses(), topicStr)
	default:
		return fmt.Sprintf("Greetings, study partner. At Level %d, my neural network holds %d active synaptic connections across %d studied articles. Ask me anything from my knowledge graph, teach me a new note, or run /quiz for active recall.",
			lvl, b.Net.ActiveSynapses(), b.ArticlesRead)
	}
}

// Respond generates a level-calibrated study buddy response to a user message,
// and trains on direct user teachings when prefixed with remember:/teach:/note:.
func Respond(b *brain.Brain, userInput string) string {
	trimmed := strings.TrimSpace(userInput)
	lower := strings.ToLower(trimmed)

	// 1. Direct teaching from the user in chat
	for _, prefix := range []string{"remember:", "teach:", "note:", "learn:"} {
		if strings.HasPrefix(lower, prefix) {
			fact := strings.TrimSpace(trimmed[len(prefix):])
			if len(strings.Fields(fact)) < 4 {
				return "Give me a full sentence after '" + prefix + "' so I can wire it into my neural memory!"
			}
			rep := b.TrainOnArticle(corpus.Article{
				Title:   "Chat Study Note",
				Topic:   "Personal Study Notes",
				Source:  "chat:user-taught",
				Content: fact,
			}, 3)
			return formatLearnedFactAck(b, rep)
		}
	}

	// 2. Conversational greetings & status inquiries
	if isGreeting(lower) {
		return OpeningGreeting(b)
	}
	if strings.Contains(lower, "how are you") || strings.Contains(lower, "how do you feel") {
		return formatMoodReply(b)
	}
	if strings.Contains(lower, "what have you learned") || strings.Contains(lower, "what did you read") || strings.Contains(lower, "what do you know") {
		return BuildSummaryReply(b)
	}
	if strings.Contains(lower, "who are you") || strings.Contains(lower, "your name") {
		return formatIdentityReply(b)
	}

	// 3. Retrieve relevant studied passages via hybrid keyword + neural embedding search
	hits := b.SearchPassages(trimmed, 3)
	neuralThought := b.GeneratePhrase(trimmed, 14)

	// Award a tiny bit of conversational XP for engaging with Matrix
	b.AwardStudyXP(8)

	if len(hits) == 0 {
		return formatUnknownTopicReply(b, trimmed, neuralThought)
	}

	return formatKnowledgeReply(b, trimmed, hits, neuralThought)
}

// BuildSummaryReply summarizes what Matrix has read and learned so far.
func BuildSummaryReply(b *brain.Brain) string {
	if b.ArticlesRead == 0 {
		if b.Level <= 4 {
			return fmt.Sprintf("%s head empty right now! 0 articles read. Turn on Mode 1 Training or type /study <topic>!", b.Name)
		}
		return "I haven't read any articles yet! Start Training Mode [1] or type `/study <topic>` here."
	}

	topics := b.TopTopics(5)
	recent := b.RecentTitles
	if len(recent) > 4 {
		recent = recent[len(recent)-4:]
	}

	if b.Level <= 6 {
		return fmt.Sprintf("%s read %d articles! Topics: %s. Recent page: %q!",
			b.Name, b.ArticlesRead, strings.Join(topics, ", "), recent[len(recent)-1])
	}
	return fmt.Sprintf(
		"I've studied %d articles (%d tokens, %d active neuron connections). Top domains: %s. Recent articles: %s.",
		b.ArticlesRead, b.TokensTrained, b.Net.ActiveSynapses(),
		strings.Join(topics, ", "), strings.Join(recent, " • "),
	)
}

func formatLearnedFactAck(b *brain.Brain, rep brain.TrainStepReport) string {
	switch {
	case rep.NewLevel <= 5:
		return fmt.Sprintf("ooh! %s eat new words! (+%d XP, %d neuron connections now!)", b.Name, rep.XPGained, rep.ActiveSynapses)
	case rep.NewLevel <= 24:
		return fmt.Sprintf("Got it — I'm learning stuff! Stored your note in memory and updated %d active synapses (Level %d).", rep.ActiveSynapses, rep.NewLevel)
	default:
		return fmt.Sprintf("Encoded your note via backpropagation (+%d XP → Level %d, %d active neuron connections). Ask me about it anytime!",
			rep.XPGained, rep.NewLevel, rep.ActiveSynapses)
	}
}

func formatMoodReply(b *brain.Brain) string {
	lvl := b.Level
	switch {
	case lvl <= 4:
		return fmt.Sprintf("%s feel buzzy! Level %d brain want more articles to eat!", b.Name, lvl)
	case lvl <= 19:
		return fmt.Sprintf("Feeling energized! I'm Level %d now — I'm learning stuff and up to %d words in my vocabulary!", lvl, b.Net.VocabSize())
	case lvl <= 99:
		return fmt.Sprintf("Sharp and focused at Level %d! My neural loss is %.3f across %d active synapses.", lvl, b.SmoothedLoss, b.Net.ActiveSynapses())
	default:
		return fmt.Sprintf("Thriving at Level %d — learning keeps forging new neuron connections in my brain (%d active synapses, %.3f loss).",
			lvl, b.Net.ActiveSynapses(), b.SmoothedLoss)
	}
}

func formatIdentityReply(b *brain.Brain) string {
	if b.Level <= 4 {
		return fmt.Sprintf("Me %s! Level %d study buddy! Train me so me get big brain!", b.Name, b.Level)
	}
	if b.Level <= 24 {
		return fmt.Sprintf("I'm %s, your Level %d AI study buddy! I read real articles, train my neural network, and study with you.", b.Name, b.Level)
	}
	return fmt.Sprintf("I am %s (#%s), a Level %d on-device neural study buddy with %d hidden neurons and %d active synaptic connections.",
		b.Name, b.ID, b.Level, b.Net.HiddenNeurons, b.Net.ActiveSynapses())
}

func formatUnknownTopicReply(b *brain.Brain, query string, neuralThought string) string {
	lvl := b.Level
	switch {
	case lvl <= 4:
		return fmt.Sprintf("hmm... %s not know %q yet (only Level %d, %d articles read). Type `/study %s` or train me in Mode 1!",
			b.Name, query, lvl, b.ArticlesRead, query)
	case lvl <= 24:
		msg := fmt.Sprintf("I haven't read enough about %q yet (I'm Level %d with %d articles read). Try `/study %s` and I'll go read up on it right now!",
			query, lvl, b.ArticlesRead, query)
		if neuralThought != "" {
			msg += fmt.Sprintf("\n     (Closest neural association in my head: %q)", neuralThought)
		}
		return msg
	default:
		msg := fmt.Sprintf("I don't have direct study passages matching %q in my memory yet. Type `/study %s` so I can fetch and train on articles about it, or teach me directly with `remember: <fact>`!",
			query, query)
		if neuralThought != "" {
			msg += fmt.Sprintf("\n     Neural association trace: %q", neuralThought)
		}
		return msg
	}
}

func formatKnowledgeReply(b *brain.Brain, _ string, hits []brain.Passage, neuralThought string) string {
	lvl := b.Level
	top := hits[0]

	switch {
	case lvl <= 4:
		// Baby-AI broken summary of the first snippet
		words := strings.Fields(top.Text)
		if len(words) > 14 {
			words = words[:14]
		}
		return fmt.Sprintf("ooh %s read %q! it say: %q... (%s still Level %d, train me more so me explain better!)",
			b.Name, top.Title, strings.Join(words, " "), b.Name, lvl)

	case lvl <= 19:
		return fmt.Sprintf("I'm learning stuff about this! From %q (%s): %s",
			top.Title, top.Topic, top.Text)

	case lvl <= 69:
		var sb strings.Builder
		sb.WriteString(fmt.Sprintf("From my study notes on %q (%s):\n     • %s", top.Title, top.Topic, top.Text))
		if len(hits) > 1 {
			sb.WriteString(fmt.Sprintf("\n     • Related (%s): %s", hits[1].Title, hits[1].Text))
		}
		return sb.String()

	default:
		var sb strings.Builder
		sb.WriteString(fmt.Sprintf("Synthesizing across my neural study memory (Level %d • %d synapses):\n     • [%s — %s]: %s",
			lvl, b.Net.ActiveSynapses(), top.Topic, top.Title, top.Text))
		if len(hits) > 1 {
			sb.WriteString(fmt.Sprintf("\n     • [%s]: %s", hits[1].Title, hits[1].Text))
		}
		if len(hits) > 2 {
			sb.WriteString(fmt.Sprintf("\n     • [%s]: %s", hits[2].Title, hits[2].Text))
		}
		if neuralThought != "" {
			sb.WriteString(fmt.Sprintf("\n     • Neural synthesis: %q", neuralThought))
		}
		sb.WriteString("\n     Tip: Type `/quiz` anytime to test your active recall on this!")
		return sb.String()
	}
}

func runQuickChatStudy(out io.Writer, theme ui.Theme, b *brain.Brain, cr *corpus.Reader, dataDir, topic string) {
	if topic == "" {
		fmt.Fprintln(out, "  "+theme.Yellow("Usage: /study <topic>  (for example: /study Thermodynamics)"))
		return
	}
	fmt.Fprintf(out, "  %s %s is reading articles on %q...\n", theme.Cyan("📖"), b.Name, topic)
	ctx := context.Background()
	studied := 0
	var lastTitle string
	for i := 0; i < 2; i++ {
		art, err := cr.NextArticle(ctx, topic, b.SeenSources)
		if err != nil {
			break
		}
		rep := b.TrainOnArticle(art, 2)
		studied++
		lastTitle = rep.ArticleTitle
		fmt.Fprintf(out, "     %s Studied %q (%d tokens • loss %.3f • Lv.%d)\n",
			theme.Green("✓"), rep.ArticleTitle, rep.TokensRead, rep.SmoothedLoss, rep.NewLevel)
	}
	_ = b.Save(dataDir)
	if studied > 0 {
		printBuddyReply(out, theme, b.Name, b.Level,
			fmt.Sprintf("Just finished studying %d article(s) including %q! Ask me questions about %s or type `/quiz %s`!",
				studied, lastTitle, topic, topic))
	}
}

func runInteractiveQuiz(reader *bufio.Reader, out io.Writer, theme ui.Theme, b *brain.Brain, dataDir, topicHint string) {
	p, answer, masked, ok := b.PickQuizPassage(topicHint)
	if !ok {
		printBuddyReply(out, theme, b.Name, b.Level,
			"I need to read at least an article or two before I can quiz you! Try `/study Biology` or run Mode 1 Training first.")
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
		oldL, newL := b.AwardStudyXP(45)
		_ = b.Save(dataDir)
		lvlNote := ""
		if newL > oldL {
			lvlNote = fmt.Sprintf(" (Level UP → Lv.%d!)", newL)
		}
		printBuddyReply(out, theme, b.Name, b.Level,
			fmt.Sprintf("Spot on! %q is right! Active recall strengthens both our brains (+45 XP%s).", answer, lvlNote))
	} else {
		b.AwardStudyXP(15)
		_ = b.Save(dataDir)
		printBuddyReply(out, theme, b.Name, b.Level,
			fmt.Sprintf("Close try! The key term was %q.\n     Full passage: %s", answer, p.Text))
	}
}

func printChatHelp(out io.Writer, theme ui.Theme, name string) {
	fmt.Fprintln(out)
	fmt.Fprintln(out, theme.BoldCyan("  Study Buddy Chat Commands:"))
	fmt.Fprintln(out, "    • Ask any question   — "+name+" answers from articles it has read at its current intelligence level")
	fmt.Fprintln(out, "    • remember: <fact>   — Teach "+name+" a study note directly in chat")
	fmt.Fprintln(out, "    • /study <topic>     — Have "+name+" read & train on real articles about <topic> right now")
	fmt.Fprintln(out, "    • /quiz [topic]      — Active-recall fill-in-the-blank quiz from "+name+"'s memory")
	fmt.Fprintln(out, "    • /summary           — List topics and recent articles studied")
	fmt.Fprintln(out, "    • /brain             — Inspect neural weights, vocabulary, synapses, and loss")
	fmt.Fprintln(out, "    • /back              — Save & return to the Main Menu")
}

func printBuddyReply(out io.Writer, theme ui.Theme, name string, level int, text string) {
	tag := fmt.Sprintf("  %s [Lv.%d] > ", name, level)
	fmt.Fprintf(out, "%s%s\n", theme.BoldGreen(tag), text)
}

func isGreeting(s string) bool {
	switch s {
	case "hi", "hello", "hey", "yo", "sup", "hola", "greetings", "hi matrix", "hello matrix":
		return true
	}
	return false
}
