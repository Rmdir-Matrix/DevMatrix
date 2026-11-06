package brain

import (
	"strings"
	"unicode"
)

var stopWords = map[string]bool{
	"the": true, "and": true, "for": true, "that": true, "this": true,
	"with": true, "from": true, "are": true, "was": true, "were": true,
	"have": true, "has": true, "had": true, "not": true, "but": true,
	"you": true, "your": true, "can": true, "will": true, "into": true,
	"their": true, "they": true, "them": true, "its": true, "about": true,
	"which": true, "when": true, "where": true, "what": true, "who": true,
	"how": true, "why": true, "also": true, "than": true, "then": true,
	"more": true, "most": true, "some": true, "such": true, "only": true,
	"over": true, "under": true, "between": true, "through": true, "during": true,
	"each": true, "other": true, "these": true, "those": true, "been": true,
	"being": true, "does": true, "did": true, "would": true, "could": true,
	"should": true, "while": true, "after": true, "before": true, "because": true,
}

// conversationalWords are everyday dialogue/grammar words that do not demand
// domain-article lookup on their own (they still must exist in the trained vocabulary!).
var conversationalWords = map[string]bool{
	"hi": true, "hello": true, "hey": true, "goodbye": true, "bye": true,
	"thanks": true, "thank": true, "please": true, "whats": true, "up": true,
	"how": true, "are": true, "you": true, "doing": true, "feel": true,
	"who": true, "what": true, "why": true, "where": true, "when": true,
	"is": true, "am": true, "be": true, "do": true, "does": true, "did": true,
	"can": true, "could": true, "would": true, "will": true, "should": true,
	"i": true, "me": true, "my": true, "we": true, "us": true, "our": true,
	"your": true, "name": true, "matrix": true, "<self>": true,
	"learn": true, "learning": true, "study": true, "studying": true,
	"read": true, "reading": true, "speak": true, "sentences": true,
	"grammar": true, "words": true, "smart": true, "help": true,
	"tell": true, "explain": true, "describe": true, "define": true,
	"about": true, "know": true, "think": true, "good": true, "morning": true,
}

// QAPair represents a prompt->response training example parsed from curriculum or self-study.
type QAPair struct {
	Prompt   []string
	Response []string
}

// ParsedCorpusItem holds structured training sequences extracted from an article or grammar lesson.
type ParsedCorpusItem struct {
	Sentences []string
	QAPairs   []QAPair
	Thoughts  [][]string
}

// ParseCorpusContent separates QA:, THOUGHT:, and prose paragraphs/sentences from training text.
func ParseCorpusContent(text string) ParsedCorpusItem {
	text = strings.ReplaceAll(text, "\r\n", "\n")
	lines := strings.Split(text, "\n")

	var item ParsedCorpusItem
	var proseBuilder strings.Builder

	for _, rawLine := range lines {
		line := strings.TrimSpace(rawLine)
		if line == "" {
			proseBuilder.WriteString("\n")
			continue
		}
		if strings.HasPrefix(line, "QA:") {
			rest := strings.TrimSpace(line[len("QA:"):])
			parts := strings.SplitN(rest, "|", 2)
			if len(parts) == 2 {
				qToks := Tokenize(strings.TrimSpace(parts[0]))
				aToks := Tokenize(strings.TrimSpace(parts[1]))
				if len(qToks) > 0 && len(aToks) > 0 {
					item.QAPairs = append(item.QAPairs, QAPair{
						Prompt:   qToks,
						Response: aToks,
					})
				}
			}
			continue
		}
		if strings.HasPrefix(line, "THOUGHT:") {
			rest := strings.TrimSpace(line[len("THOUGHT:"):])
			tToks := Tokenize(rest)
			if len(tToks) > 0 {
				item.Thoughts = append(item.Thoughts, tToks)
			}
			continue
		}
		proseBuilder.WriteString(line)
		proseBuilder.WriteString("\n")
	}

	item.Sentences = SplitSentences(proseBuilder.String())
	return item
}

// SplitSentences splits article text into clean readable sentences.
func SplitSentences(text string) []string {
	text = strings.ReplaceAll(text, "\r\n", "\n")
	paragraphs := strings.Split(text, "\n")
	var out []string

	for _, p := range paragraphs {
		p = strings.TrimSpace(p)
		if p == "" {
			continue
		}
		start := 0
		runes := []rune(p)
		for i := 0; i < len(runes); i++ {
			ch := runes[i]
			if ch == '.' || ch == '!' || ch == '?' {
				if i+1 == len(runes) || unicode.IsSpace(runes[i+1]) {
					seg := strings.TrimSpace(string(runes[start : i+1]))
					if len(seg) >= 6 {
						out = append(out, seg)
					}
					start = i + 1
				}
			}
		}
		if start < len(runes) {
			rem := strings.TrimSpace(string(runes[start:]))
			if len(rem) >= 6 {
				out = append(out, rem)
			}
		}
	}
	return out
}

// Tokenize converts text into normalized word, <self>, and punctuation tokens.
func Tokenize(text string) []string {
	// Protect <self> placeholder regardless of casing
	normalized := strings.ReplaceAll(text, "<SELF>", " <self> ")
	normalized = strings.ReplaceAll(normalized, "<Self>", " <self> ")
	normalized = strings.ReplaceAll(normalized, "<self>", " <self> ")

	var tokens []string
	var cur strings.Builder

	flush := func() {
		if cur.Len() > 0 {
			tokens = append(tokens, strings.ToLower(cur.String()))
			cur.Reset()
		}
	}

	runes := []rune(normalized)
	for i := 0; i < len(runes); i++ {
		// Detect <self>
		if runes[i] == '<' && i+6 <= len(runes) && strings.ToLower(string(runes[i:i+6])) == "<self>" {
			flush()
			tokens = append(tokens, selfToken)
			i += 5
			continue
		}
		r := runes[i]
		switch {
		case unicode.IsLetter(r) || unicode.IsDigit(r) || r == '\'':
			cur.WriteRune(r)
		case r == ',' || r == '.' || r == '!' || r == '?' || r == ';' || r == ':':
			flush()
			tokens = append(tokens, string(r))
		default:
			flush()
		}
	}
	flush()
	return tokens
}

// ExtractKeywords returns deduplicated salient content words from text.
func ExtractKeywords(text string, limit int) []string {
	toks := Tokenize(text)
	seen := make(map[string]bool)
	var kw []string
	for _, tok := range toks {
		if tok == selfToken || len(tok) < 3 || stopWords[tok] {
			continue
		}
		isAlpha := false
		for _, r := range tok {
			if unicode.IsLetter(r) {
				isAlpha = true
				break
			}
		}
		if !isAlpha || seen[tok] {
			continue
		}
		seen[tok] = true
		kw = append(kw, tok)
		if limit > 0 && len(kw) >= limit {
			break
		}
	}
	return kw
}

// ExtractTopicWords returns non-conversational, non-stopword content terms from a user query.
func ExtractTopicWords(tokens []string) []string {
	seen := make(map[string]bool)
	var out []string
	for _, tok := range tokens {
		if len(tok) < 2 || stopWords[tok] || conversationalWords[tok] {
			continue
		}
		isAlpha := false
		for _, r := range tok {
			if unicode.IsLetter(r) {
				isAlpha = true
				break
			}
		}
		if !isAlpha || seen[tok] {
			continue
		}
		seen[tok] = true
		out = append(out, tok)
	}
	return out
}
