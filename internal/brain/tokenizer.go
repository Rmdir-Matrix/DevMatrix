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
	"is": true, "am": true, "be": true, "do": true, "in": true, "on": true,
	"of": true, "to": true, "a": true, "an": true, "by": true, "or": true, "as": true,
}

var productiveSuffixes = []string{
	"ingly", "edly", "ness", "ment", "able",
	"ings", "ing", "est", "ful", "less",
	"ly", "ed", "es", "s",
}

// SFTPair represents a pure closed-book (<user> prompt -> <assistant> response) training sequence.
type SFTPair struct {
	Prompt   []string
	Response []string
}

// ParsedCorpusItem holds pretraining sentences and SFT instruction/dialogue pairs extracted from text.
type ParsedCorpusItem struct {
	Sentences []string
	SFTPairs  []SFTPair
}

// ParseCorpusContent parses DICT: lines, QA: lines, and natural prose paragraphs into pure
// GPT pretraining sentences + SFT pairs (zero lookup databases, zero synthetic thought strings).
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
		if strings.HasPrefix(line, "DICT:") {
			rest := strings.TrimSpace(line[len("DICT:"):])
			parts := strings.Split(rest, "|")
			if len(parts) >= 3 {
				word := strings.TrimSpace(parts[0])
				pos := strings.TrimSpace(parts[1])
				def := strings.TrimSpace(strings.TrimSuffix(strings.TrimSpace(parts[2]), "."))
				ex := ""
				if len(parts) >= 4 {
					ex = strings.TrimSpace(parts[3])
				}
				if word != "" && def != "" {
					articleWord := "a"
					if strings.HasPrefix(strings.ToLower(pos), "a") {
						articleWord = "an"
					}
					defSent := word + " is " + articleWord + " " + pos + " meaning " + def + "."
					proseBuilder.WriteString(defSent + "\n")
					if ex != "" {
						if !strings.HasSuffix(ex, ".") && !strings.HasSuffix(ex, "!") && !strings.HasSuffix(ex, "?") {
							ex += "."
						}
						proseBuilder.WriteString(ex + "\n")
					}
					defToks := SubwordTokenize(defSent)
					wToks := SubwordTokenize(word)
					if len(wToks) > 0 && len(defToks) > 0 {
						item.SFTPairs = append(item.SFTPairs,
							SFTPair{Prompt: append([]string{"define"}, wToks...), Response: defToks},
							SFTPair{Prompt: append(append([]string{"what", "does"}, wToks...), "mean"), Response: defToks},
						)
					}
				}
			}
			continue
		}
		if strings.HasPrefix(line, "QA:") {
			rest := strings.TrimSpace(line[len("QA:"):])
			parts := strings.SplitN(rest, "|", 2)
			if len(parts) == 2 {
				qToks := SubwordTokenize(strings.TrimSpace(parts[0]))
				aToks := SubwordTokenize(strings.TrimSpace(parts[1]))
				if len(qToks) > 0 && len(aToks) > 0 {
					item.SFTPairs = append(item.SFTPairs, SFTPair{
						Prompt:   qToks,
						Response: aToks,
					})
				}
			}
			continue
		}
		proseBuilder.WriteString(line)
		proseBuilder.WriteString("\n")
	}

	item.Sentences = SplitSentences(proseBuilder.String())
	return item
}

// SplitSentences splits prose text into clean sentences.
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

// Tokenize splits text into raw word, <self>, and punctuation tokens before subword splitting.
func Tokenize(text string) []string {
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
		case r == ',' || r == '.' || r == '!' || r == '?' || r == ';' || r == ':' || r == '(' || r == ')':
			flush()
			tokens = append(tokens, string(r))
		default:
			flush()
		}
	}
	flush()
	return tokens
}

// SubwordTokenize performs morphological WordPiece-style stem + "##suffix" tokenization.
func SubwordTokenize(text string) []string {
	raw := Tokenize(text)
	out := make([]string, 0, len(raw)+4)
	for _, tok := range raw {
		if tok == selfToken || len(tok) <= 4 || !isPureLetters(tok) || stopWords[tok] {
			out = append(out, tok)
			continue
		}
		stem, suf := splitMorpheme(tok)
		if suf != "" {
			out = append(out, stem, "##"+suf)
		} else {
			out = append(out, tok)
		}
	}
	return out
}

// EncodeWithVocab tokenizes text against an existing trained vocabulary using WordPiece
// greedy longest-match fallback, returning (subwordTokens, unknownWholeWords).
func EncodeWithVocab(text string, selfName string, knows func(tok string) bool) ([]string, []string) {
	raw := Tokenize(text)
	selfLower := strings.ToLower(strings.TrimSpace(selfName))

	var out []string
	var unknown []string

	for idx, w := range raw {
		if w == selfLower || w == "matrix" || w == selfToken {
			out = append(out, selfToken)
			continue
		}
		if !isAlphaNumericToken(w) {
			if knows(w) {
				out = append(out, w)
			}
			continue
		}
		// Allow unseen proper names in introduction slots ("my name is <X>", "call me <X>")
		isIntroNameSlot := (idx >= 2 && raw[idx-2] == "name" && raw[idx-1] == "is" && knows("name") && knows("is")) ||
			(idx >= 2 && raw[idx-2] == "call" && raw[idx-1] == "me" && knows("call") && knows("me"))
		if isIntroNameSlot && !knows(w) {
			out = append(out, unkToken)
			continue
		}
		if knows(w) {
			out = append(out, w)
			continue
		}
		stem, suf := splitMorpheme(w)
		if suf != "" && knows(stem) {
			out = append(out, stem)
			if knows("##" + suf) {
				out = append(out, "##"+suf)
			}
			continue
		}
		if pieces, ok := greedyWordPiece(w, knows); ok {
			out = append(out, pieces...)
			continue
		}
		out = append(out, unkToken)
		unknown = append(unknown, w)
	}
	return out, unknown
}

func splitMorpheme(word string) (string, string) {
	for _, suf := range productiveSuffixes {
		if len(word) >= len(suf)+4 && strings.HasSuffix(word, suf) {
			stem := word[:len(word)-len(suf)]
			last := stem[len(stem)-1]
			if suf == "s" && (last == 's' || last == 'i' || last == 'u' || last == 'x' || strings.HasSuffix(word, "ies")) {
				continue
			}
			if (suf == "es" || suf == "ed") && (last == 'i' || last == 'u') {
				continue
			}
			if len(stem) >= 4 && stem[len(stem)-1] == stem[len(stem)-2] && !strings.ContainsRune("aeioulsz", rune(stem[len(stem)-1])) {
				stem = stem[:len(stem)-1]
			}
			return stem, suf
		}
	}
	return word, ""
}

func greedyWordPiece(word string, knows func(string) bool) ([]string, bool) {
	if len(word) < 5 {
		return nil, false
	}
	for split := len(word) - 1; split >= 4; split-- {
		prefix := word[:split]
		rest := "##" + word[split:]
		if knows(prefix) && knows(rest) {
			return []string{prefix, rest}, true
		}
	}
	return nil, false
}

// DetokenizeSubwords joins subwords ("##ing") and punctuation cleanly, substituting <self> -> selfName.
func DetokenizeSubwords(tokens []string, selfName string) string {
	var out strings.Builder
	capNext := true
	prevWasOpenParen := false

	for _, tok := range tokens {
		if tok == padToken || tok == unkToken || tok == bosToken || tok == eosToken ||
			tok == userToken || tok == assistantToken {
			continue
		}
		if strings.HasPrefix(tok, "##") {
			out.WriteString(strings.TrimPrefix(tok, "##"))
			prevWasOpenParen = false
			continue
		}

		display := tok
		if tok == selfToken {
			display = selfName
		} else if tok == "i" {
			display = "I"
		} else if tok == "i'm" {
			display = "I'm"
		}

		noSpaceBefore := tok == "." || tok == "," || tok == "!" || tok == "?" || tok == ";" || tok == ":" || tok == ")"
		if out.Len() > 0 && !noSpaceBefore && !prevWasOpenParen {
			out.WriteByte(' ')
		}

		if capNext && len(display) > 0 && tok != selfToken && isAlphaNumericToken(display) {
			runes := []rune(display)
			runes[0] = unicode.ToUpper(runes[0])
			display = string(runes)
			capNext = false
		} else if isAlphaNumericToken(display) {
			capNext = false
		}

		if tok == "." || tok == "!" || tok == "?" {
			capNext = true
		}
		prevWasOpenParen = (tok == "(")
		out.WriteString(display)
	}

	res := strings.TrimSpace(out.String())
	if res != "" && !strings.HasSuffix(res, ".") && !strings.HasSuffix(res, "!") && !strings.HasSuffix(res, "?") {
		res += "."
	}
	return res
}

// ExtractKeywords returns deduplicated salient root words from text.
func ExtractKeywords(text string, limit int) []string {
	raw := Tokenize(text)
	seen := make(map[string]bool)
	var kw []string
	for _, tok := range raw {
		if tok == selfToken || len(tok) < 3 || stopWords[tok] || !isPureLetters(tok) {
			continue
		}
		stem, _ := splitMorpheme(tok)
		if stopWords[stem] {
			continue
		}
		if !seen[stem] {
			seen[stem] = true
			kw = append(kw, stem)
			if limit > 0 && len(kw) >= limit {
				break
			}
		}
	}
	return kw
}

func isPureLetters(s string) bool {
	if s == "" {
		return false
	}
	for _, r := range s {
		if !unicode.IsLetter(r) {
			return false
		}
	}
	return true
}

func isAlphaNumericToken(s string) bool {
	if s == selfToken {
		return true
	}
	for _, r := range s {
		if unicode.IsLetter(r) || unicode.IsDigit(r) {
			return true
		}
	}
	return false
}
