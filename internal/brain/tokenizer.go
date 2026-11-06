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
				// Split if at end of paragraph or followed by whitespace
				if i+1 == len(runes) || unicode.IsSpace(runes[i+1]) {
					seg := strings.TrimSpace(string(runes[start : i+1]))
					if len(seg) >= 8 {
						out = append(out, seg)
					}
					start = i + 1
				}
			}
		}
		if start < len(runes) {
			rem := strings.TrimSpace(string(runes[start:]))
			if len(rem) >= 8 {
				out = append(out, rem)
			}
		}
	}
	return out
}

// Tokenize converts a sentence into normalized word and punctuation tokens.
func Tokenize(text string) []string {
	var tokens []string
	var cur strings.Builder

	flush := func() {
		if cur.Len() > 0 {
			tokens = append(tokens, strings.ToLower(cur.String()))
			cur.Reset()
		}
	}

	for _, r := range text {
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
		if len(tok) < 3 || stopWords[tok] {
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
