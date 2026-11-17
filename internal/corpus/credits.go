package corpus

import (
	"net/url"
	"strings"
)

// SourceCredit identifies material used in training, not a guarantee of rights.
type SourceCredit struct {
	Name       string `json:"name"`
	License    string `json:"license"`
	URL        string `json:"url,omitempty"`
	LicenseURL string `json:"license_url,omitempty"`
	Note       string `json:"note,omitempty"`
}

// CreditsForArticle only makes claims supported by the source identifier. Old
// batches and user-provided material need review rather than assumed licenses.
func CreditsForArticle(a Article) []SourceCredit {
	if len(a.Credits) > 0 {
		return a.Credits
	}
	s := a.Source
	const cc = "https://creativecommons.org/licenses/by-sa/4.0/"
	switch {
	case strings.HasPrefix(s, "dictionary:core-"), strings.HasPrefix(s, "dictionary:ext-"), strings.HasPrefix(s, "dictionary:word:"):
		return []SourceCredit{{Name: "Wordset / Princeton WordNet (bundled dictionary)", License: "CC BY-SA 4.0 and WordNet notice", URL: "https://github.com/wordset/wordset-dictionary", LicenseURL: "https://github.com/wordset/wordset-dictionary/blob/master/LICENSE", Note: "Selected and transformed into dictionary chapters."}}
	case strings.HasPrefix(s, "dictionary:wiktionary:"):
		word := strings.TrimPrefix(s, "dictionary:wiktionary:")
		return []SourceCredit{{Name: "Wiktionary", License: "CC BY-SA 4.0", URL: "https://en.wiktionary.org/wiki/" + url.PathEscape(word), LicenseURL: cc, Note: "Definitions extracted and transformed for training."}}
	case strings.HasPrefix(s, "dictionary:live:") || strings.HasPrefix(s, "dictionary:live-batch-"):
		return []SourceCredit{{Name: "Earlier live dictionary material", License: "Unknown; review original entries", Note: "Older versions mixed Free Dictionary API and Wiktionary without preserving individual attribution."}}
	case strings.HasPrefix(s, "grammar:"), strings.HasPrefix(s, "curriculum:"), strings.HasPrefix(s, "embedded:"):
		return []SourceCredit{{Name: "Matrix built-in study material", License: "Not specified; review provenance", Note: "Bundled grammar or study material."}}
	case strings.HasPrefix(s, "local:"), strings.HasPrefix(s, "chat:"):
		return []SourceCredit{{Name: "User-provided notes or chat", License: "Unknown; permission must be checked", Note: "May contain private or third-party material."}}
	}
	u, err := url.Parse(s)
	if err == nil && (u.Scheme == "https" || u.Scheme == "http") && u.User == nil {
		switch strings.ToLower(u.Hostname()) {
		case "en.wikipedia.org":
			return []SourceCredit{{Name: "Wikipedia", License: "CC BY-SA 4.0", URL: s, LicenseURL: cc, Note: "Page excerpt transformed for training; check page attribution and exceptions."}}
		case "en.wikibooks.org":
			return []SourceCredit{{Name: "Wikibooks", License: "CC BY-SA 4.0", URL: s, LicenseURL: cc, Note: "Page excerpt transformed for training; check page attribution and exceptions."}}
		case "en.wiktionary.org":
			return []SourceCredit{{Name: "Wiktionary", License: "CC BY-SA 4.0", URL: s, LicenseURL: cc, Note: "Page excerpt transformed for training."}}
		case "arxiv.org", "www.arxiv.org", "export.arxiv.org":
			return []SourceCredit{{Name: "arXiv abstract metadata", License: "CC0 1.0 (metadata)", URL: s, LicenseURL: "https://creativecommons.org/publicdomain/zero/1.0/", Note: "Abstract only; full papers have separate licenses."}}
		}
	}
	if strings.HasPrefix(s, "wikipedia:") {
		title := strings.TrimPrefix(s, "wikipedia:")
		return []SourceCredit{{Name: "Wikipedia", License: "CC BY-SA 4.0", URL: "https://en.wikipedia.org/wiki/" + url.PathEscape(strings.ReplaceAll(title, " ", "_")), LicenseURL: cc, Note: "Page excerpt transformed for training; check page attribution and exceptions."}}
	}
	return []SourceCredit{{Name: "Unidentified or external training material", License: "Unknown; review before sharing"}}
}
