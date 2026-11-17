package brain

import (
	"github.com/Rmdir-Matrix/DevMatrix/internal/corpus"
	"sort"
)

const maxSourceCredits = 4096

// Retain source references independently of the bounded curriculum seen set.
func (b *Brain) RecordSourceCredits(credits []corpus.SourceCredit) {
	seen := make(map[corpus.SourceCredit]bool, len(b.SourceCredits))
	for _, c := range b.SourceCredits {
		seen[c] = true
	}
	for _, c := range credits {
		if seen[c] {
			continue
		}
		if len(b.SourceCredits) >= maxSourceCredits {
			b.SourceHistoryComplete = false
			return
		}
		b.SourceCredits = append(b.SourceCredits, c)
		seen[c] = true
	}
}

// Older saved brains may have incomplete/evicted seen-source histories. Recover
// only what remains and clearly flag that the listing may be incomplete.
func (b *Brain) SharingCredits() []corpus.SourceCredit {
	all := append([]corpus.SourceCredit(nil), b.SourceCredits...)
	seen := make(map[corpus.SourceCredit]bool, len(all))
	for _, c := range all {
		seen[c] = true
	}
	if !b.SourceHistoryComplete {
		for source, used := range b.SeenSources {
			if !used {
				continue
			}
			for _, c := range corpus.CreditsForArticle(corpus.Article{Source: source}) {
				if !seen[c] && len(all) < maxSourceCredits {
					all = append(all, c)
					seen[c] = true
				}
			}
		}
	}
	sort.Slice(all, func(i, j int) bool {
		if all[i].Name != all[j].Name {
			return all[i].Name < all[j].Name
		}
		return all[i].URL < all[j].URL
	})
	return all
}
