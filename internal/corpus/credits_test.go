package corpus

import (
	"context"
	"io"
	"net/http"
	"strings"
	"testing"
)

type creditRoundTripper func(*http.Request) (*http.Response, error)

func (f creditRoundTripper) RoundTrip(r *http.Request) (*http.Response, error) { return f(r) }
func TestWiktionaryOnlyAndBatchCredits(t *testing.T) {
	r, err := NewReader(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	var visited []string
	r.httpClient = &http.Client{Transport: creditRoundTripper(func(req *http.Request) (*http.Response, error) {
		visited = append(visited, req.URL.Host)
		if req.URL.Host != "en.wiktionary.org" {
			t.Fatalf("unexpected dictionary host: %s", req.URL.Host)
		}
		body := `{"en":[{"partOfSpeech":"noun","definitions":[{"definition":"a useful English definition"}]}]}`
		return &http.Response{StatusCode: 200, Body: io.NopCloser(strings.NewReader(body)), Header: make(http.Header)}, nil
	})}
	a, err := r.FetchDictionaryEntry(context.Background(), "inference")
	if err != nil {
		t.Fatal(err)
	}
	if c := CreditsForArticle(a); len(c) != 1 || c[0].URL != "https://en.wiktionary.org/wiki/inference" {
		t.Fatalf("credits: %+v", c)
	}
	batch, err := r.fetchLiveDictionaryBatch(context.Background(), "dictionary:live-batch-1")
	if err != nil {
		t.Fatal(err)
	}
	if len(batch.Credits) != 3 {
		t.Fatalf("batch credits: %+v", batch.Credits)
	}
	for _, c := range batch.Credits {
		if c.Name != "Wiktionary" || !strings.HasPrefix(c.URL, "https://en.wiktionary.org/wiki/") {
			t.Fatalf("bad credit: %+v", c)
		}
	}
	if len(visited) != 4 {
		t.Fatalf("requests: %v", visited)
	}
}
func TestCreditUnknownLegacyAndLocal(t *testing.T) {
	for _, source := range []string{"dictionary:live-batch-2", "dictionary:live:word", "local:private.md", "chat:user-taught", "https://en.wikipedia.org.evil.test/wiki/X"} {
		c := CreditsForArticle(Article{Source: source})
		if len(c) != 1 || !strings.Contains(c[0].License, "Unknown") {
			t.Fatalf("incorrectly attributed %q: %+v", source, c)
		}
	}
	c := CreditsForArticle(Article{Source: "https://en.wikipedia.org/wiki/Computer"})
	if len(c) != 1 || c[0].License != "CC BY-SA 4.0" {
		t.Fatal(c)
	}
}
