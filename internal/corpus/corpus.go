package corpus

import (
	"context"
	_ "embed"
	"encoding/json"
	"fmt"
	"io"
	"math/rand"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"time"
)

//go:embed dictionary_entries.json
var embeddedDictionaryJSON []byte

//go:embed grammar_curriculum.json
var embeddedGrammarJSON []byte

//go:embed seed_articles.json
var embeddedSeedJSON []byte

const (
	maxHTTPBytes       = 256 * 1024
	maxLocalFileBytes  = 512 * 1024
	httpRequestTimeout = 4 * time.Second
	userAgentHeader    = "MatrixStudyBuddy/1.0 (CLI Educational AI Study Companion)"
)

// Article represents a dictionary volume, grammar lesson, or real training article read by Matrix.
type Article struct {
	Title   string `json:"title"`
	Topic   string `json:"topic"`
	Source  string `json:"source"` // e.g. "dictionary:...", "curriculum:...", "wikipedia:...", "local:...", "embedded:..."
	Content string `json:"content"`
}

// Reader supplies real training data in natural order:
//  1. Core English Dictionaries FIRST (word meanings & parts of speech)
//  2. Foundational Grammar & Sentence Structure SECOND
//  3. Local Study Notes, Live Web Articles & Built-in Encyclopedia THIRD
type Reader struct {
	notesDir          string
	httpClient        *http.Client
	dictionaryVolumes []Article
	grammarLessons    []Article
	seedArticles      []Article
	rng               *rand.Rand
	offlineUntil      time.Time
	DisableLiveFetch  bool
}

// NewReader creates an Article Reader rooted at dataDir/study_notes.
func NewReader(dataDir string) (*Reader, error) {
	var dicts []Article
	if err := json.Unmarshal(embeddedDictionaryJSON, &dicts); err != nil {
		return nil, fmt.Errorf("load embedded dictionary corpus: %w", err)
	}

	var grammar []Article
	if err := json.Unmarshal(embeddedGrammarJSON, &grammar); err != nil {
		return nil, fmt.Errorf("load embedded grammar curriculum: %w", err)
	}

	var seeds []Article
	if err := json.Unmarshal(embeddedSeedJSON, &seeds); err != nil {
		return nil, fmt.Errorf("load embedded study corpus: %w", err)
	}

	notesDir := ""
	if strings.TrimSpace(dataDir) != "" {
		notesDir = filepath.Join(dataDir, "study_notes")
		_ = os.MkdirAll(notesDir, 0o755)
		ensureNotesReadme(notesDir)
	}

	return &Reader{
		notesDir: notesDir,
		httpClient: &http.Client{
			Timeout: httpRequestTimeout,
		},
		dictionaryVolumes: dicts,
		grammarLessons:    grammar,
		seedArticles:      seeds,
		rng:               rand.New(rand.NewSource(time.Now().UnixNano())),
	}, nil
}

// NotesDir returns the local folder where users can drop .txt/.md study notes.
func (r *Reader) NotesDir() string {
	return r.notesDir
}

// DictionaryCount returns how many sequential dictionary volumes exist in Phase 1.
func (r *Reader) DictionaryCount() int {
	return len(r.dictionaryVolumes)
}

// DictionaryCompleted counts how many dictionary volumes Matrix has trained on so far.
func (r *Reader) DictionaryCompleted(seen map[string]bool) int {
	done := 0
	for _, d := range r.dictionaryVolumes {
		if seen[d.Source] {
			done++
		}
	}
	return done
}

// GrammarCount returns how many sequential foundational grammar lessons exist in Phase 2.
func (r *Reader) GrammarCount() int {
	return len(r.grammarLessons)
}

// GrammarCompleted counts how many foundational grammar lessons Matrix has completed so far.
func (r *Reader) GrammarCompleted(seen map[string]bool) int {
	done := 0
	for _, g := range r.grammarLessons {
		if seen[g.Source] {
			done++
		}
	}
	return done
}

// SeedCount returns how many built-in encyclopedic articles are embedded in the binary.
func (r *Reader) SeedCount() int {
	return len(r.seedArticles)
}

// NextArticle fetches the next training item in strict curriculum order:
//  1. Sequential Dictionary Volumes FIRST (word definitions & parts of speech)
//  2. Sequential Grammar & Sentence Structure Lessons SECOND
//  3. Unread Local Study Notes (.txt / .md)
//  4. Live Web Wikipedia Articles (with automatic offline cooldown fallback)
//  5. Built-in Real Study Encyclopedia
func (r *Reader) NextArticle(ctx context.Context, topicFilter string, seenSources map[string]bool) (Article, error) {
	topicFilter = strings.TrimSpace(topicFilter)

	if topicFilter == "" {
		// Phase 1: Train on Dictionaries FIRST!
		for _, dictVol := range r.dictionaryVolumes {
			if !seenSources[dictVol.Source] {
				return dictVol, nil
			}
		}

		// Phase 2: Train on Grammar & Sentence Structure SECOND!
		for _, lesson := range r.grammarLessons {
			if !seenSources[lesson.Source] {
				return lesson, nil
			}
		}
	}

	// Phase 3A: Check unread local study notes (.txt / .md)
	if localArt, ok := r.nextUnreadLocalNote(topicFilter, seenSources); ok {
		return localArt, nil
	}

	// Phase 3B: Try live Wikipedia article if online and not in temporary offline cooldown
	if !r.DisableLiveFetch && time.Now().After(r.offlineUntil) {
		liveArt, err := r.fetchWikipediaArticle(ctx, topicFilter, seenSources)
		if err == nil && len(strings.Fields(liveArt.Content)) >= 35 {
			return liveArt, nil
		}
		if err != nil {
			r.offlineUntil = time.Now().Add(45 * time.Second)
		}
	}

	// Phase 3C: Select from embedded study encyclopedia (or matching dictionary volume if topic="dictionary")
	qLower := strings.ToLower(topicFilter)
	if qLower == "dictionary" || qLower == "words" || qLower == "vocabulary" {
		for _, d := range r.dictionaryVolumes {
			if !seenSources[d.Source] {
				return d, nil
			}
		}
		if len(r.dictionaryVolumes) > 0 {
			return r.dictionaryVolumes[r.rng.Intn(len(r.dictionaryVolumes))], nil
		}
	}

	if len(r.seedArticles) == 0 {
		return Article{}, fmt.Errorf("no training articles available")
	}

	candidates := make([]Article, 0, len(r.seedArticles))
	unread := make([]Article, 0, len(r.seedArticles))

	for _, a := range r.seedArticles {
		if qLower != "" {
			if !strings.Contains(strings.ToLower(a.Topic), qLower) &&
				!strings.Contains(strings.ToLower(a.Title), qLower) &&
				!strings.Contains(strings.ToLower(a.Content), qLower) {
				continue
			}
		}
		candidates = append(candidates, a)
		if !seenSources[a.Source] {
			unread = append(unread, a)
		}
	}

	if len(unread) > 0 {
		return unread[r.rng.Intn(len(unread))], nil
	}
	if len(candidates) > 0 {
		return candidates[r.rng.Intn(len(candidates))], nil
	}

	for _, a := range r.seedArticles {
		if !seenSources[a.Source] {
			return a, nil
		}
	}
	return r.seedArticles[r.rng.Intn(len(r.seedArticles))], nil
}

// FetchDictionaryEntry looks up a word's definition from the live Free Dictionary API (if online)
// or searches the embedded dictionary volumes.
func (r *Reader) FetchDictionaryEntry(ctx context.Context, word string) (Article, error) {
	cleanWord := strings.ToLower(strings.TrimSpace(word))
	if cleanWord == "" {
		return Article{}, fmt.Errorf("empty word")
	}

	if !r.DisableLiveFetch && time.Now().After(r.offlineUntil) {
		if art, err := r.fetchLiveDictionaryAPI(ctx, cleanWord); err == nil {
			return art, nil
		}
	}

	// Fallback: search embedded dictionary volumes for DICT: <word> | ...
	prefix := "dict: " + cleanWord + " |"
	for _, vol := range r.dictionaryVolumes {
		for _, line := range strings.Split(vol.Content, "\n") {
			trimmed := strings.TrimSpace(line)
			if strings.HasPrefix(strings.ToLower(trimmed), prefix) {
				return Article{
					Title:   "Dictionary Entry: " + cleanWord,
					Topic:   "English Dictionary",
					Source:  "dictionary:word:" + cleanWord,
					Content: trimmed,
				}, nil
			}
		}
	}
	return Article{}, fmt.Errorf("no dictionary entry found for %q", cleanWord)
}

func (r *Reader) fetchLiveDictionaryAPI(ctx context.Context, word string) (Article, error) {
	endpoint := "https://api.dictionaryapi.dev/api/v2/entries/en/" + url.PathEscape(word)
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, endpoint, nil)
	if err != nil {
		return Article{}, err
	}
	req.Header.Set("User-Agent", userAgentHeader)

	resp, err := r.httpClient.Do(req)
	if err != nil {
		return Article{}, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return Article{}, fmt.Errorf("dictionary api status %d", resp.StatusCode)
	}

	var entries []struct {
		Word     string `json:"word"`
		Meanings []struct {
			PartOfSpeech string `json:"partOfSpeech"`
			Definitions  []struct {
				Definition string `json:"definition"`
				Example    string `json:"example"`
			} `json:"definitions"`
		} `json:"meanings"`
	}
	body, err := io.ReadAll(io.LimitReader(resp.Body, maxHTTPBytes))
	if err != nil {
		return Article{}, err
	}
	if err := json.Unmarshal(body, &entries); err != nil || len(entries) == 0 {
		return Article{}, fmt.Errorf("decode dictionary entry failed")
	}

	var b strings.Builder
	count := 0
	for _, m := range entries[0].Meanings {
		pos := strings.TrimSpace(m.PartOfSpeech)
		if pos == "" {
			pos = "word"
		}
		for _, d := range m.Definitions {
			def := strings.TrimSpace(d.Definition)
			if def == "" {
				continue
			}
			ex := strings.TrimSpace(d.Example)
			if ex != "" {
				fmt.Fprintf(&b, "DICT: %s | %s | %s | %s\n", word, pos, def, ex)
			} else {
				fmt.Fprintf(&b, "DICT: %s | %s | %s\n", word, pos, def)
			}
			count++
			if count >= 4 {
				break
			}
		}
		if count >= 4 {
			break
		}
	}
	if count == 0 {
		return Article{}, fmt.Errorf("empty dictionary meanings for %q", word)
	}
	return Article{
		Title:   "Live Dictionary: " + entries[0].Word,
		Topic:   "English Dictionary",
		Source:  "dictionary:live:" + word,
		Content: b.String(),
	}, nil
}

func (r *Reader) nextUnreadLocalNote(topicFilter string, seen map[string]bool) (Article, bool) {
	if r.notesDir == "" {
		return Article{}, false
	}
	entries, err := os.ReadDir(r.notesDir)
	if err != nil {
		return Article{}, false
	}
	qLower := strings.ToLower(topicFilter)

	for _, entry := range entries {
		if entry.IsDir() {
			continue
		}
		name := entry.Name()
		if strings.EqualFold(name, "HOW_TO_ADD_NOTES.txt") {
			continue
		}
		ext := strings.ToLower(filepath.Ext(name))
		if ext != ".txt" && ext != ".md" {
			continue
		}
		srcKey := "local:" + name
		if seen[srcKey] {
			continue
		}

		fullPath := filepath.Join(r.notesDir, name)
		f, err := os.Open(fullPath)
		if err != nil {
			continue
		}
		raw, err := io.ReadAll(io.LimitReader(f, maxLocalFileBytes))
		_ = f.Close()
		if err != nil {
			continue
		}

		content := strings.TrimSpace(string(raw))
		if len(strings.Fields(content)) < 10 {
			continue
		}
		title := strings.TrimSuffix(name, filepath.Ext(name))
		if qLower != "" &&
			!strings.Contains(strings.ToLower(title), qLower) &&
			!strings.Contains(strings.ToLower(content), qLower) {
			continue
		}

		return Article{
			Title:   title,
			Topic:   "Personal Study Notes",
			Source:  srcKey,
			Content: content,
		}, true
	}
	return Article{}, false
}

func (r *Reader) fetchWikipediaArticle(ctx context.Context, topic string, seen map[string]bool) (Article, error) {
	title := ""
	topicLabel := "General Knowledge"

	if topic != "" {
		topicLabel = topic
		foundTitle, err := r.searchWikipediaTitle(ctx, topic, seen)
		if err != nil {
			return Article{}, err
		}
		title = foundTitle
	} else {
		randTitle, randTopic, err := r.randomWikipediaTitle(ctx)
		if err != nil {
			return Article{}, err
		}
		title = randTitle
		if randTopic != "" {
			topicLabel = randTopic
		}
	}

	extract, pageURL, err := r.fetchWikipediaExtract(ctx, title)
	if err != nil {
		return Article{}, err
	}
	src := "wikipedia:" + title
	if pageURL != "" {
		src = pageURL
	}
	return Article{
		Title:   title,
		Topic:   topicLabel,
		Source:  src,
		Content: extract,
	}, nil
}

func (r *Reader) randomWikipediaTitle(ctx context.Context) (string, string, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, "https://en.wikipedia.org/api/rest_v1/page/random/summary", nil)
	if err != nil {
		return "", "", err
	}
	req.Header.Set("User-Agent", userAgentHeader)

	resp, err := r.httpClient.Do(req)
	if err != nil {
		return "", "", err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return "", "", fmt.Errorf("wikipedia status %d", resp.StatusCode)
	}

	var payload struct {
		Title       string `json:"title"`
		Description string `json:"description"`
	}
	body, err := io.ReadAll(io.LimitReader(resp.Body, maxHTTPBytes))
	if err != nil {
		return "", "", err
	}
	if err := json.Unmarshal(body, &payload); err != nil || strings.TrimSpace(payload.Title) == "" {
		return "", "", fmt.Errorf("decode random summary failed")
	}
	return payload.Title, payload.Description, nil
}

func (r *Reader) searchWikipediaTitle(ctx context.Context, query string, seen map[string]bool) (string, error) {
	endpoint := "https://en.wikipedia.org/w/api.php?action=query&list=search&format=json&srlimit=10&srsearch=" + url.QueryEscape(query)
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, endpoint, nil)
	if err != nil {
		return "", err
	}
	req.Header.Set("User-Agent", userAgentHeader)

	resp, err := r.httpClient.Do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("wikipedia search status %d", resp.StatusCode)
	}

	var result struct {
		Query struct {
			Search []struct {
				Title string `json:"title"`
			} `json:"search"`
		} `json:"query"`
	}
	body, err := io.ReadAll(io.LimitReader(resp.Body, maxHTTPBytes))
	if err != nil {
		return "", err
	}
	if err := json.Unmarshal(body, &result); err != nil || len(result.Query.Search) == 0 {
		return "", fmt.Errorf("no search results for %q", query)
	}

	for _, hit := range result.Query.Search {
		wikiURL := "https://en.wikipedia.org/wiki/" + strings.ReplaceAll(hit.Title, " ", "_")
		if !seen[wikiURL] && !seen["wikipedia:"+hit.Title] {
			return hit.Title, nil
		}
	}
	return result.Query.Search[r.rng.Intn(len(result.Query.Search))].Title, nil
}

func (r *Reader) fetchWikipediaExtract(ctx context.Context, title string) (string, string, error) {
	endpoint := "https://en.wikipedia.org/w/api.php?action=query&prop=extracts&explaintext=1&exchars=4500&redirects=1&format=json&titles=" + url.QueryEscape(title)
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, endpoint, nil)
	if err != nil {
		return "", "", err
	}
	req.Header.Set("User-Agent", userAgentHeader)

	resp, err := r.httpClient.Do(req)
	if err != nil {
		return "", "", err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return "", "", fmt.Errorf("wikipedia extract status %d", resp.StatusCode)
	}

	var payload struct {
		Query struct {
			Pages map[string]struct {
				Title   string `json:"title"`
				Extract string `json:"extract"`
			} `json:"pages"`
		} `json:"query"`
	}
	body, err := io.ReadAll(io.LimitReader(resp.Body, maxHTTPBytes))
	if err != nil {
		return "", "", err
	}
	if err := json.Unmarshal(body, &payload); err != nil {
		return "", "", err
	}
	for _, p := range payload.Query.Pages {
		cleaned := strings.TrimSpace(p.Extract)
		if cleaned != "" {
			pageURL := "https://en.wikipedia.org/wiki/" + strings.ReplaceAll(p.Title, " ", "_")
			return cleaned, pageURL, nil
		}
	}
	return "", "", fmt.Errorf("empty wikipedia extract for %q", title)
}

func ensureNotesReadme(notesDir string) {
	p := filepath.Join(notesDir, "HOW_TO_ADD_NOTES.txt")
	if _, err := os.Stat(p); err == nil {
		return
	}
	guide := "Drop any .txt or .md study notes, custom dictionary lists (DICT: word | pos | definition | example),\n" +
		"lecture summaries, or articles into this folder!\n" +
		"After Matrix finishes its built-in Dictionaries and Grammar Primer, it will read your notes first.\n"
	_ = os.WriteFile(p, []byte(guide), 0o644)
}
