package corpus

import (
	"bytes"
	"compress/gzip"
	"context"
	_ "embed"
	"encoding/json"
	"encoding/xml"
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

//go:embed lexicon.json.gz
var embeddedLexiconGZ []byte

//go:embed grammar_curriculum.json
var embeddedGrammarJSON []byte

//go:embed seed_articles.json
var embeddedSeedJSON []byte

const (
	maxHTTPBytes       = 384 * 1024
	maxLocalFileBytes  = 512 * 1024
	httpRequestTimeout = 4 * time.Second
	userAgentHeader    = "MatrixStudyBuddy/2.0 (CLI On-Device Neural Study Model)"
)

// Article represents a dictionary chapter, grammar lesson, research abstract, or article.
type Article struct {
	Title   string `json:"title"`
	Topic   string `json:"topic"`
	Source  string `json:"source"`
	Content string `json:"content"`
}

type lexiconEntry struct {
	Word    string `json:"w"`
	POS     string `json:"p"`
	Def     string `json:"d"`
	Example string `json:"e"`
}

type lexiconChapter struct {
	Chapter int            `json:"chapter"`
	Tier    string         `json:"tier"`
	Title   string         `json:"title"`
	Entries []lexiconEntry `json:"entries"`
}

// Reader streams training items in curriculum order:
//  1. Core English Dictionary Chapters + Live Web Dictionary entries FIRST
//  2. English Grammar & Conversational Prose SECOND
//  3. Local Notes + Multi-Source Live Web (Wikipedia, Wikibooks, arXiv) + Extended 4,800-word Lexicon + Encyclopedia THIRD
type Reader struct {
	notesDir         string
	httpClient       *http.Client
	coreDicts        []Article
	extendedDicts    []Article
	dictIndex        map[string]lexiconEntry
	totalDictWords   int
	grammarLessons   []Article
	seedArticles     []Article
	rng              *rand.Rand
	offlineUntil     time.Time
	webRoundRobin    int
	stepCounter      int
	DisableLiveFetch bool
}

// NewReader decompresses the 4,800-word English lexicon, grammar chapters, and offline encyclopedia.
func NewReader(dataDir string) (*Reader, error) {
	gr, err := gzip.NewReader(bytes.NewReader(embeddedLexiconGZ))
	if err != nil {
		return nil, fmt.Errorf("decompress lexicon.json.gz: %w", err)
	}
	var rawChapters []lexiconChapter
	if err := json.NewDecoder(gr).Decode(&rawChapters); err != nil {
		_ = gr.Close()
		return nil, fmt.Errorf("decode lexicon.json.gz: %w", err)
	}
	_ = gr.Close()

	var coreDicts, extDicts []Article
	dictIndex := make(map[string]lexiconEntry, 5000)
	totalWords := 0

	for _, ch := range rawChapters {
		var sb strings.Builder
		for _, e := range ch.Entries {
			dictIndex[e.Word] = e
			totalWords++
			if e.Example != "" {
				fmt.Fprintf(&sb, "DICT: %s | %s | %s | %s\n", e.Word, e.POS, e.Def, e.Example)
			} else {
				fmt.Fprintf(&sb, "DICT: %s | %s | %s\n", e.Word, e.POS, e.Def)
			}
		}
		srcPrefix := "dictionary:ext"
		if ch.Tier == "core" {
			srcPrefix = "dictionary:core"
		}
		art := Article{
			Title:   ch.Title,
			Topic:   "English Dictionary",
			Source:  fmt.Sprintf("%s-%02d", srcPrefix, ch.Chapter),
			Content: sb.String(),
		}
		if ch.Tier == "core" {
			coreDicts = append(coreDicts, art)
		} else {
			extDicts = append(extDicts, art)
		}
	}

	var grammar []Article
	if err := json.Unmarshal(embeddedGrammarJSON, &grammar); err != nil {
		return nil, fmt.Errorf("load grammar curriculum: %w", err)
	}

	var seeds []Article
	if err := json.Unmarshal(embeddedSeedJSON, &seeds); err != nil {
		return nil, fmt.Errorf("load seed articles: %w", err)
	}

	notesDir := ""
	if strings.TrimSpace(dataDir) != "" {
		notesDir = filepath.Join(dataDir, "study_notes")
		_ = os.MkdirAll(notesDir, 0o755)
		ensureNotesReadme(notesDir)
	}

	return &Reader{
		notesDir:       notesDir,
		httpClient:     &http.Client{Timeout: httpRequestTimeout},
		coreDicts:      coreDicts,
		extendedDicts:  extDicts,
		dictIndex:      dictIndex,
		totalDictWords: totalWords,
		grammarLessons: grammar,
		seedArticles:   seeds,
		rng:            rand.New(rand.NewSource(time.Now().UnixNano())),
	}, nil
}

// NotesDir returns the local directory where users can drop .txt/.md study files.
func (r *Reader) NotesDir() string { return r.notesDir }

// DictionaryCount returns the number of mandatory Phase-1 core dictionary chapters.
func (r *Reader) DictionaryCount() int { return len(r.coreDicts) }

// TotalDictionaryWords returns total embedded dictionary headwords across all 60 chapters (4,800).
func (r *Reader) TotalDictionaryWords() int { return r.totalDictWords }

// DictionaryWordCount returns total embedded dictionary headwords (4,800).
func (r *Reader) DictionaryWordCount() int { return r.totalDictWords }

// ExtendedDictionaryCount returns the number of Phase-3 extended dictionary chapters (52).
func (r *Reader) ExtendedDictionaryCount() int { return len(r.extendedDicts) }

// DictionaryCompleted counts how many Phase-1 core dictionary chapters have been completed.
func (r *Reader) DictionaryCompleted(seen map[string]bool) int {
	done := 0
	for _, d := range r.coreDicts {
		if seen[d.Source] {
			done++
		}
	}
	return done
}

// GrammarCount returns how many sequential grammar chapters exist in Phase 2.
func (r *Reader) GrammarCount() int { return len(r.grammarLessons) }

// GrammarCompleted counts how many Phase-2 grammar chapters have been completed.
func (r *Reader) GrammarCompleted(seen map[string]bool) int {
	done := 0
	for _, g := range r.grammarLessons {
		if seen[g.Source] {
			done++
		}
	}
	return done
}

// SeedCount returns how many offline multi-paragraph study articles are embedded.
func (r *Reader) SeedCount() int { return len(r.seedArticles) }

// NextArticle returns the next training document:
//   - Phase 1: Core English Dictionary Chapters (640 words) + Live Web Dictionary batches FIRST
//   - Phase 2: English Grammar & Syntax Chapters SECOND
//   - Phase 3: Local Notes + Multi-Source Live Web (Wikipedia, Wikibooks, arXiv) + Extended 4,160-word Dictionary + Offline Articles
func (r *Reader) NextArticle(ctx context.Context, topicFilter string, seenSources map[string]bool) (Article, error) {
	topicFilter = strings.TrimSpace(topicFilter)
	r.stepCounter++

	if topicFilter == "" {
		// Phase 1: Core English Dictionaries FIRST (plus live web dictionary enrichment when online!)
		for idx, dictChap := range r.coreDicts {
			if !seenSources[dictChap.Source] {
				// Halfway through Phase 1, also try pulling a live web dictionary batch if online
				liveKey := fmt.Sprintf("dictionary:live-batch-%d", idx/3)
				if idx > 0 && idx%3 == 0 && !seenSources[liveKey] && r.canFetchLive() {
					if liveBatch, err := r.fetchLiveDictionaryBatch(ctx, liveKey); err == nil {
						return liveBatch, nil
					}
				}
				return dictChap, nil
			}
		}

		// Phase 2: English Grammar & Sentence Structure SECOND
		for _, lesson := range r.grammarLessons {
			if !seenSources[lesson.Source] {
				return lesson, nil
			}
		}
	}

	// Explicit dictionary focus topic
	qLower := strings.ToLower(topicFilter)
	if qLower == "dictionary" || qLower == "lexicon" || qLower == "vocabulary" {
		for _, d := range r.coreDicts {
			if !seenSources[d.Source] {
				return d, nil
			}
		}
		for _, d := range r.extendedDicts {
			if !seenSources[d.Source] {
				return d, nil
			}
		}
		if len(r.extendedDicts) > 0 {
			return r.extendedDicts[r.rng.Intn(len(r.extendedDicts))], nil
		}
	}

	// Phase 3A: Unread user study notes (.txt / .md)
	if localArt, ok := r.nextUnreadLocalNote(topicFilter, seenSources); ok {
		return localArt, nil
	}

	// Phase 3B: Periodically interleave unread Extended Dictionary chapters (Ch.9..60, 4,160 more words!)
	if topicFilter == "" && r.stepCounter%4 == 0 {
		for _, ext := range r.extendedDicts {
			if !seenSources[ext.Source] {
				return ext, nil
			}
		}
	}

	// Phase 3C: Multi-source Live Web Article Fetching (Wikipedia Full Text + Wikibooks + arXiv)
	if r.canFetchLive() {
		liveArt, err := r.fetchMultiSourceWebArticle(ctx, topicFilter, seenSources)
		if err == nil && len(strings.Fields(liveArt.Content)) >= 40 {
			return liveArt, nil
		}
		if err != nil {
			r.offlineUntil = time.Now().Add(45 * time.Second)
		}
	}

	// Phase 3D: Offline Encyclopedia + remaining Extended Dictionary chapters
	candidates := make([]Article, 0, len(r.seedArticles))
	unread := make([]Article, 0, len(r.seedArticles))
	for _, a := range r.seedArticles {
		if qLower != "" &&
			!strings.Contains(strings.ToLower(a.Topic), qLower) &&
			!strings.Contains(strings.ToLower(a.Title), qLower) &&
			!strings.Contains(strings.ToLower(a.Content), qLower) {
			continue
		}
		candidates = append(candidates, a)
		if !seenSources[a.Source] {
			unread = append(unread, a)
		}
	}

	if len(unread) > 0 {
		return unread[r.rng.Intn(len(unread))], nil
	}
	if topicFilter == "" {
		for _, ext := range r.extendedDicts {
			if !seenSources[ext.Source] {
				return ext, nil
			}
		}
	}
	if len(candidates) > 0 {
		return candidates[r.rng.Intn(len(candidates))], nil
	}
	return r.seedArticles[r.rng.Intn(len(r.seedArticles))], nil
}

// FetchDictionaryEntry looks up a word across live Free Dictionary API, Wiktionary REST API,
// and the embedded 4,800-word lexicon.
func (r *Reader) FetchDictionaryEntry(ctx context.Context, word string) (Article, error) {
	cleanWord := strings.ToLower(strings.TrimSpace(word))
	if cleanWord == "" {
		return Article{}, fmt.Errorf("empty word")
	}

	if r.canFetchLive() {
		if art, err := r.fetchLiveDictionaryAPI(ctx, cleanWord); err == nil {
			return art, nil
		}
		if art, err := r.fetchWiktionaryAPI(ctx, cleanWord); err == nil {
			return art, nil
		}
	}

	if e, ok := r.dictIndex[cleanWord]; ok {
		line := fmt.Sprintf("DICT: %s | %s | %s", e.Word, e.POS, e.Def)
		if e.Example != "" {
			line += " | " + e.Example
		}
		return Article{
			Title:   "Dictionary Entry: " + e.Word,
			Topic:   "English Dictionary",
			Source:  "dictionary:word:" + e.Word,
			Content: line,
		}, nil
	}

	return Article{}, fmt.Errorf("no dictionary entry found for %q", cleanWord)
}

func (r *Reader) canFetchLive() bool {
	return !r.DisableLiveFetch && time.Now().After(r.offlineUntil)
}

func (r *Reader) fetchLiveDictionaryBatch(ctx context.Context, batchKey string) (Article, error) {
	probeWords := []string{"inference", "cognition", "synthesis", "theorem", "empirical", "entropy", "algorithm", "plasticity"}
	start := r.rng.Intn(len(probeWords))
	var sb strings.Builder
	fetched := 0

	for i := 0; i < 3; i++ {
		w := probeWords[(start+i)%len(probeWords)]
		art, err := r.fetchLiveDictionaryAPI(ctx, w)
		if err != nil {
			art, err = r.fetchWiktionaryAPI(ctx, w)
		}
		if err == nil {
			sb.WriteString(art.Content)
			if !strings.HasSuffix(art.Content, "\n") {
				sb.WriteByte('\n')
			}
			fetched++
		}
	}
	if fetched == 0 {
		r.offlineUntil = time.Now().Add(45 * time.Second)
		return Article{}, fmt.Errorf("live dictionary unreachable")
	}
	return Article{
		Title:   "Live Web Dictionary Enrichment",
		Topic:   "English Dictionary",
		Source:  batchKey,
		Content: sb.String(),
	}, nil
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
		return Article{}, fmt.Errorf("empty meanings")
	}
	return Article{
		Title:   "Live Dictionary: " + entries[0].Word,
		Topic:   "English Dictionary",
		Source:  "dictionary:live:" + word,
		Content: b.String(),
	}, nil
}

func (r *Reader) fetchWiktionaryAPI(ctx context.Context, word string) (Article, error) {
	endpoint := "https://en.wiktionary.org/api/rest_v1/page/definition/" + url.PathEscape(word)
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
		return Article{}, fmt.Errorf("wiktionary status %d", resp.StatusCode)
	}

	var payload map[string][]struct {
		PartOfSpeech string `json:"partOfSpeech"`
		Definitions  []struct {
			Definition string `json:"definition"`
		} `json:"definitions"`
	}
	body, err := io.ReadAll(io.LimitReader(resp.Body, maxHTTPBytes))
	if err != nil {
		return Article{}, err
	}
	if err := json.Unmarshal(body, &payload); err != nil {
		return Article{}, err
	}

	enSections := payload["en"]
	var b strings.Builder
	count := 0
	for _, sec := range enSections {
		pos := strings.ToLower(strings.TrimSpace(sec.PartOfSpeech))
		if pos == "" {
			pos = "word"
		}
		for _, d := range sec.Definitions {
			cleanDef := stripHTMLTags(d.Definition)
			if len(cleanDef) < 6 {
				continue
			}
			fmt.Fprintf(&b, "DICT: %s | %s | %s\n", word, pos, cleanDef)
			count++
			if count >= 3 {
				break
			}
		}
		if count >= 3 {
			break
		}
	}
	if count == 0 {
		return Article{}, fmt.Errorf("no English wiktionary definitions")
	}
	return Article{
		Title:   "Wiktionary: " + word,
		Topic:   "English Dictionary",
		Source:  "dictionary:wiktionary:" + word,
		Content: b.String(),
	}, nil
}

func (r *Reader) fetchMultiSourceWebArticle(ctx context.Context, topic string, seen map[string]bool) (Article, error) {
	r.webRoundRobin++
	switch r.webRoundRobin % 3 {
	case 0:
		if art, err := r.fetchWikipediaArticle(ctx, topic, seen); err == nil {
			return art, nil
		}
	case 1:
		if art, err := r.fetchWikibooksChapter(ctx, topic, seen); err == nil {
			return art, nil
		}
		return r.fetchWikipediaArticle(ctx, topic, seen)
	case 2:
		if art, err := r.fetchArxivAbstracts(ctx, topic, seen); err == nil {
			return art, nil
		}
		return r.fetchWikipediaArticle(ctx, topic, seen)
	}
	return r.fetchWikipediaArticle(ctx, topic, seen)
}

func (r *Reader) fetchWikipediaArticle(ctx context.Context, topic string, seen map[string]bool) (Article, error) {
	title := ""
	topicLabel := "Wikipedia"

	if topic != "" {
		topicLabel = topic
		foundTitle, err := r.searchMediaWikiTitle(ctx, "https://en.wikipedia.org/w/api.php", topic, seen)
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

	extract, pageURL, err := r.fetchMediaWikiExtract(ctx, "https://en.wikipedia.org/w/api.php", "https://en.wikipedia.org/wiki/", title, 14000)
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

func (r *Reader) fetchWikibooksChapter(ctx context.Context, topic string, seen map[string]bool) (Article, error) {
	query := topic
	if query == "" {
		subjects := []string{"Physics", "Biology", "Calculus", "Algorithms", "Chemistry", "Neuroscience", "Linear Algebra"}
		query = subjects[r.rng.Intn(len(subjects))]
	}
	title, err := r.searchMediaWikiTitle(ctx, "https://en.wikibooks.org/w/api.php", query, seen)
	if err != nil {
		return Article{}, err
	}
	extract, pageURL, err := r.fetchMediaWikiExtract(ctx, "https://en.wikibooks.org/w/api.php", "https://en.wikibooks.org/wiki/", title, 12000)
	if err != nil {
		return Article{}, err
	}
	return Article{
		Title:   "Wikibooks: " + title,
		Topic:   query,
		Source:  pageURL,
		Content: extract,
	}, nil
}

func (r *Reader) fetchArxivAbstracts(ctx context.Context, topic string, seen map[string]bool) (Article, error) {
	query := topic
	if query == "" {
		subjects := []string{"machine learning", "quantum physics", "neuroscience", "mathematics", "astronomy"}
		query = subjects[r.rng.Intn(len(subjects))]
	}
	endpoint := "https://export.arxiv.org/api/query?max_results=5&search_query=all:" + url.QueryEscape(query)
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
		return Article{}, fmt.Errorf("arxiv status %d", resp.StatusCode)
	}

	var feed struct {
		Entries []struct {
			ID      string `xml:"id"`
			Title   string `xml:"title"`
			Summary string `xml:"summary"`
		} `xml:"entry"`
	}
	body, err := io.ReadAll(io.LimitReader(resp.Body, maxHTTPBytes))
	if err != nil {
		return Article{}, err
	}
	if err := xml.Unmarshal(body, &feed); err != nil || len(feed.Entries) == 0 {
		return Article{}, fmt.Errorf("no arxiv entries")
	}

	for _, e := range feed.Entries {
		id := strings.TrimSpace(e.ID)
		if id == "" || seen[id] {
			continue
		}
		cleanTitle := strings.Join(strings.Fields(e.Title), " ")
		cleanSum := strings.Join(strings.Fields(e.Summary), " ")
		if len(strings.Fields(cleanSum)) >= 35 {
			return Article{
				Title:   "arXiv: " + cleanTitle,
				Topic:   query,
				Source:  id,
				Content: cleanTitle + ".\n\n" + cleanSum,
			}, nil
		}
	}
	return Article{}, fmt.Errorf("all fetched arxiv abstracts already seen")
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

func (r *Reader) searchMediaWikiTitle(ctx context.Context, apiBase, query string, seen map[string]bool) (string, error) {
	endpoint := apiBase + "?action=query&list=search&format=json&srlimit=10&srsearch=" + url.QueryEscape(query)
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
		return "", fmt.Errorf("mediawiki search status %d", resp.StatusCode)
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
		slug := strings.ReplaceAll(hit.Title, " ", "_")
		if !seen["https://en.wikipedia.org/wiki/"+slug] && !seen["https://en.wikibooks.org/wiki/"+slug] {
			return hit.Title, nil
		}
	}
	return result.Query.Search[r.rng.Intn(len(result.Query.Search))].Title, nil
}

func (r *Reader) fetchMediaWikiExtract(ctx context.Context, apiBase, wikiPrefix, title string, maxChars int) (string, string, error) {
	endpoint := fmt.Sprintf("%s?action=query&prop=extracts&explaintext=1&exchars=%d&redirects=1&format=json&titles=%s",
		apiBase, maxChars, url.QueryEscape(title))
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
		return "", "", fmt.Errorf("mediawiki extract status %d", resp.StatusCode)
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
			pageURL := wikiPrefix + strings.ReplaceAll(p.Title, " ", "_")
			return cleaned, pageURL, nil
		}
	}
	return "", "", fmt.Errorf("empty extract for %q", title)
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
		if len(strings.Fields(content)) < 5 {
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

func stripHTMLTags(s string) string {
	var out strings.Builder
	inTag := false
	for _, r := range s {
		if r == '<' {
			inTag = true
			continue
		}
		if r == '>' {
			inTag = false
			continue
		}
		if !inTag {
			out.WriteRune(r)
		}
	}
	return strings.TrimSpace(strings.Join(strings.Fields(out.String()), " "))
}

func ensureNotesReadme(notesDir string) {
	p := filepath.Join(notesDir, "HOW_TO_ADD_NOTES.txt")
	if _, err := os.Stat(p); err == nil {
		return
	}
	guide := "Drop any .txt or .md study notes, custom dictionary lists (DICT: word | pos | definition | example),\n" +
		"lecture summaries, or articles into this folder!\n" +
		"After Matrix finishes its Core Dictionaries and Grammar Primer, it will read your notes first.\n"
	_ = os.WriteFile(p, []byte(guide), 0o644)
}
