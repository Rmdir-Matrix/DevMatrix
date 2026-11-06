package brain

import (
	"compress/gzip"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"math"
	mrand "math/rand"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/Rmdir-Matrix/DevMatrix/internal/corpus"
	"github.com/Rmdir-Matrix/DevMatrix/internal/specs"
)

const (
	maxNGramKeys        = 50000
	maxFollowersPerKey  = 64
	maxPromptCueKeys    = 12000
	maxStartersPerCue   = 24
)

// Passage stores a learned study excerpt indexed by keywords and source metadata.
type Passage struct {
	Title    string   `json:"title"`
	Topic    string   `json:"topic"`
	Source   string   `json:"source"`
	Text     string   `json:"text"`
	Keywords []string `json:"keywords"`
}

// TrainStepReport summarizes the metrics from training on a single lesson or article.
type TrainStepReport struct {
	ArticleTitle   string
	Topic          string
	Source         string
	TokensRead     int
	SentencesRead  int
	AvgLoss        float64
	SmoothedLoss   float64
	OldLevel       int
	NewLevel       int
	XPGained       int64
	ActiveSynapses int
	VocabSize      int
	SampleKeywords []string
}

// Brain encapsulates a named Matrix instance's neural network, learned transition statistics,
// semantic article memory, and progression state. Starts 100% blank with zero pre-built answers.
type Brain struct {
	mu sync.Mutex `json:"-"`

	ID             string                    `json:"id"`
	Name           string                    `json:"name"`
	CreatedAt      string                    `json:"created_at"`
	UpdatedAt      string                    `json:"updated_at"`
	Level            int                       `json:"level"`
	XP               int64                     `json:"xp"`
	ArticlesRead     int                       `json:"articles_read"`
	DictionariesDone int                       `json:"dictionaries_done"`
	GrammarDone      int                       `json:"grammar_done"`
	TokensTrained  int64                     `json:"tokens_trained"`
	TrainingSteps  int64                     `json:"training_steps"`
	InitialLoss    float64                   `json:"initial_loss"`
	SmoothedLoss   float64                   `json:"smoothed_loss"`
	MaxPassages    int                       `json:"max_passages"`
	TopicsCount    map[string]int            `json:"topics_count"`
	SeenSources    map[string]bool           `json:"seen_sources"`
	RecentTitles   []string                  `json:"recent_titles"`
	Passages       []Passage                 `json:"passages"`
	NGramCounts    map[string]map[string]int `json:"ngram_counts"`
	PromptStarters map[string]map[string]int `json:"prompt_starters"`
	Net            *NeuralNet                `json:"net"`
}

// NewBrain constructs a completely blank Level-1 Matrix Brain with zero learned words or answers.
func NewBrain(name string, prof specs.ComputeProfile) *Brain {
	if strings.TrimSpace(name) == "" {
		name = "Matrix"
	}
	maxPass := prof.MaxPassages
	if maxPass < 200 {
		maxPass = 400
	}
	now := time.Now().UTC().Format(time.RFC3339)

	return &Brain{
		ID:             randomHexID(8),
		Name:           strings.TrimSpace(name),
		CreatedAt:      now,
		UpdatedAt:      now,
		Level:          1,
		XP:             0,
		MaxPassages:    maxPass,
		TopicsCount:    make(map[string]int),
		SeenSources:    make(map[string]bool),
		RecentTitles:   make([]string, 0, 16),
		Passages:       make([]Passage, 0, 128),
		NGramCounts:    make(map[string]map[string]int, 512),
		PromptStarters: make(map[string]map[string]int, 256),
		Net:            NewNeuralNet(prof.EmbedDim, prof.HiddenNeurons, prof.MaxVocab, time.Now().UnixNano()),
	}
}

// XPForLevel returns cumulative XP required to reach a target level (>= 1).
func XPForLevel(level int) int64 {
	if level <= 1 {
		return 0
	}
	l := float64(level - 1)
	return int64(55.0*l + 0.85*l*l)
}

// LevelFromXP calculates the current Matrix level from total XP.
func LevelFromXP(xp int64) int {
	if xp <= 0 {
		return 1
	}
	lvl := 1
	for XPForLevel(lvl+1) <= xp {
		lvl++
	}
	return lvl
}

// ApplyProfileCaps updates soft memory capacities when loaded on a machine with new RAM settings.
func (b *Brain) ApplyProfileCaps(prof specs.ComputeProfile) {
	b.mu.Lock()
	defer b.mu.Unlock()

	if prof.MaxPassages >= 200 {
		b.MaxPassages = prof.MaxPassages
		if len(b.Passages) > b.MaxPassages {
			b.Passages = b.Passages[len(b.Passages)-b.MaxPassages:]
		}
	}
	if b.Net != nil && prof.MaxVocab >= 512 {
		if prof.MaxVocab > len(b.Net.IDToToken) {
			b.Net.MaxVocab = prof.MaxVocab
		} else {
			b.Net.MaxVocab = len(b.Net.IDToToken)
		}
	}
}

// TrainOnArticle ingests a grammar lesson or real article, runs neural backpropagation across
// prose sentences, conversational QA pairs, and self-thoughts, and updates learned statistics.
func (b *Brain) TrainOnArticle(art corpus.Article, epochs int) TrainStepReport {
	b.mu.Lock()
	defer b.mu.Unlock()

	if epochs < 1 {
		epochs = 2
	}
	b.ensureMaps()

	parsed := ParseCorpusContent(art.Content)
	rng := mrand.New(mrand.NewSource(time.Now().UnixNano()))

	tokenizedSentences := make([][]string, 0, len(parsed.Sentences))
	tokensRead := 0

	// 1. Register vocabulary across sentences, QA pairs, and thoughts first
	for _, sent := range parsed.Sentences {
		toks := Tokenize(sent)
		if len(toks) < 2 {
			continue
		}
		b.Net.RegisterTokens(toks, rng)
		b.recordSequenceCounts(bosToken, toks, 1)
		tokensRead += len(toks)
		tokenizedSentences = append(tokenizedSentences, toks)
	}
	for _, qa := range parsed.QAPairs {
		b.Net.RegisterTokens(qa.Prompt, rng)
		b.Net.RegisterTokens(qa.Response, rng)
		b.recordConditionedCounts(qa.Prompt, qa.Response, 3)
		tokensRead += len(qa.Prompt) + len(qa.Response)
	}
	for _, th := range parsed.Thoughts {
		b.Net.RegisterTokens(th, rng)
		stageCue := thoughtStageCue(b.Level)
		b.recordConditionedCounts([]string{thoughtToken, stageCue}, th, 3)
		tokensRead += len(th)
	}

	// Self-supervised study conditioning from real article sentences:
	// Every informative sentence teaches Matrix how to answer prompts about its keywords!
	type condSample struct {
		cond []string
		resp []string
	}
	var selfStudy []condSample
	titleKWs := ExtractKeywords(art.Title, 4)

	for _, sent := range parsed.Sentences {
		toks := Tokenize(sent)
		if len(toks) < 5 || len(toks) > 60 {
			continue
		}
		kw := ExtractKeywords(sent, 8)
		if len(kw) >= 2 {
			cues := append([]string(nil), kw...)
			for _, tk := range titleKWs {
				cues = append(cues, tk)
			}
			selfStudy = append(selfStudy, condSample{cond: cues, resp: toks})
			b.recordConditionedCounts(cues, toks, 2)

			if !strings.HasPrefix(art.Source, "curriculum:") {
				b.storePassage(Passage{
					Title:    art.Title,
					Topic:    art.Topic,
					Source:   art.Source,
					Text:     sent,
					Keywords: kw,
				})
			}
		}
	}

	// Once dictionary & grammar foundations are learned, reading real articles also trains new article-grounded thoughts
	sampleKW := ExtractKeywords(art.Title+" "+art.Content, 6)
	isFoundational := strings.HasPrefix(art.Source, "dictionary:") || strings.HasPrefix(art.Source, "curriculum:")
	if !isFoundational && len(sampleKW) >= 2 && b.Net.KnowsToken("learning") {
		var dynThought []string
		if b.Level >= 75 && b.Net.KnowsToken("connections") {
			dynThought = Tokenize(fmt.Sprintf(
				"I am learning about %s and %s which makes more neuron connections in my brain.",
				sampleKW[0], sampleKW[1],
			))
		} else if b.Level >= 20 && b.Net.KnowsToken("connect") {
			dynThought = Tokenize(fmt.Sprintf(
				"Reading about %s helps me connect %s and %s in my memory.",
				sampleKW[0], sampleKW[0], sampleKW[1],
			))
		} else {
			dynThought = Tokenize(fmt.Sprintf("I'm learning stuff about %s and %s!", sampleKW[0], sampleKW[1]))
		}
		b.Net.RegisterTokens(dynThought, rng)
		stageCue := thoughtStageCue(b.Level)
		b.recordConditionedCounts([]string{thoughtToken, stageCue, sampleKW[0]}, dynThought, 2)
		parsed.Thoughts = append(parsed.Thoughts, dynThought)
	}

	// 2. Run neural backpropagation across prose, QA pairs, self-study associations, and thoughts
	lr := float32(0.13 / (1.0 + 0.00008*float64(b.TrainingSteps)))
	if lr < 0.025 {
		lr = 0.025
	}

	var lossSum float64
	var predSteps int

	for ep := 0; ep < epochs; ep++ {
		for _, toks := range tokenizedSentences {
			l, st := b.Net.TrainSentence(toks, lr, rng)
			lossSum += l
			predSteps += st
		}
		for _, qa := range parsed.QAPairs {
			l, st := b.Net.TrainConditioned(qa.Prompt, qa.Response, lr*1.15, rng)
			lossSum += l
			predSteps += st
		}
		for _, ss := range selfStudy {
			l, st := b.Net.TrainConditioned(ss.cond, ss.resp, lr, rng)
			lossSum += l
			predSteps += st
		}
		stageCue := thoughtStageCue(b.Level)
		for _, th := range parsed.Thoughts {
			l, st := b.Net.TrainConditioned([]string{thoughtToken, stageCue}, th, lr*1.1, rng)
			lossSum += l
			predSteps += st
		}
	}

	avgLoss := 0.0
	if predSteps > 0 {
		avgLoss = lossSum / float64(predSteps)
	}
	if b.InitialLoss == 0 && avgLoss > 0 {
		b.InitialLoss = avgLoss
		b.SmoothedLoss = avgLoss
	} else if avgLoss > 0 {
		b.SmoothedLoss = 0.80*b.SmoothedLoss + 0.20*avgLoss
	}

	oldLevel := b.Level
	xpGain := int64(48 + minInt(tokensRead/3, 185))
	if !b.SeenSources[art.Source] && art.Source != "" {
		xpGain += 35
		if strings.HasPrefix(art.Source, "dictionary:") {
			b.DictionariesDone++
		} else if strings.HasPrefix(art.Source, "curriculum:") {
			b.GrammarDone++
		}
	}

	b.XP += xpGain
	b.Level = LevelFromXP(b.XP)
	b.ArticlesRead++
	b.TokensTrained += int64(tokensRead)
	b.TrainingSteps += int64(predSteps)
	if art.Source != "" {
		b.SeenSources[art.Source] = true
	}
	topic := strings.TrimSpace(art.Topic)
	if topic == "" {
		topic = "General Knowledge"
	}
	b.TopicsCount[topic]++
	b.pushRecentTitle(art.Title)
	b.UpdatedAt = time.Now().UTC().Format(time.RFC3339)

	return TrainStepReport{
		ArticleTitle:   art.Title,
		Topic:          topic,
		Source:         art.Source,
		TokensRead:     tokensRead,
		SentencesRead:  len(parsed.Sentences) + len(parsed.QAPairs),
		AvgLoss:        avgLoss,
		SmoothedLoss:   b.SmoothedLoss,
		OldLevel:       oldLevel,
		NewLevel:       b.Level,
		XPGained:       xpGain,
		ActiveSynapses: b.Net.ActiveSynapses(),
		VocabSize:      b.Net.LearnedWordCount(),
		SampleKeywords: sampleKW,
	}
}

func thoughtStageCue(level int) string {
	switch {
	case level <= 3:
		return "stage_1"
	case level <= 8:
		return "stage_2"
	case level <= 24:
		return "stage_3"
	case level <= 74:
		return "stage_4"
	default:
		return "stage_5"
	}
}

func (b *Brain) ensureMaps() {
	if b.TopicsCount == nil {
		b.TopicsCount = make(map[string]int)
	}
	if b.SeenSources == nil {
		b.SeenSources = make(map[string]bool)
	}
	if b.NGramCounts == nil {
		b.NGramCounts = make(map[string]map[string]int)
	}
	if b.PromptStarters == nil {
		b.PromptStarters = make(map[string]map[string]int)
	}
}

func (b *Brain) recordConditionedCounts(prompt []string, resp []string, weight int) {
	if len(resp) == 0 {
		return
	}
	b.recordSequenceCounts(sepToken, resp, weight)

	starter := resp[0]
	if len(resp) >= 2 {
		starter = resp[0] + " " + resp[1]
	}

	// Index exact short prompt phrase + individual prompt tokens -> response starter
	if len(prompt) <= 5 {
		phraseKey := "p:" + strings.Join(prompt, "_")
		b.addStarterCount(phraseKey, starter, weight*3)
	}
	for i, pt := range prompt {
		b.addStarterCount(pt, starter, weight)
		if i+1 < len(prompt) {
			b.addStarterCount(pt+"_"+prompt[i+1], starter, weight*2)
		}
	}
}

func (b *Brain) recordSequenceCounts(startMarker string, tokens []string, weight int) {
	prev3 := padToken
	prev2 := padToken
	prev1 := startMarker
	for _, tok := range tokens {
		b.addNGramCount(prev3+" "+prev2+" "+prev1, tok, weight*2)
		b.addNGramCount(prev2+" "+prev1, tok, weight)
		b.addNGramCount(prev1, tok, weight)
		prev3 = prev2
		prev2 = prev1
		prev1 = tok
	}
	b.addNGramCount(prev3+" "+prev2+" "+prev1, eosToken, weight*2)
	b.addNGramCount(prev2+" "+prev1, eosToken, weight)
	b.addNGramCount(prev1, eosToken, weight)
}

func (b *Brain) addNGramCount(key, next string, weight int) {
	sub, ok := b.NGramCounts[key]
	if !ok {
		if len(b.NGramCounts) >= maxNGramKeys {
			return
		}
		sub = make(map[string]int, 4)
		b.NGramCounts[key] = sub
	}
	if _, exists := sub[next]; !exists && len(sub) >= maxFollowersPerKey {
		return
	}
	sub[next] += weight
}

func (b *Brain) addStarterCount(cue, starter string, weight int) {
	sub, ok := b.PromptStarters[cue]
	if !ok {
		if len(b.PromptStarters) >= maxPromptCueKeys {
			return
		}
		sub = make(map[string]int, 4)
		b.PromptStarters[cue] = sub
	}
	if _, exists := sub[starter]; !exists && len(sub) >= maxStartersPerCue {
		return
	}
	sub[starter] += weight
}

func (b *Brain) storePassage(p Passage) {
	for _, existing := range b.Passages {
		if existing.Text == p.Text {
			return
		}
	}
	if len(b.Passages) >= b.MaxPassages {
		drop := b.MaxPassages / 10
		if drop < 1 {
			drop = 1
		}
		b.Passages = append([]Passage(nil), b.Passages[drop:]...)
	}
	b.Passages = append(b.Passages, p)
}

func (b *Brain) pushRecentTitle(title string) {
	title = strings.TrimSpace(title)
	if title == "" {
		return
	}
	for _, t := range b.RecentTitles {
		if t == title {
			return
		}
	}
	if len(b.RecentTitles) >= 15 {
		b.RecentTitles = b.RecentTitles[1:]
	}
	b.RecentTitles = append(b.RecentTitles, title)
}

// SearchPassages retrieves top matching learned facts using hybrid keyword overlap +
// learned neural embedding cosine similarity.
func (b *Brain) SearchPassages(query string, topK int) []Passage {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.searchPassagesLocked(query, topK)
}

func (b *Brain) searchPassagesLocked(query string, topK int) []Passage {
	if len(b.Passages) == 0 || topK <= 0 {
		return nil
	}

	qKeywords := ExtractKeywords(query, 12)
	qSet := make(map[string]bool, len(qKeywords))
	for _, kw := range qKeywords {
		qSet[kw] = true
	}
	qVec := b.Net.EmbedText(query)

	type scored struct {
		p     Passage
		score float64
	}
	var matches []scored

	for _, p := range b.Passages {
		overlap := 0.0
		for _, pk := range p.Keywords {
			if qSet[pk] {
				overlap += 2.2
			} else {
				for qk := range qSet {
					if len(qk) >= 5 && len(pk) >= 5 && (strings.HasPrefix(pk, qk[:5]) || strings.HasPrefix(qk, pk[:5])) {
						overlap += 1.0
						break
					}
				}
			}
		}
		titleLower := strings.ToLower(p.Title)
		topicLower := strings.ToLower(p.Topic)
		textLower := strings.ToLower(p.Text)
		for qk := range qSet {
			if strings.HasPrefix(textLower, qk+", ") || strings.HasPrefix(textLower, qk+" ") {
				overlap += 3.5
			}
			if strings.Contains(titleLower, qk) {
				overlap += 1.8
			}
			if strings.Contains(topicLower, qk) {
				overlap += 1.0
			}
		}

		if overlap <= 0 {
			continue
		}
		pVec := b.Net.EmbedText(p.Text)
		sim := float64(CosineSimilarity(qVec, pVec))
		totalScore := overlap + math.Max(0, sim)*1.5
		if totalScore > 0.8 {
			matches = append(matches, scored{p: p, score: totalScore})
		}
	}

	sort.Slice(matches, func(i, j int) bool {
		return matches[i].score > matches[j].score
	})
	if len(matches) > topK {
		matches = matches[:topK]
	}
	res := make([]Passage, len(matches))
	for i, m := range matches {
		res[i] = m.p
	}
	return res
}

// TopTopics returns the Matrix's most-studied subjects sorted by article count.
func (b *Brain) TopTopics(limit int) []string {
	b.mu.Lock()
	defer b.mu.Unlock()

	type kv struct {
		k string
		v int
	}
	list := make([]kv, 0, len(b.TopicsCount))
	for k, v := range b.TopicsCount {
		list = append(list, kv{k, v})
	}
	sort.Slice(list, func(i, j int) bool {
		if list[i].v == list[j].v {
			return list[i].k < list[j].k
		}
		return list[i].v > list[j].v
	})
	if limit > 0 && len(list) > limit {
		list = list[:limit]
	}
	out := make([]string, len(list))
	for i, item := range list {
		out[i] = fmt.Sprintf("%s (%d)", item.k, item.v)
	}
	return out
}

// BrainFilePath returns the save file location inside dataDir/brains for a Matrix name.
func BrainFilePath(dataDir, name string) string {
	slug := slugify(name)
	return filepath.Join(dataDir, "brains", slug+".brain.gz")
}

// Save persists the Brain atomically as a gzip-compressed state file.
func (b *Brain) Save(dataDir string) error {
	b.mu.Lock()
	defer b.mu.Unlock()

	brainsDir := filepath.Join(dataDir, "brains")
	if err := os.MkdirAll(brainsDir, 0o755); err != nil {
		return err
	}

	target := BrainFilePath(dataDir, b.Name)
	tmp := target + ".tmp"

	f, err := os.Create(tmp)
	if err != nil {
		return err
	}
	gw, _ := gzip.NewWriterLevel(f, gzip.BestSpeed)
	enc := json.NewEncoder(gw)
	if err := enc.Encode(b); err != nil {
		_ = gw.Close()
		_ = f.Close()
		_ = os.Remove(tmp)
		return err
	}
	if err := gw.Close(); err != nil {
		_ = f.Close()
		_ = os.Remove(tmp)
		return err
	}
	if err := f.Close(); err != nil {
		_ = os.Remove(tmp)
		return err
	}
	return os.Rename(tmp, target)
}

// LoadOrCreate loads an existing Brain from disk or creates a new blank Level-1 Brain.
func LoadOrCreate(dataDir, name string, prof specs.ComputeProfile) (*Brain, bool, error) {
	target := BrainFilePath(dataDir, name)
	f, err := os.Open(target)
	if err != nil {
		if os.IsNotExist(err) {
			b := NewBrain(name, prof)
			return b, false, nil
		}
		return nil, false, err
	}
	defer f.Close()

	gr, err := gzip.NewReader(f)
	if err != nil {
		return nil, false, err
	}
	defer gr.Close()

	var b Brain
	if err := json.NewDecoder(gr).Decode(&b); err != nil {
		return nil, false, err
	}
	b.ensureMaps()
	b.ApplyProfileCaps(prof)
	return &b, true, nil
}

func slugify(s string) string {
	s = strings.ToLower(strings.TrimSpace(s))
	var out strings.Builder
	for _, r := range s {
		if (r >= 'a' && r <= 'z') || (r >= '0' && r <= '9') {
			out.WriteRune(r)
		} else if r == ' ' || r == '-' || r == '_' {
			out.WriteRune('_')
		}
	}
	res := strings.Trim(out.String(), "_")
	if res == "" {
		return "matrix"
	}
	return res
}

func randomHexID(numBytes int) string {
	buf := make([]byte, numBytes)
	if _, err := rand.Read(buf); err != nil {
		return fmt.Sprintf("%x", time.Now().UnixNano())
	}
	return hex.EncodeToString(buf)
}

func minInt(a, b int) int {
	if a < b {
		return a
	}
	return b
}
