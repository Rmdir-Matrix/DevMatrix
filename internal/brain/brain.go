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

const maxTransitionsKeys = 12000

// Passage stores a learned factual study excerpt indexed by keywords and source metadata.
type Passage struct {
	Title    string   `json:"title"`
	Topic    string   `json:"topic"`
	Source   string   `json:"source"`
	Text     string   `json:"text"`
	Keywords []string `json:"keywords"`
}

// TrainStepReport summarizes the metrics from training on a single article.
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

// Brain encapsulates a named Matrix instance's neural network, phrase transitions,
// semantic article memory, and progression state.
type Brain struct {
	mu sync.Mutex `json:"-"`

	ID                string             `json:"id"`
	Name              string             `json:"name"`
	CreatedAt         string             `json:"created_at"`
	UpdatedAt         string             `json:"updated_at"`
	Level             int                `json:"level"`
	XP                int64              `json:"xp"`
	ArticlesRead      int                `json:"articles_read"`
	TokensTrained     int64              `json:"tokens_trained"`
	TrainingSteps     int64              `json:"training_steps"`
	InitialLoss       float64            `json:"initial_loss"`
	SmoothedLoss      float64            `json:"smoothed_loss"`
	MaxPassages       int                `json:"max_passages"`
	TopicsCount       map[string]int     `json:"topics_count"`
	SeenSources       map[string]bool    `json:"seen_sources"`
	RecentTitles      []string           `json:"recent_titles"`
	Passages          []Passage          `json:"passages"`
	Transitions       map[string][]string `json:"transitions"`
	Net               *NeuralNet         `json:"net"`
}

// NewBrain constructs a fresh Level-1 Matrix Brain sized according to the hardware profile.
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
		ID:           randomHexID(8),
		Name:         strings.TrimSpace(name),
		CreatedAt:    now,
		UpdatedAt:    now,
		Level:        1,
		XP:           0,
		MaxPassages:  maxPass,
		TopicsCount:  make(map[string]int),
		SeenSources:  make(map[string]bool),
		RecentTitles: make([]string, 0, 16),
		Passages:     make([]Passage, 0, 128),
		Transitions:  make(map[string][]string, 512),
		Net:          NewNeuralNet(prof.EmbedDim, prof.HiddenNeurons, prof.MaxVocab, time.Now().UnixNano()),
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

// TrainOnArticle ingests a real article, runs multi-pass neural backpropagation, updates
// syntactic transitions and semantic study memory, and advances XP & Level.
func (b *Brain) TrainOnArticle(art corpus.Article, epochs int) TrainStepReport {
	b.mu.Lock()
	defer b.mu.Unlock()

	if epochs < 1 {
		epochs = 2
	}
	if b.TopicsCount == nil {
		b.TopicsCount = make(map[string]int)
	}
	if b.SeenSources == nil {
		b.SeenSources = make(map[string]bool)
	}
	if b.Transitions == nil {
		b.Transitions = make(map[string][]string)
	}

	sentences := SplitSentences(art.Content)
	rng := mrand.New(mrand.NewSource(time.Now().UnixNano()))

	tokenizedSentences := make([][]string, 0, len(sentences))
	tokensRead := 0
	for _, sent := range sentences {
		toks := Tokenize(sent)
		if len(toks) < 3 {
			continue
		}
		b.Net.RegisterTokens(toks, rng)
		b.learnTransitions(toks)
		tokensRead += len(toks)
		tokenizedSentences = append(tokenizedSentences, toks)
	}

	// Adaptive learning rate that gently anneals as training steps accumulate
	lr := float32(0.14 / (1.0 + 0.0001*float64(b.TrainingSteps)))
	if lr < 0.02 {
		lr = 0.02
	}

	var lossSum float64
	var predSteps int

	for ep := 0; ep < epochs; ep++ {
		for _, toks := range tokenizedSentences {
			l, st := b.Net.TrainSentence(toks, lr, rng)
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
		b.SmoothedLoss = 0.82*b.SmoothedLoss + 0.18*avgLoss
	}

	// Index concise, informative passages into bounded semantic memory
	for _, sent := range sentences {
		words := strings.Fields(sent)
		if len(words) < 7 || len(words) > 65 {
			continue
		}
		kw := ExtractKeywords(sent, 12)
		if len(kw) < 3 {
			continue
		}
		b.storePassage(Passage{
			Title:    art.Title,
			Topic:    art.Topic,
			Source:   art.Source,
			Text:     sent,
			Keywords: kw,
		})
	}

	oldLevel := b.Level
	xpGain := int64(45 + minInt(tokensRead/3, 180))
	if !b.SeenSources[art.Source] && art.Source != "" {
		xpGain += 35 // Novel article discovery bonus
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

	sampleKW := ExtractKeywords(art.Title+" "+art.Content, 6)
	synapses := b.Net.ActiveSynapses()

	return TrainStepReport{
		ArticleTitle:   art.Title,
		Topic:          topic,
		Source:         art.Source,
		TokensRead:     tokensRead,
		SentencesRead:  len(sentences),
		AvgLoss:        avgLoss,
		SmoothedLoss:   b.SmoothedLoss,
		OldLevel:       oldLevel,
		NewLevel:       b.Level,
		XPGained:       xpGain,
		ActiveSynapses: synapses,
		VocabSize:      b.Net.VocabSize(),
		SampleKeywords: sampleKW,
	}
}

func (b *Brain) learnTransitions(tokens []string) {
	if len(b.Transitions) >= maxTransitionsKeys {
		return
	}
	prev1 := bosToken
	prev2 := bosToken
	for _, tok := range tokens {
		key2 := prev2 + " " + prev1
		b.appendUniqueFollower(key2, tok, 10)
		b.appendUniqueFollower(prev1, tok, 14)
		prev2 = prev1
		prev1 = tok
	}
}

func (b *Brain) appendUniqueFollower(key, next string, capPerKey int) {
	list := b.Transitions[key]
	for _, existing := range list {
		if existing == next {
			return
		}
	}
	if len(list) >= capPerKey {
		return
	}
	if len(list) == 0 && len(b.Transitions) >= maxTransitionsKeys {
		return
	}
	b.Transitions[key] = append(list, next)
}

func (b *Brain) storePassage(p Passage) {
	// Deduplicate identical sentences
	for _, existing := range b.Passages {
		if existing.Text == p.Text {
			return
		}
	}
	if len(b.Passages) >= b.MaxPassages {
		// Evict oldest 10% slice so memory stays bounded without O(n) shift on every insert
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
		for qk := range qSet {
			if strings.Contains(titleLower, qk) {
				overlap += 1.8
			}
			if strings.Contains(topicLower, qk) {
				overlap += 1.0
			}
		}

		pVec := b.Net.EmbedText(p.Text)
		sim := float64(CosineSimilarity(qVec, pVec))
		totalScore := overlap + math.Max(0, sim)*1.5
		if totalScore > 0.35 {
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

// LoadOrCreate loads an existing Brain from disk or creates a new Level-1 Brain.
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
	if b.TopicsCount == nil {
		b.TopicsCount = make(map[string]int)
	}
	if b.SeenSources == nil {
		b.SeenSources = make(map[string]bool)
	}
	if b.Transitions == nil {
		b.Transitions = make(map[string][]string)
	}
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
