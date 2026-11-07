package brain

import (
	"bytes"
	"compress/gzip"
	"crypto/rand"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"math"
	mrand "math/rand"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"github.com/Rmdir-Matrix/DevMatrix/internal/corpus"
	"github.com/Rmdir-Matrix/DevMatrix/internal/specs"
)

const binaryMagicHeader = "MXB2"

// KnowledgePassage stores a learned study passage and its neural vector embedding for RAG + quizzes.
// Note: Embedding is serialized as raw IEEE-754 float32 binary, never bloated ASCII JSON.
type KnowledgePassage struct {
	Title      string    `json:"title"`
	Topic      string    `json:"topic"`
	Source     string    `json:"source"`
	Text       string    `json:"text"`
	Keywords   []string  `json:"keywords"`
	AccessTick int64     `json:"access_tick"`
	Embedding  []float32 `json:"-"`
}

// DictEntry stores a learned word definition in bounded, LRU-evicting lexical memory.
type DictEntry struct {
	Word       string `json:"word"`
	POS        string `json:"pos"`
	Definition string `json:"definition"`
	Example    string `json:"example,omitempty"`
	AccessTick int64  `json:"access_tick"`
}

// TrainReport summarizes one completed training step on a dictionary, grammar lesson, or article.
type TrainReport struct {
	ArticleTitle   string
	ArticleExcerpt string
	Topic          string
	Source         string
	TokensRead     int
	AvgLoss        float64
	SmoothedLoss   float64
	XPGained       int64
	OldLevel       int
	NewLevel       int
	LeveledUp      bool
	ActiveSynapses int
	VocabSize      int
}

// Brain holds Matrix's Causal Self-Attention + GRU neural model, bounded study memory, and telemetry.
type Brain struct {
	ID             string                  `json:"id"`
	Name           string                  `json:"name"`
	CreatedAt      string                  `json:"created_at"`
	UpdatedAt      string                  `json:"updated_at"`
	Level          int                     `json:"level"`
	XP             int64                   `json:"xp"`
	ArticlesRead   int                     `json:"articles_read"`
	TokensTrained  int64                   `json:"tokens_trained"`
	SmoothedLoss   float64                 `json:"smoothed_loss"`
	MaxPassages    int                     `json:"max_passages"`
	MaxDictEntries int                     `json:"max_dict_entries"`
	BatchSize      int                     `json:"batch_size"`
	WorkerThreads  int                     `json:"worker_threads"`
	ClockTick      int64                   `json:"clock_tick"`
	TopicCounts    map[string]int          `json:"topic_counts"`
	SeenSources    map[string]bool         `json:"seen_sources"`
	RecentTitles   []string                `json:"recent_titles"`
	Dictionary     map[string]DictEntry    `json:"dictionary"`
	Passages       []KnowledgePassage      `json:"passages"`
	CoreReplay     []TrainingSample        `json:"core_replay,omitempty"`
	RecentReplay   []TrainingSample        `json:"recent_replay,omitempty"`
	Net            *NeuralNet              `json:"-"`
}

// NewBrain initializes a brand-new Level 1 Matrix sized to the hardware ComputeProfile.
func NewBrain(name string, prof specs.ComputeProfile) *Brain {
	if strings.TrimSpace(name) == "" {
		name = "Matrix"
	}
	now := time.Now().UTC().Format(time.RFC3339)
	seed := time.Now().UnixNano()

	maxPass := prof.MaxPassages
	if maxPass < 100 {
		maxPass = 400
	}
	maxDict := prof.MaxVocab
	if maxDict < 512 {
		maxDict = 4096
	}
	batchSz := prof.BatchSize
	if batchSz < 4 {
		batchSz = 16
	}
	workers := prof.WorkerThreads
	if workers < 1 {
		workers = 1
	}

	return &Brain{
		ID:             randomHexID(4),
		Name:           strings.TrimSpace(name),
		CreatedAt:      now,
		UpdatedAt:      now,
		Level:          1,
		XP:             0,
		SmoothedLoss:   0,
		MaxPassages:    maxPass,
		MaxDictEntries: maxDict,
		BatchSize:      batchSz,
		WorkerThreads:  workers,
		TopicCounts:    make(map[string]int),
		SeenSources:    make(map[string]bool),
		Dictionary:     make(map[string]DictEntry),
		Passages:       make([]KnowledgePassage, 0, 64),
		Net:            NewNeuralNet(prof.EmbedDim, prof.HiddenNeurons, prof.MaxVocab, seed),
	}
}

// ApplyProfileCaps adapts all memory structures AND neural vocabulary tensors safely
// when loading or importing a Matrix onto lower-RAM hardware.
func (b *Brain) ApplyProfileCaps(prof specs.ComputeProfile) {
	if prof.MaxPassages >= 100 {
		b.MaxPassages = prof.MaxPassages
	}
	if prof.MaxVocab >= 512 {
		b.MaxDictEntries = prof.MaxVocab
	}
	if prof.BatchSize >= 1 {
		b.BatchSize = prof.BatchSize
	}
	if prof.WorkerThreads >= 1 {
		b.WorkerThreads = prof.WorkerThreads
	}
	if b.TopicCounts == nil {
		b.TopicCounts = make(map[string]int)
	}
	if b.SeenSources == nil {
		b.SeenSources = make(map[string]bool)
	}
	if b.Dictionary == nil {
		b.Dictionary = make(map[string]DictEntry)
	}
	if b.Net != nil && prof.MaxVocab >= 512 {
		prevVocab := b.Net.VocabSize()
		b.Net.PruneToMaxVocab(prof.MaxVocab)
		if b.Net.VocabSize() < prevVocab {
			for i := range b.Passages {
				b.Passages[i].Embedding = b.Net.EmbedText(b.Passages[i].Title + " " + b.Passages[i].Text)
			}
			// Clear replay sequences whose old token IDs were remapped
			b.CoreReplay = nil
			b.RecentReplay = nil
		}
	}
	b.evictPassagesToCap()
	b.evictDictionaryToCap()
	b.evictSeenSourcesToCap(8192)
}

// TrainOnArticle parses a dictionary chapter, grammar lesson, local note, or web article
// into subword sequences and trains the Causal Self-Attention + GRU model using multi-core AdamW.
func (b *Brain) TrainOnArticle(art corpus.Article, epochs int) TrainReport {
	if epochs < 1 {
		epochs = 1
	}
	oldLevel := b.Level
	b.ClockTick++

	parsed := ParseCorpusContent(art.Content)
	rng := mrand.New(mrand.NewSource(time.Now().UnixNano() + int64(b.ArticlesRead*131)))

	// Store structured dictionary entries in bounded LRU lexical memory
	for _, dr := range parsed.DictRecords {
		key := strings.ToLower(strings.TrimSpace(dr.Word))
		if key != "" {
			b.ClockTick++
			b.Dictionary[key] = DictEntry{
				Word:       key,
				POS:        dr.POS,
				Definition: dr.Definition,
				Example:    dr.Example,
				AccessTick: b.ClockTick,
			}
		}
	}
	b.evictDictionaryToCap()

	isCurriculum := strings.HasPrefix(art.Source, "curriculum:") || strings.HasPrefix(art.Source, "grammar:")
	isDict := strings.HasPrefix(art.Source, "dictionary:")

	// Build subword training samples (100% from real document content — zero synthetic templates!)
	var samples []TrainingSample
	rawTokensRead := 0
	var firstExcerpt string

	titleSubwords := SubwordTokenize(art.Title)
	if len(titleSubwords) > 0 {
		b.Net.RegisterTokens(titleSubwords, rng)
	}

	for _, sent := range parsed.Sentences {
		sw := SubwordTokenize(sent)
		if len(sw) < 3 {
			continue
		}
		if firstExcerpt == "" {
			firstExcerpt = sent
		}
		rawTokensRead += len(sw)
		if len(sw) > 36 {
			sw = sw[:36]
		}
		sentIDs := b.Net.RegisterTokens(sw, rng)

		// 1. Autoregressive language modeling sequence: <bos> w_1 ... w_k <eos>
		lmSeq := make([]int, 0, len(sentIDs)+2)
		lmSeq = append(lmSeq, bosID)
		lmSeq = append(lmSeq, sentIDs...)
		lmSeq = append(lmSeq, eosID)
		samples = append(samples, TrainingSample{IDs: lmSeq, LossStart: 1})

		// 2. Self-supervised RAG comprehension on real domain sentences:
		//    <ctx> [sentence] <ask> [salient topic keywords] <ans> [sentence] <eos>
		if !isCurriculum && !isDict && len(sentIDs) >= 5 && len(sentIDs) <= 24 {
			kws := ExtractKeywords(art.Title+" "+sent, 3)
			if len(kws) > 0 {
				askToks := SubwordTokenize(strings.Join(kws, " "))
				askIDs := b.Net.RegisterTokens(askToks, rng)
				if seq, ls, ok := packRAGSample(sentIDs, askIDs, sentIDs); ok {
					samples = append(samples, TrainingSample{IDs: seq, LossStart: ls})
				}
				whatToks := append([]string{"what", "is"}, SubwordTokenize(kws[0])...)
				whatIDs := b.Net.RegisterTokens(whatToks, rng)
				if seq, ls, ok := packRAGSample(sentIDs, whatIDs, sentIDs); ok {
					samples = append(samples, TrainingSample{IDs: seq, LossStart: ls})
				}
			}
		}
	}

	// 3. Explicit QA / Dictionary pairs: [<ctx> context] <ask> prompt <ans> response <eos>
	for _, qa := range parsed.QAPairs {
		if len(qa.Prompt) == 0 || len(qa.Response) == 0 {
			continue
		}
		rawTokensRead += len(qa.Prompt) + len(qa.Response)
		var ctxIDs []int
		if len(qa.Context) > 0 {
			cToks := qa.Context
			if len(cToks) > 22 {
				cToks = cToks[:22]
			}
			ctxIDs = b.Net.RegisterTokens(cToks, rng)
		}
		pToks := qa.Prompt
		if len(pToks) > 12 {
			pToks = pToks[:12]
		}
		rToks := qa.Response
		if len(rToks) > 24 {
			rToks = rToks[:24]
		}
		pIDs := b.Net.RegisterTokens(pToks, rng)
		rIDs := b.Net.RegisterTokens(rToks, rng)

		if seq, ls, ok := packRAGSample(ctxIDs, pIDs, rIDs); ok {
			smp := TrainingSample{IDs: seq, LossStart: ls}
			reps := 1
			if isCurriculum {
				reps = 3
				if len(qa.Context) == 0 && len(b.CoreReplay) < 96 {
					b.CoreReplay = append(b.CoreReplay, smp)
				}
			}
			for r := 0; r < reps; r++ {
				samples = append(samples, smp)
			}
		}
	}

	// Continual-learning Experience Replay: blend learned dialogue & recent knowledge
	// into each training batch to prevent catastrophic forgetting on single-topic updates.
	newDocSamples := append([]TrainingSample(nil), samples...)
	if len(b.CoreReplay) > 0 {
		limit := minInt(len(b.CoreReplay), maxInt(6, len(newDocSamples)/3))
		if limit > 24 {
			limit = 24
		}
		for i := 0; i < limit; i++ {
			samples = append(samples, b.CoreReplay[rng.Intn(len(b.CoreReplay))])
		}
	}
	if len(b.RecentReplay) > 0 {
		limit := minInt(len(b.RecentReplay), maxInt(6, len(newDocSamples)/3))
		if limit > 24 {
			limit = 24
		}
		for i := 0; i < limit; i++ {
			samples = append(samples, b.RecentReplay[rng.Intn(len(b.RecentReplay))])
		}
	}

	// Update bounded RecentReplay ring buffer (capped at 192 sequences)
	for i := 0; i < len(newDocSamples); i += maxInt(1, len(newDocSamples)/8) {
		b.RecentReplay = append(b.RecentReplay, newDocSamples[i])
	}
	if len(b.RecentReplay) > 192 {
		b.RecentReplay = append([]TrainingSample(nil), b.RecentReplay[len(b.RecentReplay)-192:]...)
	}

	// Execute multi-goroutine mini-batch AdamW across epochs with stable GRU learning rate
	baseLR := float32(0.0045 / (1.0 + 0.0008*float64(b.ArticlesRead)))
	if baseLR < 0.0018 {
		baseLR = 0.0018
	}

	effectiveBatch := b.BatchSize
	if len(samples) < effectiveBatch*4 {
		effectiveBatch = len(samples) / 4
		if effectiveBatch < 4 {
			effectiveBatch = 4
		}
	}

	var lossAcc float64
	validEpochs := 0
	for ep := 0; ep < epochs; ep++ {
		if len(samples) > 1 {
			rng.Shuffle(len(samples), func(i, j int) { samples[i], samples[j] = samples[j], samples[i] })
		}
		epLoss, steps := b.Net.TrainMiniBatch(samples, effectiveBatch, b.WorkerThreads, baseLR, rng.Int63())
		if steps > 0 {
			lossAcc += epLoss
			validEpochs++
		}
	}

	avgLoss := 0.0
	if validEpochs > 0 {
		avgLoss = lossAcc / float64(validEpochs)
	}
	if b.SmoothedLoss == 0 {
		b.SmoothedLoss = avgLoss
	} else if avgLoss > 0 {
		b.SmoothedLoss = 0.80*b.SmoothedLoss + 0.20*avgLoss
	}

	// Index domain passages with updated neural embeddings for RAG retrieval & active-recall quizzes
	if !isCurriculum {
		b.storeArticlePassages(art, parsed.Sentences)
	}

	b.ArticlesRead++
	b.TokensTrained += int64(rawTokensRead)
	if art.Topic != "" && !isCurriculum && !isDict {
		b.TopicCounts[art.Topic]++
	}
	if art.Source != "" {
		b.SeenSources[art.Source] = true
		b.evictSeenSourcesToCap(8192)
	}
	if art.Title != "" && !isCurriculum && !isDict {
		b.RecentTitles = appendUniqueTail(b.RecentTitles, art.Title, 16)
	}

	xpGain := int64(40 + minInt(rawTokensRead/4, 130))
	b.XP += xpGain
	b.Level = LevelFromXP(b.XP)
	b.UpdatedAt = time.Now().UTC().Format(time.RFC3339)

	return TrainReport{
		ArticleTitle:   art.Title,
		ArticleExcerpt: firstExcerpt,
		Topic:          art.Topic,
		Source:         art.Source,
		TokensRead:     rawTokensRead,
		AvgLoss:        avgLoss,
		SmoothedLoss:   b.SmoothedLoss,
		XPGained:       xpGain,
		OldLevel:       oldLevel,
		NewLevel:       b.Level,
		LeveledUp:      b.Level > oldLevel,
		ActiveSynapses: b.Net.ActiveSynapses(),
		VocabSize:      b.Net.LearnedWordCount(),
	}
}

func packRAGSample(ctxIDs, askIDs, ansIDs []int) ([]int, int, bool) {
	if len(askIDs) == 0 || len(ansIDs) == 0 {
		return nil, 0, false
	}
	// Budget total sequence <= maxSeqLen (64) so <ctx> never truncates off the left!
	maxAns := 24
	if len(ansIDs) > maxAns {
		ansIDs = ansIDs[:maxAns]
	}
	maxAsk := 12
	if len(askIDs) > maxAsk {
		askIDs = askIDs[:maxAsk]
	}
	overhead := 3 + len(askIDs) + len(ansIDs) // <ask> + ask + <ans> + ans + <eos>
	if len(ctxIDs) > 0 {
		overhead++ // <ctx>
		availCtx := maxSeqLen - overhead
		if availCtx < 4 {
			ctxIDs = nil
			overhead--
		} else if len(ctxIDs) > availCtx {
			ctxIDs = ctxIDs[:availCtx]
		}
	}
	if overhead+len(ctxIDs) > maxSeqLen {
		return nil, 0, false
	}

	seq := make([]int, 0, overhead+len(ctxIDs))
	if len(ctxIDs) > 0 {
		seq = append(seq, ctxID)
		seq = append(seq, ctxIDs...)
	}
	seq = append(seq, askID)
	seq = append(seq, askIDs...)
	ansPos := len(seq)
	seq = append(seq, ansID)
	seq = append(seq, ansIDs...)
	seq = append(seq, eosID)
	return seq, ansPos + 1, true
}

func (b *Brain) storeArticlePassages(art corpus.Article, sentences []string) {
	kept := 0
	for _, s := range sentences {
		if len(s) < 25 || len(s) > 360 {
			continue
		}
		kws := ExtractKeywords(s, 8)
		if len(kws) < 2 {
			continue
		}
		b.ClockTick++
		emb := b.Net.EmbedText(art.Title + " " + art.Topic + " " + s)

		replaced := false
		for i := range b.Passages {
			if b.Passages[i].Text == s {
				b.Passages[i].Embedding = emb
				b.Passages[i].AccessTick = b.ClockTick
				replaced = true
				break
			}
		}
		if !replaced {
			b.Passages = append(b.Passages, KnowledgePassage{
				Title:      art.Title,
				Topic:      art.Topic,
				Source:     art.Source,
				Text:       s,
				Keywords:   kws,
				AccessTick: b.ClockTick,
				Embedding:  emb,
			})
		}
		kept++
		if kept >= 5 {
			break
		}
	}
	b.evictPassagesToCap()
}

func (b *Brain) evictPassagesToCap() {
	if b.MaxPassages <= 0 || len(b.Passages) <= b.MaxPassages {
		return
	}
	// Evict least-recently-used passages
	sort.SliceStable(b.Passages, func(i, j int) bool {
		return b.Passages[i].AccessTick < b.Passages[j].AccessTick
	})
	b.Passages = append([]KnowledgePassage(nil), b.Passages[len(b.Passages)-b.MaxPassages:]...)
}

func (b *Brain) evictDictionaryToCap() {
	if b.MaxDictEntries <= 0 || len(b.Dictionary) <= b.MaxDictEntries {
		return
	}
	type kv struct {
		k    string
		tick int64
	}
	all := make([]kv, 0, len(b.Dictionary))
	for k, v := range b.Dictionary {
		all = append(all, kv{k: k, tick: v.AccessTick})
	}
	sort.Slice(all, func(i, j int) bool { return all[i].tick < all[j].tick })
	drop := len(b.Dictionary) - b.MaxDictEntries
	for i := 0; i < drop; i++ {
		delete(b.Dictionary, all[i].k)
	}
}

func (b *Brain) evictSeenSourcesToCap(limit int) {
	if len(b.SeenSources) <= limit {
		return
	}
	for k := range b.SeenSources {
		if strings.HasPrefix(k, "dictionary:core") || strings.HasPrefix(k, "curriculum:") {
			continue
		}
		delete(b.SeenSources, k)
		if len(b.SeenSources) <= limit {
			break
		}
	}
}

// LookupDictionary performs an O(1) + stem lookup in learned dictionary memory and updates LRU tick.
func (b *Brain) LookupDictionary(word string) (DictEntry, bool) {
	w := strings.ToLower(strings.TrimSpace(word))
	if e, ok := b.Dictionary[w]; ok {
		b.ClockTick++
		e.AccessTick = b.ClockTick
		b.Dictionary[w] = e
		return e, true
	}
	stem, _ := splitMorpheme(w)
	if stem != w {
		if e, ok := b.Dictionary[stem]; ok {
			b.ClockTick++
			e.AccessTick = b.ClockTick
			b.Dictionary[stem] = e
			return e, true
		}
	}
	return DictEntry{}, false
}

// ScoredPassage pairs a retrieved passage with its hybrid neural-semantic + lexical score.
type ScoredPassage struct {
	Passage KnowledgePassage
	Score   float64
}

// SearchPassagesWithScore ranks indexed study passages using neural cosine similarity + root overlap.
func (b *Brain) SearchPassagesWithScore(query string, topK int) []ScoredPassage {
	if len(b.Passages) == 0 || topK <= 0 {
		return nil
	}
	qVec := b.Net.EmbedText(query)
	qKWs := ExtractKeywords(query, 8)

	var scored []ScoredPassage
	for idx := range b.Passages {
		p := &b.Passages[idx]
		cos := float64(CosineSimilarity(qVec, p.Embedding))
		kwHits := 0
		titleLower := strings.ToLower(p.Title + " " + p.Topic)
		for _, qk := range qKWs {
			if strings.Contains(titleLower, qk) {
				kwHits += 2
			}
			for _, pk := range p.Keywords {
				if pk == qk || strings.HasPrefix(pk, qk) || strings.HasPrefix(qk, pk) {
					kwHits++
					break
				}
			}
		}
		total := cos*0.55 + float64(kwHits)*0.30
		if total > 0.14 {
			scored = append(scored, ScoredPassage{Passage: *p, Score: total})
		}
	}

	sort.Slice(scored, func(i, j int) bool { return scored[i].Score > scored[j].Score })
	if len(scored) > topK {
		scored = scored[:topK]
	}
	// Update LRU ticks on retrieved passages
	for _, sp := range scored {
		for i := range b.Passages {
			if b.Passages[i].Text == sp.Passage.Text {
				b.ClockTick++
				b.Passages[i].AccessTick = b.ClockTick
				break
			}
		}
	}
	return scored
}

// SearchPassages returns topK passages matching query.
func (b *Brain) SearchPassages(query string, topK int) []KnowledgePassage {
	sp := b.SearchPassagesWithScore(query, topK)
	out := make([]KnowledgePassage, len(sp))
	for i, item := range sp {
		out[i] = item.Passage
	}
	return out
}

// PickQuizPassage selects a studied domain passage and masks a salient concept for active-recall quizzing.
func (b *Brain) PickQuizPassage(topicHint string) (KnowledgePassage, string, string, bool) {
	var pool []KnowledgePassage
	if strings.TrimSpace(topicHint) != "" {
		pool = b.SearchPassages(topicHint, 8)
	}
	if len(pool) == 0 {
		for _, p := range b.Passages {
			if !strings.HasPrefix(p.Source, "dictionary:") {
				pool = append(pool, p)
			}
		}
	}
	if len(pool) == 0 {
		return KnowledgePassage{}, "", "", false
	}

	rng := mrand.New(mrand.NewSource(time.Now().UnixNano()))
	chosen := pool[rng.Intn(len(pool))]
	rawWords := strings.Fields(chosen.Text)
	for _, w := range rawWords {
		clean := strings.Trim(w, ".,!?;:()\"'")
		if len(clean) >= 5 && !stopWords[strings.ToLower(clean)] && isPureLetters(clean) {
			masked := strings.Replace(chosen.Text, clean, "________", 1)
			return chosen, clean, masked, true
		}
	}
	return KnowledgePassage{}, "", "", false
}

// AwardStudyXP awards XP from quizzes or study interactions and updates Level.
func (b *Brain) AwardStudyXP(xp int64) bool {
	if xp <= 0 {
		return false
	}
	old := b.Level
	b.XP += xp
	b.Level = LevelFromXP(b.XP)
	b.UpdatedAt = time.Now().UTC().Format(time.RFC3339)
	return b.Level > old
}

// TopTopics returns the user's most-studied subject categories.
func (b *Brain) TopTopics(limit int) []string {
	type kv struct {
		k string
		v int
	}
	var pairs []kv
	for k, v := range b.TopicCounts {
		pairs = append(pairs, kv{k, v})
	}
	sort.Slice(pairs, func(i, j int) bool {
		if pairs[i].v == pairs[j].v {
			return pairs[i].k < pairs[j].k
		}
		return pairs[i].v > pairs[j].v
	})
	var res []string
	for i := 0; i < len(pairs) && i < limit; i++ {
		res = append(res, fmt.Sprintf("%s (%d)", pairs[i].k, pairs[i].v))
	}
	return res
}

// LevelFromXP maps cumulative XP to Matrix's intelligence level.
func LevelFromXP(xp int64) int {
	if xp <= 0 {
		return 1
	}
	lvl := int(math.Floor(math.Sqrt(float64(xp)/70.0))) + 1
	if lvl < 1 {
		return 1
	}
	return lvl
}

// XPForLevel returns total XP required to reach targetLevel.
func XPForLevel(targetLevel int) int64 {
	if targetLevel <= 1 {
		return 0
	}
	d := int64(targetLevel - 1)
	return d * d * 70
}

// MarshalCompactBinary serializes Brain metadata + IEEE-754 float32 neural tensors + passage embeddings.
func (b *Brain) MarshalCompactBinary() ([]byte, error) {
	var out bytes.Buffer
	out.WriteString(binaryMagicHeader)

	metaBytes, err := json.Marshal(b)
	if err != nil {
		return nil, err
	}
	if err := binary.Write(&out, binary.LittleEndian, uint32(len(metaBytes))); err != nil {
		return nil, err
	}
	if _, err := out.Write(metaBytes); err != nil {
		return nil, err
	}

	if err := b.Net.WriteBinary(&out); err != nil {
		return nil, err
	}

	// Write compact binary embeddings for indexed passages
	if err := binary.Write(&out, binary.LittleEndian, uint32(len(b.Passages))); err != nil {
		return nil, err
	}
	for i := range b.Passages {
		if err := writeFloat32Slice(&out, b.Passages[i].Embedding); err != nil {
			return nil, err
		}
	}
	return out.Bytes(), nil
}

// UnmarshalCompactBinary reconstructs a Brain from compact binary format.
func UnmarshalCompactBinary(data []byte) (*Brain, error) {
	r := bytes.NewReader(data)
	magic := make([]byte, 4)
	if _, err := io.ReadFull(r, magic); err != nil {
		return nil, err
	}
	if string(magic) != binaryMagicHeader {
		return nil, fmt.Errorf("unsupported brain binary header %q", string(magic))
	}

	var metaLen uint32
	if err := binary.Read(r, binary.LittleEndian, &metaLen); err != nil {
		return nil, err
	}
	if metaLen > 64*1024*1024 {
		return nil, fmt.Errorf("metadata payload too large: %d", metaLen)
	}
	metaBuf := make([]byte, int(metaLen))
	if _, err := io.ReadFull(r, metaBuf); err != nil {
		return nil, err
	}

	var loaded Brain
	if err := json.Unmarshal(metaBuf, &loaded); err != nil {
		return nil, err
	}

	nn, err := ReadNeuralNetBinary(r)
	if err != nil {
		return nil, err
	}
	loaded.Net = nn

	var pCount uint32
	if err := binary.Read(r, binary.LittleEndian, &pCount); err != nil {
		return nil, err
	}
	for i := 0; i < int(pCount) && i < len(loaded.Passages); i++ {
		emb, err := readFloat32Slice(r)
		if err != nil {
			return nil, err
		}
		loaded.Passages[i].Embedding = emb
	}
	return &loaded, nil
}

// Save writes Matrix's compact binary checkpoint atomically inside dataDir/brains/<slug>.brain.gz.
func (b *Brain) Save(dataDir string) error {
	brainsDir := filepath.Join(dataDir, "brains")
	if err := os.MkdirAll(brainsDir, 0o755); err != nil {
		return err
	}

	raw, err := b.MarshalCompactBinary()
	if err != nil {
		return err
	}

	target := BrainFilePath(dataDir, b.Name)
	tmp := target + ".tmp"
	f, err := os.Create(tmp)
	if err != nil {
		return err
	}

	gw, _ := gzip.NewWriterLevel(f, gzip.BestSpeed)
	if _, err := gw.Write(raw); err != nil {
		_ = gw.Close()
		_ = f.Close()
		return err
	}
	if err := gw.Close(); err != nil {
		_ = f.Close()
		return err
	}
	if err := f.Close(); err != nil {
		return err
	}
	return os.Rename(tmp, target)
}

// LoadFile loads a saved .brain.gz checkpoint and applies hardware profile caps.
func LoadFile(filePath string, prof specs.ComputeProfile) (*Brain, error) {
	f, err := os.Open(filePath)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	gr, err := gzip.NewReader(f)
	if err != nil {
		return nil, err
	}
	defer gr.Close()

	payload, err := io.ReadAll(gr)
	if err != nil {
		return nil, err
	}
	loaded, err := UnmarshalCompactBinary(payload)
	if err != nil {
		return nil, err
	}
	loaded.ApplyProfileCaps(prof)
	return loaded, nil
}

// LoadOrCreate loads an existing Brain by name, or creates a brand-new Level 1 Brain.
func LoadOrCreate(dataDir, name string, prof specs.ComputeProfile) (*Brain, bool, error) {
	path := BrainFilePath(dataDir, name)
	if _, err := os.Stat(path); os.IsNotExist(err) {
		return NewBrain(name, prof), false, nil
	}
	loaded, err := LoadFile(path, prof)
	if err != nil {
		return nil, false, err
	}
	return loaded, true, nil
}

// BrainFilePath returns the canonical path for a named Matrix checkpoint.
func BrainFilePath(dataDir, name string) string {
	slug := strings.ToLower(strings.TrimSpace(name))
	var out strings.Builder
	for _, r := range slug {
		if (r >= 'a' && r <= 'z') || (r >= '0' && r <= '9') || r == '-' || r == '_' {
			out.WriteRune(r)
		} else if r == ' ' {
			out.WriteByte('_')
		}
	}
	clean := out.String()
	if clean == "" {
		clean = "matrix"
	}
	return filepath.Join(dataDir, "brains", clean+".brain.gz")
}

func appendUniqueTail(list []string, val string, maxLen int) []string {
	var filtered []string
	for _, item := range list {
		if item != val {
			filtered = append(filtered, item)
		}
	}
	filtered = append(filtered, val)
	if len(filtered) > maxLen {
		filtered = filtered[len(filtered)-maxLen:]
	}
	return filtered
}

func randomHexID(numBytes int) string {
	buf := make([]byte, numBytes)
	if _, err := rand.Read(buf); err != nil {
		return "0101a0b1"
	}
	return hex.EncodeToString(buf)
}

func minInt(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func maxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}
