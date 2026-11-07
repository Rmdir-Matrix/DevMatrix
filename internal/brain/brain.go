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

const binaryMagicHeader = "MXG3" // Matrix Closed-Book GPT v3 Binary Checkpoint

// ChatTurn represents one message pair inside the live ChatGPT-style context window.
type ChatTurn struct {
	User      string
	Assistant string
}

// TrainReport summarizes one completed GPT training step on a dictionary, grammar lesson, or article.
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
	GrowthMessage  string
}

// Brain holds Matrix's Closed-Book Multi-Layer, Multi-Head GPT Transformer and training telemetry.
// ZERO text/definition/passage lookup databases exist inside Brain — 100% of knowledge lives in Net weights!
type Brain struct {
	ID             string           `json:"id"`
	Name           string           `json:"name"`
	CreatedAt      string           `json:"created_at"`
	UpdatedAt      string           `json:"updated_at"`
	Level          int              `json:"level"`
	XP             int64            `json:"xp"`
	ArticlesRead   int              `json:"articles_read"`
	TokensTrained  int64            `json:"tokens_trained"`
	SmoothedLoss   float64          `json:"smoothed_loss"`
	MaxReplay      int              `json:"max_replay"`
	BatchSize      int              `json:"batch_size"`
	WorkerThreads  int              `json:"worker_threads"`
	TopicCounts    map[string]int   `json:"topic_counts"`
	SeenSources    map[string]bool  `json:"seen_sources"`
	RecentTitles   []string         `json:"recent_titles"`
	CoreReplay     []TrainingSample `json:"core_replay,omitempty"`   // training-only SGD replay (never used at inference)
	RecentReplay   []TrainingSample `json:"recent_replay,omitempty"` // training-only SGD replay (never used at inference)
	TrainingItems  int              `json:"training_items,omitempty"`
	GrowthEvents   int              `json:"growth_events,omitempty"`
	GrowthMaxWidth int              `json:"-"`
	GrowthRAMMB    int              `json:"-"`
	Net            *NeuralNet       `json:"-"`
}

// NewBrain initializes a brand-new Level 1 Closed-Book GPT Matrix sized to ComputeProfile.
func NewBrain(name string, prof specs.ComputeProfile) *Brain {
	if strings.TrimSpace(name) == "" {
		name = "Matrix"
	}
	now := time.Now().UTC().Format(time.RFC3339)
	seed := time.Now().UnixNano()

	maxRep := prof.MaxPassages
	if maxRep < 128 {
		maxRep = 256
	}
	batchSz := prof.BatchSize
	if batchSz < 4 {
		batchSz = 12
	}
	workers := prof.WorkerThreads
	if workers < 1 {
		workers = 1
	}
	layers := prof.NumLayers
	if layers < 2 {
		layers = 3
	}
	heads := prof.NumHeads
	if heads < 2 {
		heads = 4
	}

	return &Brain{
		ID:             randomHexID(4),
		Name:           strings.TrimSpace(name),
		CreatedAt:      now,
		UpdatedAt:      now,
		Level:          1,
		XP:             0,
		SmoothedLoss:   0,
		MaxReplay:      maxRep,
		BatchSize:      batchSz,
		WorkerThreads:  workers,
		GrowthMaxWidth: prof.HiddenNeurons * 4,
		GrowthRAMMB:    prof.RAMBudgetMB,
		TopicCounts:    make(map[string]int),
		SeenSources:    make(map[string]bool),
		Net:            NewNeuralNet(layers, heads, prof.EmbedDim, prof.HiddenNeurons, prof.MaxVocab, seed),
	}
}

// ApplyProfileCaps adapts training buffers AND prunes vocabulary tensors safely
// when loading or importing a Matrix onto lower-RAM hardware.
func (b *Brain) ApplyProfileCaps(prof specs.ComputeProfile) {
	b.GrowthRAMMB = prof.RAMBudgetMB
	b.GrowthMaxWidth = prof.HiddenNeurons * 4
	if prof.MaxPassages >= 128 {
		b.MaxReplay = prof.MaxPassages
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
	if b.Net != nil && prof.MaxVocab >= 512 {
		prevVocab := b.Net.VocabSize()
		b.Net.PruneToMaxVocab(prof.MaxVocab)
		if b.Net.VocabSize() < prevVocab {
			b.CoreReplay = nil
			b.RecentReplay = nil
		}
	}
	if len(b.RecentReplay) > b.MaxReplay {
		b.RecentReplay = append([]TrainingSample(nil), b.RecentReplay[len(b.RecentReplay)-b.MaxReplay:]...)
	}
	b.evictSeenSourcesToCap(8192)
}

// TrainOnArticle trains the Closed-Book GPT Transformer directly on pretraining + SFT token
// sequences via multi-core AdamW backpropagation (zero lookup tables stored for inference).
func (b *Brain) TrainOnArticle(art corpus.Article, epochs int) TrainReport {
	if epochs < 1 {
		epochs = 1
	}
	oldLevel := b.Level

	parsed := ParseCorpusContent(art.Content)
	rng := mrand.New(mrand.NewSource(time.Now().UnixNano() + int64(b.ArticlesRead*131)))

	isCurriculum := strings.HasPrefix(art.Source, "curriculum:") || strings.HasPrefix(art.Source, "grammar:")
	isDict := strings.HasPrefix(art.Source, "dictionary:")

	var docSamples []TrainingSample
	rawTokensRead := 0
	var firstExcerpt string

	titleSubwords := SubwordTokenize(art.Title)
	if len(titleSubwords) > 0 {
		b.Net.RegisterTokens(titleSubwords, rng)
	}
	titleKWs := ExtractKeywords(art.Title, 3)

	// 1. Pretraining (<bos> sentence <eos>) + Self-Supervised Closed-Book SFT (<user> ask <assistant> sentence <eos>)
	for sIdx, sent := range parsed.Sentences {
		sw := SubwordTokenize(sent)
		if len(sw) < 3 {
			continue
		}
		if firstExcerpt == "" {
			firstExcerpt = sent
		}
		rawTokensRead += len(sw)
		if len(sw) > 38 {
			sw = sw[:38]
		}
		sentIDs := b.Net.RegisterTokens(sw, rng)

		// GPT Pretraining sequence: <bos> w_1 ... w_k <eos>
		lmSeq := make([]int, 0, len(sentIDs)+2)
		lmSeq = append(lmSeq, bosID)
		lmSeq = append(lmSeq, sentIDs...)
		lmSeq = append(lmSeq, eosID)
		docSamples = append(docSamples, TrainingSample{IDs: lmSeq, LossStart: 1})

		// Closed-Book SFT comprehension sequences for real articles & user-taught facts
		if !isCurriculum && !isDict && len(sentIDs) >= 5 && len(sentIDs) <= 32 {
			kws := ExtractKeywords(sent, 4)
			if len(kws) >= 2 {
				kwPair := kws[0] + " " + kws[1]
				for _, promptStr := range []string{kwPair, "explain " + kwPair} {
					pIDs := b.Net.RegisterTokens(SubwordTokenize(promptStr), rng)
					if seq, ls, ok := packSFTSample(pIDs, sentIDs); ok {
						docSamples = append(docSamples, TrainingSample{IDs: seq, LossStart: ls})
					}
				}
			}
			rawWords := Tokenize(sent)
			leadNoun := ""
			for _, rw := range rawWords {
				if len(rw) >= 4 && !stopWords[rw] && isPureLetters(rw) {
					leadNoun = rw
					break
				}
			}
			if leadNoun != "" {
				qVerb := "is"
				if strings.HasSuffix(leadNoun, "s") && !strings.HasSuffix(leadNoun, "ss") && !strings.HasSuffix(leadNoun, "is") && !strings.HasSuffix(leadNoun, "us") {
					qVerb = "are"
				}
				for _, promptStr := range []string{
					"what " + qVerb + " " + leadNoun,
					"explain " + leadNoun,
					leadNoun,
				} {
					pIDs := b.Net.RegisterTokens(SubwordTokenize(promptStr), rng)
					if seq, ls, ok := packSFTSample(pIDs, sentIDs); ok {
						docSamples = append(docSamples, TrainingSample{IDs: seq, LossStart: ls})
					}
				}
			}
			if sIdx == 0 && len(titleKWs) > 0 {
				tPrompt := strings.Join(titleKWs, " ")
				for _, promptStr := range []string{tPrompt, "explain " + tPrompt} {
					pIDs := b.Net.RegisterTokens(SubwordTokenize(promptStr), rng)
					if seq, ls, ok := packSFTSample(pIDs, sentIDs); ok {
						docSamples = append(docSamples, TrainingSample{IDs: seq, LossStart: ls})
					}
				}
			}
		}
	}

	// 2. Explicit SFT Pairs (Grammar conversations + Dictionary definitions)
	for _, sft := range parsed.SFTPairs {
		if len(sft.Prompt) == 0 || len(sft.Response) == 0 {
			continue
		}
		rawTokensRead += len(sft.Prompt) + len(sft.Response)
		pToks := sft.Prompt
		if len(pToks) > 16 {
			pToks = pToks[:16]
		}
		rToks := sft.Response
		if len(rToks) > 32 {
			rToks = rToks[:32]
		}
		pIDs := b.Net.RegisterTokens(pToks, rng)
		rIDs := b.Net.RegisterTokens(rToks, rng)

		if seq, ls, ok := packSFTSample(pIDs, rIDs); ok {
			smp := TrainingSample{IDs: seq, LossStart: ls}
			reps := 1
			if isCurriculum {
				reps = 3
				if len(b.CoreReplay) < 192 {
					b.CoreReplay = append(b.CoreReplay, smp)
				}
				// Generalize proper-name introductions ("my name is <unk>", "call me <unk>") directly in GPT weights
				if len(pToks) >= 3 && ((pToks[0] == "my" && pToks[1] == "name" && pToks[2] == "is") || (pToks[0] == "call" && pToks[1] == "me")) {
					unkPIDs := append([]int(nil), pIDs...)
					unkPIDs[len(unkPIDs)-1] = unkID
					if uSeq, uLs, uOk := packSFTSample(unkPIDs, rIDs); uOk {
						uSmp := TrainingSample{IDs: uSeq, LossStart: uLs}
						docSamples = append(docSamples, uSmp, uSmp)
						if len(b.CoreReplay) < 192 {
							b.CoreReplay = append(b.CoreReplay, uSmp)
						}
					}
				}
			}
			for r := 0; r < reps; r++ {
				docSamples = append(docSamples, smp)
			}
		}
	}

	// Scale optimizer passes so focused articles/definitions internalize cleanly without overfitting
	effectiveEpochs := epochs + 1
	if len(docSamples) > 0 && len(docSamples) <= 16 {
		effectiveEpochs = epochs + 3
		if effectiveEpochs > 6 {
			effectiveEpochs = 6
		}
	}

	baseLR := float32(0.0035 / (1.0 + 0.0006*float64(b.ArticlesRead)))
	if baseLR < 0.0015 {
		baseLR = 0.0015
	}

	var lossAcc float64
	validEpochs := 0
	for ep := 0; ep < effectiveEpochs; ep++ {
		epochBatch := make([]TrainingSample, 0, len(docSamples)+48)
		epochBatch = append(epochBatch, docSamples...)

		// Mix balanced SGD replay sequences during training so AdamW preserves earlier skills in weights
		if len(b.CoreReplay) > 0 {
			nCore := minInt(len(b.CoreReplay), maxInt(10, len(docSamples)))
			if nCore > 24 {
				nCore = 24
			}
			for i := 0; i < nCore; i++ {
				epochBatch = append(epochBatch, b.CoreReplay[rng.Intn(len(b.CoreReplay))])
			}
		}
		if len(b.RecentReplay) > 0 {
			nRec := minInt(len(b.RecentReplay), maxInt(14, (len(docSamples)*3)/2))
			if nRec > 32 {
				nRec = 32
			}
			for i := 0; i < nRec; i++ {
				epochBatch = append(epochBatch, b.RecentReplay[rng.Intn(len(b.RecentReplay))])
			}
		}

		if len(epochBatch) > 1 {
			rng.Shuffle(len(epochBatch), func(i, j int) { epochBatch[i], epochBatch[j] = epochBatch[j], epochBatch[i] })
		}

		effBS := b.BatchSize
		if len(epochBatch) < effBS*3 {
			effBS = maxInt(4, len(epochBatch)/3)
		}

		epLoss, steps := b.Net.TrainMiniBatch(epochBatch, effBS, b.WorkerThreads, baseLR, rng.Int63())
		if steps > 0 {
			lossAcc += epLoss
			validEpochs++
		}
	}

	// Update bounded training-only replay buffer
	stepStride := maxInt(1, len(docSamples)/18)
	for i := 0; i < len(docSamples); i += stepStride {
		b.RecentReplay = append(b.RecentReplay, docSamples[i])
	}
	if b.MaxReplay > 0 && len(b.RecentReplay) > b.MaxReplay {
		b.RecentReplay = append([]TrainingSample(nil), b.RecentReplay[len(b.RecentReplay)-b.MaxReplay:]...)
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

	xpGain := int64(45 + minInt(rawTokensRead/4, 135))
	b.XP += xpGain
	b.Level = LevelFromXP(b.XP)
	b.UpdatedAt = time.Now().UTC().Format(time.RFC3339)

	var growthMessage string
	if validEpochs > 0 && len(docSamples) > 0 {
		b.TrainingItems++
		if msg, grown := b.TryGrow(); grown {
			growthMessage = msg
		}
	}

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
		GrowthMessage:  growthMessage,
	}
}

func packSFTSample(promptIDs, respIDs []int) ([]int, int, bool) {
	if len(promptIDs) == 0 || len(respIDs) == 0 {
		return nil, 0, false
	}
	if len(promptIDs) > 18 {
		promptIDs = promptIDs[len(promptIDs)-18:]
	}
	maxResp := maxSeqLen - 3 - len(promptIDs) // <user> + prompt + <assistant> + resp + <eos>
	if maxResp < 3 {
		return nil, 0, false
	}
	if len(respIDs) > maxResp {
		respIDs = respIDs[:maxResp]
	}
	seq := make([]int, 0, 3+len(promptIDs)+len(respIDs))
	seq = append(seq, userID)
	seq = append(seq, promptIDs...)
	ansPos := len(seq)
	seq = append(seq, assistantID)
	seq = append(seq, respIDs...)
	seq = append(seq, eosID)
	return seq, ansPos + 1, true
}

func (b *Brain) evictSeenSourcesToCap(limit int) {
	if len(b.SeenSources) <= limit {
		return
	}
	for k := range b.SeenSources {
		if strings.HasPrefix(k, "dictionary:core") || strings.HasPrefix(k, "curriculum:") || strings.HasPrefix(k, "grammar:") {
			continue
		}
		delete(b.SeenSources, k)
		if len(b.SeenSources) <= limit {
			break
		}
	}
}

// PickQuizConcept selects a learned domain concept from vocabulary so the user and GPT can compare explanations.
func (b *Brain) PickQuizConcept(topicHint string) (string, bool) {
	if b.Net == nil || b.Net.LearnedWordCount() < 20 {
		return "", false
	}
	hintKWs := ExtractKeywords(topicHint, 3)
	for _, kw := range hintKWs {
		if b.Net.KnowsToken(kw) {
			return kw, true
		}
	}

	var candidates []string
	for id := reservedTokensCount; id < len(b.Net.IDToToken); id++ {
		tok := b.Net.IDToToken[id]
		if len(tok) >= 6 && !strings.HasPrefix(tok, "##") && !stopWords[tok] && isPureLetters(tok) && b.Net.TokenCounts[id] >= 3 {
			candidates = append(candidates, tok)
		}
	}
	if len(candidates) == 0 {
		return "", false
	}
	rng := mrand.New(mrand.NewSource(time.Now().UnixNano()))
	return candidates[rng.Intn(len(candidates))], true
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

// writeCompactBinary streams a checkpoint without buffering all tensor data.
func (b *Brain) writeCompactBinary(w io.Writer) error {
	if _, err := io.WriteString(w, binaryMagicHeader); err != nil {
		return err
	}
	meta, err := json.Marshal(b)
	if err != nil {
		return err
	}
	if err := binary.Write(w, binary.LittleEndian, uint32(len(meta))); err != nil {
		return err
	}
	if _, err := w.Write(meta); err != nil {
		return err
	}
	return b.Net.WriteBinary(w)
}

// MarshalCompactBinary serializes Brain metadata and float32 tensors.
func (b *Brain) MarshalCompactBinary() ([]byte, error) {
	var out bytes.Buffer
	if err := b.writeCompactBinary(&out); err != nil {
		return nil, err
	}
	return out.Bytes(), nil
}

// UnmarshalCompactBinary reconstructs a model from compact binary format.
func UnmarshalCompactBinary(data []byte) (*Brain, error) {
	return readCompactBinary(bytes.NewReader(data))
}

func readCompactBinary(r io.Reader) (*Brain, error) {
	magic := make([]byte, 4)
	if _, err := io.ReadFull(r, magic); err != nil {
		return nil, err
	}
	if string(magic) != binaryMagicHeader {
		return nil, fmt.Errorf("checkpoint uses an older architecture (%q); start a fresh GPT brain or re-run --setup", string(magic))
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
	return &loaded, nil
}

// Save writes Matrix's compact binary GPT checkpoint atomically inside dataDir/brains/<slug>.brain.gz.
func (b *Brain) Save(dataDir string) error {
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
	if err := b.writeCompactBinary(gw); err != nil {
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

// LoadFile loads a saved .brain.gz GPT checkpoint and applies hardware profile caps.
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

	loaded, err := readCompactBinary(gr)
	if err != nil {
		return nil, err
	}
	if _, err := io.Copy(io.Discard, gr); err != nil {
		return nil, err
	}
	loaded.ApplyProfileCaps(prof)
	return loaded, nil
}

// LoadOrCreate loads an existing GPT Brain by name, or creates a brand-new Level 1 Brain.
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
