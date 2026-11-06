package brain

import (
	"fmt"
	"math"
	mrand "math/rand"
	"strings"
	"time"
	"unicode"
)

// CheckKnowledge verifies whether the model has actually learned the words/concepts in userInput.
// Returns (true, "") when the model knows enough to generate an answer, or (false, honestReason)
// when untrained or asked about unknown words/topics.
func (b *Brain) CheckKnowledge(userInput string) (bool, string) {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.checkKnowledgeLocked(userInput)
}

func (b *Brain) checkKnowledgeLocked(userInput string) (bool, string) {
	if b.TokensTrained == 0 || b.Net == nil || b.Net.LearnedWordCount() == 0 {
		return false, "I don't know any words or grammar yet (0 tokens trained). Start Mode 1 (Training) first!"
	}

	rawToks := b.normalizeSelfTokens(Tokenize(userInput))
	var wordToks []string
	for _, t := range rawToks {
		if isWordToken(t) {
			wordToks = append(wordToks, t)
		}
	}
	if len(wordToks) == 0 {
		return false, "I didn't catch any words I recognize."
	}

	topicWords := ExtractTopicWords(wordToks)
	if len(topicWords) > 0 {
		knownTopics := 0
		firstUnknown := ""
		for _, tw := range topicWords {
			if b.knowsWordOrStemLocked(tw) {
				knownTopics++
			} else if firstUnknown == "" {
				firstUnknown = tw
			}
		}
		// Require single-concept queries to be known, and multi-word queries to know at least half the topic terms
		if knownTopics == 0 || (len(topicWords) <= 2 && firstUnknown != "") || float64(knownTopics)/float64(len(topicWords)) < 0.5 {
			return false, fmt.Sprintf("I don't know about %q yet — I haven't read any training data covering it. Try `/study %s` or train me more in Mode 1!", firstUnknown, firstUnknown)
		}
		return true, ""
	}

	// Conversational / grammar query (e.g. "hello", "how are you", "who are you"):
	// Every core word (or >= 65% for longer sentences) must exist in the model's learned vocabulary!
	knownConv := 0
	firstMissing := ""
	for _, w := range wordToks {
		if w == selfToken || b.knowsWordOrStemLocked(w) {
			knownConv++
		} else if firstMissing == "" {
			firstMissing = w
		}
	}
	if knownConv == 0 || float64(knownConv)/float64(len(wordToks)) < 0.65 {
		return false, fmt.Sprintf("I haven't learned the word %q yet — train me a little more so I learn more words and grammar!", firstMissing)
	}
	return true, ""
}

// GenerateModelReply checks whether Matrix knows the words/topic in userInput; if unknown, returns
// known=false with an honest unknown message. Otherwise, decodes a response 100% token-by-token
// from the trained neural network and learned transition distributions.
func (b *Brain) GenerateModelReply(userInput string) (reply string, known bool, unknownMsg string) {
	b.mu.Lock()
	defer b.mu.Unlock()

	ok, reason := b.checkKnowledgeLocked(userInput)
	if !ok {
		return "", false, reason
	}

	qToks := b.normalizeSelfTokens(Tokenize(userInput))
	var wordToks []string
	for _, t := range qToks {
		if isWordToken(t) {
			wordToks = append(wordToks, t)
		}
	}

	// Gather relevant learned passages only when query asks about domain/dictionary target words
	topicWords := ExtractTopicWords(wordToks)
	var hits []Passage
	if len(topicWords) > 0 {
		hits = b.searchPassagesLocked(strings.Join(topicWords, " "), 2)
		if len(hits) == 0 {
			hasCue := false
			for _, tw := range topicWords {
				if len(b.PromptStarters[tw]) > 0 {
					hasCue = true
					break
				}
			}
			if !hasCue {
				return "", false, fmt.Sprintf("I recognize the word %q, but I haven't learned enough facts about it yet. Try `/study %s`!", topicWords[0], topicWords[0])
			}
		}
	}

	condToks := append([]string(nil), wordToks...)
	boostSet := make(map[string]bool)
	var passageStarterHints []string

	for idx, h := range hits {
		pToks := Tokenize(h.Text)
		for _, pt := range pToks {
			if isWordToken(pt) && !stopWords[pt] {
				boostSet[pt] = true
			}
		}
		if idx == 0 && len(pToks) >= 2 {
			passageStarterHints = append(passageStarterHints, pToks[0]+" "+pToks[1])
			condToks = append(condToks, h.Keywords...)
		}
	}

	starterCues := wordToks
	if len(topicWords) > 0 {
		starterCues = topicWords
	}

	maxWords := b.maxDecodeLengthForLevel()
	decoded := b.decodeSequenceLocked(starterCues, condToks, boostSet, passageStarterHints, maxWords)
	if strings.TrimSpace(decoded) == "" {
		return "", false, "I don't have enough trained connections to form an answer for that yet — train me a bit more!"
	}

	// If high-level Matrix has multiple distinct relevant passages, optionally decode a second sentence!
	if b.Level >= 35 && len(hits) >= 2 && len(topicWords) > 0 {
		h2Toks := Tokenize(hits[1].Text)
		if len(h2Toks) >= 4 {
			boost2 := make(map[string]bool)
			for _, pt := range h2Toks {
				if isWordToken(pt) && !stopWords[pt] {
					boost2[pt] = true
				}
			}
			cond2 := append([]string(nil), topicWords...)
			cond2 = append(cond2, hits[1].Keywords...)
			second := b.decodeSequenceLocked(hits[1].Keywords, cond2, boost2, []string{h2Toks[0] + " " + h2Toks[1]}, maxWords)
			if second != "" && !strings.EqualFold(second, decoded) {
				decoded = decoded + " " + second
			}
		}
	}

	return decoded, true, ""
}

// GenerateModelThought decodes an unscripted training reflection directly from the neural model
// conditioned on [<thought>, stageCue, optionalTopicKeyword].
func (b *Brain) GenerateModelThought(sampleKeywords []string) string {
	b.mu.Lock()
	defer b.mu.Unlock()

	if b.TokensTrained == 0 || b.Net == nil || b.Net.LearnedWordCount() == 0 {
		return ""
	}

	stage := thoughtStageCue(b.Level)
	cues := []string{thoughtToken, stage}
	boost := make(map[string]bool)
	for _, kw := range sampleKeywords {
		if b.Net.KnowsToken(kw) {
			cues = append(cues, kw)
			boost[kw] = true
			break
		}
	}

	maxLen := b.maxDecodeLengthForLevel()
	return b.decodeSequenceLocked(cues, cues, boost, nil, maxLen)
}

func (b *Brain) maxDecodeLengthForLevel() int {
	switch {
	case b.Level <= 3:
		return 10
	case b.Level <= 9:
		return 16
	case b.Level <= 25:
		return 24
	default:
		return 34
	}
}

func (b *Brain) temperatureForLevel() float64 {
	switch {
	case b.Level <= 3:
		return 0.55
	case b.Level <= 12:
		return 0.42
	case b.Level <= 50:
		return 0.34
	default:
		return 0.28
	}
}

func (b *Brain) decodeSequenceLocked(
	cueTokens []string,
	condTokens []string,
	boostTokens map[string]bool,
	extraStarters []string,
	maxSteps int,
) string {
	rng := mrand.New(mrand.NewSource(time.Now().UnixNano()))
	temp := b.temperatureForLevel()

	condIDs := make([]int, 0, len(condTokens))
	for _, ct := range condTokens {
		if id, ok := b.Net.TokenToID[ct]; ok {
			condIDs = append(condIDs, id)
		}
	}
	if len(condIDs) == 0 {
		condIDs = []int{bosID}
	}

	// 1. Gather candidate starter phrases from learned PromptStarters
	starterWeights := make(map[string]float64)
	if len(cueTokens) > 0 && len(cueTokens) <= 5 {
		exactKey := "p:" + strings.Join(cueTokens, "_")
		for st, cnt := range b.PromptStarters[exactKey] {
			starterWeights[st] += float64(cnt) * 16.0
		}
	}
	if len(starterWeights) == 0 {
		for i := 0; i+1 < len(cueTokens); i++ {
			bg := cueTokens[i] + "_" + cueTokens[i+1]
			for st, cnt := range b.PromptStarters[bg] {
				starterWeights[st] += float64(cnt) * 5.0
			}
		}
	}
	if len(starterWeights) == 0 {
		for _, ct := range cueTokens {
			if len(cueTokens) > 1 && stopWords[ct] {
				continue
			}
			for st, cnt := range b.PromptStarters[ct] {
				starterWeights[st] += float64(cnt) * 1.5
			}
		}
	}
	for _, ex := range extraStarters {
		if strings.TrimSpace(ex) != "" {
			starterWeights[ex] += 12.0
		}
	}

	// Fallback to learned <sep> starters only if cues were conversational and no direct cue hit
	if len(starterWeights) == 0 {
		for tok, cnt := range b.NGramCounts[sepToken] {
			if tok != eosToken {
				starterWeights[tok] += float64(cnt)
			}
		}
	}
	if len(starterWeights) == 0 {
		return ""
	}

	type starterCand struct {
		tokens []string
		first  int
		weight float64
	}
	var sCands []starterCand
	var sFirstIDs []int
	for st, w := range starterWeights {
		parts := strings.Fields(st)
		if len(parts) == 0 {
			continue
		}
		fid, ok := b.Net.TokenToID[parts[0]]
		if !ok {
			continue
		}
		sCands = append(sCands, starterCand{tokens: parts, first: fid, weight: w})
		sFirstIDs = append(sFirstIDs, fid)
	}
	if len(sCands) == 0 {
		return ""
	}

	sNeural := b.Net.ForwardLogits(condIDs, padID, padID, sepID, sFirstIDs)
	sScores := make([]float32, len(sCands))
	for i, sc := range sCands {
		bonus := float32(0)
		for _, pt := range sc.tokens {
			if boostTokens[pt] {
				bonus += 2.2
			}
		}
		sScores[i] = sNeural[i] + float32(1.45*math.Log(sc.weight+1.0)) + bonus
	}

	pickedIdx := sampleIndexFromLogits(sScores, temp, rng)
	history := append([]string(nil), sCands[pickedIdx].tokens...)
	usedBigrams := make(map[string]int)
	if len(history) >= 2 {
		usedBigrams[history[0]+" "+history[1]] = 1
	}

	// 2. Autoregressive token-by-token decoding using NeuralNet + learned transition counts
	hardCap := maxSteps + 14
	for len(history) < hardCap {
		pastSoftLimit := len(history) >= maxSteps
		prev1 := history[len(history)-1]
		prev2 := sepToken
		if len(history) >= 2 {
			prev2 = history[len(history)-2]
		}
		prev3 := padToken
		if len(history) >= 3 {
			prev3 = history[len(history)-3]
		} else if len(history) == 2 {
			prev3 = sepToken
		}

		quadMap := b.NGramCounts[prev3+" "+prev2+" "+prev1]
		triMap := b.NGramCounts[prev2+" "+prev1]
		biMap := b.NGramCounts[prev1]

		candWeights := make(map[string]float64)
		for tok, cnt := range quadMap {
			candWeights[tok] += float64(cnt) * 10.0
		}
		for tok, cnt := range triMap {
			candWeights[tok] += float64(cnt) * 3.0
		}
		if len(candWeights) == 0 {
			for tok, cnt := range biMap {
				candWeights[tok] += float64(cnt)
			}
		}

		if len(candWeights) == 0 {
			break
		}

		var cTokens []string
		var cIDs []int
		for tok := range candWeights {
			if tok == eosToken && len(history) < 3 && len(candWeights) > 1 {
				continue
			}
			id, ok := b.Net.TokenToID[tok]
			if !ok {
				continue
			}
			cTokens = append(cTokens, tok)
			cIDs = append(cIDs, id)
		}
		if len(cTokens) == 0 {
			break
		}

		p3ID := b.tokenIDOrPad(prev3)
		p2ID := b.tokenIDOrPad(prev2)
		p1ID := b.tokenIDOrPad(prev1)

		nLogits := b.Net.ForwardLogits(condIDs, p3ID, p2ID, p1ID, cIDs)
		combined := make([]float32, len(cTokens))
		for i, tok := range cTokens {
			score := nLogits[i] + float32(1.50*math.Log(candWeights[tok]+1.0))
			if boostTokens[tok] {
				score += 3.0
			}
			if pastSoftLimit && (tok == eosToken || tok == "." || tok == "!" || tok == "?") {
				score += 4.5
			}
			if reps := usedBigrams[prev1+" "+tok]; reps > 0 {
				score -= float32(reps) * 3.2
			}
			if tok == prev1 {
				score -= 4.0
			}
			combined[i] = score
		}

		nextIdx := sampleIndexFromLogits(combined, temp, rng)
		chosen := cTokens[nextIdx]
		if chosen == eosToken || chosen == padToken {
			break
		}

		usedBigrams[prev1+" "+chosen]++
		history = append(history, chosen)

		if (chosen == "." || chosen == "!" || chosen == "?") && len(history) >= 5 {
			break
		}
	}

	return b.detokenizeWithSelf(history)
}

func (b *Brain) tokenIDOrPad(tok string) int {
	if id, ok := b.Net.TokenToID[tok]; ok {
		return id
	}
	return padID
}

func (b *Brain) normalizeSelfTokens(tokens []string) []string {
	selfLower := strings.ToLower(strings.TrimSpace(b.Name))
	out := make([]string, len(tokens))
	for i, t := range tokens {
		if t == selfLower || t == "matrix" {
			out[i] = selfToken
		} else {
			out[i] = t
		}
	}
	return out
}

func (b *Brain) knowsWordOrStemLocked(word string) bool {
	if b.Net.KnowsToken(word) {
		return true
	}
	// Check simple plural/singular suffix match in learned vocabulary
	if strings.HasSuffix(word, "s") && len(word) > 4 && b.Net.KnowsToken(strings.TrimSuffix(word, "s")) {
		return true
	}
	if b.Net.KnowsToken(word + "s") {
		return true
	}
	if len(word) >= 6 {
		prefix := word[:len(word)-1]
		for tok, id := range b.Net.TokenToID {
			if id >= reservedTokensCount && b.Net.TokenCounts[id] > 0 && strings.HasPrefix(tok, prefix) {
				return true
			}
		}
	}
	return false
}

func (b *Brain) detokenizeWithSelf(tokens []string) string {
	var out strings.Builder
	capNext := true

	for i, tok := range tokens {
		if tok == eosToken || tok == padToken || tok == bosToken || tok == sepToken || tok == thoughtToken {
			continue
		}
		display := tok
		if tok == selfToken {
			display = b.Name
		} else if tok == "i" {
			display = "I"
		} else if tok == "i'm" {
			display = "I'm"
		}

		isPunct := tok == "." || tok == "," || tok == "!" || tok == "?" || tok == ";" || tok == ":"
		if i > 0 && out.Len() > 0 && !isPunct {
			out.WriteByte(' ')
		}
		if capNext && len(display) > 0 && tok != selfToken {
			runes := []rune(display)
			runes[0] = unicode.ToUpper(runes[0])
			display = string(runes)
			capNext = false
		} else if !isPunct {
			capNext = false
		}
		if tok == "." || tok == "!" || tok == "?" {
			capNext = true
		}
		out.WriteString(display)
	}

	res := strings.TrimSpace(out.String())
	if res != "" && !strings.HasSuffix(res, ".") && !strings.HasSuffix(res, "!") && !strings.HasSuffix(res, "?") {
		res += "."
	}
	return res
}

// PickQuizPassage chooses a learned passage and masks one salient keyword for active-recall quizzing.
func (b *Brain) PickQuizPassage(topicHint string) (passage Passage, answer string, maskedText string, ok bool) {
	b.mu.Lock()
	defer b.mu.Unlock()

	if len(b.Passages) == 0 {
		return Passage{}, "", "", false
	}

	rng := mrand.New(mrand.NewSource(time.Now().UnixNano()))
	pool := b.Passages
	if strings.TrimSpace(topicHint) != "" {
		q := strings.ToLower(strings.TrimSpace(topicHint))
		var filtered []Passage
		for _, p := range b.Passages {
			if strings.Contains(strings.ToLower(p.Topic), q) || strings.Contains(strings.ToLower(p.Title), q) {
				filtered = append(filtered, p)
			}
		}
		if len(filtered) > 0 {
			pool = filtered
		}
	}

	for tries := 0; tries < 10; tries++ {
		cand := pool[rng.Intn(len(pool))]
		if len(cand.Keywords) == 0 {
			continue
		}
		ans := cand.Keywords[rng.Intn(len(cand.Keywords))]
		if len(ans) < 4 {
			continue
		}
		masked := maskWordCaseInsensitive(cand.Text, ans)
		if masked != cand.Text {
			return cand, ans, masked, true
		}
	}
	return Passage{}, "", "", false
}

// AwardStudyXP grants bonus XP from interactive Chat Mode studying or quizzes.
func (b *Brain) AwardStudyXP(xp int64) (oldLevel, newLevel int) {
	b.mu.Lock()
	defer b.mu.Unlock()

	oldLevel = b.Level
	if xp > 0 {
		b.XP += xp
		b.Level = LevelFromXP(b.XP)
	}
	return oldLevel, b.Level
}

func sampleIndexFromLogits(logits []float32, temp float64, rng *mrand.Rand) int {
	if len(logits) <= 1 {
		return 0
	}
	if temp <= 0.05 {
		best := 0
		for i := 1; i < len(logits); i++ {
			if logits[i] > logits[best] {
				best = i
			}
		}
		return best
	}
	maxL := float64(logits[0])
	for _, l := range logits[1:] {
		if float64(l) > maxL {
			maxL = float64(l)
		}
	}
	weights := make([]float64, len(logits))
	var sum float64
	for i, l := range logits {
		w := math.Exp((float64(l) - maxL) / temp)
		weights[i] = w
		sum += w
	}
	r := rng.Float64() * sum
	var acc float64
	for i, w := range weights {
		acc += w
		if r <= acc {
			return i
		}
	}
	return len(logits) - 1
}

func isWordToken(tok string) bool {
	if tok == selfToken {
		return true
	}
	for _, r := range tok {
		if unicode.IsLetter(r) || unicode.IsDigit(r) {
			return true
		}
	}
	return false
}

func maskWordCaseInsensitive(sentence, target string) string {
	lowerSent := strings.ToLower(sentence)
	lowerTarg := strings.ToLower(target)
	idx := strings.Index(lowerSent, lowerTarg)
	if idx < 0 {
		return sentence
	}
	return sentence[:idx] + "________" + sentence[idx+len(target):]
}
