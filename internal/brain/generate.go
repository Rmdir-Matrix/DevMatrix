package brain

import (
	"math"
	mrand "math/rand"
	"strings"
	"time"
)

// GeneratePhrase synthesizes a short continuation phrase by scoring learned n-gram
// candidates through the trained NeuralNet forward pass.
func (b *Brain) GeneratePhrase(seed string, maxWords int) string {
	b.mu.Lock()
	defer b.mu.Unlock()

	if b.Net == nil || b.Net.VocabSize() < 12 || len(b.Transitions) == 0 {
		return ""
	}
	if maxWords < 5 {
		maxWords = 12
	}

	rng := mrand.New(mrand.NewSource(time.Now().UnixNano()))
	kws := ExtractKeywords(seed, 6)

	startTok := ""
	for _, kw := range kws {
		if len(b.Transitions[kw]) > 0 {
			startTok = kw
			break
		}
	}
	if startTok == "" {
		// Pick a start token from "<bos>" transitions
		starts := b.Transitions[bosToken]
		if len(starts) == 0 {
			return ""
		}
		startTok = starts[rng.Intn(len(starts))]
	}

	history := []string{startTok}
	for step := 0; step < maxWords; step++ {
		prev1 := history[len(history)-1]
		prev2 := bosToken
		if len(history) >= 2 {
			prev2 = history[len(history)-2]
		}

		cands := b.Transitions[prev2+" "+prev1]
		if len(cands) == 0 {
			cands = b.Transitions[prev1]
		}
		if len(cands) == 0 {
			break
		}

		scores := b.Net.ScoreCandidates(history, cands)
		chosen := sampleFromLogits(cands, scores, 0.75, rng)
		if chosen == "" || chosen == eosToken || chosen == padToken {
			break
		}
		history = append(history, chosen)
		if (chosen == "." || chosen == "!" || chosen == "?") && len(history) >= 6 {
			break
		}
	}

	return detokenize(history)
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

func sampleFromLogits(cands []string, logits []float32, temp float64, rng *mrand.Rand) string {
	if len(cands) == 0 {
		return ""
	}
	if len(cands) == 1 {
		return cands[0]
	}
	maxL := float64(logits[0])
	for _, l := range logits[1:] {
		if float64(l) > maxL {
			maxL = float64(l)
		}
	}
	weights := make([]float64, len(cands))
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
			return cands[i]
		}
	}
	return cands[len(cands)-1]
}

func detokenize(tokens []string) string {
	var out strings.Builder
	for i, tok := range tokens {
		isPunct := tok == "." || tok == "," || tok == "!" || tok == "?" || tok == ";" || tok == ":"
		if i > 0 && !isPunct {
			out.WriteByte(' ')
		}
		if i == 0 && len(tok) > 0 {
			out.WriteString(strings.ToUpper(tok[:1]) + tok[1:])
		} else {
			out.WriteString(tok)
		}
	}
	res := strings.TrimSpace(out.String())
	if res != "" && !strings.HasSuffix(res, ".") && !strings.HasSuffix(res, "!") && !strings.HasSuffix(res, "?") {
		res += "."
	}
	return res
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
