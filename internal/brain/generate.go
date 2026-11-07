package brain

import (
	"fmt"
	"math"
	"math/rand"
	"sort"
	"strings"
	"time"
)

// GenerateModelReply runs pure closed-book GPT inference without prior chat turns.
func (b *Brain) GenerateModelReply(userInput string) (string, bool, string) {
	return b.GenerateModelReplyWithHistory(nil, userInput)
}

// GenerateModelReplyWithHistory decodes a response 100% from the Closed-Book Multi-Layer,
// Multi-Head GPT Transformer weights — ZERO lookup tables, ZERO RAG passage/dictionary retrieval!
func (b *Brain) GenerateModelReplyWithHistory(history []ChatTurn, userInput string) (string, bool, string) {
	trimmed := strings.TrimSpace(userInput)
	if trimmed == "" {
		return "", false, "Say something or ask a study question."
	}

	if b.Net == nil || b.Net.AdamStep == 0 || b.Net.LearnedWordCount() < 12 {
		return "", false, fmt.Sprintf(
			"My GPT Transformer weights are untrained (%d words, 0 learned synapses). Start Mode 1 (Training Mode) so I can train my layers on dictionaries, grammar, and articles first.",
			b.Net.LearnedWordCount(),
		)
	}

	// 1. Subword WordPiece tokenization + OOV check
	promptSubwords, unknownWords := EncodeWithVocab(trimmed, b.Name, b.Net.KnowsToken)
	if len(unknownWords) > 0 {
		shown := unknownWords
		if len(shown) > 4 {
			shown = shown[:4]
		}
		return "", false, fmt.Sprintf(
			"I have not trained on %q yet (%d words in my GPT vocabulary). Try '/define %s' or '/study %s' so I can learn it in my weights!",
			strings.Join(shown, ", "), b.Net.LearnedWordCount(), shown[0], shown[0],
		)
	}
	if len(promptSubwords) == 0 {
		return "", false, "I could not parse any known tokens from that input."
	}

	// 2. Build Closed-Book GPT Causal Context Window: [optional prior turn] + <user> prompt <assistant>
	askSub := promptSubwords
	if len(askSub) > 18 {
		askSub = askSub[len(askSub)-18:]
	}
	askIDs := b.Net.LookupTokens(askSub)

	var condSeq []int
	if len(history) > 0 && isFollowUpTurn(trimmed) {
		last := history[len(history)-1]
		prevU, uUnk := EncodeWithVocab(last.User, b.Name, b.Net.KnowsToken)
		prevA, aUnk := EncodeWithVocab(last.Assistant, b.Name, b.Net.KnowsToken)
		if len(uUnk) == 0 && len(aUnk) == 0 && len(prevU)+len(prevA)+len(askIDs)+6 <= maxSeqLen-24 {
			condSeq = append(condSeq, userID)
			condSeq = append(condSeq, b.Net.LookupTokens(prevU)...)
			condSeq = append(condSeq, assistantID)
			condSeq = append(condSeq, b.Net.LookupTokens(prevA)...)
			condSeq = append(condSeq, eosID)
		}
	}
	condSeq = append(condSeq, userID)
	condSeq = append(condSeq, askIDs...)
	condSeq = append(condSeq, assistantID)

	// 3. Forward pass through all Multi-Head GPT Transformer blocks (100% pure weights!)
	probs, logits, entropy, maxProb := b.Net.ForwardFullVocab(condSeq)
	if len(probs) == 0 {
		return "", false, "My GPT vocabulary is empty."
	}

	V := float64(b.Net.LearnedWordCount() + 2)
	maxPossibleEntropy := math.Log(math.Max(V, 4.0))
	normEntropy := entropy / maxPossibleEntropy
	perplexity := math.Exp(entropy)

	// 4. Pure Neural Softmax Confidence / Perplexity Gate
	if maxProb < 0.065 || normEntropy > 0.85 {
		kws := ExtractKeywords(trimmed, 5)
		topicHint := trimmed
		if len(kws) > 0 {
			topicHint = strings.Join(kws, " ")
		}
		return "", false, fmt.Sprintf(
			"My GPT weights are still uncertain about %q (confidence %.1f%%, perplexity %.1f). Train me more in Mode 1 or run '/study %s'!",
			topicHint, maxProb*100, perplexity, topicHint,
		)
	}

	// 5. Closed-Book Autoregressive Nucleus (Top-P / Top-K) Decoding
	rng := rand.New(rand.NewSource(time.Now().UnixNano()))
	temperature := float32(0.35)
	if b.Level < 5 {
		temperature = 0.50
	}

	generated := b.decodeAutoregressive(condSeq, logits, 28, temperature, 10, 0.82, rng)
	if len(generated) == 0 {
		return "", false, "My GPT layers need more training to complete that thought — run Mode 1 for a few more items!"
	}

	reply := DetokenizeSubwords(generated, b.Name)
	if strings.TrimSpace(reply) == "" {
		return "", false, "My decoded output was empty; please train me further in Mode 1."
	}
	return reply, true, ""
}

// GenerateThoughtFromWeights decodes an emergent status thought during Training Mode
// 100% from the GPT Transformer weights starting from <bos> + 1..2 seed subwords.
func (b *Brain) GenerateThoughtFromWeights(recentTitle, recentExcerpt string) string {
	if b.Net == nil || b.Net.AdamStep == 0 || b.Net.LearnedWordCount() < 6 {
		return ""
	}

	rng := rand.New(rand.NewSource(time.Now().UnixNano() + int64(b.ArticlesRead*79)))
	seedSource := recentExcerpt
	if strings.TrimSpace(seedSource) == "" {
		seedSource = recentTitle
	}
	subwords := SubwordTokenize(seedSource)

	seq := []int{bosID}
	limitSeed := 2
	if b.Level <= 2 {
		limitSeed = 1
	}
	added := 0
	var outTokens []string
	for _, sw := range subwords {
		if b.Net.KnowsToken(sw) && isAlphaNumericToken(sw) {
			seq = append(seq, b.Net.TokenToID[sw])
			outTokens = append(outTokens, sw)
			added++
			if added >= limitSeed {
				break
			}
		}
	}

	_, logits, _, _ := b.Net.ForwardFullVocab(seq)
	if len(logits) == 0 {
		return ""
	}

	maxToks := 10 + minInt(b.Level, 14)
	temp := float32(0.65)
	if b.Level >= 6 {
		temp = 0.45
	}

	cont := b.decodeAutoregressive(seq, logits, maxToks, temp, 10, 0.85, rng)
	outTokens = append(outTokens, cont...)
	if len(outTokens) < 3 {
		return ""
	}
	return DetokenizeSubwords(outTokens, b.Name)
}

func isFollowUpTurn(input string) bool {
	l := strings.ToLower(strings.TrimSpace(input))
	for _, p := range []string{
		"tell me more", "more about that", "elaborate", "explain more",
		"why is that", "and why", "give an example", "what else",
	} {
		if strings.HasPrefix(l, p) {
			return true
		}
	}
	return false
}

func (b *Brain) decodeAutoregressive(
	initSeq []int,
	firstLogits []float32,
	maxSteps int,
	temperature float32,
	topK int,
	topP float64,
	rng *rand.Rand,
) []string {
	seq := make([]int, len(initSeq), len(initSeq)+maxSteps)
	copy(seq, initSeq)

	seenCounts := make(map[int]int, maxSteps)
	var out []string
	prevID := -1

	curLogits := firstLogits
	for step := 0; step < maxSteps; step++ {
		if step > 0 {
			_, curLogits, _, _ = b.Net.ForwardFullVocab(seq)
			if len(curLogits) == 0 {
				break
			}
		}

		nextID := sampleTopKTopP(curLogits, seenCounts, prevID, step, temperature, topK, topP, b.Net.IDToToken, rng)
		if nextID == eosID || nextID <= bosID {
			if step >= 3 {
				break
			}
			continue
		}

		tok := b.Net.IDToToken[nextID]
		out = append(out, tok)
		seq = append(seq, nextID)
		seenCounts[nextID]++
		prevID = nextID

		if (tok == "." || tok == "!" || tok == "?") && len(out) >= 6 {
			// Stop cleanly if <eos> is strong on the next step or clause is complete
			nextProbs, _, _, _ := b.Net.ForwardFullVocab(seq)
			if len(nextProbs) > eosID && (nextProbs[eosID] >= 0.15 || len(out) >= 14) {
				break
			}
		}
	}
	return out
}

func sampleTopKTopP(
	logits []float32,
	seenCounts map[int]int,
	prevID int,
	step int,
	temperature float32,
	topK int,
	topP float64,
	idToTok []string,
	rng *rand.Rand,
) int {
	if temperature < 0.05 {
		temperature = 0.05
	}
	invTemp := float32(1.0) / temperature

	prevWasPunct := false
	prevWasSubword := false
	if prevID >= 0 && prevID < len(idToTok) {
		pt := idToTok[prevID]
		prevWasPunct = !isAlphaNumericToken(pt) && !strings.HasPrefix(pt, "##")
		prevWasSubword = strings.HasPrefix(pt, "##")
	}

	// Find max logit for subword probability check
	maxRawL := float32(-1e30)
	for _, l := range logits {
		if l > maxRawL {
			maxRawL = l
		}
	}

	type cand struct {
		id    int
		logit float32
		prob  float64
	}
	pool := make([]cand, 0, len(logits))

	for vid, rawL := range logits {
		if rawL <= -1e8 {
			continue
		}
		if vid == prevID {
			continue
		}
		if vid == eosID {
			if step < 3 {
				continue
			}
			pool = append(pool, cand{id: vid, logit: rawL * invTemp})
			continue
		}

		tok := idToTok[vid]
		isSub := strings.HasPrefix(tok, "##")
		isPunct := !isAlphaNumericToken(tok) && !isSub

		if step == 0 && (isSub || isPunct) {
			continue
		}
		if isSub && (prevWasPunct || prevWasSubword || (maxRawL-rawL) > 1.5) {
			continue
		}
		if isPunct && prevWasPunct {
			continue
		}

		adj := rawL
		if cnt := seenCounts[vid]; cnt > 0 {
			if isPunct {
				adj -= float32(cnt) * 2.5
			} else if stopWords[tok] {
				if cnt >= 2 {
					adj -= float32(cnt-1) * 1.8
				} else {
					adj -= 0.4
				}
			} else {
				adj -= float32(cnt) * 2.5
			}
		}
		pool = append(pool, cand{id: vid, logit: adj * invTemp})
	}

	if len(pool) == 0 {
		return eosID
	}

	sort.Slice(pool, func(i, j int) bool {
		return pool[i].logit > pool[j].logit
	})
	if topK > 0 && len(pool) > topK {
		pool = pool[:topK]
	}

	maxL := pool[0].logit
	var sumExp float64
	for i := range pool {
		e := math.Exp(float64(pool[i].logit - maxL))
		pool[i].prob = e
		sumExp += e
	}
	for i := range pool {
		pool[i].prob /= sumExp
	}

	if topP > 0 && topP < 1.0 {
		var cum float64
		keep := 0
		for i := range pool {
			cum += pool[i].prob
			keep++
			if cum >= topP {
				break
			}
		}
		if keep < 1 {
			keep = 1
		}
		pool = pool[:keep]
		var renorm float64
		for i := range pool {
			renorm += pool[i].prob
		}
		for i := range pool {
			pool[i].prob /= renorm
		}
	}

	r := rng.Float64()
	var acc float64
	for _, c := range pool {
		acc += c.prob
		if r <= acc {
			return c.id
		}
	}
	return pool[len(pool)-1].id
}
