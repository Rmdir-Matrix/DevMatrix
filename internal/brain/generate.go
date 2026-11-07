package brain

import (
	"fmt"
	"math"
	"math/rand"
	"sort"
	"strings"
	"time"
)

// GenerateModelReply decodes a response 100% from the Causal Self-Attention + GRU language model
// over the full vocabulary (zero n-gram lookup tables, zero hardcoded reply maps).
// Returns (reply, known=false, reason) when subword OOV or neural perplexity detects unknown input.
func (b *Brain) GenerateModelReply(userInput string) (string, bool, string) {
	trimmed := strings.TrimSpace(userInput)
	if trimmed == "" {
		return "", false, "Say something or ask a study question."
	}

	if b.Net == nil || b.Net.AdamStep == 0 || b.Net.LearnedWordCount() < 12 {
		return "", false, fmt.Sprintf(
			"My neural weights are untrained (%d words, 0 learned synapses). Start Mode 1 (Training Mode) so I can study my dictionaries and grammar first.",
			b.Net.LearnedWordCount(),
		)
	}

	// 1. Subword WordPiece tokenization + OOV check (no hardcoded conversational word list!)
	promptSubwords, unknownWords := EncodeWithVocab(trimmed, b.Name, b.Net.KnowsToken)
	if len(unknownWords) > 0 {
		shown := unknownWords
		if len(shown) > 4 {
			shown = shown[:4]
		}
		return "", false, fmt.Sprintf(
			"I do not know %q yet (%d words in my vocabulary). Try '/define %s' or '/study %s' to teach me!",
			strings.Join(shown, ", "), b.Net.LearnedWordCount(), shown[0], shown[0],
		)
	}
	if len(promptSubwords) == 0 {
		return "", false, "I could not parse any known words from that input."
	}

	// 2. Retrieve relevant <ctx> subwords from learned Dictionary or Semantic Passages
	ctxSubwords := b.retrieveContextSubwords(trimmed)

	// 3. Build neural conditioning sequence: [<ctx> ctxSubwords] <ask> promptSubwords <ans>
	askSub := promptSubwords
	if len(askSub) > 14 {
		askSub = askSub[len(askSub)-14:]
	}
	askIDs := b.Net.LookupTokens(askSub)

	var condSeq []int
	if len(ctxSubwords) > 0 {
		if len(ctxSubwords) > 24 {
			ctxSubwords = ctxSubwords[:24]
		}
		condSeq = append(condSeq, ctxID)
		condSeq = append(condSeq, b.Net.LookupTokens(ctxSubwords)...)
	}
	condSeq = append(condSeq, askID)
	condSeq = append(condSeq, askIDs...)
	condSeq = append(condSeq, ansID)

	// 4. Neural Perplexity / Predictive Entropy Uncertainty Gate
	probs, logits, entropy, maxProb := b.Net.ForwardFullVocab(condSeq)
	if len(probs) == 0 {
		return "", false, "My neural vocabulary is empty."
	}

	V := float64(b.Net.LearnedWordCount() + 2)
	maxPossibleEntropy := math.Log(math.Max(V, 4.0))
	normEntropy := entropy / maxPossibleEntropy
	perplexity := math.Exp(entropy)

	// When no <ctx> passage/definition matched, require strong direct neural confidence
	// (e.g. learned conversational QA patterns have high maxProb & low entropy; unstudied topics have flat distributions).
	if len(ctxSubwords) == 0 {
		kws := ExtractKeywords(trimmed, 6)
		// If asking about domain keywords with zero retrieved memory and low direct neural confidence, report unknown
		if maxProb < 0.075 || normEntropy > 0.84 {
			topicHint := trimmed
			if len(kws) > 0 {
				topicHint = strings.Join(kws, " ")
			}
			return "", false, fmt.Sprintf(
				"My neural uncertainty is too high on %q (perplexity %.1f, confidence %.1f%%) and I have no studied notes on it yet. Try '/study %s' or 'remember: <fact>'!",
				topicHint, perplexity, maxProb*100, topicHint,
			)
		}
	} else if maxProb < 0.025 {
		return "", false, fmt.Sprintf(
			"My neural weights are not confident enough to form an answer yet (perplexity %.1f). Train me further in Mode 1!",
			perplexity,
		)
	}

	// 5. Full-Vocabulary Autoregressive Nucleus (Top-P / Top-K) Decoding
	rng := rand.New(rand.NewSource(time.Now().UnixNano()))
	maxGenTokens := 26
	temperature := float32(0.45)
	if b.Level < 5 {
		temperature = 0.55
	}

	generated := b.decodeAutoregressive(condSeq, probs, logits, maxGenTokens, temperature, 14, 0.85, rng)
	if len(generated) == 0 {
		return "", false, "I could not decode a complete sentence from my neural weights yet. Train me a bit more in Mode 1!"
	}

	reply := DetokenizeSubwords(generated, b.Name)
	if strings.TrimSpace(reply) == "" {
		return "", false, "My decoded output was empty; please train me on more grammar and articles."
	}
	return reply, true, ""
}

// GenerateThoughtFromWeights decodes an emergent status thought during Training Mode
// 100% from the Causal Self-Attention + GRU neural weights (zero fmt.Sprintf templates!).
func (b *Brain) GenerateThoughtFromWeights(recentTitle, recentExcerpt string) string {
	if b.Net == nil || b.Net.AdamStep == 0 || b.Net.LearnedWordCount() < 6 {
		return ""
	}

	rng := rand.New(rand.NewSource(time.Now().UnixNano() + int64(b.ArticlesRead*79)))

	// Seed the neural decoder with <bos> + 1..2 subwords from the newly learned item so the
	// GRU + Self-Attention completes a genuine thought from its current weights.
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

	probs, logits, _, _ := b.Net.ForwardFullVocab(seq)
	if len(probs) == 0 {
		return ""
	}

	// Decoding horizon and sharpness naturally reflect model maturity
	maxToks := 10 + minInt(b.Level, 14)
	temp := float32(0.70)
	if b.Level >= 6 {
		temp = 0.50
	}

	cont := b.decodeAutoregressive(seq, probs, logits, maxToks, temp, 12, 0.85, rng)
	outTokens = append(outTokens, cont...)
	if len(outTokens) < 3 {
		return ""
	}
	return DetokenizeSubwords(outTokens, b.Name)
}

func (b *Brain) retrieveContextSubwords(userInput string) []string {
	lower := strings.ToLower(strings.TrimSpace(userInput))
	kws := ExtractKeywords(userInput, 6)

	isDefAsk := strings.HasPrefix(lower, "define ") ||
		strings.Contains(lower, "what does ") ||
		strings.Contains(lower, "meaning of ") ||
		(strings.HasPrefix(lower, "what is ") && len(kws) == 1)

	// 1. Check lexical dictionary first for explicit word definitions
	if isDefAsk {
		for _, kw := range kws {
			if kw == "define" || kw == "mean" || kw == "meaning" {
				continue
			}
			if entry, ok := b.LookupDictionary(kw); ok {
				return SubwordTokenize(entry.Definition)
			}
		}
		rawToks := Tokenize(userInput)
		for _, rt := range rawToks {
			if rt == "define" || rt == "mean" || rt == "meaning" || stopWords[rt] {
				continue
			}
			if entry, ok := b.LookupDictionary(rt); ok {
				return SubwordTokenize(entry.Definition)
			}
		}
	}

	// 2. Search studied domain passages using topic keywords (avoiding question-word noise)
	if len(kws) > 0 {
		searchQuery := strings.Join(kws, " ")
		hits := b.SearchPassagesWithScore(searchQuery, 2)
		if len(hits) > 0 && hits[0].Score >= 0.32 {
			// For multi-keyword queries, verify topical relevance
			if len(kws) == 1 || hits[0].Score >= 0.45 {
				return SubwordTokenize(hits[0].Passage.Text)
			}
		}
	}

	// 3. Fallback to dictionary entry ONLY when prompt is a question or concept inquiry (not bare greetings)
	isInquiry := strings.Contains(lower, "?") ||
		strings.HasPrefix(lower, "what ") ||
		strings.HasPrefix(lower, "explain ") ||
		strings.HasPrefix(lower, "describe ") ||
		strings.HasPrefix(lower, "tell ")
	if isInquiry && len(kws) == 1 {
		if entry, ok := b.LookupDictionary(kws[0]); ok {
			return SubwordTokenize(entry.Definition)
		}
	}
	return nil
}

func (b *Brain) decodeAutoregressive(
	initSeq []int,
	firstProbs []float32,
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

		// Stop cleanly at sentence-ending punctuation once a complete clause is formed
		if (tok == "." || tok == "!" || tok == "?") && len(out) >= 6 {
			break
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

	type cand struct {
		id    int
		logit float32
		prob  float64
	}
	pool := make([]cand, 0, len(logits))

	for vid, rawL := range logits {
		if rawL <= -27.5 {
			continue
		}
		// Never repeat the exact same token immediately in a row
		if vid == prevID {
			continue
		}
		if vid == eosID {
			if step < 4 {
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
		// Subword suffix "##..." can only follow an alphanumeric root word when genuinely high-probability
		if isSub && (prevWasPunct || prevWasSubword || rawL < -1.35) {
			continue
		}
		// Prevent consecutive punctuation ("..", ".,", etc.)
		if isPunct && prevWasPunct {
			continue
		}

		adj := rawL // rawL is log(prob) <= 0
		if cnt := seenCounts[vid]; cnt > 0 {
			if isPunct {
				adj -= float32(cnt) * 2.2
			} else if stopWords[tok] {
				if cnt >= 2 {
					adj -= float32(cnt-1) * 1.8
				} else {
					adj -= 0.45
				}
			} else {
				adj -= float32(cnt) * 2.4
			}
		}
		pool = append(pool, cand{id: vid, logit: adj * invTemp})
	}

	if len(pool) == 0 {
		return eosID
	}

	sort.Slice(pool, func(i, j int) bool {
		return pool[i].logit > pool[j].logit
	} )
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

	// Nucleus Top-P cutoff
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
