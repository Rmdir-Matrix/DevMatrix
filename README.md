# Matrix — On-Device AI Study Buddy CLI

**Matrix** is a standalone, single-executable CLI AI study buddy written in pure Go (`CGO_ENABLED=0`). Matrix uses an experimental small **100% Closed-Book Multi-Layer, Multi-Head Decoder-Only GPT Transformer** (`Pre-RMSNorm` + `Multi-Head Causal Self-Attention` + `SiLU Feed-Forward MLP` + `Exact Full-Vocabulary Weight-Tied Softmax`). Zero text databases, zero passage lookup tables, and zero RAG retrieval are used at chat time — model-generated chat text comes from trained Transformer `float32` weights; uncertainty messages and CLI tools are application text. It is not comparable in capability to ChatGPT.

---

## Quick Start (Single `.exe` Output)

Run the standalone Windows executable directly:

```powershell
.\matrix.exe
```

Useful startup flags:
```powershell
.\matrix.exe --train                        # Power on & jump straight into Training Mode
.\matrix.exe --chat                         # Jump straight into Study Buddy Chat Mode
.\matrix.exe --setup                        # Re-run Hardware RAM & Matrix Name Setup Wizard
.\matrix.exe --specs                        # Print auto-detected RAM/CPU specs & AI tier
.\matrix.exe --export my_buddy.matrix --export-name "NeoMatrix"
.\matrix.exe --import friend_buddy.matrix
```

---

## Architecture & Key Features

### 1. Closed-Book Decoder-Only GPT Architecture & Multi-Core AdamW
- **Multi-Layer, Multi-Head Causal Transformer (`GPT`)**: Stacked Pre-RMSNorm Transformer blocks (`WQ, WK, WV, WO` multi-head causal self-attention + `W1, W2` SiLU feed-forward key-value memory layers) with exact full-vocabulary cross-entropy backpropagation and zero inference-time lookup tables.
- **Subword Morpheme / WordPiece Tokenization**: Shares root representations across inflected forms (`learn`, `learning`, `learned`) and detects out-of-vocabulary words + high GPT perplexity without any hardcoded conversational word list.
- **Multi-Goroutine Mini-Batch AdamW**: Scales Transformer depth (`2–4 layers`), attention heads (`4–6 heads`), embedding/FFN width, `BatchSize`, and `WorkerThreads` directly to your hardware profile while tracking genuinely gradient-updated synaptic weights (`0` at birth).

### 2. Automatic Hardware Detection & RAM Targets
On first launch, Matrix probes your system's **Total RAM**, **Available Free RAM**, and **CPU Cores** (`GlobalMemoryStatusEx` on Windows, `/proc/meminfo` + cgroups on Linux) and lets you choose:
- **`[1] Auto-Optimize`** *(Recommended)* — chooses a conservative target based on total and available RAM.
- **`[2] Enter PC Total RAM`** — calculates a safe budget from your entered RAM.
- **`[3] Set a RAM Target`** — e.g. `256MB`, `512MB`, `1024MB`, `2GB` (target clamped to 75% of detected physical RAM).

Your chosen budget sets a Go runtime **soft memory target** (`debug.SetMemoryLimit` + `GOMAXPROCS`), bounds replay buffers, and selects neural dimensions. It is not a hard process-RAM limit: OS overhead, temporary buffers, imports, and other applications can exceed it. On a detected 4 GB machine the automatic target is at most 512 MB; use a smaller custom target if needed.

### 3. Mode 1 — Dictionaries First → Grammar → Multi-Source Real Articles
Press **`Enter`** (or **`1`**) from the Main Menu to turn Matrix ON:
- **Phase 1 — English Lexicon & Live Web Dictionaries (`[DICT]`)**: Trains first on **8 core chapters (640 foundational words)** from the embedded **4,800-word English lexicon** (`lexicon.json.gz`), enriched with live web definitions from `dictionaryapi.dev` and Wiktionary.
- **Phase 2 — Foundational Grammar (`8 sequential chapters`, `[GRAMMAR]`)**: Trains on English syntax, pronouns, tenses, adjectives/adverbs, question formation, conjunctions, and natural dialogue.
- **Phase 3 — Multi-Source Articles & Extended Lexicon (`[NOTES]`, `[WEB]`, `[DICT]`, `[ARTICLE]`)**: Reads personal `.txt`/`.md` notes in `./matrix_data/study_notes/`, full **Wikipedia** articles (up to 14k chars), **Wikibooks** chapters, **arXiv** research abstracts, the remaining **4,160 extended lexicon words**, and the built-in 23-article study encyclopedia.
- **Physical growth**: After every eight successfully trained items, the dense transformer can add FFN neurons, subject to a conservative estimated-memory check and a profile-dependent width ceiling. New output weights start at zero so growth initially preserves predictions; further training learns them. `/brain` reports parameter counts and `/grow` retries an earned milestone. More parameters do not guarantee better answers. This is an experimental small model, not ChatGPT.
- **Emergent Neural Thoughts**: Periodic reflections during training are decoded directly from the live neural weights — zero planted `THOUGHT:` strings or milestone scripts.

### 4. Mode 2 — Full-Vocabulary Neural Study Buddy Chat Mode
Choose **`[2] Chat Mode`** to talk and study with your Matrix:
- **Zero built-in fixed answers**: Every reply is decoded token-by-token from Matrix's trained neural weights over the full vocabulary.
- **Subword OOV + Neural Perplexity Unknown Gate**: When Matrix encounters unknown words or high predictive entropy / perplexity on an unstudied topic, it honestly tells you what it hasn't learned yet.
- Built-in study tools:
  - `/define <word>` — look up & train on an embedded or live web dictionary definition for `<word>`
  - `remember: <fact>` — train Matrix's neural weights directly on a new sentence
  - `/study <topic>` — fetch & train on multi-source articles about `<topic>` immediately
  - `/quiz [topic]` — active-recall fill-in-the-blank quiz generated from studied passages
  - `/summary` — list studied subjects and recent articles
  - `/brain` — inspect live neural architecture, parameters, learned vocabulary, active synapses, and loss
  - `/grow` — retry an earned growth milestone
  - `/train` — switch straight back into Training Mode

### 5. Compact Binary Checkpoints & Cross-RAM `.matrix` Sharing
Choose **`[3] Share / Export`** to:
- **Export** your Matrix under any custom name into a single gzipped, SHA-256 verified `.matrix` archive backed by **IEEE-754 little-endian binary `float32` tensor serialization (`MXG3`)**.
- **Import** a friend's `.matrix` file — pruning vocabulary and replay buffers after loading. Model layers and FFN width are not downsized, and import temporarily needs extra memory; large models may not fit on a smaller machine.
- **Rename** or **Switch / Create** multiple saved Matrixes.

### Viewing an exported `.matrix` file

Open `matrix-viewer.html` in a recent Chrome, Edge, or Firefox browser, then choose or drag in a `matrix-share-v2` `.matrix` export. The standalone viewer works offline and can download a readable `.txt` report with the metadata, saved replay samples, vocabulary, and tensor summaries. Replay may contain private training text; weights are numbers, not a recoverable transcript of everything learned. The viewer checks the archive checksum when Web Crypto is available, and limits files to 32 MB compressed / 96 MB expanded to reduce memory use. It does not modify the original or create an importable `.matrix` file.

To test the viewer: `node tests/viewer.test.cjs` (Node 22+).

**Current limitation:** All model weights and optimizer state are in RAM. SSD expert paging, sparse MoE routing, and billion-parameter growth are not implemented; saving to disk is not SSD paging. Training and inference get slower as the dense model grows.

---

## Compiling for Windows & Linux

Requires Go 1.22+:

```bash
# Build single standalone Windows executable (matrix.exe)
GOOS=windows GOARCH=amd64 CGO_ENABLED=0 go build -trimpath -ldflags="-s -w" -o matrix.exe ./cmd/matrix

# Build single standalone Linux executable (matrix)
GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go build -trimpath -ldflags="-s -w" -o matrix ./cmd/matrix

# Run full test suite
go test -v ./...
```
