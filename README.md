# Matrix — On-Device AI Study Buddy CLI

**Matrix** is a standalone, single-executable CLI AI study buddy written in pure Go (`CGO_ENABLED=0`). A brand-new Matrix starts with **0 words** and **zero hardcoded answers**: when you turn it on, it first learns **foundational English words and grammar**, then reads **real articles** (live Wikipedia articles, your own `.txt`/`.md` study notes, and a built-in 23-article study encyclopedia), training an on-device positional neural language model via backpropagation and decoding all responses word-by-word from its trained model.

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

## Key Features

### 1. Automatic Hardware Detection & Safe RAM Setup
On first launch, Matrix probes your system's **Total RAM**, **Available Free RAM**, and **CPU Cores** (`GlobalMemoryStatusEx` on Windows, `/proc/meminfo` + cgroups on Linux) and lets you choose:
- **`[1] Auto-Optimize`** *(Recommended)* — safely allocates ~15–20% of free RAM so Matrix never overwhelms your PC.
- **`[2] Enter PC Total RAM`** — calculates a safe budget from your entered RAM.
- **`[3] Set Exact Max RAM Cap`** — e.g. `256MB`, `512MB`, `1024MB`, `2GB` (guarded so it can never exceed 75% of physical RAM).

Your chosen budget is enforced in the Go runtime (`debug.SetMemoryLimit` + `GOMAXPROCS`) and scales neural network dimensions (`Featherweight` → `Lite` → `Balanced` → `Performance` → `Ultra`).

### 2. Mode 1 — Words & Grammar First Curriculum + Real Article Training
Press **`Enter`** (or **`1`**) from the Main Menu to turn Matrix ON:
- **Phase 1 — Foundational Words & Grammar (`10 sequential lessons`)**: Before reading complex encyclopedias, a fresh Matrix learns core English words, greetings, pronouns, verbs, tenses, adjectives, prepositions, question words, and conjunctions.
- **Phase 2 — Real Articles**: Once its grammar foundation is complete, Matrix reads live web articles from **Wikipedia's API**, personal notes dropped into `./matrix_data/study_notes/` (`.txt` / `.md`), and the built-in 23-article study encyclopedia.
- **100% Model-Decoded Training Thoughts**: Every periodic message spoken during training (`"Matrix is learn now!"` → `"I'm learning stuff..."` → `"I am learning which makes more neuron connections in my brain."` and article-specific reflections) is decoded word-by-word directly from the neural model.
- Type `chat` + `Enter` anytime during training to jump straight into Chat Mode, or press `Enter` to pause cleanly.

### 3. Mode 2 — Pure Model-Driven Study Buddy Chat Mode
Choose **`[2] Chat Mode`** to talk and study with your Matrix:
- **Zero built-in fixed answers**: Every reply (including `"hello"` and `"how are you"`) is decoded autoregressively word-by-word from Matrix's trained neural language model and learned transition distributions.
- **Honest Unknown Detection**: If Matrix hasn't trained yet or you ask about a word/concept it has never read about, it never fakes a canned answer — it tells you honestly what it doesn't know yet.
- Built-in study tools:
  - `remember: <fact>` — train Matrix's neural weights directly on a new sentence
  - `/study <topic>` — have Matrix fetch & train on articles about `<topic>` immediately
  - `/quiz [topic]` — active-recall fill-in-the-blank quiz generated from studied articles
  - `/summary` — list studied subjects and recent articles
  - `/brain` — inspect live neural architecture, learned words, active synapses, and loss
  - `/train` — jump straight back into Training Mode

### 4. Export Name & Share Your Matrixes (`.matrix` Files)
Choose **`[3] Share / Export`** to:
- **Export** your Matrix under any custom name into a single compressed, SHA-256 verified `.matrix` file to send to friends.
- **Import** a friend's `.matrix` file (automatically adapting memory caps safely to *your* PC's RAM).
- **Rename** or **Switch / Create** multiple saved Matrixes.

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
