package pretrained

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"regexp"

	"github.com/Rmdir-Matrix/DevMatrix/internal/specs"
)

// Catalog entries are pinned to a specific GGUF revision and SHA-256.
type modelSpec struct {
	ID, Name, Filename, URL, SHA, Format, License string
	SizeMB                                        int
}

var smallModel = modelSpec{
	ID: "smollm2-135m", Name: "SmolLM2 135M Instruct Q4", Filename: "SmolLM2-135M-Instruct.Q4_K_M.gguf",
	URL: "https://huggingface.co/QuantFactory/SmolLM2-135M-Instruct-GGUF/resolve/a6b7826ed969c3926a98359e39912af137b464cb/SmolLM2-135M-Instruct.Q4_K_M.gguf",
	SHA: "8030f04528538d47bda434f6f0bdf3952c40a58123e4d5e755332f23731a8684", Format: "chatml", License: "Apache 2.0: https://www.apache.org/licenses/LICENSE-2.0", SizeMB: 105,
}
var gemmaModel = modelSpec{
	ID: "gemma-270m", Name: "Gemma 3 270M Instruct Q4", Filename: "gemma-3-270m-it-Q4_K_M.gguf",
	URL: "https://huggingface.co/unsloth/gemma-3-270m-it-GGUF/resolve/37cbb6e/gemma-3-270m-it-Q4_K_M.gguf",
	SHA: "b1baabd6b729e4041822220d3e648e00d99cac5df86b10dffb77bcccf0688e39", Format: "gemma", License: "Gemma terms: https://ai.google.dev/gemma/terms", SizeMB: 253,
}

// Selection is deliberately conservative. Free RAM varies, so it is checked
// during actual execution too. Neither choice guarantees that training fits.
func recommendedModel(hw specs.HardwareInfo) modelSpec {
	if hw.Detected && hw.TotalRAMMB >= 6144 && hw.AvailableRAMMB >= 2560 {
		return gemmaModel
	}
	return smallModel
}
func RecommendedModelName(hw specs.HardwareInfo) string { return recommendedModel(hw).Name }
func modelID(s Settings) string {
	if s.ModelID != "" {
		return s.ModelID
	}
	switch s.ModelHash {
	case smallModel.SHA:
		return smallModel.ID
	case gemmaModel.SHA:
		return gemmaModel.ID
	}
	// Old versions only supported Gemma; preserve their prompt formatting.
	return ""
}
func modelFormat(s Settings) string {
	if s.ChatFormat != "" {
		return s.ChatFormat
	}
	if modelID(s) == smallModel.ID {
		return "chatml"
	}
	return "gemma"
}

var shaPattern = regexp.MustCompile(`^[a-f0-9]{64}$`)

func profilePath(dir, hash string) (string, error) {
	if !shaPattern.MatchString(hash) {
		return "", fmt.Errorf("invalid model hash for profile")
	}
	return filepath.Join(dir, "model_profiles", hash+".json"), nil
}

// Save an independent snapshot before switching models. Adapters are model-specific.
func saveProfile(dir string, s Settings) error {
	if s.ModelHash == "" {
		return nil
	}
	p, e := profilePath(dir, s.ModelHash)
	if e != nil {
		return e
	}
	if e = os.MkdirAll(filepath.Dir(p), 0700); e != nil {
		return e
	}
	b, e := json.MarshalIndent(s, "", "  ")
	if e != nil {
		return e
	}
	f, e := os.CreateTemp(filepath.Dir(p), "profile-*.tmp")
	if e != nil {
		return e
	}
	defer os.Remove(f.Name())
	if _, e = f.Write(b); e != nil {
		f.Close()
		return e
	}
	if e = f.Close(); e != nil {
		return e
	}
	return os.Rename(f.Name(), p)
}
func loadProfile(dir, hash string) (Settings, bool, error) {
	var s Settings
	p, e := profilePath(dir, hash)
	if e != nil {
		return s, false, e
	}
	b, e := os.ReadFile(p)
	if os.IsNotExist(e) {
		return s, false, nil
	}
	if e != nil {
		return s, false, e
	}
	if e = json.Unmarshal(b, &s); e != nil {
		return s, false, e
	}
	if s.ModelHash != hash {
		return s, false, fmt.Errorf("model profile hash mismatch")
	}
	return s, true, nil
}

// Link a new selection with its own earlier adapter, never another model's.
func applyModelProfile(dir string, old Settings, next *Settings) error {
	if old.ModelHash == "" && old.Adapter != "" {
		return fmt.Errorf("cannot switch models safely: previous adapter has no recorded base-model hash")
	}
	if old.ModelHash != "" && old.ModelHash != next.ModelHash {
		if err := saveProfile(dir, old); err != nil {
			return err
		}
	}
	previous := old
	if old.ModelHash != next.ModelHash {
		p, found, err := loadProfile(dir, next.ModelHash)
		if err != nil {
			return err
		}
		if !found {
			return nil
		}
		previous = p
	}
	if previous.ModelHash == next.ModelHash {
		if previous.Adapter != "" {
			if err := ggufFile(previous.Adapter); err != nil {
				return fmt.Errorf("existing model adapter is unavailable: %w", err)
			}
		}
		next.Adapter = previous.Adapter
		next.Seen = previous.Seen
		next.Sessions = previous.Sessions
		if previous.ChatFormat != "" {
			next.ChatFormat = previous.ChatFormat
		}
		if previous.ModelID != "" {
			next.ModelID = previous.ModelID
		}
	}
	return nil
}
