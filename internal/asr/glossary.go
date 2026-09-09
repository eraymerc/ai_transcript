package asr

import (
	"os"
	"path/filepath"
	"sort"
	"strings"
)

// Glossary is the vocabulary a recording is expected to contain: names, place
// names, technical terms, recurring jargon.
//
// It is fed to whisper as its initial prompt, which biases the decoder toward
// these spellings. On a corpus of lectures the same terms recur in every
// recording, so a short list prevents the same handful of errors again and
// again -- much cheaper than finding and fixing them one at a time afterwards.
type Glossary struct {
	Terms []string
}

// ParseGlossary reads one term per line. Blank lines and lines starting with #
// are ignored, so a list can be commented and grouped.
func ParseGlossary(text string) Glossary {
	var g Glossary
	seen := map[string]bool{}
	for _, line := range strings.Split(text, "\n") {
		line = strings.TrimSpace(line)
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		// A line may hold several comma-separated terms.
		for _, term := range strings.Split(line, ",") {
			term = strings.TrimSpace(term)
			if term == "" {
				continue
			}
			key := strings.ToLower(term)
			if seen[key] {
				continue
			}
			seen[key] = true
			g.Terms = append(g.Terms, term)
		}
	}
	return g
}

// Prompt renders the terms as whisper expects an initial prompt: ordinary text
// in the target language, not a list of instructions.
func (g Glossary) Prompt() string {
	if len(g.Terms) == 0 {
		return ""
	}
	return strings.Join(g.Terms, ", ") + "."
}

// Sorted returns the terms in a stable order for display.
func (g Glossary) Sorted() []string {
	out := append([]string(nil), g.Terms...)
	sort.Slice(out, func(i, j int) bool {
		return strings.ToLower(out[i]) < strings.ToLower(out[j])
	})
	return out
}

// LoadGlossary reads the list from disk. A missing file is an empty glossary.
func LoadGlossary(path string) (string, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return "", nil
		}
		return "", err
	}
	return string(raw), nil
}

// SaveGlossary writes the list atomically.
func SaveGlossary(path, text string) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	tmp := path + ".tmp"
	if err := os.WriteFile(tmp, []byte(text), 0o644); err != nil {
		return err
	}
	return os.Rename(tmp, path)
}
