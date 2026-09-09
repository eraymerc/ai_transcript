package web

import (
	"encoding/json"
	"os"
	"path/filepath"
	"runtime"
	"sync"
)

// Prefs are the transcription settings the one-click pipeline uses.
//
// They live on the server rather than in the page because the pipeline starts
// from the Library, which does not have the Transcribe page's controls loaded.
// Storing them also means the choice survives a restart, so a volunteer sets it
// once rather than every session.
type Prefs struct {
	Models   []string `json:"models"` // model file paths
	VADModel string   `json:"vadModel"`
	UseVAD   bool     `json:"useVad"`
	Language string   `json:"language"`
	Threads  int      `json:"threads"`
	BeamSize int      `json:"beamSize"`
	UseGPU   bool     `json:"useGpu"`
	Backend  string   `json:"backend"`
	Device   int      `json:"device"`
	Loudnorm bool     `json:"loudnorm"`
}

func defaultPrefs() Prefs {
	return Prefs{
		UseVAD:   true,
		Language: "tr",
		Threads:  runtime.NumCPU(),
		BeamSize: 5,
		Loudnorm: true,
	}
}

type prefStore struct {
	path string
	mu   sync.Mutex
	p    Prefs
}

func openPrefs(dir string) *prefStore {
	ps := &prefStore{path: filepath.Join(dir, "prefs.json"), p: defaultPrefs()}
	if raw, err := os.ReadFile(ps.path); err == nil {
		var loaded Prefs
		if json.Unmarshal(raw, &loaded) == nil {
			ps.p = loaded
			if ps.p.Language == "" {
				ps.p.Language = "tr"
			}
			if ps.p.Threads <= 0 {
				ps.p.Threads = runtime.NumCPU()
			}
			if ps.p.BeamSize <= 0 {
				ps.p.BeamSize = 5
			}
		}
	}
	return ps
}

func (ps *prefStore) get() Prefs {
	ps.mu.Lock()
	defer ps.mu.Unlock()
	return ps.p
}

func (ps *prefStore) set(p Prefs) {
	ps.mu.Lock()
	ps.p = p
	raw, err := json.MarshalIndent(p, "", "  ")
	path := ps.path
	ps.mu.Unlock()
	if err != nil {
		return
	}
	_ = os.MkdirAll(filepath.Dir(path), 0o755)
	tmp := path + ".tmp"
	if os.WriteFile(tmp, raw, 0o644) == nil {
		_ = os.Rename(tmp, path)
	}
}

// fillDefaults completes a partial request with the stored preferences, so the
// pipeline can run with nothing specified at all.
func (ps *prefStore) fillDefaults(models []string, vad string) ([]string, string) {
	p := ps.get()
	if len(models) == 0 {
		models = p.Models
	}
	if vad == "" {
		vad = p.VADModel
	}
	return models, vad
}
