package web

import (
	"encoding/json"
	"os"
	"path/filepath"
	"time"
)

// Work that outlives the process.
//
// A transcription runs for tens of minutes. Losing the queue because the window
// was closed is the most likely way somebody loses real work, so the job list is
// written to disk and reloaded at startup: finished jobs come back as history,
// and anything that was still running is queued again. Neither ffmpeg nor
// whisper can resume mid-file, so re-running is the only honest recovery -- but
// re-running beats silently losing the job.

type persistedJob struct {
	ID      string   `json:"id"`
	Type    JobType  `json:"type"`
	Kind    Kind     `json:"kind"`
	Model   string   `json:"model,omitempty"`
	Name    string   `json:"name"`
	Source  string   `json:"source"`
	Output  string   `json:"output"`
	State   JobState `json:"state"`
	Percent float64  `json:"percent"`
	Error   string   `json:"error,omitempty"`
	OutSize int64    `json:"outSize"`
	Elapsed float64  `json:"elapsed"`

	// Enough to start the work again after a restart.
	Opts persistedOpts `json:"opts"`
}

type persistedOpts struct {
	Loudnorm  bool   `json:"loudnorm"`
	ModelPath string `json:"modelPath,omitempty"`
	VADModel  string `json:"vadModel,omitempty"`
	Language  string `json:"language,omitempty"`
	Threads   int    `json:"threads,omitempty"`
	BeamSize  int    `json:"beamSize,omitempty"`
	Backend   string `json:"backend,omitempty"`
	Device    int    `json:"device,omitempty"`
	Prompt    string `json:"prompt,omitempty"`
	OutDir    string `json:"outDir,omitempty"`
	OutBase   string `json:"outBase,omitempty"`
	Chain     bool   `json:"chain,omitempty"` // transcribe once conversion finishes
}

type persistedState struct {
	Jobs []persistedJob `json:"jobs"`
	Seq  int            `json:"seq"`
	// Rate is how many seconds of audio this machine processes per second of
	// wall clock, learned from finished work and used to estimate the wait
	// before any progress has been reported.
	Rate map[string]float64 `json:"rate,omitempty"`
}

func (s *Server) statePath() string {
	return filepath.Join(s.stateDir, "jobs.json")
}

// saveJobs writes the queue. Called on every state change, so it is throttled
// by the caller rather than doing IO on each progress tick.
func (s *Server) saveJobs() {
	s.mu.Lock()
	st := persistedState{Seq: s.seq, Rate: map[string]float64{}}
	for k, v := range s.rate {
		st.Rate[k] = v
	}
	for _, id := range s.order {
		j := s.jobs[id]
		st.Jobs = append(st.Jobs, persistedJob{
			ID: j.ID, Type: j.Type, Kind: j.Kind, Model: j.Model,
			Name: j.Name, Source: j.Source, Output: j.Output,
			State: j.State, Percent: j.Percent, Error: j.Error,
			OutSize: j.OutSize, Elapsed: j.Elapsed, Opts: j.opts,
		})
	}
	path := s.statePath()
	s.mu.Unlock()

	raw, err := json.MarshalIndent(st, "", "  ")
	if err != nil {
		return
	}
	_ = os.MkdirAll(filepath.Dir(path), 0o755)
	tmp := path + ".tmp"
	if os.WriteFile(tmp, raw, 0o644) == nil {
		_ = os.Rename(tmp, path)
	}
}

// loadJobs restores the queue at startup and returns the jobs to restart.
func (s *Server) loadJobs() []*Job {
	raw, err := os.ReadFile(s.statePath())
	if err != nil {
		return nil
	}
	var st persistedState
	if json.Unmarshal(raw, &st) != nil {
		return nil
	}

	var resume []*Job
	s.mu.Lock()
	s.seq = st.Seq
	if st.Rate != nil {
		s.rate = st.Rate
	}
	for _, pj := range st.Jobs {
		j := &Job{
			ID: pj.ID, Type: pj.Type, Kind: pj.Kind, Model: pj.Model,
			Name: pj.Name, Source: pj.Source, Output: pj.Output,
			State: pj.State, Percent: pj.Percent, Error: pj.Error,
			OutSize: pj.OutSize, Elapsed: pj.Elapsed, opts: pj.Opts,
		}
		// Anything interrupted by the shutdown goes back in the queue.
		if j.State == StateRunning || j.State == StateQueued {
			j.State = StateQueued
			j.Percent = 0
			j.Error = ""
			resume = append(resume, j)
		}
		s.jobs[j.ID] = j
		s.order = append(s.order, j.ID)
	}
	s.mu.Unlock()
	return resume
}

// noteRate records how fast this machine got through a job, keyed by the kind
// of work, so the next one can be estimated before it reports any progress.
func (s *Server) noteRate(key string, audioSeconds, wall float64) {
	if audioSeconds <= 0 || wall <= 0 {
		return
	}
	r := audioSeconds / wall
	s.mu.Lock()
	if s.rate == nil {
		s.rate = map[string]float64{}
	}
	if prev, ok := s.rate[key]; ok {
		// A rolling average, so one unusual file does not skew the estimate.
		s.rate[key] = prev*0.7 + r*0.3
	} else {
		s.rate[key] = r
	}
	s.mu.Unlock()
}

// eta estimates the seconds remaining. Callers hold mu.
func (s *Server) etaLocked(j *Job) float64 {
	if j.State != StateRunning || j.started.IsZero() {
		return 0
	}
	elapsed := time.Since(j.started).Seconds()

	// Once there is real progress, measured beats predicted.
	if j.Percent > 2 {
		return elapsed * (100 - j.Percent) / j.Percent
	}
	// Otherwise fall back on what this machine has managed before.
	key := string(j.Type)
	if j.Type == JobTranscribe {
		key += ":" + j.Model
	}
	if r, ok := s.rate[key]; ok && r > 0 && j.audioSeconds > 0 {
		remaining := j.audioSeconds/r - elapsed
		if remaining > 0 {
			return remaining
		}
	}
	return 0
}
