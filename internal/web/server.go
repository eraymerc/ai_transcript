// Package web serves the local UI.
//
// A localhost HTTP server plus the browser is used instead of a GUI toolkit:
// it keeps the program pure Go (CGO_ENABLED=0), so it cross-compiles to Windows
// from anywhere, which Wails, Fyne and every other toolkit would prevent.
//
// Nothing is ever uploaded. The browser cannot see real filesystem paths, so
// files are located server-side instead: the audio and video libraries are
// listed from disk, and anything elsewhere is picked by path through
// /api/browse. Media is read in place, never copied.
package web

import (
	"context"
	"embed"
	"encoding/json"
	"fmt"
	"io"
	"io/fs"
	"log"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"regexp"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	"aitranscript/internal/asr"
	"aitranscript/internal/audio"
	"aitranscript/internal/dataset"
	"aitranscript/internal/review"
)

//go:embed assets
var assets embed.FS

// Kind is which library a file belongs to. It decides which list a file appears
// in and which output subdirectory it converts into, so that an audio and a
// video file sharing a base name cannot overwrite one another's WAV.
type Kind string

const (
	KindAudio Kind = "audio"
	KindVideo Kind = "video"
)

// Extensions we offer to convert. ffmpeg reads far more than this; the lists
// exist to keep unrelated files out of the picker and to classify sources.
var (
	audioExt = map[string]bool{
		// Opus lands in several containers: .opus and .oga are Ogg, .weba is
		// audio-only WebM, which is what yt-dlp and browsers commonly produce.
		".opus": true, ".ogg": true, ".oga": true, ".ogx": true, ".weba": true,
		".caf": true,
		// MPEG layer II and the generic MPEG-audio extensions.
		".mp2": true, ".m2a": true, ".mpa": true, ".mpga": true,
		".m4a": true, ".m4b": true, ".mp3": true, ".aac": true, ".flac": true,
		".wma": true, ".wav": true, ".aiff": true, ".aif": true, ".ape": true,
		".amr": true,
	}
	videoExt = map[string]bool{
		".mp4": true, ".mkv": true, ".webm": true, ".mov": true, ".avi": true,
		".ts": true, ".m4v": true, ".flv": true, ".3gp": true, ".wmv": true,
		".mpg": true, ".mpeg": true, ".ogv": true, ".mts": true, ".m2ts": true,
	}
)

// kindFor classifies a file by extension. .webm, .ogg and .ogv can hold either
// audio or video; the guess only decides which output subdirectory is used,
// since conversion drops any video stream regardless.
func kindFor(name string) (Kind, bool) {
	switch ext := strings.ToLower(filepath.Ext(name)); {
	case audioExt[ext]:
		return KindAudio, true
	case videoExt[ext]:
		return KindVideo, true
	default:
		return "", false
	}
}

// JobState is where a conversion has got to.
type JobState string

const (
	StateQueued   JobState = "queued"
	StateRunning  JobState = "running"
	StateDone     JobState = "done"
	StateFailed   JobState = "failed"
	StateCanceled JobState = "canceled"
)

// JobType distinguishes the two kinds of work that share the job list, the
// progress stream and the cancel button.
type JobType string

const (
	JobConvert    JobType = "convert"
	JobTranscribe JobType = "transcribe"
)

// Job is one unit of work: a conversion or a transcription.
type Job struct {
	ID      string   `json:"id"`
	Type    JobType  `json:"type"`
	Model   string   `json:"model,omitempty"`
	Kind    Kind     `json:"kind"`
	Name    string   `json:"name"`
	Source  string   `json:"source"`
	Output  string   `json:"output"`
	State   JobState `json:"state"`
	Percent float64  `json:"percent"` // -1 when the duration could not be probed
	Error   string   `json:"error,omitempty"`
	OutSize int64    `json:"outSize"`
	Elapsed float64  `json:"elapsed"`
	ETA     float64  `json:"eta"` // seconds remaining, 0 when unknown

	cancel       context.CancelFunc
	started      time.Time
	total        time.Duration
	audioSeconds float64       // length of the input, for estimating the wait
	opts         persistedOpts // enough to run this job again after a restart
}

// Server holds the UI state. Everything under mu is touched by both HTTP
// handlers and job goroutines.
type Server struct {
	tools    audio.Tools
	asrTools asr.Tools
	asrErr   string

	sem chan struct{}
	// Transcription saturates every core, so two at once is slower than two in
	// sequence. It gets its own slot of one rather than sharing the ffmpeg pool.
	asrSem chan struct{}

	mu            sync.Mutex
	audioDir      string
	videoDir      string
	outputDir     string
	modelDir      string
	transcriptDir string
	datasetDir    string
	// engines is nil until probed; probing costs a model load, so it is cached.
	engines  []asr.Engine
	rate     map[string]float64 // learned speed per job kind, for the ETA
	prefs    *prefStore
	stateDir string
	added    []string // absolute paths picked from elsewhere, in pick order
	jobs     map[string]*Job
	order    []string
	subs     map[chan struct{}]struct{}
	seq      int
	lastPush time.Time
}

// Config is where the server reads from and writes to.
type Config struct {
	AudioDir      string
	VideoDir      string
	OutputDir     string
	ModelDir      string
	TranscriptDir string
	DatasetDir    string
	StateDir      string
	Concurrency   int
}

// NewServer returns a server. Concurrency caps simultaneous ffmpeg processes;
// transcription is always serialised regardless.
func NewServer(tools audio.Tools, cfg Config) *Server {
	if cfg.Concurrency < 1 {
		cfg.Concurrency = 1
	}
	s := &Server{
		tools:         tools,
		sem:           make(chan struct{}, cfg.Concurrency),
		asrSem:        make(chan struct{}, 1),
		audioDir:      cfg.AudioDir,
		videoDir:      cfg.VideoDir,
		outputDir:     cfg.OutputDir,
		modelDir:      cfg.ModelDir,
		transcriptDir: cfg.TranscriptDir,
		datasetDir:    cfg.DatasetDir,
		stateDir:      cfg.StateDir,
		rate:          map[string]float64{},
		prefs:         openPrefs(cfg.StateDir),
		jobs:          map[string]*Job{},
		subs:          map[chan struct{}]struct{}{},
	}
	// A missing engine disables transcription but must not stop the program:
	// conversion is still useful on its own.
	if t, err := asr.Locate(); err != nil {
		s.asrErr = err.Error()
	} else {
		s.asrTools = t
	}

	// Bring back the queue from the last run and restart whatever the shutdown
	// interrupted. Neither engine can resume mid-file, so this re-runs them.
	for _, j := range s.loadJobs() {
		s.restart(j)
	}
	return s
}

// restart puts a job recovered from disk back to work.
func (s *Server) restart(j *Job) {
	ctx, cancel := context.WithCancel(context.Background())
	j.cancel = cancel

	switch j.Type {
	case JobConvert:
		opts := audio.DefaultWAVOptions()
		opts.Loudnorm = j.opts.Loudnorm
		go s.run(ctx, j, j.Source, j.Output, opts)
	case JobTranscribe:
		go s.runTranscribe(ctx, j, j.Source, asr.Options{
			ModelPath: j.opts.ModelPath, VADModel: j.opts.VADModel,
			Language: j.opts.Language, Threads: j.opts.Threads,
			BeamSize: j.opts.BeamSize, OutDir: j.opts.OutDir,
			OutBase: j.opts.OutBase, Backend: asr.Backend(j.opts.Backend),
			Device: j.opts.Device, Prompt: j.opts.Prompt,
		})
	default:
		cancel()
	}
}

// Handler builds the routes.
func (s *Server) Handler() http.Handler {
	mux := http.NewServeMux()

	sub, err := fs.Sub(assets, "assets")
	if err != nil {
		panic(err) // the embedded tree is fixed at build time
	}
	mux.Handle("/", http.FileServer(http.FS(sub)))

	mux.HandleFunc("/api/status", s.handleStatus)
	mux.HandleFunc("/api/config", s.handleConfig)
	mux.HandleFunc("/api/files", s.handleFiles)
	mux.HandleFunc("/api/browse", s.handleBrowse)
	mux.HandleFunc("/api/sources/add", s.handleSourcesAdd)
	mux.HandleFunc("/api/sources/remove", s.handleSourcesRemove)
	mux.HandleFunc("/api/wavs", s.handleWavs)
	mux.HandleFunc("/api/models", s.handleModels)
	mux.HandleFunc("/api/gpu", s.handleGPU)
	mux.HandleFunc("/api/review/list", s.handleReviewList)
	mux.HandleFunc("/api/review/spans", s.handleReviewSpans)
	mux.HandleFunc("/api/review/text", s.handleReviewText)
	mux.HandleFunc("/api/review/clip", s.handleReviewClip)
	mux.HandleFunc("/api/review/decide", s.handleReviewDecide)
	mux.HandleFunc("/api/review/segment", s.handleReviewSegment)
	mux.HandleFunc("/api/review/download", s.handleReviewDownload)
	mux.HandleFunc("/api/review/delete", s.handleReviewDelete)
	mux.HandleFunc("/api/dataset", s.handleDataset)
	mux.HandleFunc("/api/dataset/export", s.handleDatasetExport)
	mux.HandleFunc("/api/transcribe", s.handleTranscribe)
	mux.HandleFunc("/api/convert", s.handleConvert)
	mux.HandleFunc("/api/pipeline", s.handlePipeline)
	mux.HandleFunc("/api/glossary", s.handleGlossary)
	mux.HandleFunc("/api/prefs", s.handlePrefs)
	mux.HandleFunc("/api/cancel", s.handleCancel)
	mux.HandleFunc("/api/clear", s.handleClear)
	mux.HandleFunc("/api/events", s.handleEvents)
	return mux
}

// --- paths -----------------------------------------------------------------

// outputPath keeps each kind's WAVs in their own subdirectory.
func (s *Server) outputPath(kind Kind, name string) string {
	s.mu.Lock()
	out := s.outputDir
	s.mu.Unlock()
	return filepath.Join(out, string(kind), wavName(name))
}

// wavName keeps the source extension in the output name, so lecture.mp3 and
// lecture.m4a become lecture.mp3.wav and lecture.m4a.wav rather than colliding
// on a single lecture.wav. The name is what transcripts and review state are
// keyed on, so a collision would attach one recording's corrections to another.
func wavName(name string) string {
	return name + ".wav"
}

// resolveSource turns a user-supplied path into a checked absolute path.
//
// Arbitrary absolute paths are accepted on purpose: this is a single-user tool
// on the user's own machine, and reading media from wherever it already lives
// is the entire point. The checks are for clear errors, not for containment.
func resolveSource(path string) (string, Kind, error) {
	if strings.TrimSpace(path) == "" {
		return "", "", fmt.Errorf("empty path")
	}
	abs, err := filepath.Abs(expandHome(strings.TrimSpace(path)))
	if err != nil {
		return "", "", err
	}
	fi, err := os.Stat(abs)
	if err != nil {
		return "", "", fmt.Errorf("%s: %w", abs, err)
	}
	if fi.IsDir() {
		return "", "", fmt.Errorf("%s is a folder, not a file", abs)
	}
	if !fi.Mode().IsRegular() {
		return "", "", fmt.Errorf("%s is not a regular file", abs)
	}
	kind, ok := kindFor(abs)
	if !ok {
		return "", "", fmt.Errorf("%s: unsupported file type", filepath.Base(abs))
	}
	return abs, kind, nil
}

func expandHome(p string) string {
	if p == "~" || strings.HasPrefix(p, "~/") || strings.HasPrefix(p, `~\`) {
		if home, err := os.UserHomeDir(); err == nil {
			return filepath.Join(home, strings.TrimPrefix(p[1:], string(filepath.Separator)))
		}
	}
	return p
}

// roots lists the top of the filesystem for the picker: drive letters on
// Windows, "/" elsewhere.
func roots() []string {
	if runtime.GOOS != "windows" {
		return []string{"/"}
	}
	var out []string
	for c := 'A'; c <= 'Z'; c++ {
		p := string(c) + `:\`
		if _, err := os.Stat(p); err == nil {
			out = append(out, p)
		}
	}
	if len(out) == 0 {
		out = []string{`C:\`}
	}
	return out
}

// --- notifications ---------------------------------------------------------

func (s *Server) subscribe() chan struct{} {
	ch := make(chan struct{}, 1)
	s.mu.Lock()
	s.subs[ch] = struct{}{}
	s.mu.Unlock()
	return ch
}

func (s *Server) unsubscribe(ch chan struct{}) {
	s.mu.Lock()
	delete(s.subs, ch)
	s.mu.Unlock()
}

// notify wakes every SSE client. Callers must not hold mu.
func (s *Server) notify() {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.notifyLocked()
}

func (s *Server) notifyLocked() {
	s.lastPush = time.Now()
	for ch := range s.subs {
		select {
		case ch <- struct{}{}:
		default: // a wake-up is already pending; state is read fresh anyway
		}
	}
}

// --- helpers ---------------------------------------------------------------

func writeJSON(w http.ResponseWriter, code int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)
	_ = json.NewEncoder(w).Encode(v)
}

func writeErr(w http.ResponseWriter, code int, format string, args ...any) {
	writeJSON(w, code, map[string]string{"error": fmt.Sprintf(format, args...)})
}

func decodeBody(w http.ResponseWriter, r *http.Request, v any) bool {
	if r.Method != http.MethodPost {
		writeErr(w, http.StatusMethodNotAllowed, "POST only")
		return false
	}
	if err := json.NewDecoder(r.Body).Decode(v); err != nil {
		writeErr(w, http.StatusBadRequest, "bad request body: %v", err)
		return false
	}
	return true
}

// --- status and config -----------------------------------------------------

func (s *Server) handleStatus(w http.ResponseWriter, r *http.Request) {
	s.mu.Lock()
	a, v, o := s.audioDir, s.videoDir, s.outputDir
	s.mu.Unlock()

	home, _ := os.UserHomeDir()
	cwd, _ := os.Getwd()

	writeJSON(w, http.StatusOK, map[string]any{
		"ffmpeg":        s.tools.FFmpeg,
		"ffprobe":       s.tools.FFprobe,
		"audioDir":      a,
		"videoDir":      v,
		"outputDir":     o,
		"modelDir":      s.modelDir,
		"transcriptDir": s.transcriptDir,
		"whisper":       s.asrTools.WhisperCLI,
		"whisperError":  s.asrErr,
		"cores":         runtime.NumCPU(),
		"home":          home,
		"cwd":           cwd,
		"roots":         roots(),
		"sep":           string(filepath.Separator),
		"sampleRate":    16000,
		"channels":      1,
		"codec":         "pcm_s16le",
	})
}

func (s *Server) handleConfig(w http.ResponseWriter, r *http.Request) {
	var req struct {
		AudioDir  string `json:"audioDir"`
		VideoDir  string `json:"videoDir"`
		OutputDir string `json:"outputDir"`
	}
	if !decodeBody(w, r, &req) {
		return
	}

	// Resolve and validate everything before storing any of it, so a bad value
	// cannot leave the configuration half-applied.
	type update struct {
		dst  *string
		path string
	}
	var updates []update

	for _, in := range []struct {
		raw   string
		field *string
		label string
	}{
		{req.AudioDir, &s.audioDir, "audio folder"},
		{req.VideoDir, &s.videoDir, "video folder"},
	} {
		if in.raw == "" {
			continue
		}
		abs, err := filepath.Abs(expandHome(in.raw))
		if err != nil {
			writeErr(w, http.StatusBadRequest, "%s: %v", in.label, err)
			return
		}
		if fi, err := os.Stat(abs); err != nil || !fi.IsDir() {
			writeErr(w, http.StatusBadRequest, "%s is not a directory: %s", in.label, abs)
			return
		}
		updates = append(updates, update{in.field, abs})
	}

	if req.OutputDir != "" {
		abs, err := filepath.Abs(expandHome(req.OutputDir))
		if err != nil {
			writeErr(w, http.StatusBadRequest, "output folder: %v", err)
			return
		}
		if err := os.MkdirAll(abs, 0o755); err != nil {
			writeErr(w, http.StatusBadRequest, "output folder: %v", err)
			return
		}
		updates = append(updates, update{&s.outputDir, abs})
	}

	s.mu.Lock()
	for _, u := range updates {
		*u.dst = u.path
	}
	s.notifyLocked()
	s.mu.Unlock()

	s.handleStatus(w, r)
}

// --- file listing ----------------------------------------------------------

type fileInfo struct {
	Kind      Kind   `json:"kind"`
	Name      string `json:"name"`
	Source    string `json:"source"`
	Size      int64  `json:"size"`
	Output    string `json:"output"`
	Converted bool   `json:"converted"`
	OutSize   int64  `json:"outSize"`
	Missing   bool   `json:"missing,omitempty"`
}

func (s *Server) handleFiles(w http.ResponseWriter, r *http.Request) {
	s.mu.Lock()
	audioDir, videoDir, out := s.audioDir, s.videoDir, s.outputDir
	added := append([]string(nil), s.added...)
	s.mu.Unlock()

	resp := map[string]any{}
	for kind, dir := range map[Kind]string{KindAudio: audioDir, KindVideo: videoDir} {
		files, err := listDir(dir, kind, out)
		if err != nil {
			// A missing folder is reported per-list rather than failing the
			// whole request, so one bad path does not hide the other library.
			resp[string(kind)] = []fileInfo{}
			resp[string(kind)+"Error"] = err.Error()
			continue
		}
		resp[string(kind)] = files
	}

	// Files picked from elsewhere. Anything that actually lives in one of the
	// libraries is skipped here so it is not listed twice.
	extra := []fileInfo{}
	for _, p := range added {
		parent := filepath.Dir(p)
		if parent == audioDir || parent == videoDir {
			continue
		}
		kind, ok := kindFor(p)
		if !ok {
			continue
		}
		f := fileInfo{Kind: kind, Name: filepath.Base(p), Source: p, Output: wavName(filepath.Base(p))}
		if fi, err := os.Stat(p); err == nil {
			f.Size = fi.Size()
		} else {
			f.Missing = true
		}
		if fi, err := os.Stat(filepath.Join(out, string(kind), f.Output)); err == nil {
			f.Converted = true
			f.OutSize = fi.Size()
		}
		extra = append(extra, f)
	}
	resp["added"] = extra

	writeJSON(w, http.StatusOK, resp)
}

func listDir(dir string, kind Kind, outputDir string) ([]fileInfo, error) {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return nil, err
	}
	files := []fileInfo{}
	for _, e := range entries {
		if e.IsDir() || strings.HasPrefix(e.Name(), ".") {
			continue
		}
		// Only list files belonging to this library, so a stray .mp4 in the
		// audio folder does not appear as convertible audio.
		if k, ok := kindFor(e.Name()); !ok || k != kind {
			continue
		}
		f := fileInfo{
			Kind:   kind,
			Name:   e.Name(),
			Source: filepath.Join(dir, e.Name()),
			Output: wavName(e.Name()),
		}
		if fi, err := e.Info(); err == nil {
			f.Size = fi.Size()
		}
		if fi, err := os.Stat(filepath.Join(outputDir, string(kind), f.Output)); err == nil {
			f.Converted = true
			f.OutSize = fi.Size()
		}
		files = append(files, f)
	}
	sort.Slice(files, func(i, j int) bool { return files[i].Name < files[j].Name })
	return files, nil
}

// --- converted WAVs --------------------------------------------------------

type wavInfo struct {
	Kind       Kind    `json:"kind"`
	Name       string  `json:"name"`
	Path       string  `json:"path"`
	Size       int64   `json:"size"`
	SampleRate int     `json:"sampleRate"`
	Channels   int     `json:"channels"`
	Bits       int     `json:"bits"`
	Seconds    float64 `json:"seconds"`
	Ready      bool    `json:"ready"`
	Error      string  `json:"error,omitempty"`
}

// handleWavs lists everything already converted, which is what the transcribe
// stage will consume.
func (s *Server) handleWavs(w http.ResponseWriter, r *http.Request) {
	s.mu.Lock()
	out := s.outputDir
	s.mu.Unlock()

	files := []wavInfo{}
	for _, kind := range []Kind{KindAudio, KindVideo} {
		dir := filepath.Join(out, string(kind))
		entries, err := os.ReadDir(dir)
		if err != nil {
			continue // the subdirectory only exists once something is converted
		}
		for _, e := range entries {
			if e.IsDir() || !strings.EqualFold(filepath.Ext(e.Name()), ".wav") {
				continue
			}
			full := filepath.Join(dir, e.Name())
			f := wavInfo{Kind: kind, Name: e.Name(), Path: full}
			if fi, err := e.Info(); err == nil {
				f.Size = fi.Size()
			}
			if info, err := audio.InspectWAV(full); err != nil {
				f.Error = err.Error()
			} else {
				f.SampleRate = info.SampleRate
				f.Channels = info.Channels
				f.Bits = info.Bits
				f.Seconds = info.Duration.Seconds()
				f.Ready = info.Ready()
			}
			files = append(files, f)
		}
	}
	sort.Slice(files, func(i, j int) bool {
		if files[i].Kind != files[j].Kind {
			return files[i].Kind < files[j].Kind
		}
		return files[i].Name < files[j].Name
	})
	writeJSON(w, http.StatusOK, files)
}

// glossaryPath is where the vocabulary list lives.
func (s *Server) glossaryPath() string {
	s.mu.Lock()
	defer s.mu.Unlock()
	return filepath.Join(s.stateDir, "glossary.txt")
}

// transcribeOptsFor builds a transcription request from the stored preferences,
// which is what lets the pipeline run without the Transcribe page's controls.
func (s *Server) transcribeOptsFor(wav string, model asr.Model) asr.Options {
	p := s.prefs.get()

	s.mu.Lock()
	transcripts := s.transcriptDir
	s.mu.Unlock()

	backend := asr.BackendCPU
	device := 0
	if p.UseGPU && p.Backend != "" {
		backend = asr.Backend(p.Backend)
		device = p.Device
	}
	vad := ""
	if p.UseVAD {
		vad = p.VADModel
	}
	text, _ := asr.LoadGlossary(s.glossaryPath())

	base := strings.TrimSuffix(filepath.Base(wav), filepath.Ext(wav))
	outDir := filepath.Join(transcripts, model.Name)
	return asr.Options{
		ModelPath: model.Path,
		VADModel:  vad,
		Language:  p.Language,
		Threads:   p.Threads,
		BeamSize:  p.BeamSize,
		OutDir:    outDir,
		OutBase:   nextRunBase(outDir, base),
		Backend:   backend,
		Device:    device,
		Prompt:    asr.ParseGlossary(text).Prompt(),
	}
}

// chainTranscribe starts transcription for a WAV the pipeline just produced.
func (s *Server) chainTranscribe(wav string, kind Kind) error {
	if s.asrTools.WhisperCLI() == "" {
		return fmt.Errorf("no transcription engine installed")
	}
	s.mu.Lock()
	modelDir := s.modelDir
	s.mu.Unlock()

	known, err := asr.ListModels(modelDir)
	if err != nil {
		return err
	}
	wanted, _ := s.prefs.fillDefaults(nil, "")

	// With no model chosen yet, take the smallest that is installed: a first
	// run should produce something rather than stopping to ask.
	var models []asr.Model
	for _, m := range known {
		if m.VAD {
			continue
		}
		for _, w := range wanted {
			if w == m.Path {
				models = append(models, m)
			}
		}
	}
	if len(models) == 0 {
		// Nothing chosen yet: take the recommended model, falling back to the
		// smallest so a first run still produces something.
		var best *asr.Model
		for i, m := range known {
			if m.VAD {
				continue
			}
			if m.Recommended {
				best = &known[i]
				break
			}
			if best == nil || m.Size < best.Size {
				best = &known[i]
			}
		}
		if best == nil {
			return fmt.Errorf("no transcription model in %s", modelDir)
		}
		models = []asr.Model{*best}
	}

	for _, m := range models {
		s.startTranscribe(wav, m.Name, s.transcribeOptsFor(wav, m))
	}
	s.notify()
	return nil
}

// handleModels lists the ggml models on disk, split into transcription models
// and voice-activity models.
func (s *Server) handleModels(w http.ResponseWriter, r *http.Request) {
	s.mu.Lock()
	dir := s.modelDir
	s.mu.Unlock()

	models, err := asr.ListModels(dir)
	if err != nil {
		writeJSON(w, http.StatusOK, map[string]any{
			"asr": []asr.Model{}, "vad": []asr.Model{},
			"dir": dir, "error": err.Error(),
		})
		return
	}
	asrModels, vadModels := []asr.Model{}, []asr.Model{}
	for _, m := range models {
		if m.VAD {
			vadModels = append(vadModels, m)
		} else {
			asrModels = append(asrModels, m)
		}
	}
	// Anything in the catalog that is not on disk is offered rather than left
	// to be discovered.
	have := map[string]bool{}
	for _, m := range models {
		have[strings.ToLower(m.File)] = true
	}
	available := []asr.CatalogEntry{}
	for _, c := range asr.Catalog {
		if !have[strings.ToLower(c.File)] {
			available = append(available, c)
		}
	}

	writeJSON(w, http.StatusOK, map[string]any{
		"asr": asrModels, "vad": vadModels, "available": available, "dir": dir,
	})
}

// smallestModel picks the cheapest model to load for a probe.
func (s *Server) smallestModel() string {
	s.mu.Lock()
	dir := s.modelDir
	s.mu.Unlock()

	models, err := asr.ListModels(dir)
	if err != nil {
		return ""
	}
	best, bestSize := "", int64(0)
	for _, m := range models {
		if m.VAD {
			continue
		}
		if best == "" || m.Size < bestSize {
			best, bestSize = m.Path, m.Size
		}
	}
	return best
}

// probeEngines reports every backend, caching the result. Probing costs one
// model load per backend, so it happens once unless the user re-checks.
func (s *Server) probeEngines(ctx context.Context, force bool) []asr.Engine {
	s.mu.Lock()
	if s.engines != nil && !force {
		out := append([]asr.Engine(nil), s.engines...)
		s.mu.Unlock()
		return out
	}
	s.mu.Unlock()

	engines := s.asrTools.ProbeAll(ctx, s.smallestModel())

	s.mu.Lock()
	s.engines = engines
	s.notifyLocked()
	s.mu.Unlock()
	return engines
}

func (s *Server) handleGPU(w http.ResponseWriter, r *http.Request) {
	ctx, cancel := context.WithTimeout(r.Context(), 3*time.Minute)
	defer cancel()

	engines := s.probeEngines(ctx, r.URL.Query().Get("refresh") == "1")
	accel := []asr.Engine{}
	for _, e := range engines {
		if e.Backend != asr.BackendCPU {
			accel = append(accel, e)
		}
	}
	writeJSON(w, http.StatusOK, map[string]any{
		"engines":  accel,
		"anyReady": anyAvailable(accel),
	})
}

func anyAvailable(engines []asr.Engine) bool {
	for _, e := range engines {
		if e.Available {
			return true
		}
	}
	return false
}

// handleTranscribe queues one job per (WAV x model) pair, which is what makes
// running several models over the same audio — to compare them, or to get the
// second opinion the correction stage wants — a single button press.
func (s *Server) handleTranscribe(w http.ResponseWriter, r *http.Request) {
	var req struct {
		Paths    []string `json:"paths"`
		Models   []string `json:"models"` // model file paths
		VADModel string   `json:"vadModel"`
		UseVAD   bool     `json:"useVad"`
		Language string   `json:"language"`
		Threads  int      `json:"threads"`
		BeamSize int      `json:"beamSize"`
		UseGPU   bool     `json:"useGpu"`
		Backend  string   `json:"backend"`
		Device   int      `json:"device"`
	}
	if !decodeBody(w, r, &req) {
		return
	}
	if s.asrTools.WhisperCLI() == "" {
		writeErr(w, http.StatusServiceUnavailable,
			"whisper-cli was not found, so transcription is unavailable. %s", s.asrErr)
		return
	}
	if len(req.Paths) == 0 {
		writeErr(w, http.StatusBadRequest, "no audio selected")
		return
	}
	if len(req.Models) == 0 {
		writeErr(w, http.StatusBadRequest, "no model selected")
		return
	}

	backend := asr.BackendCPU
	if req.UseGPU {
		backend = asr.Backend(req.Backend)
		if backend == "" || backend == asr.BackendCPU {
			writeErr(w, http.StatusBadRequest, "GPU was requested but no driver was chosen")
			return
		}
		var chosen *asr.Engine
		for _, e := range s.probeEngines(r.Context(), false) {
			if e.Backend == backend {
				chosen = &e
				break
			}
		}
		if chosen == nil {
			writeErr(w, http.StatusBadRequest, "unknown driver: %s", req.Backend)
			return
		}
		if !chosen.Available {
			writeErr(w, http.StatusBadRequest,
				"the %s driver is not usable: %s", backend, chosen.Reason)
			return
		}
		known := false
		for _, d := range chosen.Devices {
			if d.Index == req.Device {
				known = true
				break
			}
		}
		if !known {
			writeErr(w, http.StatusBadRequest,
				"device %d is not one the %s driver reported", req.Device, backend)
			return
		}
	}

	s.mu.Lock()
	modelDir, transcriptDir := s.modelDir, s.transcriptDir
	s.mu.Unlock()

	known, err := asr.ListModels(modelDir)
	if err != nil {
		writeErr(w, http.StatusBadRequest, "reading models from %s: %v", modelDir, err)
		return
	}
	byPath := map[string]asr.Model{}
	for _, m := range known {
		byPath[m.Path] = m
	}

	// Validate the whole batch before starting any of it.
	type target struct {
		wav   string
		model asr.Model
	}
	var targets []target
	for _, p := range req.Paths {
		abs, err := filepath.Abs(p)
		if err != nil {
			writeErr(w, http.StatusBadRequest, "%v", err)
			return
		}
		if fi, err := os.Stat(abs); err != nil || !fi.Mode().IsRegular() {
			writeErr(w, http.StatusBadRequest, "not a readable file: %s", abs)
			return
		}
		for _, mp := range req.Models {
			m, ok := byPath[mp]
			if !ok {
				writeErr(w, http.StatusBadRequest, "unknown model: %s", mp)
				return
			}
			if m.VAD {
				writeErr(w, http.StatusBadRequest, "%s is a VAD model, not a transcription model", m.Name)
				return
			}
			targets = append(targets, target{abs, m})
		}
	}

	vadPath := ""
	if req.UseVAD {
		if req.VADModel == "" {
			writeErr(w, http.StatusBadRequest, "VAD is enabled but no VAD model was chosen")
			return
		}
		m, ok := byPath[req.VADModel]
		if !ok || !m.VAD {
			writeErr(w, http.StatusBadRequest, "unknown VAD model: %s", req.VADModel)
			return
		}
		vadPath = m.Path
	}

	ids := make([]string, 0, len(targets))
	for _, t := range targets {
		base := strings.TrimSuffix(filepath.Base(t.wav), filepath.Ext(t.wav))
		outDir := filepath.Join(transcriptDir, t.model.Name)
		opts := asr.Options{
			ModelPath: t.model.Path,
			VADModel:  vadPath,
			Language:  req.Language,
			Threads:   req.Threads,
			BeamSize:  req.BeamSize,
			// Each model writes into its own subdirectory so that running
			// several over the same audio does not overwrite one another, and
			// repeat runs of the same model are numbered rather than replaced.
			OutDir:  outDir,
			OutBase: nextRunBase(outDir, base),
			Backend: backend,
			Device:  req.Device,
		}
		ids = append(ids, s.startTranscribe(t.wav, t.model.Name, opts))
	}
	s.notify()
	writeJSON(w, http.StatusOK, map[string]any{"ids": ids})
}

func (s *Server) startTranscribe(wav, modelName string, opts asr.Options) string {
	ctx, cancel := context.WithCancel(context.Background())

	s.mu.Lock()
	s.seq++
	id := fmt.Sprintf("j%d", s.seq)
	job := &Job{
		ID:    id,
		Type:  JobTranscribe,
		Model: modelName,
		opts: persistedOpts{
			ModelPath: opts.ModelPath, VADModel: opts.VADModel,
			Language: opts.Language, Threads: opts.Threads, BeamSize: opts.BeamSize,
			Backend: string(opts.Backend), Device: opts.Device,
			Prompt: opts.Prompt, OutDir: opts.OutDir, OutBase: opts.OutBase,
		},
		Name:    filepath.Base(wav),
		Source:  wav,
		Output:  filepath.Join(opts.OutDir, opts.OutBase+".json"),
		State:   StateQueued,
		Percent: 0,
		cancel:  cancel,
	}
	s.jobs[id] = job
	s.order = append(s.order, id)
	s.mu.Unlock()

	go s.runTranscribe(ctx, job, wav, opts)
	go s.saveJobs()
	return id
}

func (s *Server) runTranscribe(ctx context.Context, job *Job, wav string, opts asr.Options) {
	defer job.cancel()

	select {
	case s.asrSem <- struct{}{}:
		defer func() { <-s.asrSem }()
	case <-ctx.Done():
		s.finish(job, StateCanceled, "canceled before it started")
		return
	}

	if info, err := audio.InspectWAV(wav); err == nil && info.Duration > 0 {
		s.mu.Lock()
		job.audioSeconds = info.Duration.Seconds()
		s.mu.Unlock()
	}

	s.mu.Lock()
	job.State = StateRunning
	job.started = time.Now()
	s.notifyLocked()
	s.mu.Unlock()
	s.saveJobs()

	err := s.asrTools.Transcribe(ctx, wav, opts, func(pct int) {
		s.mu.Lock()
		job.Percent = float64(pct)
		if time.Since(s.lastPush) > 200*time.Millisecond {
			s.notifyLocked()
		}
		s.mu.Unlock()
	})

	switch {
	case err == nil:
		var size int64
		if fi, statErr := os.Stat(job.Output); statErr == nil {
			size = fi.Size()
		}
		s.mu.Lock()
		job.OutSize = size
		job.Percent = 100
		s.mu.Unlock()
		s.finish(job, StateDone, "")
	case ctx.Err() != nil:
		s.finish(job, StateCanceled, "canceled")
	default:
		msg := err.Error()
		if opts.Backend != asr.BackendCPU {
			// A driver fault kills the engine, not this program -- which is why
			// it runs as a subprocess. Point at the switch that avoids it.
			msg += fmt.Sprintf("  (this run used the %s driver; switch GPU off on the "+
				"Transcribe page to retry on CPU, or pick another device)", opts.Backend)
		}
		s.finish(job, StateFailed, msg)
	}
}

// --- transcript runs -------------------------------------------------------

// runSuffix matches the " (2)" a repeat run adds to a transcript's name.
var runSuffix = regexp.MustCompile(`^(.*) \((\d+)\)$`)

// splitRun separates a transcript stem into its source name and run number.
func splitRun(stem string) (string, int) {
	if m := runSuffix.FindStringSubmatch(stem); m != nil {
		if n, err := strconv.Atoi(m[2]); err == nil && n > 1 {
			return m[1], n
		}
	}
	return stem, 1
}

// nextRunBase picks the output stem for a new run.
//
// Transcribing the same audio again is a normal thing to do -- read the first
// result, collect the terms it got wrong into the glossary, run it again -- so a
// second run must not overwrite the first. Runs are numbered instead, and both
// stay available to compare and to review.
func nextRunBase(dir, base string) string {
	if _, err := os.Stat(filepath.Join(dir, base+".json")); os.IsNotExist(err) {
		return base
	}
	for n := 2; n < 1000; n++ {
		candidate := fmt.Sprintf("%s (%d)", base, n)
		if _, err := os.Stat(filepath.Join(dir, candidate+".json")); os.IsNotExist(err) {
			return candidate
		}
	}
	return base
}

// --- review ----------------------------------------------------------------

// reviewThreshold reads the confidence cutoff from the query. It is the
// reviewer's call, not ours: a careful pass over a poor recording wants a high
// cutoff, a quick sanity check on a clean one wants a low one.
func reviewThreshold(r *http.Request) review.Options {
	opts := review.DefaultOptions()
	if t := r.URL.Query().Get("threshold"); t != "" {
		if v, err := strconv.ParseFloat(t, 64); err == nil && v > 0 && v <= 1 {
			opts.Threshold = v
		}
	}
	return opts
}

type reviewItem struct {
	Name       string `json:"name"`
	Source     string `json:"source"` // the recording, without the run suffix
	Run        int    `json:"run"`
	Model      string `json:"model"`
	Path       string `json:"path"`
	Audio      string `json:"audio"`
	AudioFound bool   `json:"audioFound"`
	Spans      int    `json:"spans"`
	Reviewed   int    `json:"reviewed"`
	Aligned    bool   `json:"aligned"`
}

// findAudio locates the WAV a transcript came from. Transcripts are named after
// their source, so this is a lookup by stem in the two output subdirectories.
func (s *Server) findAudio(stem string) string {
	s.mu.Lock()
	out := s.outputDir
	s.mu.Unlock()
	for _, kind := range []Kind{KindAudio, KindVideo} {
		p := filepath.Join(out, string(kind), stem+".wav")
		if fi, err := os.Stat(p); err == nil && fi.Mode().IsRegular() {
			return p
		}
	}
	return ""
}

func (s *Server) handleReviewList(w http.ResponseWriter, r *http.Request) {
	s.mu.Lock()
	dir := s.transcriptDir
	s.mu.Unlock()

	opts := reviewThreshold(r)

	items := []reviewItem{}
	models, err := os.ReadDir(dir)
	if err != nil {
		writeJSON(w, http.StatusOK, items)
		return
	}
	for _, m := range models {
		if !m.IsDir() {
			continue
		}
		files, err := os.ReadDir(filepath.Join(dir, m.Name()))
		if err != nil {
			continue
		}
		for _, f := range files {
			name := f.Name()
			if f.IsDir() || !strings.EqualFold(filepath.Ext(name), ".json") ||
				strings.HasSuffix(name, ".review.json") {
				continue
			}
			full := filepath.Join(dir, m.Name(), name)
			stem := strings.TrimSuffix(name, filepath.Ext(name))

			spans, err := review.Detect(full, opts)
			if err != nil {
				continue
			}
			st, _ := review.LoadState(full)
			done := 0
			for _, sp := range spans {
				if _, ok := st.Decisions[sp.ID]; ok {
					done++
				}
			}
			source, run := splitRun(stem)
			audio := s.findAudio(source)
			items = append(items, reviewItem{
				Name: stem, Source: source, Run: run, Model: m.Name(), Path: full,
				Audio: audio, AudioFound: audio != "",
				Spans: len(spans), Reviewed: done,
				Aligned: review.Aligned(full),
			})
		}
	}
	// Newest run of each recording first, so the latest attempt is at hand.
	sort.Slice(items, func(i, j int) bool {
		if items[i].Source != items[j].Source {
			return items[i].Source < items[j].Source
		}
		if items[i].Model != items[j].Model {
			return items[i].Model < items[j].Model
		}
		return items[i].Run > items[j].Run
	})
	writeJSON(w, http.StatusOK, items)
}

func (s *Server) handleReviewSpans(w http.ResponseWriter, r *http.Request) {
	path := r.URL.Query().Get("path")
	if path == "" {
		writeErr(w, http.StatusBadRequest, "no transcript given")
		return
	}
	opts := reviewThreshold(r)
	if o := r.URL.Query().Get("order"); o == string(review.ByRank) {
		opts.Order = review.ByRank
	}
	spans, err := review.Detect(path, opts)
	if err != nil {
		writeErr(w, http.StatusBadRequest, "%v", err)
		return
	}
	st, _ := review.LoadState(path)
	writeJSON(w, http.StatusOK, map[string]any{
		"spans": review.Apply(spans, st),
		"total": len(spans),
	})
}

// handleReviewText returns the whole transcript for reading, with flagged words
// marked and accepted corrections already applied.
func (s *Server) handleReviewText(w http.ResponseWriter, r *http.Request) {
	path := r.URL.Query().Get("path")
	if path == "" {
		writeErr(w, http.StatusBadRequest, "no transcript given")
		return
	}
	st, _ := review.LoadState(path)
	segs, err := review.View(path, reviewThreshold(r), st)
	if err != nil {
		writeErr(w, http.StatusBadRequest, "%v", err)
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"segments": segs})
}

// handleReviewClip returns just the seconds around a span, as its own WAV.
func (s *Server) handleReviewClip(w http.ResponseWriter, r *http.Request) {
	q := r.URL.Query()
	wav := q.Get("wav")
	start, _ := strconv.ParseFloat(q.Get("start"), 64)
	end, _ := strconv.ParseFloat(q.Get("end"), 64)
	pad := 1.5
	if p, err := strconv.ParseFloat(q.Get("pad"), 64); err == nil && p >= 0 {
		pad = p
	}
	if wav == "" || end <= 0 {
		writeErr(w, http.StatusBadRequest, "need wav, start and end")
		return
	}

	clip, err := audio.SliceWAV(wav,
		time.Duration((start-pad)*float64(time.Second)),
		time.Duration((end+pad)*float64(time.Second)))
	if err != nil {
		writeErr(w, http.StatusBadRequest, "%v", err)
		return
	}
	w.Header().Set("Content-Type", "audio/wav")
	w.Header().Set("Cache-Control", "no-store")
	_, _ = w.Write(clip)
}

func (s *Server) handleReviewDecide(w http.ResponseWriter, r *http.Request) {
	var req struct {
		Path       string   `json:"path"`
		SpanID     string   `json:"spanId"`
		Decision   string   `json:"decision"`
		Corrected  string   `json:"corrected"`
		Original   string   `json:"original"`
		Model      string   `json:"model"`
		Video      string   `json:"video"`
		Audio      string   `json:"audio"`
		Start      float64  `json:"start"`
		End        float64  `json:"end"`
		Confidence float64  `json:"confidence"`
		Flags      []string `json:"flags"`
	}
	if !decodeBody(w, r, &req) {
		return
	}
	if req.Path == "" || req.SpanID == "" {
		writeErr(w, http.StatusBadRequest, "need path and spanId")
		return
	}
	switch req.Decision {
	case "accepted", "edited", "not_an_error", "skipped":
	default:
		writeErr(w, http.StatusBadRequest, "unknown decision: %q", req.Decision)
		return
	}
	if req.Decision == "edited" && strings.TrimSpace(req.Corrected) == "" {
		writeErr(w, http.StatusBadRequest, "an edit needs replacement text")
		return
	}

	if err := review.SaveDecision(req.Path, req.SpanID, review.Decision{
		Decision: req.Decision, Corrected: req.Corrected,
	}); err != nil {
		writeErr(w, http.StatusInternalServerError, "%v", err)
		return
	}

	// Only decisions that carry information about the audio are worth training
	// on: a correction, or a confirmation that the engine was right.
	if req.Decision == "edited" || req.Decision == "accepted" {
		s.recordForDataset(req.Audio, dataset.Record{
			Video: req.Video, Model: req.Model,
			Start: req.Start, End: req.End,
			Original: req.Original, Corrected: req.Corrected,
			Decision: dataset.Decision(req.Decision),
			Flags:    req.Flags, Confidence: req.Confidence,
		})
	}
	writeJSON(w, http.StatusOK, map[string]bool{"ok": true})
}

// handleReviewSegment rewrites a whole line, for when the error is not confined
// to the flagged word -- several words wrong, or one dropped or invented.
func (s *Server) handleReviewSegment(w http.ResponseWriter, r *http.Request) {
	var req struct {
		Path     string  `json:"path"`
		Segment  int     `json:"segment"`
		Text     string  `json:"text"`
		Original string  `json:"original"`
		Revert   bool    `json:"revert"`
		Model    string  `json:"model"`
		Video    string  `json:"video"`
		Audio    string  `json:"audio"`
		Start    float64 `json:"start"`
		End      float64 `json:"end"`
	}
	if !decodeBody(w, r, &req) {
		return
	}
	if req.Path == "" {
		writeErr(w, http.StatusBadRequest, "no transcript given")
		return
	}
	if req.Segment < 0 {
		writeErr(w, http.StatusBadRequest, "bad segment")
		return
	}

	text := req.Text
	if req.Revert {
		text = "\x00" // the sentinel that restores the engine's own words
	}
	if err := review.SaveSegment(req.Path, req.Segment, text); err != nil {
		writeErr(w, http.StatusInternalServerError, "%v", err)
		return
	}

	// A rewritten line paired with its audio is the most useful kind of
	// training example, so it is worth recording -- but an emptied line says
	// only "this was not speech", which is not.
	if !req.Revert && strings.TrimSpace(req.Text) != "" &&
		strings.TrimSpace(req.Text) != strings.TrimSpace(req.Original) {
		s.recordForDataset(req.Audio, dataset.Record{
			Video: req.Video, Model: req.Model,
			Start: req.Start, End: req.End,
			Original: req.Original, Corrected: req.Text,
			Decision: dataset.Edited,
			Flags:    []string{"line_rewrite"},
		})
	}
	writeJSON(w, http.StatusOK, map[string]bool{"ok": true})
}

// recordForDataset stores the record and its clip. Failures are logged, never
// surfaced: losing a training sample must not interrupt someone's review.
func (s *Server) recordForDataset(wav string, rec dataset.Record) {
	s.mu.Lock()
	dir := s.datasetDir
	s.mu.Unlock()

	st, err := dataset.Open(dir)
	if err != nil {
		log.Printf("dataset: %v", err)
		return
	}
	if wav != "" {
		clip, err := audio.SliceWAV(wav,
			time.Duration((rec.Start-1.0)*float64(time.Second)),
			time.Duration((rec.End+1.0)*float64(time.Second)))
		if err == nil {
			name := fmt.Sprintf("%s-%.0f.wav", sanitise(rec.Video), rec.Start*1000)
			if saved, err := st.SaveClip(name, clip); err == nil {
				rec.Clip = saved
			}
		}
	}
	if err := st.Append(rec); err != nil {
		log.Printf("dataset: %v", err)
	}
}

var unsafeName = regexp.MustCompile(`[^\w.-]+`)

func sanitise(s string) string {
	s = unsafeName.ReplaceAllString(s, "_")
	if len(s) > 60 {
		s = s[:60]
	}
	return strings.Trim(s, "_")
}

// handleReviewDownload sends the corrected transcript to the browser.
func (s *Server) handleReviewDownload(w http.ResponseWriter, r *http.Request) {
	path := r.URL.Query().Get("path")
	format := r.URL.Query().Get("format")
	if path == "" {
		writeErr(w, http.StatusBadRequest, "no transcript given")
		return
	}
	if format != "txt" && format != "srt" {
		writeErr(w, http.StatusBadRequest, "format must be txt or srt")
		return
	}

	st, _ := review.LoadState(path)
	data, err := review.Export(path, st, format)
	if err != nil {
		writeErr(w, http.StatusBadRequest, "%v", err)
		return
	}

	name := strings.TrimSuffix(filepath.Base(path), filepath.Ext(path)) + "." + format
	mime := "text/plain; charset=utf-8"
	if format == "srt" {
		mime = "application/x-subrip; charset=utf-8"
	}
	w.Header().Set("Content-Type", mime)
	// The name is quoted and also given as UTF-8 so Turkish titles survive.
	w.Header().Set("Content-Disposition",
		"attachment; filename*=UTF-8''"+url.PathEscape(name))
	w.Header().Set("Content-Length", strconv.Itoa(len(data)))
	_, _ = w.Write(data)
}

// handleReviewDelete removes a transcript and everything derived from it.
//
// Repeat runs accumulate, and a run made with the wrong settings is just
// clutter. Only generated files go: the source audio and the collected training
// data are left alone, since neither can be reproduced by transcribing again.
func (s *Server) handleReviewDelete(w http.ResponseWriter, r *http.Request) {
	var req struct {
		Path string `json:"path"`
	}
	if !decodeBody(w, r, &req) {
		return
	}
	abs, err := filepath.Abs(req.Path)
	if err != nil {
		writeErr(w, http.StatusBadRequest, "%v", err)
		return
	}

	s.mu.Lock()
	root := s.transcriptDir
	s.mu.Unlock()

	// Only ever delete inside the transcripts directory.
	rel, err := filepath.Rel(root, abs)
	if err != nil || strings.HasPrefix(rel, "..") {
		writeErr(w, http.StatusBadRequest, "that file is not a transcript")
		return
	}
	if !strings.EqualFold(filepath.Ext(abs), ".json") || strings.HasSuffix(abs, ".review.json") {
		writeErr(w, http.StatusBadRequest, "that file is not a transcript")
		return
	}

	stem := strings.TrimSuffix(abs, filepath.Ext(abs))
	removed := []string{}
	for _, ext := range []string{".json", ".txt", ".srt", ".vtt", ".review.json", ".corrected.txt"} {
		f := stem + ext
		if err := os.Remove(f); err == nil {
			removed = append(removed, filepath.Base(f))
		}
	}
	if len(removed) == 0 {
		writeErr(w, http.StatusNotFound, "nothing to delete")
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"removed": removed})
}

// --- dataset ---------------------------------------------------------------

func (s *Server) store() (*dataset.Store, string, error) {
	s.mu.Lock()
	dir, transcripts := s.datasetDir, s.transcriptDir
	s.mu.Unlock()
	st, err := dataset.Open(dir)
	return st, transcripts, err
}

// handleDataset reports what an export would contain, so nothing is sent
// without the sender having seen the size and the source list first.
func (s *Server) handleDataset(w http.ResponseWriter, r *http.Request) {
	st, transcripts, err := s.store()
	if err != nil {
		writeErr(w, http.StatusInternalServerError, "%v", err)
		return
	}
	sum := st.Summarise(transcripts)
	writeJSON(w, http.StatusOK, map[string]any{
		"records":     sum.Records,
		"clips":       sum.Clips,
		"transcripts": sum.Transcripts,
		"videos":      sum.Videos,
		"bytes":       sum.Bytes,
		"empty":       sum.Empty,
		"dir":         st.Dir(),
	})
}

// handleDatasetExport streams a zip straight to the browser's downloads.
func (s *Server) handleDatasetExport(w http.ResponseWriter, r *http.Request) {
	st, transcripts, err := s.store()
	if err != nil {
		writeErr(w, http.StatusInternalServerError, "%v", err)
		return
	}
	if sum := st.Summarise(transcripts); sum.Empty {
		writeErr(w, http.StatusBadRequest,
			"there is nothing to export yet - transcribe something first")
		return
	}

	name := "transcript-dataset-" + time.Now().Format("2006-01-02") + ".zip"
	w.Header().Set("Content-Type", "application/zip")
	w.Header().Set("Content-Disposition", `attachment; filename="`+name+`"`)
	// The zip is streamed rather than staged on disk, so a large dataset costs
	// no extra space and the download starts immediately.
	if err := st.WriteZip(w, transcripts, r.URL.Query().Get("note")); err != nil {
		// Headers are already sent by now, so there is no way to turn this into
		// an HTTP error; the truncated zip will fail its own integrity check.
		log.Printf("dataset export failed partway: %v", err)
	}
}

// --- filesystem picker -----------------------------------------------------

type browseEntry struct {
	Name string `json:"name"`
	Path string `json:"path"`
	Dir  bool   `json:"dir"`
	Size int64  `json:"size"`
	Kind Kind   `json:"kind,omitempty"`
}

// handleBrowse lists a directory for the in-page picker.
//
// The picker exists because a browser will not tell JavaScript a real
// filesystem path, for either <input type="file"> or a drop. Since the server
// is the user's own machine, it can read the directory itself and hand back
// paths, which is what conversion actually needs.
func (s *Server) handleBrowse(w http.ResponseWriter, r *http.Request) {
	path := strings.TrimSpace(r.URL.Query().Get("path"))
	if path == "" {
		if home, err := os.UserHomeDir(); err == nil {
			path = home
		} else {
			path = roots()[0]
		}
	}
	abs, err := filepath.Abs(expandHome(path))
	if err != nil {
		writeErr(w, http.StatusBadRequest, "%v", err)
		return
	}
	// A file was given rather than a folder: open its containing folder.
	if fi, err := os.Stat(abs); err == nil && !fi.IsDir() {
		abs = filepath.Dir(abs)
	}

	entries, err := os.ReadDir(abs)
	if err != nil {
		writeErr(w, http.StatusBadRequest, "%v", err)
		return
	}

	showHidden := r.URL.Query().Get("hidden") == "1"
	dirs, files := []browseEntry{}, []browseEntry{}
	for _, e := range entries {
		if !showHidden && strings.HasPrefix(e.Name(), ".") {
			continue
		}
		full := filepath.Join(abs, e.Name())
		if e.IsDir() {
			dirs = append(dirs, browseEntry{Name: e.Name(), Path: full, Dir: true})
			continue
		}
		kind, ok := kindFor(e.Name())
		if !ok {
			continue // only convertible media is worth showing here
		}
		f := browseEntry{Name: e.Name(), Path: full, Kind: kind}
		if fi, err := e.Info(); err == nil {
			f.Size = fi.Size()
		}
		files = append(files, f)
	}
	less := func(s []browseEntry) func(i, j int) bool {
		return func(i, j int) bool { return strings.ToLower(s[i].Name) < strings.ToLower(s[j].Name) }
	}
	sort.Slice(dirs, less(dirs))
	sort.Slice(files, less(files))

	parent := filepath.Dir(abs)
	if parent == abs {
		parent = "" // already at a root
	}

	writeJSON(w, http.StatusOK, map[string]any{
		"path":    abs,
		"parent":  parent,
		"entries": append(dirs, files...),
	})
}

// handleSourcesAdd registers files picked from outside the two libraries.
// A folder is expanded to the convertible media directly inside it.
func (s *Server) handleSourcesAdd(w http.ResponseWriter, r *http.Request) {
	var req struct {
		Paths []string `json:"paths"`
	}
	if !decodeBody(w, r, &req) {
		return
	}

	var resolved []string
	var problems []string

	for _, raw := range req.Paths {
		p := expandHome(strings.TrimSpace(raw))
		if p == "" {
			continue
		}
		abs, err := filepath.Abs(p)
		if err != nil {
			problems = append(problems, fmt.Sprintf("%s: %v", raw, err))
			continue
		}
		fi, err := os.Stat(abs)
		if err != nil {
			problems = append(problems, fmt.Sprintf("%s: %v", raw, err))
			continue
		}
		if fi.IsDir() {
			entries, err := os.ReadDir(abs)
			if err != nil {
				problems = append(problems, fmt.Sprintf("%s: %v", abs, err))
				continue
			}
			found := 0
			for _, e := range entries {
				if e.IsDir() || strings.HasPrefix(e.Name(), ".") {
					continue
				}
				if _, ok := kindFor(e.Name()); !ok {
					continue
				}
				resolved = append(resolved, filepath.Join(abs, e.Name()))
				found++
			}
			if found == 0 {
				problems = append(problems, fmt.Sprintf("%s: no convertible media in this folder", abs))
			}
			continue
		}
		src, _, err := resolveSource(abs)
		if err != nil {
			problems = append(problems, err.Error())
			continue
		}
		resolved = append(resolved, src)
	}

	s.mu.Lock()
	seen := make(map[string]bool, len(s.added))
	for _, p := range s.added {
		seen[p] = true
	}
	addedNow := []string{}
	for _, p := range resolved {
		if seen[p] {
			continue
		}
		seen[p] = true
		s.added = append(s.added, p)
		addedNow = append(addedNow, p)
	}
	s.notifyLocked()
	s.mu.Unlock()

	code := http.StatusOK
	if len(addedNow) == 0 && len(problems) > 0 {
		code = http.StatusBadRequest
	}
	writeJSON(w, code, map[string]any{"added": addedNow, "problems": problems})
}

func (s *Server) handleSourcesRemove(w http.ResponseWriter, r *http.Request) {
	var req struct {
		Paths []string `json:"paths"`
		All   bool     `json:"all"`
	}
	if !decodeBody(w, r, &req) {
		return
	}

	drop := make(map[string]bool, len(req.Paths))
	for _, p := range req.Paths {
		drop[p] = true
	}

	s.mu.Lock()
	if req.All {
		s.added = nil
	} else {
		kept := s.added[:0]
		for _, p := range s.added {
			if !drop[p] {
				kept = append(kept, p)
			}
		}
		s.added = kept
	}
	s.notifyLocked()
	s.mu.Unlock()

	writeJSON(w, http.StatusOK, map[string]bool{"ok": true})
}

// --- conversion ------------------------------------------------------------

func (s *Server) handleConvert(w http.ResponseWriter, r *http.Request) {
	var req struct {
		Paths    []string `json:"paths"`
		Loudnorm bool     `json:"loudnorm"`
	}
	if !decodeBody(w, r, &req) {
		return
	}
	if len(req.Paths) == 0 {
		writeErr(w, http.StatusBadRequest, "no files given")
		return
	}

	opts := audio.DefaultWAVOptions()
	opts.Loudnorm = req.Loudnorm

	// Validate the whole batch before starting any of it.
	type target struct {
		src  string
		kind Kind
	}
	targets := make([]target, 0, len(req.Paths))
	for _, p := range req.Paths {
		src, kind, err := resolveSource(p)
		if err != nil {
			writeErr(w, http.StatusBadRequest, "%v", err)
			return
		}
		targets = append(targets, target{src, kind})
	}

	ids := make([]string, 0, len(targets))
	for _, t := range targets {
		ids = append(ids, s.start(t.kind, t.src, opts, false))
	}
	s.notify()
	writeJSON(w, http.StatusOK, map[string]any{"ids": ids})
}

// handlePipeline is the one-click path: hand it media and it converts, then
// transcribes, with no further instruction. The WAV is an implementation
// detail of whisper's input format, so it should not be a step the user takes.
func (s *Server) handlePipeline(w http.ResponseWriter, r *http.Request) {
	var req struct {
		Paths []string `json:"paths"`
	}
	if !decodeBody(w, r, &req) {
		return
	}
	if len(req.Paths) == 0 {
		writeErr(w, http.StatusBadRequest, "no files given")
		return
	}
	if s.asrTools.WhisperCLI() == "" {
		writeErr(w, http.StatusServiceUnavailable,
			"no transcription engine is installed, so only conversion is possible. %s", s.asrErr)
		return
	}

	opts := audio.DefaultWAVOptions()
	opts.Loudnorm = s.prefs.get().Loudnorm

	type target struct {
		src  string
		kind Kind
	}
	targets := make([]target, 0, len(req.Paths))
	for _, p := range req.Paths {
		src, kind, err := resolveSource(p)
		if err != nil {
			writeErr(w, http.StatusBadRequest, "%v", err)
			return
		}
		targets = append(targets, target{src, kind})
	}

	ids := make([]string, 0, len(targets))
	for _, t := range targets {
		// Already converted? Skip straight to transcription.
		out := s.outputPath(t.kind, filepath.Base(t.src))
		if fi, err := os.Stat(out); err == nil && fi.Size() > 0 {
			if err := s.chainTranscribe(out, t.kind); err != nil {
				writeErr(w, http.StatusBadRequest, "%v", err)
				return
			}
			continue
		}
		ids = append(ids, s.start(t.kind, t.src, opts, true))
	}
	s.notify()
	s.saveJobs()
	writeJSON(w, http.StatusOK, map[string]any{"ids": ids})
}

// handleGlossary reads and writes the vocabulary list fed to whisper.
func (s *Server) handleGlossary(w http.ResponseWriter, r *http.Request) {
	path := s.glossaryPath()

	if r.Method == http.MethodPost {
		var req struct {
			Text string `json:"text"`
		}
		if !decodeBody(w, r, &req) {
			return
		}
		if err := asr.SaveGlossary(path, req.Text); err != nil {
			writeErr(w, http.StatusInternalServerError, "%v", err)
			return
		}
	}

	text, err := asr.LoadGlossary(path)
	if err != nil {
		writeErr(w, http.StatusInternalServerError, "%v", err)
		return
	}
	g := asr.ParseGlossary(text)
	prompt := g.Prompt()
	writeJSON(w, http.StatusOK, map[string]any{
		"text":    text,
		"terms":   len(g.Terms),
		"prompt":  asr.TrimPrompt(prompt),
		"limit":   asr.PromptLimit,
		"trimmed": len(prompt) > asr.PromptLimit,
		"path":    path,
	})
}

// handlePrefs stores the transcription settings the pipeline runs with.
func (s *Server) handlePrefs(w http.ResponseWriter, r *http.Request) {
	if r.Method == http.MethodPost {
		var p Prefs
		if !decodeBody(w, r, &p) {
			return
		}
		s.prefs.set(p)
	}
	writeJSON(w, http.StatusOK, s.prefs.get())
}

func (s *Server) handleCancel(w http.ResponseWriter, r *http.Request) {
	var req struct {
		ID string `json:"id"`
	}
	if !decodeBody(w, r, &req) {
		return
	}
	s.mu.Lock()
	job, ok := s.jobs[req.ID]
	s.mu.Unlock()
	if !ok {
		writeErr(w, http.StatusNotFound, "no such job: %s", req.ID)
		return
	}
	job.cancel()
	writeJSON(w, http.StatusOK, map[string]bool{"ok": true})
}

// handleClear drops finished jobs from the list, leaving running ones alone.
func (s *Server) handleClear(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeErr(w, http.StatusMethodNotAllowed, "POST only")
		return
	}
	s.mu.Lock()
	kept := s.order[:0]
	for _, id := range s.order {
		switch s.jobs[id].State {
		case StateQueued, StateRunning:
			kept = append(kept, id)
		default:
			delete(s.jobs, id)
		}
	}
	s.order = kept
	s.notifyLocked()
	s.mu.Unlock()
	s.saveJobs()
	writeJSON(w, http.StatusOK, map[string]bool{"ok": true})
}

// --- events ----------------------------------------------------------------

func (s *Server) handleEvents(w http.ResponseWriter, r *http.Request) {
	flusher, ok := w.(http.Flusher)
	if !ok {
		writeErr(w, http.StatusInternalServerError, "streaming unsupported")
		return
	}
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.Header().Set("X-Accel-Buffering", "no")

	ch := s.subscribe()
	defer s.unsubscribe(ch)

	send := func() bool {
		b, err := json.Marshal(s.snapshot())
		if err != nil {
			return false
		}
		if _, err := fmt.Fprintf(w, "data: %s\n\n", b); err != nil {
			return false
		}
		flusher.Flush()
		return true
	}

	if !send() {
		return
	}
	// Keeps intermediaries and idle connections from dropping the stream.
	ping := time.NewTicker(20 * time.Second)
	defer ping.Stop()

	for {
		select {
		case <-r.Context().Done():
			return
		case <-ch:
			if !send() {
				return
			}
		case <-ping.C:
			if _, err := io.WriteString(w, ": ping\n\n"); err != nil {
				return
			}
			flusher.Flush()
		}
	}
}

func (s *Server) snapshot() map[string]any {
	s.mu.Lock()
	defer s.mu.Unlock()

	jobs := make([]Job, 0, len(s.order))
	for _, id := range s.order {
		j := s.jobs[id]
		snap := *j
		if j.State == StateRunning && !j.started.IsZero() {
			snap.Elapsed = time.Since(j.started).Seconds()
			snap.ETA = s.etaLocked(j)
		}
		jobs = append(jobs, snap)
	}
	return map[string]any{
		"jobs":      jobs,
		"audioDir":  s.audioDir,
		"videoDir":  s.videoDir,
		"outputDir": s.outputDir,
		"added":     len(s.added),
	}
}

// --- job execution ---------------------------------------------------------

func (s *Server) start(kind Kind, src string, opts audio.WAVOptions, chain bool) string {
	ctx, cancel := context.WithCancel(context.Background())

	name := filepath.Base(src)
	out := s.outputPath(kind, name)

	s.mu.Lock()
	s.seq++
	id := fmt.Sprintf("j%d", s.seq)
	job := &Job{
		ID:      id,
		Type:    JobConvert,
		Kind:    kind,
		opts:    persistedOpts{Loudnorm: opts.Loudnorm, Chain: chain},
		Name:    name,
		Source:  src,
		Output:  out,
		State:   StateQueued,
		Percent: -1,
		cancel:  cancel,
	}
	s.jobs[id] = job
	s.order = append(s.order, id)
	s.mu.Unlock()

	go s.run(ctx, job, src, out, opts)
	// Persisted at creation, not only at completion: a job interrupted while
	// running is exactly the one worth recovering.
	go s.saveJobs()
	return id
}

func (s *Server) run(ctx context.Context, job *Job, in, out string, opts audio.WAVOptions) {
	defer job.cancel()

	// Wait for a slot, but stay cancellable while queued.
	select {
	case s.sem <- struct{}{}:
		defer func() { <-s.sem }()
	case <-ctx.Done():
		s.finish(job, StateCanceled, "canceled before it started")
		return
	}

	s.mu.Lock()
	job.State = StateRunning
	job.started = time.Now()
	s.notifyLocked()
	s.mu.Unlock()
	s.saveJobs()

	// Without a duration there is nothing to compute a percentage against, so
	// the UI shows an indeterminate bar instead.
	if d, err := s.tools.Duration(ctx, in); err == nil && d > 0 {
		s.mu.Lock()
		job.total = d
		job.audioSeconds = d.Seconds()
		job.Percent = 0
		s.mu.Unlock()
	}

	err := s.tools.ConvertToWAV(ctx, in, out, opts, func(done time.Duration) {
		s.mu.Lock()
		if job.total > 0 {
			pct := float64(done) / float64(job.total) * 100
			if pct > 100 {
				pct = 100
			}
			job.Percent = pct
		}
		// ffmpeg reports several times a second; throttle the fan-out.
		if time.Since(s.lastPush) > 200*time.Millisecond {
			s.notifyLocked()
		}
		s.mu.Unlock()
	})

	switch {
	case err == nil:
		var size int64
		if fi, statErr := os.Stat(out); statErr == nil {
			size = fi.Size()
		}
		s.mu.Lock()
		job.OutSize = size
		job.Percent = 100
		chain := job.opts.Chain
		s.mu.Unlock()
		s.finish(job, StateDone, "")

		// One-click pipeline: the WAV exists only because whisper needs it, so
		// having produced it, carry straight on to the transcription.
		if chain {
			if err := s.chainTranscribe(out, job.Kind); err != nil {
				log.Printf("pipeline: %v", err)
			}
		}
	case ctx.Err() != nil:
		s.finish(job, StateCanceled, "canceled")
	default:
		s.finish(job, StateFailed, err.Error())
	}
}

func (s *Server) finish(job *Job, state JobState, msg string) {
	s.mu.Lock()
	job.State = state
	job.Error = msg
	job.ETA = 0
	if !job.started.IsZero() {
		job.Elapsed = time.Since(job.started).Seconds()
	}
	key := string(job.Type)
	if job.Type == JobTranscribe {
		key += ":" + job.Model
	}
	audio, wall := job.audioSeconds, job.Elapsed
	s.notifyLocked()
	s.mu.Unlock()

	// Learn this machine's speed, so the next job can be estimated before it
	// reports any progress of its own.
	if state == StateDone {
		s.noteRate(key, audio, wall)
	}
	s.saveJobs()
}
