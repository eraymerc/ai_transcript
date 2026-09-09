// Package asr drives whisper.cpp's whisper-cli for transcription.
//
// Like ffmpeg, the engine is a subprocess rather than a linked library: it
// keeps this program pure Go so it cross-compiles to Windows, lets the engine
// be updated independently, and stops a crash in the model runner from taking
// the application with it.
package asr

import (
	"bufio"
	"context"
	_ "embed"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"runtime"
	"sort"
	"strconv"
	"strings"
)

// EnvWhisper overrides the location of the plain CPU binary.
const EnvWhisper = "AI_TRANSCRIPT_WHISPER"

// Backend is an acceleration path. Each is a separate whisper-cli executable
// built against a different ggml backend, so a driver fault in one cannot
// affect the others -- they are different processes.
type Backend string

const (
	BackendCPU    Backend = "cpu"
	BackendCUDA   Backend = "cuda"
	BackendVulkan Backend = "vulkan"
)

// Backends in preference order: fastest first, CPU last as the always-works case.
var Backends = []Backend{BackendCUDA, BackendVulkan, BackendCPU}

// binaryFor is the file name each backend is installed under.
func binaryFor(b Backend) string {
	switch b {
	case BackendCUDA:
		return exeName("whisper-cli-cuda")
	case BackendVulkan:
		return exeName("whisper-cli-vulkan")
	default:
		return exeName("whisper-cli")
	}
}

// Tools maps each backend to the executable that provides it. A backend absent
// from the map simply was not built or shipped for this platform.
type Tools struct {
	Engines map[Backend]string
}

// WhisperCLI is the CPU engine, which every install is expected to have.
func (t Tools) WhisperCLI() string { return t.Engines[BackendCPU] }

// Has reports whether an executable exists for a backend.
func (t Tools) Has(b Backend) bool { return t.Engines[b] != "" }

func exeName(base string) string {
	if runtime.GOOS == "windows" {
		return base + ".exe"
	}
	return base
}

func searchDirs() []string {
	var dirs []string
	if exe, err := os.Executable(); err == nil {
		d := filepath.Dir(exe)
		dirs = append(dirs, d, filepath.Join(d, "binaries", runtime.GOOS))
	}
	if wd, err := os.Getwd(); err == nil {
		dirs = append(dirs, filepath.Join(wd, "binaries", runtime.GOOS), wd)
	}
	// The Windows CUDA build ships beside its own CUDA DLLs, so it lives in a
	// subdirectory rather than next to the other engines.
	for _, d := range append([]string{}, dirs...) {
		dirs = append(dirs, filepath.Join(d, "cuda"))
	}
	return dirs
}

func findBinary(name string) string {
	for _, d := range searchDirs() {
		p := filepath.Join(d, name)
		if fi, err := os.Stat(p); err == nil && !fi.IsDir() {
			return p
		}
	}
	if p, err := exec.LookPath(name); err == nil {
		return p
	}
	return ""
}

// Locate finds whichever engine binaries are installed. It is an error only
// when none is found at all.
func Locate() (Tools, error) {
	t := Tools{Engines: map[Backend]string{}}

	if p := os.Getenv(EnvWhisper); p != "" {
		if _, err := os.Stat(p); err != nil {
			return t, fmt.Errorf("%s points at %q: %w", EnvWhisper, p, err)
		}
		t.Engines[BackendCPU] = p
	}
	for _, b := range Backends {
		if t.Engines[b] != "" {
			continue
		}
		if p := findBinary(binaryFor(b)); p != "" {
			t.Engines[b] = p
		}
	}
	if len(t.Engines) == 0 {
		return t, fmt.Errorf("no whisper-cli found in %s or on PATH", strings.Join(searchDirs(), ", "))
	}
	return t, nil
}

// Model is one ggml model file on disk.
type Model struct {
	Name string `json:"name"` // display name, also the output subdirectory
	Path string `json:"path"`
	File string `json:"file"`
	Size int64  `json:"size"`
	VAD  bool   `json:"vad"`

	// Who made it and on what terms. A file name alone tells a user nothing
	// useful when choosing between two half-gigabyte models.
	Display     string  `json:"display"`
	Publisher   string  `json:"publisher"`
	License     string  `json:"license"`
	Notes       string  `json:"notes,omitempty"`
	WER         float64 `json:"wer,omitempty"`
	WERSource   string  `json:"werSource,omitempty"`
	Known       bool    `json:"known"`
	Recommended bool    `json:"recommended,omitempty"`

	order int // position in the curated catalog, for sorting
}

// ListModels finds ggml models in dir, separating voice-activity models from
// transcription models — they are selected in different places in the UI.
func ListModels(dir string) ([]Model, error) {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return nil, err
	}
	models := []Model{}
	for _, e := range entries {
		name := e.Name()
		if e.IsDir() || !strings.HasSuffix(strings.ToLower(name), ".bin") {
			continue
		}
		if strings.HasPrefix(name, ".") {
			continue
		}
		m := Model{Name: displayName(name), File: name, Path: filepath.Join(dir, name)}
		if fi, err := e.Info(); err == nil {
			m.Size = fi.Size()
		}
		lower := strings.ToLower(name)
		m.VAD = strings.Contains(lower, "silero") || strings.Contains(lower, "vad")

		if c, ok := Lookup(name); ok {
			m.Display, m.Publisher, m.License = c.Display, c.Publisher, c.License
			m.Notes, m.WER, m.WERSource = c.Notes, c.WER, c.WERSource
			m.VAD, m.Known, m.Recommended = c.VAD, true, c.Recommended
			m.order = Index(name)
		} else {
			// A model dropped in by hand is still usable; it just cannot be
			// described, and saying so is better than inventing a publisher.
			m.Display = m.Name
			m.Publisher = "unknown"
			m.order = len(Catalog) // unknown models sort after the known ones
		}
		models = append(models, m)
	}
	// Recommended first, then the curated order, then anything unrecognised.
	sort.SliceStable(models, func(i, j int) bool {
		a, b := models[i], models[j]
		if a.Recommended != b.Recommended {
			return a.Recommended
		}
		if a.order != b.order {
			return a.order < b.order
		}
		return a.Name < b.Name
	})
	return models, nil
}

// displayName turns "ggml-large-v3-turbo-q5_0.bin" into "large-v3-turbo-q5_0".
func displayName(file string) string {
	n := strings.TrimSuffix(file, filepath.Ext(file))
	return strings.TrimPrefix(n, "ggml-")
}

// dtwPreset maps a model file name to the alignment preset whisper-cli expects.
//
// Without -dtw the per-token timestamps in the JSON are interpolated guesses
// that drift by seconds, which makes word-level review clips unusable. The
// preset must match the model or whisper-cli refuses to start.
func dtwPreset(modelPath string) string {
	n := strings.ToLower(filepath.Base(modelPath))
	switch {
	case strings.Contains(n, "large-v3-turbo"), strings.Contains(n, "large-v3_turbo"):
		return "large.v3.turbo"
	case strings.Contains(n, "large-v3"):
		return "large.v3"
	case strings.Contains(n, "large-v2"):
		return "large.v2"
	case strings.Contains(n, "large-v1"), strings.Contains(n, "large"):
		return "large.v1"
	case strings.Contains(n, "medium.en"):
		return "medium.en"
	case strings.Contains(n, "medium"):
		return "medium"
	case strings.Contains(n, "small.en"):
		return "small.en"
	case strings.Contains(n, "small"):
		return "small"
	case strings.Contains(n, "base.en"):
		return "base.en"
	case strings.Contains(n, "base"):
		return "base"
	case strings.Contains(n, "tiny.en"):
		return "tiny.en"
	case strings.Contains(n, "tiny"):
		return "tiny"
	}
	return "" // unknown model: skip DTW rather than fail the run
}

// Options configures one transcription run.
type Options struct {
	ModelPath string
	VADModel  string // empty disables VAD
	Language  string // "tr"
	Threads   int
	BeamSize  int
	OutDir    string // where the .json/.txt/.srt land
	OutBase   string // file name stem, without extension

	// Backend selects which engine executable runs. Device is the index within
	// that backend, which matters on hybrid laptops where device 0 is often the
	// integrated GPU rather than the discrete one.
	Backend Backend
	Device  int

	// NoDTW disables cross-attention alignment. Leave false: without it the
	// token timestamps are unusable for review.
	NoDTW bool

	// Prompt primes the decoder with vocabulary it should expect -- names,
	// jargon, recurring terms. Preventing an error costs nothing at inference
	// time, which is cheaper than detecting and repairing it afterwards.
	Prompt string
}

// PromptLimit is roughly how much prompt whisper will accept. The decoder takes
// at most n_text_ctx/2 tokens; this is a conservative character budget for it,
// since Turkish runs a little over 3 characters per token.
const PromptLimit = 700

// TrimPrompt cuts a prompt to what whisper will actually read, on a separator
// so a term is never sliced in half.
func TrimPrompt(s string) string {
	s = strings.TrimSpace(s)
	if len(s) <= PromptLimit {
		return s
	}
	cut := s[:PromptLimit]
	if i := strings.LastIndexAny(cut, ",;\n "); i > 0 {
		cut = cut[:i]
	}
	return strings.TrimRight(strings.TrimSpace(cut), ",;")
}

var progressRe = regexp.MustCompile(`progress\s*=\s*(\d+)%`)

// Transcribe runs whisper-cli over one WAV and writes json/txt/srt into OutDir.
//
// Output goes to a temporary directory first and is moved into place only on
// success, so a cancelled run cannot leave a half-written transcript that
// later stages would treat as complete.
func (t Tools) Transcribe(ctx context.Context, wav string, opts Options, onProgress func(pct int)) error {
	if opts.Backend == "" {
		opts.Backend = BackendCPU
	}
	engine := t.Engines[opts.Backend]
	if engine == "" {
		return fmt.Errorf("no engine installed for the %s backend", opts.Backend)
	}
	if opts.Language == "" {
		opts.Language = "tr"
	}
	if opts.Threads <= 0 {
		opts.Threads = runtime.NumCPU()
	}
	if opts.BeamSize <= 0 {
		opts.BeamSize = 5
	}
	if err := os.MkdirAll(opts.OutDir, 0o755); err != nil {
		return err
	}

	tmpDir, err := os.MkdirTemp(opts.OutDir, ".partial-")
	if err != nil {
		return err
	}
	defer os.RemoveAll(tmpDir)

	args := []string{
		"-m", opts.ModelPath,
		"-f", wav,
		"-l", opts.Language,
		// whisper-cli has no --no-context; -mc 0 is the equivalent, and it is
		// the single biggest lever against repetition-loop hallucination.
		"-mc", "0",
		"-t", strconv.Itoa(opts.Threads),
		"-bs", strconv.Itoa(opts.BeamSize),
		"-pp",
	}
	if opts.Backend == BackendCPU {
		args = append(args, "-ng")
	} else {
		args = append(args, "-dev", strconv.Itoa(opts.Device))
	}
	if opts.VADModel != "" {
		args = append(args, "--vad", "-vm", opts.VADModel)
	}
	if p := TrimPrompt(opts.Prompt); p != "" {
		args = append(args, "--prompt", p)
	}
	if !opts.NoDTW {
		if preset := dtwPreset(opts.ModelPath); preset != "" {
			args = append(args, "-dtw", preset)
		}
	}
	// -ojf carries the per-token probabilities the error-detection stage runs
	// on. Plain -oj would drop them, so it is never used alone.
	args = append(args, "-oj", "-ojf", "-otxt", "-osrt",
		"-of", filepath.Join(tmpDir, opts.OutBase))

	cmd := exec.CommandContext(ctx, engine, args...)
	stderr, err := cmd.StderrPipe()
	if err != nil {
		return err
	}
	cmd.Stdout = nil

	if err := cmd.Start(); err != nil {
		return fmt.Errorf("starting whisper-cli: %w", err)
	}

	// whisper-cli reports progress on stderr; keep the tail for error messages.
	var tail []string
	sc := bufio.NewScanner(stderr)
	sc.Buffer(make([]byte, 0, 64*1024), 1024*1024)
	for sc.Scan() {
		line := sc.Text()
		if m := progressRe.FindStringSubmatch(line); m != nil {
			if pct, err := strconv.Atoi(m[1]); err == nil && onProgress != nil {
				onProgress(pct)
			}
			continue
		}
		if s := strings.TrimSpace(line); s != "" {
			tail = append(tail, s)
			if len(tail) > 12 {
				tail = tail[1:]
			}
		}
	}

	if err := cmd.Wait(); err != nil {
		if ctxErr := ctx.Err(); ctxErr != nil {
			return ctxErr
		}
		if len(tail) > 0 {
			return fmt.Errorf("whisper-cli: %s", strings.Join(tail, "; "))
		}
		return fmt.Errorf("whisper-cli: %w", err)
	}

	produced, err := os.ReadDir(tmpDir)
	if err != nil {
		return err
	}
	if len(produced) == 0 {
		return fmt.Errorf("whisper-cli produced no output")
	}
	for _, e := range produced {
		from := filepath.Join(tmpDir, e.Name())
		to := filepath.Join(opts.OutDir, e.Name())
		_ = os.Remove(to)
		if err := os.Rename(from, to); err != nil {
			return fmt.Errorf("moving %s into place: %w", e.Name(), err)
		}
	}
	return nil
}

// --- device probing --------------------------------------------------------

// probeWAV is a fraction of a second of silence, just enough to make
// whisper-cli initialise its backends and report what hardware it found.
//
//go:embed probe.wav
var probeWAV []byte

// Device is one accelerator a backend can target.
type Device struct {
	Index int    `json:"index"`
	Name  string `json:"name"`
	// Integrated marks unified-memory devices, which share bandwidth with the
	// CPU. Since the encoder is bandwidth-bound, these are often no faster than
	// running on the CPU, so the UI warns rather than recommending them.
	Integrated bool `json:"integrated"`
}

// Engine is one backend's availability, as reported by the engine itself.
type Engine struct {
	Backend   Backend  `json:"backend"`
	Installed bool     `json:"installed"`
	Available bool     `json:"available"`
	Devices   []Device `json:"devices"`
	Reason    string   `json:"reason,omitempty"`
}

var (
	// ggml_vulkan: 1 = NVIDIA GeForce RTX 4060 Laptop GPU (NVIDIA) | uma: 0 | ...
	vulkanDevRe = regexp.MustCompile(`ggml_vulkan:\s+(\d+) = (.+?) \(([^)]*)\)\s*\|\s*uma:\s*(\d)`)
	//   Device 0: NVIDIA GeForce RTX 4060 Laptop GPU, compute capability 8.9
	cudaDevRe = regexp.MustCompile(`Device (\d+): (.+?), compute capability`)
	// Generic fallback: whisper_backend_init_gpu: device 0: Vulkan0 (type: 2)
	genericDevRe = regexp.MustCompile(`whisper_backend_init_gpu: device (\d+): (.+?) \(type: (\d+)\)`)
	noGPURe      = regexp.MustCompile(`no GPU found`)
)

// Probe asks one backend's engine what hardware it can actually use.
//
// This deliberately does not inspect drivers or vendor strings from Go. Whether
// a device is usable depends on how the engine was built, which backends its
// runtime loads, and whether the driver works -- and only the engine knows all
// three at once. So it is run over a fragment of silence and its report parsed.
func (t Tools) Probe(ctx context.Context, backend Backend, modelPath string) Engine {
	e := Engine{Backend: backend}

	engine := t.Engines[backend]
	if engine == "" {
		e.Reason = "no engine binary is installed for this backend"
		return e
	}
	e.Installed = true

	if backend == BackendCPU {
		e.Available = true
		return e
	}
	if modelPath == "" {
		e.Reason = "no transcription model is available to probe with"
		return e
	}

	tmp, err := os.CreateTemp("", "gpuprobe-*.wav")
	if err != nil {
		e.Reason = err.Error()
		return e
	}
	defer os.Remove(tmp.Name())
	if _, err := tmp.Write(probeWAV); err != nil {
		tmp.Close()
		e.Reason = err.Error()
		return e
	}
	tmp.Close()

	out, runErr := exec.CommandContext(ctx, engine,
		"-m", modelPath, "-f", tmp.Name(), "-l", "tr", "-t", "1", "-nt").CombinedOutput()
	text := string(out)

	e.Devices = parseDevices(backend, text)

	switch {
	case len(e.Devices) > 0:
		e.Available = true
	case noGPURe.MatchString(text):
		e.Reason = "the engine ran but found no usable device for this backend"
	case runErr != nil:
		e.Reason = "the engine failed to start: " + firstLine(text)
	default:
		e.Reason = "the engine reported no device"
	}
	return e
}

func parseDevices(backend Backend, text string) []Device {
	var devs []Device

	// Backend-specific lines carry the real product name and, for Vulkan, the
	// unified-memory flag. The generic line only has "Vulkan0", which is useless
	// for telling an iGPU from a discrete card.
	switch backend {
	case BackendVulkan:
		for _, m := range vulkanDevRe.FindAllStringSubmatch(text, -1) {
			idx, err := strconv.Atoi(m[1])
			if err != nil {
				continue
			}
			devs = append(devs, Device{Index: idx, Name: m[2], Integrated: m[4] == "1"})
		}
	case BackendCUDA:
		for _, m := range cudaDevRe.FindAllStringSubmatch(text, -1) {
			idx, err := strconv.Atoi(m[1])
			if err != nil {
				continue
			}
			devs = append(devs, Device{Index: idx, Name: m[2]})
		}
	}
	if len(devs) > 0 {
		return devs
	}

	for _, m := range genericDevRe.FindAllStringSubmatch(text, -1) {
		if m[3] == "0" { // type 0 is the CPU device
			continue
		}
		idx, err := strconv.Atoi(m[1])
		if err != nil {
			continue
		}
		devs = append(devs, Device{Index: idx, Name: m[2]})
	}
	return devs
}

// ProbeAll reports every backend, in preference order.
func (t Tools) ProbeAll(ctx context.Context, modelPath string) []Engine {
	out := make([]Engine, 0, len(Backends))
	for _, b := range Backends {
		out = append(out, t.Probe(ctx, b, modelPath))
	}
	return out
}

func firstLine(s string) string {
	for _, l := range strings.Split(s, "\n") {
		if l = strings.TrimSpace(l); l != "" {
			return l
		}
	}
	return "no output"
}
