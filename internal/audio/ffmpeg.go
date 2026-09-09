// Package audio drives ffmpeg and ffprobe to prepare audio for the ASR stage.
//
// The binaries are invoked as subprocesses rather than linked in. That keeps
// ffmpeg's LGPL obligation at the process boundary, lets it be updated
// independently of this program, and means a crash in a decoder cannot take
// the application down with it.
package audio

import (
	"bufio"
	"bytes"
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"time"
)

// Environment overrides for the binary locations, checked before any search.
const (
	EnvFFmpeg  = "AI_TRANSCRIPT_FFMPEG"
	EnvFFprobe = "AI_TRANSCRIPT_FFPROBE"
)

// Tools holds the resolved paths of the external binaries.
type Tools struct {
	FFmpeg  string
	FFprobe string
}

// WAVOptions describes the output format.
//
// whisper.cpp accepts only 16-bit signed PCM, and Whisper itself works on
// 16 kHz mono samples, so the defaults are the only values the pipeline should
// use. They are fields rather than constants to make experiments possible.
type WAVOptions struct {
	SampleRate int
	Channels   int
	Loudnorm   bool
}

// DefaultWAVOptions returns what the ASR stage requires.
func DefaultWAVOptions() WAVOptions {
	return WAVOptions{SampleRate: 16000, Channels: 1, Loudnorm: true}
}

func exeName(base string) string {
	if runtime.GOOS == "windows" {
		return base + ".exe"
	}
	return base
}

// searchDirs lists directories checked before falling back to PATH: next to our
// own executable, and the repo's binaries/<goos> directory whether we are run
// from an install or from the source tree.
func searchDirs() []string {
	var dirs []string
	if exe, err := os.Executable(); err == nil {
		d := filepath.Dir(exe)
		dirs = append(dirs, d, filepath.Join(d, "binaries", runtime.GOOS))
	}
	if wd, err := os.Getwd(); err == nil {
		dirs = append(dirs, filepath.Join(wd, "binaries", runtime.GOOS), wd)
	}
	return dirs
}

func locate(base, env string) (string, error) {
	if p := os.Getenv(env); p != "" {
		if _, err := os.Stat(p); err != nil {
			return "", fmt.Errorf("%s points at %q: %w", env, p, err)
		}
		return p, nil
	}
	name := exeName(base)
	for _, d := range searchDirs() {
		p := filepath.Join(d, name)
		if fi, err := os.Stat(p); err == nil && !fi.IsDir() {
			return p, nil
		}
	}
	// Nothing bundled: on Linux this is the normal case, since ffmpeg comes
	// from the package manager and only Windows binaries are vendored.
	if p, err := exec.LookPath(base); err == nil {
		return p, nil
	}
	return "", fmt.Errorf("%s not found in %s or on PATH", name, strings.Join(searchDirs(), ", "))
}

// Locate resolves ffmpeg and ffprobe. A missing ffprobe is not an error: it only
// costs the progress percentage, since that is what supplies the total duration.
func Locate() (Tools, error) {
	var t Tools
	var err error
	if t.FFmpeg, err = locate("ffmpeg", EnvFFmpeg); err != nil {
		return t, err
	}
	t.FFprobe, _ = locate("ffprobe", EnvFFprobe)
	return t, nil
}

// Duration reports the length of an audio or video file.
func (t Tools) Duration(ctx context.Context, path string) (time.Duration, error) {
	if t.FFprobe == "" {
		return 0, errors.New("ffprobe unavailable")
	}
	out, err := exec.CommandContext(ctx, t.FFprobe,
		"-v", "error",
		"-show_entries", "format=duration",
		"-of", "default=noprint_wrappers=1:nokey=1",
		path,
	).Output()
	if err != nil {
		return 0, fmt.Errorf("ffprobe: %w", err)
	}
	secs, err := strconv.ParseFloat(strings.TrimSpace(string(out)), 64)
	if err != nil {
		return 0, fmt.Errorf("ffprobe returned %q: %w", strings.TrimSpace(string(out)), err)
	}
	return time.Duration(secs * float64(time.Second)), nil
}

// ConvertToWAV decodes any ffmpeg-readable input into PCM WAV.
//
// Note that this is a decode, not a re-encode: PCM is uncompressed, so a lossy
// source loses nothing further here.
//
// onProgress, if non-nil, is called with the position reached so far.
func (t Tools) ConvertToWAV(ctx context.Context, in, out string, opts WAVOptions, onProgress func(done time.Duration)) error {
	if opts.SampleRate <= 0 {
		opts.SampleRate = 16000
	}
	if opts.Channels <= 0 {
		opts.Channels = 1
	}
	if abs, err := filepath.Abs(in); err == nil {
		if absOut, err := filepath.Abs(out); err == nil && abs == absOut {
			return errors.New("input and output are the same file")
		}
	}
	if err := os.MkdirAll(filepath.Dir(out), 0o755); err != nil {
		return err
	}

	// Write to a sidecar and rename only on success, so an interrupted run never
	// leaves a truncated WAV that later stages would treat as complete.
	tmp := out + ".part"
	_ = os.Remove(tmp)

	args := []string{
		"-nostdin", "-hide_banner", "-loglevel", "error", "-y",
		"-i", in,
		"-vn",                 // drop any video stream
		"-map_metadata", "-1", // keep output byte-identical across runs
		"-ac", strconv.Itoa(opts.Channels),
		"-ar", strconv.Itoa(opts.SampleRate),
		"-c:a", "pcm_s16le",
	}
	if opts.Loudnorm {
		// Single-pass EBU R128. Two-pass measures first and is more accurate,
		// but for clean single-speaker audio the difference is not worth the
		// extra full read of the file.
		args = append(args, "-af", "loudnorm=I=-16:TP=-1.5:LRA=11")
	}
	// -f wav is required because the .part suffix hides the real container.
	args = append(args, "-progress", "pipe:1", "-f", "wav", tmp)

	cmd := exec.CommandContext(ctx, t.FFmpeg, args...)
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return err
	}
	var stderr bytes.Buffer
	cmd.Stderr = &stderr

	if err := cmd.Start(); err != nil {
		return fmt.Errorf("starting ffmpeg: %w", err)
	}

	// -progress writes key=value lines; out_time_us and the misnamed out_time_ms
	// both carry microseconds, so report whichever moves and skip repeats.
	var last time.Duration
	sc := bufio.NewScanner(stdout)
	for sc.Scan() {
		key, val, ok := strings.Cut(sc.Text(), "=")
		if !ok {
			continue
		}
		switch strings.TrimSpace(key) {
		case "out_time_us", "out_time_ms":
			n, err := strconv.ParseInt(strings.TrimSpace(val), 10, 64)
			if err != nil || n < 0 {
				continue
			}
			if d := time.Duration(n) * time.Microsecond; d != last {
				last = d
				if onProgress != nil {
					onProgress(d)
				}
			}
		}
	}

	if err := cmd.Wait(); err != nil {
		_ = os.Remove(tmp)
		if ctxErr := ctx.Err(); ctxErr != nil {
			return ctxErr
		}
		if msg := strings.TrimSpace(stderr.String()); msg != "" {
			return fmt.Errorf("ffmpeg: %s", firstLines(msg, 5))
		}
		return fmt.Errorf("ffmpeg: %w", err)
	}

	if err := os.Rename(tmp, out); err != nil {
		_ = os.Remove(tmp)
		return err
	}
	return nil
}

func firstLines(s string, n int) string {
	lines := strings.Split(s, "\n")
	if len(lines) > n {
		lines = append(lines[:n], "...")
	}
	return strings.Join(lines, "; ")
}

// WAVInfo is what a WAV file's header declares.
type WAVInfo struct {
	SampleRate int
	Channels   int
	Bits       int
	DataBytes  int64
	Duration   time.Duration
}

// Ready reports whether whisper.cpp will accept this file: it reads only
// 16-bit PCM, and Whisper works on 16 kHz mono samples.
func (w WAVInfo) Ready() bool {
	return w.SampleRate == 16000 && w.Channels == 1 && w.Bits == 16
}

// InspectWAV reads a WAV header without decoding the audio.
//
// ffprobe would answer the same question, but this runs on every file in a
// listing, and a subprocess each is not worth it for 44 bytes of header.
func InspectWAV(path string) (WAVInfo, error) {
	var info WAVInfo

	f, err := os.Open(path)
	if err != nil {
		return info, err
	}
	defer f.Close()

	var riff [12]byte
	if _, err := io.ReadFull(f, riff[:]); err != nil {
		return info, fmt.Errorf("reading header: %w", err)
	}
	if string(riff[0:4]) != "RIFF" || string(riff[8:12]) != "WAVE" {
		return info, errors.New("not a RIFF/WAVE file")
	}

	var haveFmt, haveData bool
	for !(haveFmt && haveData) {
		var ch [8]byte
		if _, err := io.ReadFull(f, ch[:]); err != nil {
			break // ran out of chunks
		}
		id := string(ch[0:4])
		size := int64(binary.LittleEndian.Uint32(ch[4:8]))

		switch id {
		case "fmt ":
			n := size
			if n > 40 {
				n = 40 // PCM fmt chunks are 16, 18 or 40 bytes
			}
			buf := make([]byte, n)
			if _, err := io.ReadFull(f, buf); err != nil {
				return info, fmt.Errorf("truncated fmt chunk: %w", err)
			}
			if len(buf) < 16 {
				return info, errors.New("short fmt chunk")
			}
			info.Channels = int(binary.LittleEndian.Uint16(buf[2:4]))
			info.SampleRate = int(binary.LittleEndian.Uint32(buf[4:8]))
			info.Bits = int(binary.LittleEndian.Uint16(buf[14:16]))
			haveFmt = true
			if rest := size - n; rest > 0 {
				if _, err := f.Seek(rest, io.SeekCurrent); err != nil {
					return info, err
				}
			}
		case "data":
			info.DataBytes = size
			haveData = true
			if _, err := f.Seek(size, io.SeekCurrent); err != nil {
				// A truncated or streamed file: fall back to what is actually there.
				if fi, statErr := f.Stat(); statErr == nil {
					if pos, posErr := f.Seek(0, io.SeekCurrent); posErr == nil {
						info.DataBytes = fi.Size() - pos
					}
				}
			}
		default:
			if _, err := f.Seek(size, io.SeekCurrent); err != nil {
				return info, err
			}
		}

		// RIFF chunks are word-aligned; an odd size carries a pad byte.
		if size%2 == 1 {
			if _, err := f.Seek(1, io.SeekCurrent); err != nil {
				break
			}
		}
	}

	if !haveFmt {
		return info, errors.New("no fmt chunk")
	}
	// Some writers store 0 or 0xFFFFFFFF for a streamed length.
	if fi, err := f.Stat(); err == nil {
		if info.DataBytes <= 0 || info.DataBytes > fi.Size() {
			info.DataBytes = fi.Size() - 44
		}
	}
	if bps := info.SampleRate * info.Channels * info.Bits / 8; bps > 0 && info.DataBytes > 0 {
		info.Duration = time.Duration(float64(info.DataBytes) / float64(bps) * float64(time.Second))
	}
	return info, nil
}

// SliceWAV extracts a time range from a PCM WAV as a standalone WAV file.
//
// No decoding and no ffmpeg call: in linear PCM a byte offset is a time offset,
// so a span is found by arithmetic and copied out. That is what makes clip
// playback in the review editor instant, which is what makes reviewing quick
// enough to be worth doing.
func SliceWAV(path string, start, end time.Duration) ([]byte, error) {
	info, err := InspectWAV(path)
	if err != nil {
		return nil, err
	}
	blockAlign := info.Channels * info.Bits / 8
	byteRate := info.SampleRate * blockAlign
	if byteRate <= 0 {
		return nil, errors.New("unusable WAV header")
	}
	if start < 0 {
		start = 0
	}
	if end <= start {
		return nil, errors.New("empty range")
	}

	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	// Locate the data chunk rather than assuming the canonical 44-byte header.
	dataOff, dataLen, err := findDataChunk(f)
	if err != nil {
		return nil, err
	}

	toBytes := func(d time.Duration) int64 {
		b := int64(d.Seconds() * float64(byteRate))
		return b - b%int64(blockAlign) // never cut mid-sample
	}
	from, to := toBytes(start), toBytes(end)
	if from > dataLen {
		return nil, errors.New("range starts past the end of the audio")
	}
	if to > dataLen {
		to = dataLen
	}

	pcm := make([]byte, to-from)
	if _, err := f.ReadAt(pcm, dataOff+from); err != nil && err != io.EOF {
		return nil, err
	}
	return buildWAV(info, pcm), nil
}

func findDataChunk(f *os.File) (offset, length int64, err error) {
	var riff [12]byte
	if _, err := io.ReadFull(f, riff[:]); err != nil {
		return 0, 0, err
	}
	pos := int64(12)
	for {
		var ch [8]byte
		if _, err := f.ReadAt(ch[:], pos); err != nil {
			return 0, 0, errors.New("no data chunk")
		}
		size := int64(binary.LittleEndian.Uint32(ch[4:8]))
		if string(ch[0:4]) == "data" {
			return pos + 8, size, nil
		}
		pos += 8 + size
		if size%2 == 1 {
			pos++
		}
	}
}

func buildWAV(info WAVInfo, pcm []byte) []byte {
	blockAlign := info.Channels * info.Bits / 8
	byteRate := info.SampleRate * blockAlign

	out := make([]byte, 44, 44+len(pcm))
	copy(out[0:4], "RIFF")
	binary.LittleEndian.PutUint32(out[4:8], uint32(36+len(pcm)))
	copy(out[8:12], "WAVE")
	copy(out[12:16], "fmt ")
	binary.LittleEndian.PutUint32(out[16:20], 16)
	binary.LittleEndian.PutUint16(out[20:22], 1) // PCM
	binary.LittleEndian.PutUint16(out[22:24], uint16(info.Channels))
	binary.LittleEndian.PutUint32(out[24:28], uint32(info.SampleRate))
	binary.LittleEndian.PutUint32(out[28:32], uint32(byteRate))
	binary.LittleEndian.PutUint16(out[32:34], uint16(blockAlign))
	binary.LittleEndian.PutUint16(out[34:36], uint16(info.Bits))
	copy(out[36:40], "data")
	binary.LittleEndian.PutUint32(out[40:44], uint32(len(pcm)))
	return append(out, pcm...)
}
