// Package dataset collects the labelled corrections a reviewer produces and
// packages them for the maintainer to fine-tune on.
//
// A record pairs a short audio clip with the text a human confirmed against it,
// which is the standard shape for ASR training data. It comes from real target
// domain speech rather than read-aloud corpora, which is what makes it worth
// collecting at all.
package dataset

import (
	"archive/zip"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"
)

// Decision is what the reviewer concluded about a flagged span.
type Decision string

const (
	Accepted   Decision = "accepted"     // engine output was right
	Edited     Decision = "edited"       // reviewer rewrote it
	NotAnError Decision = "not_an_error" // the detector was wrong to flag it
	Skipped    Decision = "skipped"
)

// Record is one reviewed span.
type Record struct {
	ID        string    `json:"id"`
	CreatedAt time.Time `json:"created_at"`

	Video string  `json:"video"` // source WAV stem
	Model string  `json:"model"`
	Start float64 `json:"start"` // seconds into the source audio
	End   float64 `json:"end"`

	Original  string   `json:"original"`
	Corrected string   `json:"corrected"`
	Decision  Decision `json:"decision"`

	// Why the span was flagged, and how sure the engine was. Kept so detector
	// precision can be measured per flag type rather than in aggregate.
	Flags      []string `json:"flags,omitempty"`
	Confidence float64  `json:"confidence"`
	Alternate  string   `json:"alternate,omitempty"` // second model's reading

	Clip string `json:"clip,omitempty"` // file name inside clips/
}

// Store is the on-disk dataset.
type Store struct {
	dir string
}

const (
	recordsFile = "dataset.jsonl"
	clipsDir    = "clips"
)

// Open prepares the dataset directory.
func Open(dir string) (*Store, error) {
	if err := os.MkdirAll(filepath.Join(dir, clipsDir), 0o755); err != nil {
		return nil, err
	}
	return &Store{dir: dir}, nil
}

func (s *Store) Dir() string { return s.dir }

// Append writes one record. Records are append-only JSONL so a crash can never
// corrupt more than the last line, and so the file stays readable by hand.
func (s *Store) Append(r Record) error {
	if r.CreatedAt.IsZero() {
		r.CreatedAt = time.Now().UTC()
	}
	if r.ID == "" {
		r.ID = fmt.Sprintf("%d", time.Now().UnixNano())
	}
	line, err := json.Marshal(r)
	if err != nil {
		return err
	}
	f, err := os.OpenFile(filepath.Join(s.dir, recordsFile),
		os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o644)
	if err != nil {
		return err
	}
	defer f.Close()
	_, err = f.Write(append(line, '\n'))
	return err
}

// SaveClip stores the audio for a record and returns the name to reference.
func (s *Store) SaveClip(name string, wav []byte) (string, error) {
	name = filepath.Base(name)
	if err := os.WriteFile(filepath.Join(s.dir, clipsDir, name), wav, 0o644); err != nil {
		return "", err
	}
	return name, nil
}

// Summary is what the UI shows before anyone downloads anything.
type Summary struct {
	Records     int      `json:"records"`
	Clips       int      `json:"clips"`
	Transcripts int      `json:"transcripts"`
	Videos      []string `json:"videos"`
	Bytes       int64    `json:"bytes"`
	Empty       bool     `json:"empty"`
}

// Summarise reports what an export would contain.
func (s *Store) Summarise(transcriptDir string) Summary {
	var sum Summary

	seen := map[string]bool{}
	if f, err := os.Open(filepath.Join(s.dir, recordsFile)); err == nil {
		defer f.Close()
		dec := json.NewDecoder(f)
		for {
			var r Record
			if err := dec.Decode(&r); err != nil {
				break
			}
			sum.Records++
			if r.Video != "" && !seen[r.Video] {
				seen[r.Video] = true
				sum.Videos = append(sum.Videos, r.Video)
			}
		}
		if fi, err := os.Stat(filepath.Join(s.dir, recordsFile)); err == nil {
			sum.Bytes += fi.Size()
		}
	}
	sort.Strings(sum.Videos)

	if entries, err := os.ReadDir(filepath.Join(s.dir, clipsDir)); err == nil {
		for _, e := range entries {
			if e.IsDir() {
				continue
			}
			sum.Clips++
			if fi, err := e.Info(); err == nil {
				sum.Bytes += fi.Size()
			}
		}
	}

	for _, f := range transcriptFiles(transcriptDir) {
		sum.Transcripts++
		if fi, err := os.Stat(f); err == nil {
			sum.Bytes += fi.Size()
		}
	}

	sum.Empty = sum.Records == 0 && sum.Transcripts == 0
	return sum
}

// transcriptFiles lists the engine's JSON output, which carries the per-token
// probabilities and so is useful for analysis even before any review happens.
func transcriptFiles(root string) []string {
	var out []string
	if root == "" {
		return out
	}
	_ = filepath.WalkDir(root, func(path string, d os.DirEntry, err error) error {
		if err != nil {
			return nil
		}
		if !d.IsDir() && strings.EqualFold(filepath.Ext(path), ".json") {
			out = append(out, path)
		}
		return nil
	})
	sort.Strings(out)
	return out
}

// WriteZip streams the export. It is written straight to the response rather
// than staged on disk, so a large dataset costs no extra space.
func (s *Store) WriteZip(w io.Writer, transcriptDir, note string) error {
	zw := zip.NewWriter(w)
	defer zw.Close()

	sum := s.Summarise(transcriptDir)

	if err := addBytes(zw, "README.txt", []byte(readme(sum, note))); err != nil {
		return err
	}
	manifest, _ := json.MarshalIndent(map[string]any{
		"exported_at": time.Now().UTC().Format(time.RFC3339),
		"records":     sum.Records,
		"clips":       sum.Clips,
		"transcripts": sum.Transcripts,
		"videos":      sum.Videos,
		"note":        note,
	}, "", "  ")
	if err := addBytes(zw, "manifest.json", manifest); err != nil {
		return err
	}

	if err := addFile(zw, filepath.Join(s.dir, recordsFile), "corrections/dataset.jsonl"); err != nil {
		return err
	}
	if entries, err := os.ReadDir(filepath.Join(s.dir, clipsDir)); err == nil {
		for _, e := range entries {
			if e.IsDir() {
				continue
			}
			src := filepath.Join(s.dir, clipsDir, e.Name())
			if err := addFile(zw, src, "corrections/clips/"+e.Name()); err != nil {
				return err
			}
		}
	}

	for _, f := range transcriptFiles(transcriptDir) {
		rel, err := filepath.Rel(transcriptDir, f)
		if err != nil {
			rel = filepath.Base(f)
		}
		if err := addFile(zw, f, "transcripts/"+filepath.ToSlash(rel)); err != nil {
			return err
		}
	}
	return nil
}

func addBytes(zw *zip.Writer, name string, data []byte) error {
	w, err := zw.Create(name)
	if err != nil {
		return err
	}
	_, err = w.Write(data)
	return err
}

// addFile skips anything missing rather than failing the whole export.
func addFile(zw *zip.Writer, src, name string) error {
	f, err := os.Open(src)
	if err != nil {
		if os.IsNotExist(err) {
			return nil
		}
		return err
	}
	defer f.Close()
	w, err := zw.Create(name)
	if err != nil {
		return err
	}
	_, err = io.Copy(w, f)
	return err
}

func readme(sum Summary, note string) string {
	var b strings.Builder
	fmt.Fprintf(&b, "Turkish transcript pipeline - dataset export\n")
	fmt.Fprintf(&b, "exported %s\n\n", time.Now().UTC().Format("2006-01-02 15:04 UTC"))
	fmt.Fprintf(&b, "Contents\n")
	fmt.Fprintf(&b, "  corrections/dataset.jsonl   %d reviewed spans\n", sum.Records)
	fmt.Fprintf(&b, "  corrections/clips/          %d audio clips (16 kHz mono WAV)\n", sum.Clips)
	fmt.Fprintf(&b, "  transcripts/                %d engine outputs, with token probabilities\n", sum.Transcripts)
	if len(sum.Videos) > 0 {
		fmt.Fprintf(&b, "\nSources\n")
		for _, v := range sum.Videos {
			fmt.Fprintf(&b, "  %s\n", v)
		}
	}
	if strings.TrimSpace(note) != "" {
		fmt.Fprintf(&b, "\nNote from the sender\n  %s\n", strings.TrimSpace(note))
	}
	fmt.Fprintf(&b, `
Each line of dataset.jsonl pairs a clip with the text a human confirmed against
it after listening. "decision" says what the reviewer did:

  accepted      the engine was already right
  edited        the reviewer corrected it; "corrected" holds the true text
  not_an_error  the detector flagged it wrongly
  skipped       left undecided

"flags" records why the span was flagged, so detector precision can be measured
per flag type rather than only in aggregate.
`)
	return b.String()
}
