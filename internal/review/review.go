// Package review turns an engine transcript into a ranked list of spans worth
// a human's attention, and records what the human decided.
//
// This is Stage 3 of the pipeline in its first form: word-level aggregation,
// noise filtering and confidence ranking. The lexicon (FST) and second-model
// disagreement detectors slot in beside them later without changing the shape
// of a Span.
package review

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"time"
)

// --- the engine's JSON ------------------------------------------------------

type wOffsets struct {
	From int64 `json:"from"` // milliseconds
	To   int64 `json:"to"`
}

type wToken struct {
	Text    string   `json:"text"`
	Offsets wOffsets `json:"offsets"`
	P       float64  `json:"p"`
}

type wSegment struct {
	Offsets wOffsets `json:"offsets"`
	Text    string   `json:"text"`
	Tokens  []wToken `json:"tokens"`
}

type wTranscript struct {
	Transcription []wSegment `json:"transcription"`
}

// --- spans ------------------------------------------------------------------

// Span is one word the detectors think is worth checking.
type Span struct {
	ID      string `json:"id"`
	Segment int    `json:"segment"`
	Word    int    `json:"word"`

	Text  string  `json:"text"`  // the flagged word
	Start float64 `json:"start"` // seconds
	End   float64 `json:"end"`

	// ClipStart/ClipEnd bound the audio to play. They are snapped to whole
	// words rather than a fixed time pad: starting mid-word gives the listener
	// a fragment of a syllable, which is harder to follow than no context.
	ClipStart float64 `json:"clipStart"`
	ClipEnd   float64 `json:"clipEnd"`

	// Context for display: the sentence, split around the flagged word so the
	// UI can highlight it without doing string matching.
	Before string `json:"before"`
	After  string `json:"after"`

	Confidence float64  `json:"confidence"` // minimum across the word's tokens
	Rank       float64  `json:"rank"`       // adjusted score, lower is worse
	Flags      []string `json:"flags"`

	// Filled in from saved review state.
	Decision  string `json:"decision,omitempty"`
	Corrected string `json:"corrected,omitempty"`
}

// Options tunes detection.
// Order decides the sequence spans are presented in.
type Order string

const (
	// ByDocument follows the recording. Consecutive spans then share context,
	// so the reviewer stays inside one train of thought instead of being thrown
	// around the timeline, and the audio clips overlap into something coherent.
	ByDocument Order = "document"
	// ByRank puts the least confident first, for a quick pass at the worst.
	ByRank Order = "rank"
)

type Options struct {
	Threshold float64 // flag words scoring below this
	Max       int     // cap the list; 0 means no cap
	Order     Order
}

func DefaultOptions() Options { return Options{Threshold: 0.5, Max: 0, Order: ByDocument} }

// Fillers are removed by the readability stage anyway, so flagging them spends
// review time on words that will not survive to the output.
var fillers = map[string]bool{
	"yani": true, "işte": true, "hani": true, "şey": true,
	"ııı": true, "eee": true, "ee": true, "ıı": true, "hı": true,
}

var wordish = regexp.MustCompile(`\p{L}|\p{N}`)

type word struct {
	text    string
	start   float64
	end     float64
	minP    float64
	initial bool // first word of its segment
	index   int  // position within its segment
	seg     int
}

// splitWords groups whisper's subword tokens back into words.
//
// A word's confidence is the MINIMUM across its pieces, not the mean: Turkish
// agglutination puts errors in the suffix, so "Bağ"+"lık" scores 0.999 and
// 0.361, and averaging would hide the only part that was wrong.
func splitWords(seg wSegment) []word {
	var words []word
	cur := word{minP: 1}
	started := false

	flush := func() {
		if started && strings.TrimSpace(cur.text) != "" {
			cur.text = strings.TrimSpace(cur.text)
			cur.index = len(words)
			cur.initial = len(words) == 0
			words = append(words, cur)
		}
	}

	for _, t := range seg.Tokens {
		if strings.HasPrefix(t.Text, "[_") { // special tokens
			continue
		}
		// whisper marks a word boundary with a leading space on the token.
		if strings.HasPrefix(t.Text, " ") || !started {
			flush()
			cur = word{minP: 1, start: float64(t.Offsets.From) / 1000}
			started = true
		}
		cur.text += t.Text
		cur.end = float64(t.Offsets.To) / 1000
		if t.P < cur.minP {
			cur.minP = t.P
		}
	}
	flush()
	return words
}

// Words of context to include on each side of a flagged word, and the floor on
// clip length so a short word in a fast passage still gets something to hear.
const (
	contextWords = 2
	edgePad      = 0.15 // seconds, so consonant onsets are not clipped
	minClip      = 2.5  // seconds
	// trailPad is how much audio must follow the flagged word. What comes next
	// often disambiguates it -- a suffix, or the word it agrees with -- so the
	// clip keeps running past it rather than stopping on it.
	trailPad = 5.0
)

// Detect reads a transcript and returns the spans worth reviewing, worst first.
func Detect(transcriptPath string, opts Options) ([]Span, error) {
	raw, err := os.ReadFile(transcriptPath)
	if err != nil {
		return nil, err
	}
	var doc wTranscript
	if err := json.Unmarshal(raw, &doc); err != nil {
		return nil, err
	}
	if opts.Threshold <= 0 {
		opts.Threshold = 0.5
	}

	base := strings.TrimSuffix(filepath.Base(transcriptPath), filepath.Ext(transcriptPath))

	// Token timestamps are relied on directly: transcription always passes -dtw,
	// without which whisper.cpp's offsets are interpolated guesses that drift by
	// seconds and put review clips nowhere near the word.
	//
	// One flat list across the whole transcript. Segment boundaries are where
	// context is needed most -- a segment-initial word has none within its own
	// segment -- so neighbours are taken globally, not per segment.
	var all []word
	perSeg := make([][]word, len(doc.Transcription))
	for si, seg := range doc.Transcription {
		ws := splitWords(seg)
		for i := range ws {
			ws[i].seg = si
		}
		perSeg[si] = ws
		all = append(all, ws...)
	}

	// Where each global index sits, so a flagged word can find its neighbours.
	globalOf := map[[2]int]int{}
	for gi, w := range all {
		globalOf[[2]int{w.seg, w.index}] = gi
	}

	var spans []Span
	for si, words := range perSeg {
		for wi, w := range words {
			bare := strings.Trim(w.text, ".,!?;:\"'()[]…-–—")

			// Punctuation-only tokens were 15% of the low-confidence tail and
			// are never worth a human's time.
			if !wordish.MatchString(bare) {
				continue
			}
			if fillers[strings.ToLower(bare)] {
				continue
			}
			if w.minP >= opts.Threshold {
				continue
			}

			// Segment-initial words are measurably noisier (2.30% vs 1.12%
			// below 0.5) without being more often wrong, because the sentence
			// boundary itself is what the model is unsure about. Down-weight
			// rather than exclude: they were still only 14% of the tail.
			rank := w.minP
			flags := []string{"low_confidence"}
			if w.initial {
				rank *= 1.4
				flags = append(flags, "segment_initial")
			}

			clipStart, clipEnd := clipBounds(all, globalOf[[2]int{si, wi}])

			spans = append(spans, Span{
				ID:         base + ":" + itoa(si) + ":" + itoa(wi),
				Segment:    si,
				Word:       wi,
				Text:       w.text,
				Start:      w.start,
				End:        w.end,
				ClipStart:  clipStart,
				ClipEnd:    clipEnd,
				Before:     strings.Join(wordsText(words[:wi]), " "),
				After:      strings.Join(wordsText(words[wi+1:]), " "),
				Confidence: w.minP,
				Rank:       rank,
				Flags:      flags,
			})
		}
	}

	if opts.Order == ByRank {
		sort.SliceStable(spans, func(i, j int) bool { return spans[i].Rank < spans[j].Rank })
	} else {
		// Spans are built in reading order already; sort explicitly so that
		// stays true if detection ever grows a second pass.
		sort.SliceStable(spans, func(i, j int) bool {
			if spans[i].Segment != spans[j].Segment {
				return spans[i].Segment < spans[j].Segment
			}
			return spans[i].Word < spans[j].Word
		})
	}
	if opts.Max > 0 && len(spans) > opts.Max {
		spans = spans[:opts.Max]
	}
	return spans, nil
}

// clipBounds snaps playback to whole words around index gi, then guarantees
// enough audio after the flagged word to hear how the sentence continues.
func clipBounds(all []word, gi int) (float64, float64) {
	if len(all) == 0 {
		return 0, 0
	}
	if gi < 0 || gi >= len(all) {
		gi = 0
	}
	lo, hi := gi-contextWords, gi+contextWords
	if lo < 0 {
		lo = 0
	}
	if hi > len(all)-1 {
		hi = len(all) - 1
	}

	startAt := func(i int) float64 {
		v := all[i].start - edgePad
		if v < 0 {
			return 0
		}
		return v
	}
	start, end := startAt(lo), all[hi].end+edgePad

	// Keep taking whole following words until the trailing window is covered.
	needEnd := all[gi].end + trailPad
	for end < needEnd && hi < len(all)-1 {
		hi++
		end = all[hi].end + edgePad
	}
	if end < needEnd {
		end = needEnd // ran out of transcript; extend on time alone
	}

	// Widen by whole words until the clip is long enough to follow.
	for end-start < minClip && (lo > 0 || hi < len(all)-1) {
		if lo > 0 {
			lo--
			start = startAt(lo)
		}
		if end-start >= minClip {
			break
		}
		if hi < len(all)-1 {
			hi++
			end = all[hi].end + edgePad
		}
	}
	return start, end
}

func wordsText(ws []word) []string {
	out := make([]string, 0, len(ws))
	for _, w := range ws {
		out = append(out, w.text)
	}
	return out
}

func itoa(n int) string {
	if n == 0 {
		return "0"
	}
	var b []byte
	for n > 0 {
		b = append([]byte{byte('0' + n%10)}, b...)
		n /= 10
	}
	return string(b)
}

// Aligned reports whether a transcript's token timestamps can be trusted.
//
// This is a diagnostic, not a fallback: a transcript produced without -dtw has
// token offsets that are interpolated guesses drifting by seconds, so review
// clips play the wrong part of the recording while the on-screen timestamp,
// which comes from the segment, still looks right. Detecting it lets the UI say
// so instead of quietly playing the wrong audio.
//
// The test is the invariant that must hold: a token lies inside its segment.
func Aligned(transcriptPath string) bool {
	raw, err := os.ReadFile(transcriptPath)
	if err != nil {
		return false
	}
	var doc wTranscript
	if err := json.Unmarshal(raw, &doc); err != nil {
		return false
	}
	var total, inside int
	for _, seg := range doc.Transcription {
		lo := float64(seg.Offsets.From)/1000 - 0.05
		hi := float64(seg.Offsets.To)/1000 + 0.05
		for _, t := range seg.Tokens {
			if strings.HasPrefix(t.Text, "[_") {
				continue
			}
			total++
			if f, to := float64(t.Offsets.From)/1000, float64(t.Offsets.To)/1000; f >= lo && to <= hi {
				inside++
			}
		}
	}
	return total > 0 && float64(inside)/float64(total) >= 0.9
}

// --- saved decisions --------------------------------------------------------

// Decision is what a reviewer concluded about one span.
type Decision struct {
	Decision  string    `json:"decision"` // accepted | edited | not_an_error | skipped
	Corrected string    `json:"corrected,omitempty"`
	At        time.Time `json:"at"`
}

// SegmentEdit replaces a whole line. Word-level fixes cannot help when the
// error spans several words, or when the engine dropped or invented one, so a
// reviewer can rewrite the line outright. An empty string removes the line.
type SegmentEdit struct {
	Text string    `json:"text"`
	At   time.Time `json:"at"`
}

// State is the review progress for one transcript, stored beside it so that
// deleting a transcript takes its review with it.
type State struct {
	Decisions map[string]Decision    `json:"decisions"`
	Segments  map[string]SegmentEdit `json:"segments,omitempty"`
}

func statePath(transcriptPath string) string {
	return strings.TrimSuffix(transcriptPath, filepath.Ext(transcriptPath)) + ".review.json"
}

// LoadState reads saved decisions; an absent file is an empty state, not an error.
func LoadState(transcriptPath string) (State, error) {
	st := State{Decisions: map[string]Decision{}}
	raw, err := os.ReadFile(statePath(transcriptPath))
	if err != nil {
		if os.IsNotExist(err) {
			return st, nil
		}
		return st, err
	}
	if err := json.Unmarshal(raw, &st); err != nil {
		return State{Decisions: map[string]Decision{}}, err
	}
	if st.Decisions == nil {
		st.Decisions = map[string]Decision{}
	}
	if st.Segments == nil {
		st.Segments = map[string]SegmentEdit{}
	}
	return st, nil
}

// SaveSegment records a rewritten line, or clears the rewrite when text is the
// sentinel "\x00" so the engine's own words come back.
func SaveSegment(transcriptPath string, segment int, text string) error {
	st, err := LoadState(transcriptPath)
	if err != nil {
		return err
	}
	key := itoa(segment)
	if text == "\x00" {
		delete(st.Segments, key)
	} else {
		st.Segments[key] = SegmentEdit{Text: text, At: time.Now().UTC()}
	}
	return writeState(transcriptPath, st)
}

// SaveDecision records one decision, rewriting the state file atomically so an
// interrupted save cannot lose the whole review.
func SaveDecision(transcriptPath, spanID string, d Decision) error {
	st, err := LoadState(transcriptPath)
	if err != nil {
		return err
	}
	if d.At.IsZero() {
		d.At = time.Now().UTC()
	}
	st.Decisions[spanID] = d
	return writeState(transcriptPath, st)
}

// writeState rewrites the file atomically, so an interrupted save cannot lose
// the whole review.
func writeState(transcriptPath string, st State) error {
	raw, err := json.MarshalIndent(st, "", "  ")
	if err != nil {
		return err
	}
	p := statePath(transcriptPath)
	tmp := p + ".tmp"
	if err := os.WriteFile(tmp, raw, 0o644); err != nil {
		return err
	}
	return os.Rename(tmp, p)
}

// Apply merges saved decisions into a span list.
func Apply(spans []Span, st State) []Span {
	for i := range spans {
		if d, ok := st.Decisions[spans[i].ID]; ok {
			spans[i].Decision = d.Decision
			spans[i].Corrected = d.Corrected
		}
	}
	return spans
}

// CorrectedText rebuilds the transcript with every accepted edit applied,
// working from the token offsets so that timestamps stay intact.
func CorrectedText(transcriptPath string, st State) (string, error) {
	raw, err := os.ReadFile(transcriptPath)
	if err != nil {
		return "", err
	}
	var doc wTranscript
	if err := json.Unmarshal(raw, &doc); err != nil {
		return "", err
	}
	base := strings.TrimSuffix(filepath.Base(transcriptPath), filepath.Ext(transcriptPath))

	var out []string
	for si, seg := range doc.Transcription {
		words := splitWords(seg)
		parts := make([]string, 0, len(words))
		for wi, w := range words {
			text := w.text
			if d, ok := st.Decisions[base+":"+itoa(si)+":"+itoa(wi)]; ok &&
				d.Decision == "edited" && strings.TrimSpace(d.Corrected) != "" {
				text = strings.TrimSpace(d.Corrected)
			}
			parts = append(parts, text)
		}
		out = append(out, strings.Join(parts, " "))
	}
	return strings.Join(out, "\n"), nil
}

// --- full-text view ---------------------------------------------------------

// ViewWord is one word as rendered in the read-through view.
type ViewWord struct {
	Text       string  `json:"text"`
	SpanID     string  `json:"spanId,omitempty"` // set only on flagged words
	Confidence float64 `json:"confidence,omitempty"`
	Decision   string  `json:"decision,omitempty"`
	Edited     bool    `json:"edited,omitempty"`
}

// ViewSegment is one timestamped line.
type ViewSegment struct {
	Index  int        `json:"index"`
	Start  float64    `json:"start"`
	End    float64    `json:"end"`
	Words  []ViewWord `json:"words"`
	Text   string     `json:"text"`             // the line as it currently reads
	Edited bool       `json:"edited,omitempty"` // rewritten by hand
}

// View returns the whole transcript word by word, with the flagged words
// marked and any accepted corrections already substituted, so the reader sees
// the text as it currently stands rather than as the engine first produced it.
func View(transcriptPath string, opts Options, st State) ([]ViewSegment, error) {
	raw, err := os.ReadFile(transcriptPath)
	if err != nil {
		return nil, err
	}
	var doc wTranscript
	if err := json.Unmarshal(raw, &doc); err != nil {
		return nil, err
	}
	if opts.Threshold <= 0 {
		opts.Threshold = 0.5
	}
	base := strings.TrimSuffix(filepath.Base(transcriptPath), filepath.Ext(transcriptPath))

	out := make([]ViewSegment, 0, len(doc.Transcription))
	for si, seg := range doc.Transcription {
		words := splitWords(seg)
		vs := ViewSegment{
			Index: si,
			Start: float64(seg.Offsets.From) / 1000,
			End:   float64(seg.Offsets.To) / 1000,
			Words: make([]ViewWord, 0, len(words)),
		}

		// A rewritten line replaces the engine's words wholesale. Word-level
		// flags are dropped for it: their indices no longer describe anything.
		if e, ok := st.Segments[itoa(si)]; ok {
			vs.Edited = true
			vs.Text = e.Text
			for _, w := range strings.Fields(e.Text) {
				vs.Words = append(vs.Words, ViewWord{Text: w, Edited: true})
			}
			out = append(out, vs)
			continue
		}
		for wi, w := range words {
			vw := ViewWord{Text: w.text}
			id := base + ":" + itoa(si) + ":" + itoa(wi)

			bare := strings.Trim(w.text, ".,!?;:\"'()[]…-–—")
			flagged := wordish.MatchString(bare) &&
				!fillers[strings.ToLower(bare)] &&
				w.minP < opts.Threshold

			if d, ok := st.Decisions[id]; ok {
				vw.Decision = d.Decision
				if d.Decision == "edited" && strings.TrimSpace(d.Corrected) != "" {
					vw.Text = strings.TrimSpace(d.Corrected)
					vw.Edited = true
				}
			}
			if flagged {
				vw.SpanID = id
				vw.Confidence = w.minP
			}
			vs.Words = append(vs.Words, vw)
		}
		vs.Text = segmentText(vs)
		out = append(out, vs)
	}
	return out, nil
}

// --- export -----------------------------------------------------------------

func srtTime(sec float64) string {
	if sec < 0 {
		sec = 0
	}
	ms := int64(sec*1000 + 0.5)
	h := ms / 3600000
	m := (ms % 3600000) / 60000
	s := (ms % 60000) / 1000
	return fmt.Sprintf("%02d:%02d:%02d,%03d", h, m, s, ms%1000)
}

// Export renders the transcript with every correction applied.
//
// It is built from View, so what is exported is exactly what the reviewer sees:
// there is no second code path that could drift from the on-screen text.
func Export(transcriptPath string, st State, format string) ([]byte, error) {
	segs, err := View(transcriptPath, DefaultOptions(), st)
	if err != nil {
		return nil, err
	}

	var b strings.Builder
	switch format {
	case "srt":
		n := 0
		for _, seg := range segs {
			text := segmentText(seg)
			if text == "" {
				continue // an empty cue is invalid SRT
			}
			n++
			fmt.Fprintf(&b, "%d\n%s --> %s\n%s\n\n",
				n, srtTime(seg.Start), srtTime(seg.End), text)
		}
	case "txt":
		for _, seg := range segs {
			if text := segmentText(seg); text != "" {
				b.WriteString(text)
				b.WriteByte('\n')
			}
		}
	default:
		return nil, fmt.Errorf("unknown format %q", format)
	}
	return []byte(b.String()), nil
}

func segmentText(seg ViewSegment) string {
	parts := make([]string, 0, len(seg.Words))
	for _, w := range seg.Words {
		if t := strings.TrimSpace(w.Text); t != "" {
			parts = append(parts, t)
		}
	}
	return strings.Join(parts, " ")
}
