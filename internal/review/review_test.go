package review

import (
	"path/filepath"
	"testing"
)

// findTranscript returns a real engine transcript to test against, if one has
// been produced. There is no fixture in the repo: transcripts are large, and
// the properties worth checking are about real output.
func findTranscript(t *testing.T) string {
	t.Helper()
	all, _ := filepath.Glob("../../work/transcripts/*/*.json")
	for _, p := range all {
		if filepath.Ext(p) == ".json" && !hasSuffix(p, ".review.json") {
			return p
		}
	}
	t.Skip("no transcript in work/transcripts to test against")
	return ""
}

func hasSuffix(s, suf string) bool {
	return len(s) >= len(suf) && s[len(s)-len(suf):] == suf
}

// The review queue and the underlined words in the document are produced by two
// different functions. If they ever disagree, the reviewer sees a count that
// does not match what is marked on screen, so they must stay in step.
func TestDetectAndViewAgree(t *testing.T) {
	path := findTranscript(t)
	st, err := LoadState(path)
	if err != nil {
		t.Fatalf("LoadState: %v", err)
	}

	for _, th := range []float64{0.2, 0.5, 0.8} {
		opts := DefaultOptions()
		opts.Threshold = th

		spans, err := Detect(path, opts)
		if err != nil {
			t.Fatalf("Detect at %.1f: %v", th, err)
		}
		inQueue := map[string]bool{}
		for _, s := range spans {
			inQueue[s.ID] = true
		}

		segs, err := View(path, opts, st)
		if err != nil {
			t.Fatalf("View at %.1f: %v", th, err)
		}
		inDoc := map[string]bool{}
		for _, seg := range segs {
			if seg.Edited {
				continue // a rewritten line drops its word flags on purpose
			}
			for _, w := range seg.Words {
				if w.SpanID != "" {
					inDoc[w.SpanID] = true
				}
			}
		}

		for id := range inQueue {
			if !inDoc[id] {
				t.Errorf("threshold %.1f: %s is in the queue but not underlined", th, id)
			}
		}
		for id := range inDoc {
			if !inQueue[id] {
				t.Errorf("threshold %.1f: %s is underlined but not in the queue", th, id)
			}
		}
		t.Logf("threshold %.1f: %d flagged, queue and document agree", th, len(inQueue))
	}
}

// A lower cutoff must never flag more words than a higher one.
func TestThresholdIsMonotonic(t *testing.T) {
	path := findTranscript(t)
	prev := -1
	for _, th := range []float64{0.1, 0.3, 0.5, 0.7, 0.9} {
		opts := DefaultOptions()
		opts.Threshold = th
		spans, err := Detect(path, opts)
		if err != nil {
			t.Fatalf("Detect at %.1f: %v", th, err)
		}
		if prev >= 0 && len(spans) < prev {
			t.Errorf("threshold %.1f flagged %d, fewer than the stricter cutoff's %d", th, len(spans), prev)
		}
		prev = len(spans)
	}
}

// Every clip window must contain the word it was built for.
func TestClipContainsItsWord(t *testing.T) {
	path := findTranscript(t)
	spans, err := Detect(path, DefaultOptions())
	if err != nil {
		t.Fatalf("Detect: %v", err)
	}
	for _, s := range spans {
		if s.ClipStart > s.Start || s.ClipEnd < s.End {
			t.Errorf("%s: clip %.2f-%.2f does not contain the word at %.2f-%.2f",
				s.ID, s.ClipStart, s.ClipEnd, s.Start, s.End)
		}
	}
}
