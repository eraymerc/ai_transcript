package asr

import (
	"strings"
)

// Catalog describes the models this program knows about.
//
// A ggml file name says nothing about who made it, what it was trained on or
// how it is licensed, and those are exactly the things someone choosing between
// two 500 MB files needs to know. The catalog supplies them, and also lists
// models that are not installed yet so they can be offered rather than
// discovered by accident.
type CatalogEntry struct {
	// File is the name on disk once installed.
	File string `json:"file"`

	Display   string `json:"display"`
	Publisher string `json:"publisher"`
	License   string `json:"license"`
	Notes     string `json:"notes"`

	// Turkish word error rate, where a comparable published figure exists, and
	// where it came from. Numbers from different eval sets are not comparable,
	// so the source is always named alongside.
	WER       float64 `json:"wer,omitempty"`
	WERSource string  `json:"werSource,omitempty"`

	// Speed relative to the others in the same benchmark: lower is faster.
	RTF float64 `json:"rtf,omitempty"`

	Bytes int64 `json:"bytes,omitempty"`

	// URL is a direct download of the ready ggml file, when one exists.
	URL string `json:"url,omitempty"`
	// Source is the upstream repository, for models that must be converted.
	Source string `json:"source,omitempty"`
	// Install is the command that produces the file when there is no direct
	// download; an empty string means the URL can simply be fetched.
	Install string `json:"install,omitempty"`

	VAD bool `json:"vad,omitempty"`

	// Recommended marks the model a user should pick unless they have a reason
	// not to. Exactly one transcription model should carry it.
	Recommended bool `json:"recommended,omitempty"`
}

const hfWhisper = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/"

// Known models, best Turkish first. The benchmark cited for the Turkish figures
// covers 1,060 clips over 20 models and is published by the same group that
// trained the fine-tune, so the comparison between those rows is like for like.
var Catalog = []CatalogEntry{
	{
		File:      "ggml-large-v3-turkish-general-q5_0.bin",
		Display:   "large-v3 Turkish (general)",
		Publisher: "TurkMedSTT",
		License:   "apache-2.0",
		Notes: "whisper large-v3 adapted to Turkish with LoRA over ~140 h and merged back. " +
			"Much more accurate on Turkish, but roughly four times slower than turbo, and it " +
			"was trained mostly on read speech, so test it on your own recordings.",
		WER: 0.078, WERSource: "turkmedstt/turkish-asr-benchmark, 1060 clips",
		RTF:     0.146,
		Bytes:   1_080_000_000,
		Source:  "https://huggingface.co/turkmedstt/whisper-large-v3-turkish-general",
		Install: "make model-turkish",
	},
	{
		File:      "ggml-buzzasr-turkish-q5_0.bin",
		Display:   "BuzzASR Turkish",
		Publisher: "BuzzASR (LEMN Lab)",
		License:   "mit",
		Notes: "whisper large-v3 fine-tuned for Turkish, from a 102-language academic suite. " +
			"Its published gain over stock large-v3 is modest, and it was trained on FLEURS and " +
			"Common Voice, which are read speech rather than lectures.",
		WER: 0.0799, WERSource: "own card, FLEURS test set — not comparable with the rows below",
		Bytes:       1_030_000_000,
		Source:      "https://huggingface.co/BuzzASR/turkish",
		Install:     "make model-buzz",
		Recommended: true,
	},
	{
		File:      "ggml-large-v3.bin",
		Display:   "large-v3",
		Publisher: "OpenAI",
		License:   "mit",
		Notes:     "The full model. More accurate than turbo on Turkish, and slower.",
		WER:       0.135, WERSource: "turkmedstt/turkish-asr-benchmark, 1060 clips",
		RTF:   0.135,
		Bytes: 3_095_033_483,
		URL:   hfWhisper + "ggml-large-v3.bin",
	},
	{
		File:      "ggml-large-v3-turbo.bin",
		Display:   "large-v3-turbo",
		Publisher: "OpenAI",
		License:   "mit",
		Notes:     "Distilled decoder: much faster, some accuracy given up.",
		WER:       0.201, WERSource: "turkmedstt/turkish-asr-benchmark, 1060 clips",
		RTF:   0.034,
		Bytes: 1_624_555_275,
		URL:   hfWhisper + "ggml-large-v3-turbo.bin",
	},
	{
		File:      "ggml-large-v3-turbo-q5_0.bin",
		Display:   "large-v3-turbo (q5_0)",
		Publisher: "OpenAI",
		License:   "mit",
		Notes:     "Quantised turbo. The default: fastest, smallest, and adequate on clean speech.",
		WER:       0.201, WERSource: "turkmedstt/turkish-asr-benchmark, 1060 clips (unquantised turbo)",
		RTF:   0.034,
		Bytes: 574_041_195,
		URL:   hfWhisper + "ggml-large-v3-turbo-q5_0.bin",
	},
	{
		File:      "ggml-silero-v5.1.2.bin",
		Display:   "Silero VAD v5.1.2",
		Publisher: "Silero",
		License:   "mit",
		Notes:     "Finds the speech, so silence is never transcribed. Not a transcription model.",
		Bytes:     885_098,
		URL:       "https://huggingface.co/ggml-org/whisper-vad/resolve/main/ggml-silero-v5.1.2.bin",
		VAD:       true,
	},
}

// Index is the model's position in the curated order, or -1 when unknown. The
// list is arranged best-for-Turkish first, which is more useful to someone
// choosing than alphabetical order would be.
func Index(file string) int {
	for i, e := range Catalog {
		if strings.EqualFold(e.File, file) {
			return i
		}
	}
	return -1
}

// Lookup returns what is known about a model file.
func Lookup(file string) (CatalogEntry, bool) {
	for _, e := range Catalog {
		if strings.EqualFold(e.File, file) {
			return e, true
		}
	}
	return CatalogEntry{}, false
}
