# ai_transcript

Local Turkish transcript pipeline. Design and rationale:
[`helper_programs/plans/turkish-transcript-pipeline.md`](helper_programs/plans/turkish-transcript-pipeline.md).

Current state: **Stage 1 only** — media decoded to the PCM WAV that whisper.cpp
requires. The ASR and correction stages are not built yet.

## Layout

```
cmd/transcript/       the program (local web UI)
internal/audio/       ffmpeg + ffprobe wrapper
internal/web/         HTTP server, SSE, embedded UI page
audios/               default audio library
videos/               default video library
work/wav/audio/       WAVs from the audio library
work/wav/video/       WAVs from the video library
binaries/windows/     vendored ffmpeg for Windows
helper_programs/      yt_downloader, plans
```

## Run

```
go run ./cmd/transcript
```

It serves <http://127.0.0.1:8765> and opens a browser. Flags:

| Flag | Default | |
|---|---|---|
| `-audio` | `audios` | default audio library folder |
| `-video` | `videos` | default video library folder |
| `-output` | `work/wav` | where WAVs go, under `audio/` and `video/` |
| `-addr` | `127.0.0.1:8765` | falls back to a free port if taken |
| `-jobs` | `2` | concurrent ffmpeg processes |
| `-no-open` | | don't launch a browser |

Bound to `127.0.0.1` only, never a public interface. Missing folders are
created on startup, so a first run just works.

### Pages

**Library** lists `audios/` and `videos/` separately, plus a third list for
files picked from anywhere else. **Jobs** shows conversions with live progress.
**Folders** points the three directories somewhere else.

### Using files from outside the libraries

Nothing is ever uploaded or copied &mdash; media is read where it already lives.
Three ways to point at a file:

- **Browse&hellip;** &mdash; an in-page filesystem browser. The server reads the
  directory and returns real paths.
- **Paste a path** into the box. `~` is expanded. A folder adds the convertible
  media directly inside it.
- **Drag a file in** from your file manager.

The picker exists because of a browser limitation worth knowing: neither
`<input type="file">` nor a drop will tell JavaScript a real filesystem path
&mdash; a browser offers only the file *contents*. Drag-and-drop works here only
when the file manager also puts a `file://` URI in the drag data, which GTK and
KDE file managers usually do and Windows Explorer usually does not. When the URI
is missing, the page says so and points at Browse instead, rather than silently
failing.

## Build for Windows

No cgo anywhere, so this cross-compiles from Linux:

```
CGO_ENABLED=0 GOOS=windows GOARCH=amd64 go build -o transcript.exe ./cmd/transcript
```

That is the reason the UI is a localhost web page rather than a native window —
every Go GUI toolkit (Wails, Fyne) needs cgo and a per-platform build host.

## Requirements

`ffmpeg`, plus `ffprobe` for progress percentages. Resolution order:

1. `AI_TRANSCRIPT_FFMPEG` / `AI_TRANSCRIPT_FFPROBE`
2. next to the executable, and `<exe dir>/binaries/<goos>/`
3. `./binaries/<goos>/` and the working directory
4. `PATH`

So Windows picks up `binaries/windows/` automatically, and Linux uses the
package-manager copy. A missing `ffprobe` is not fatal — progress bars just go
indeterminate.

## Output format

`pcm_s16le`, **16 kHz, mono**, video and metadata stripped, optional single-pass
EBU R128 loudness normalisation.

Output goes to `<output>/audio/` or `<output>/video/` by source kind, so a
`lecture.mp4` and a `lecture.opus` cannot overwrite each other's WAV.

Those are not preferences. `whisper-cli` reads only 16-bit PCM WAV, and Whisper
works on 16 kHz mono samples, so anything else either errors or silently
transcribes badly.

Decoding Opus to PCM loses nothing further — it is a decode, not a re-encode,
and PCM is uncompressed. Keep the `.opus` as the archival source; `work/wav/` is
a cache. The WAV is worth keeping around, though: the plan's Stage 4 re-decodes
short suspect spans, and in 16 kHz mono PCM a span is byte-offset arithmetic
(`44 + t*32000`) rather than a seek into a compressed stream.

Expect roughly **63 MB of WAV per 33 minutes** of audio.
