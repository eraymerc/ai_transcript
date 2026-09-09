# Windows binaries — provenance

Linux users install ffmpeg via their package manager; only Windows binaries are vendored here.

## ffmpeg / ffprobe

| | |
|---|---|
| Build | `ffmpeg-n9.0.1-27-g9b0578816c-win64-lgpl-9.0` |
| Source of build | https://github.com/BtbN/FFmpeg-Builds (release tag `autobuild-2026-09-09-14-51`) |
| License | **LGPL** (chosen over the GPL build so the bundle stays LGPL-only) |
| Linkage | static — single `.exe`, no DLLs to ship |
| Arch | win64 (x86-64) |
| Retrieved | 2026-09-09 |

SHA-256:

```
c09c9818d91357e6c8ecbba6d832377ca309690dda35baac6019e716070bd934  ffmpeg.exe
02739830b1ffeb4a8357de1b069cbee81df1c9654ede15a35d53069f7a372291  ffprobe.exe
```

Upstream FFmpeg license text: `FFMPEG-LICENSE.txt`.

## LGPL compliance notes

These are unmodified upstream builds, invoked as separate processes (never linked into our
binary), so our own code is not a derivative work. What redistribution still requires:

1. Ship `FFMPEG-LICENSE.txt` alongside the binaries in the installer.
2. Offer the corresponding source: FFmpeg at tag `n9.0.1` plus the BtbN build scripts at
   release tag `autobuild-2026-09-09-14-51`. A link in the app's About/licenses screen is
   sufficient; record the exact tags above so the offer stays accurate.
3. Do not switch to a GPL build — it would make the whole distributed bundle GPL.

## Size

~132 MB each, and they are already stripped. This is the dominant fixed cost of the
installer after the models.

Reduction options, in order of effort:

- **Drop `ffprobe.exe`** (−132 MB). Duration and stream info can be read from `ffmpeg -i`
  stderr instead. Do this if the installer size becomes a problem.
- **Build a minimal ffmpeg** with `--disable-everything` plus only the audio decoders,
  demuxers, the `aresample`/`loudnorm` filters, and the WAV muxer. Realistically gets under
  ~5 MB. Requires setting up an MSYS2/mingw-w64 cross-build in CI — worth doing before a
  public release, not before the pipeline works.
