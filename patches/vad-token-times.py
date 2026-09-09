#!/usr/bin/env python3
"""Make whisper-cli's JSON output report token times on the original timeline.

With --vad, whisper.cpp transcribes a copy of the audio with the silence cut
out. Segment times are mapped back to the real recording by
whisper_full_get_segment_t0/t1_from_state, but the JSON writer reads token times
straight off whisper_full_get_token_data, which is still on the compressed
timeline. The error is the total silence removed so far, so it grows through the
file -- eleven seconds by the middle of a 33-minute lecture.

The library already exposes the mapped values as whisper_full_get_token_t0/t1.
This patch makes the JSON writer use them. Tokens are merged for multi-byte
UTF-8, so it also tracks which original token index each merged entry starts and
ends at, since the mapped accessors take an index rather than a value.

Idempotent: running it twice is a no-op.
"""
import sys
from pathlib import Path

MARK = "// patched: VAD-mapped token times"

OLD_STRUCT = """                        struct merged_token {
                            std::string        text;
                            whisper_token_data data;
                            int64_t            t1;
                        };"""

NEW_STRUCT = """                        struct merged_token {
                            std::string        text;
                            whisper_token_data data;
                            int64_t            t1;
                            int                j0; %s
                            int                j1;
                        };""" % MARK

OLD_LOOP = """                        for (int j = 0; j < n; ) {
                            auto tok = whisper_full_get_token_data(ctx, i, j);
                            merged_token m{ whisper_token_to_str(ctx, tok.id), tok, tok.t1 };
                            ++j;
                            while (j < n && utf8_trailing_bytes_needed(m.text) > 0) {
                                auto tok_next = whisper_full_get_token_data(ctx, i, j);
                                m.text += whisper_token_to_str(ctx, tok_next.id);
                                if (tok_next.t1 > -1) {
                                    m.t1 = tok_next.t1;
                                }
                                ++j;
                            }
                            merged.push_back(std::move(m));
                        }"""

NEW_LOOP = """                        for (int j = 0; j < n; ) {
                            const int j_first = j;
                            auto tok = whisper_full_get_token_data(ctx, i, j);
                            merged_token m{ whisper_token_to_str(ctx, tok.id), tok, tok.t1, j_first, j_first };
                            ++j;
                            while (j < n && utf8_trailing_bytes_needed(m.text) > 0) {
                                auto tok_next = whisper_full_get_token_data(ctx, i, j);
                                m.text += whisper_token_to_str(ctx, tok_next.id);
                                if (tok_next.t1 > -1) {
                                    m.t1 = tok_next.t1;
                                    m.j1 = j;
                                }
                                ++j;
                            }
                            merged.push_back(std::move(m));
                        }"""

OLD_EMIT = """                                if (mt.data.t0 > -1 && mt.t1 > -1) {
                                    // If we have per-token timestamps, write them out
                                    times_o(mt.data.t0, mt.t1, false);
                                }"""

NEW_EMIT = """                                if (mt.data.t0 > -1 && mt.t1 > -1) {
                                    // Mapped back onto the original recording, which
                                    // matters whenever --vad removed silence.
                                    times_o(whisper_full_get_token_t0(ctx, i, mt.j0),
                                            whisper_full_get_token_t1(ctx, i, mt.j1), false);
                                }"""


def main(path: Path) -> int:
    src = path.read_text()
    if MARK in src:
        print(f"already patched: {path}")
        return 0
    for old, new, what in (
        (OLD_STRUCT, NEW_STRUCT, "merged_token struct"),
        (OLD_LOOP, NEW_LOOP, "token merge loop"),
        (OLD_EMIT, NEW_EMIT, "token timestamp output"),
    ):
        if src.count(old) != 1:
            print(f"error: could not find the {what} in {path}", file=sys.stderr)
            print("       upstream has changed; review the patch by hand", file=sys.stderr)
            return 1
        src = src.replace(old, new)
    path.write_text(src)
    print(f"patched: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(Path(sys.argv[1])))
