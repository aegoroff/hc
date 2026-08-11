#!/usr/bin/env python3
"""Generate kernels/NAME.cl.h from kernels/NAME.cl (string-escape join)."""
import sys
from pathlib import Path

def gen_inc(cl_path: Path) -> None:
    text = cl_path.read_text()
    out_lines = []
    for line in text.splitlines(True):
        esc = line.replace("\\", "\\\\").replace('"', '\\"')
        if esc.endswith("\n"):
            esc = esc[:-1] + "\\n"
        out_lines.append(f'"{esc}"\\')
    inc = cl_path.with_suffix(".cl.h")
    # last line should not have trailing \ for C string concat? Looking at existing:
    # every line ends with \ including last - and then closing "; in the .c include
    # Existing format: each line is "....\n"\
    # Final line also has \
    # The C file does: static const char k[] = \n #include "x.cl.h" \n ;
    if out_lines:
        # keep trailing backslash on all lines as existing files do
        pass
    inc.write_text("\n".join(out_lines) + "\n")
    print(f"wrote {inc} ({len(out_lines)} lines)")

if __name__ == "__main__":
    for p in sys.argv[1:]:
        gen_inc(Path(p))
