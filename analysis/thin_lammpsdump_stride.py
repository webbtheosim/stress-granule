#!/usr/bin/env python3
"""Keep every Nth frame from a LAMMPS dump without loading the file into memory.

Streaming frame-stride utility for LAMMPS trajectory dumps. Reads the input
line by line, detecting frame boundaries by the ``ITEM: TIMESTEP`` marker, and
copies through only the kept frames so arbitrarily large dumps are thinned with
constant memory.

Role: a general-purpose trajectory preprocessing tool (no figure of its own).

Key inputs (CLI flags):
    --input   path to the source LAMMPS dump
    --output  path for the thinned dump
    --stride  keep every Nth frame (must be > 0)
    --offset  frame offset applied before striding (default 0 keeps 0, N, 2N, ...)

Key outputs: the thinned dump at ``--output``; a one-line summary to stdout.

Exact CLI invocation:
    python thin_lammpsdump_stride.py --input IN --output OUT --stride N [--offset K]
"""

import argparse


def main() -> None:
    """Parse CLI args and stream-copy every Nth frame to the output dump.

    Validates ``--stride`` (> 0) and ``--offset`` (>= 0), then walks the input
    keeping only frames at indices ``offset, offset+stride, offset+2*stride, ...``
    and prints a summary line.
    """
    parser = argparse.ArgumentParser(description="Thin a LAMMPS dump by frame stride.")
    parser.add_argument("--input", required=True, help="Input LAMMPS dump")
    parser.add_argument("--output", required=True, help="Output thinned dump")
    parser.add_argument("--stride", type=int, required=True, help="Keep every Nth frame")
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Frame offset before applying stride (default: 0 keeps frames 0, N, 2N, ...)",
    )
    args = parser.parse_args()

    if args.stride <= 0:
        raise ValueError("--stride must be > 0")
    if args.offset < 0:
        raise ValueError("--offset must be >= 0")

    frame_idx = -1
    keep = False

    with open(args.input, "r") as fin, open(args.output, "w") as fout:
        for line in fin:
            if line.startswith("ITEM: TIMESTEP"):
                frame_idx += 1
                keep = frame_idx >= args.offset and ((frame_idx - args.offset) % args.stride == 0)
                if keep:
                    fout.write(line)
                continue
            if keep:
                fout.write(line)

    print(
        "Wrote thinned dump: input={} output={} stride={} offset={}".format(
            args.input, args.output, args.stride, args.offset
        )
    )


if __name__ == "__main__":
    main()
