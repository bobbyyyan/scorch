#!/usr/bin/env python3
"""Every key in a collection's cache, one per line, sorted -- the frame screen_draw.py samples from.
usage: full_keys.py <cache_dir> <corpus> <out_file>"""
import os, sys
d = os.path.join(sys.argv[1], sys.argv[2]); ks = []
for root, _, files in os.walk(d):
    for f in files:
        if f.endswith(".npz"):
            ks.append(os.path.relpath(os.path.join(root, f[:-4]), d))
ks = sorted(set(ks)); open(sys.argv[3], "w").write("\n".join(ks) + "\n"); print(sys.argv[2], len(ks))
