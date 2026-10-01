"""Resolve a corpus key to a cached .npz path. ONE implementation, imported by every consumer.

WHY THIS FILE EXISTS. The corpus list uses flat SuiteSparse keys (`ss/lhr71/lhr71`). redwood's cache
is built that way; MKT's is built with the SuiteSparse GROUP in the path
(`ss/Mallya/lhr71/lhr71.npz`). Three separate consumers each did their own
`os.path.exists(cache + key + ".npz")`:

  * mkkeys_split.py  -- silently skipped 50 of 298 keys on MKT for months, counting but never
                        describing them; the 50 were every real-world FEM/circuit matrix and took
                        the entire 16-32 mean-degree band to zero, which is why a two-host
                        confirmation passed over a 12% width-1 regression living in that band.
  * a stage's band census -- reported the 16-32 band EMPTY on a corpus the selector had just
                        resolved 37 target matrices from, and refused at its own gate.
  * mrgate.py        -- refused 24 of 61 keys AFTER eight timing passes had run, wasting the job.

Each fix in one place moved the failure to the next place. So the resolution lives here and nowhere
else, and a consumer that cannot import it must refuse rather than fall back to the exact path --
falling back is precisely the bug.
"""
import os

_INDEX = {}


def _index_for(cache):
    """basename -> [paths], built once per cache directory."""
    idx = _INDEX.get(cache)
    if idx is None:
        idx = {}
        for root, _dirs, files in os.walk(cache):
            for f in files:
                if f.endswith(".npz"):
                    idx.setdefault(f[:-4], []).append(os.path.join(root, f))
        _INDEX[cache] = idx
    return idx


def resolve(cache, key):
    """(path, how). how is 'exact', 'basename', 'absent' or 'ambiguous(n)'.

    Exact path first, so a cache matching the corpus list costs nothing and behaves as before.
    Then a UNIQUE basename match; an ambiguous one is refused rather than guessed, because picking
    one of several same-named matrices silently would be worse than not resolving at all.
    """
    p = os.path.join(cache, key + ".npz")
    if os.path.exists(p):
        return p, "exact"
    hits = _index_for(cache).get(key.split("/")[-1], [])
    if len(hits) == 1:
        return hits[0], "basename"
    if len(hits) > 1:
        return None, "ambiguous(%d)" % len(hits)
    return None, "absent"


def resolve_all(cache, keys):
    """(paths_by_key, how_counts, unresolved) over an iterable of keys."""
    paths, how, bad = {}, {}, []
    for k in keys:
        p, h = resolve(cache, k)
        how[h] = how.get(h, 0) + 1
        if p is None:
            bad.append((k, h))
        else:
            paths[k] = p
    return paths, how, bad


def describe(how, bad, label="keys"):
    """Lines a caller should print. The COMPOSITION of a drop is the finding, not its count."""
    out = ["  %s resolution: %s" % (label, ", ".join("%s %d" % kv for kv in sorted(how.items())))]
    if how.get("basename"):
        out.append("  NOTE: %d resolved only by basename -- this cache's path convention differs "
                   "from the corpus list" % how["basename"])
    if bad:
        from collections import Counter
        grp = Counter(k.split("/")[1] if k.startswith("ss/") and len(k.split("/")) > 2
                      else k.split("/")[0] for k, _ in bad)
        out.append("  UNRESOLVED %d, and what they are made of:" % len(bad))
        out.append("        by top group: %s" % dict(grp.most_common(8)))
        out.append("        by reason:    %s" % dict(Counter(w for _, w in bad)))
    return out


def read_keys(path):
    """The corpus file: one key per line, blanks skipped. Here so no consumer re-parses it."""
    with open(path) as fh:
        return [ln.strip() for ln in fh if ln.strip()]
